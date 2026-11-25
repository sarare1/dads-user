"""Training a sequence model (background job with cross-validation) and testing it."""

import asyncio
import copy
import hashlib
import io
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.data.generator import frequency_to_band
from src.data.sequence_features import (
    cv_split, DEFAULT_SEQ_FEATURES, DEFAULT_WINDOW, load_sequence_dataset_for_training, order_columns,
    required_seq_columns, SEQ_FEATURES, sequence_categorical_columns, sequence_dataset_columns, upgrade_columns,
    validate_seq_features, WINDOW_MODES,
)
from src.data.sequence_generator import build_sequence_dataset_rows
from src.models.checkpoint import load_sequence_checkpoint, save_sequence_checkpoint
from src.models.explain import branch_ablation, evaluation_report
from src.models.sequence_eval import (
    calibrate_distance_threshold, evaluate_tracks, relabel_by_name, sequence_architecture_report,
    sequence_feature_importance,
)
from src.models.sequence_models import SEQ_BACKBONES, SequenceDualHeadModel
from src.storage.training_store import get_all_runs, get_run, get_saved_runs, insert_run, insert_test_run
from src.training import cancel
from src.training.cancel import TrainingCancelled
from src.training.trainer import build_distance_fn, seed_training, train_model
from src.api.core import (
    _begin_job, _end_job, _explain_error, _public_error, _set_phase, APP_VERSION, current_config, datasets,
    EMBEDDING_DIM_RANGE, jobs, log, registry, training_status,
)
from src.api.common import (
    _check_run_dataset, _check_sequence_file, _ensure_class_ids, _load_sequence_dataset, _population_split,
    _read_upload, _reused_seed_and_ranges, _simulator_gap, SEQUENCE_TRAIN_CHECKPOINT, TEST_RESULTS_DIR,
)
router = APIRouter()


class TrainSequenceRequest(BaseModel):
    epochs: int = 20
    backbone_type: str = "seq_hybrid"
    features: Optional[List[str]] = None
    window: int = DEFAULT_WINDOW
    # "windows": overlapping windows of k = `window` pulses, voted per track. "track": no windowing - ONE input per
    # track = its first `window` pulses (the cap; 0 = no cap, up to MAX_TRACK_PULSES); shorter tracks are padded + masked.
    window_mode: str = "windows"
    # Each sequence model gets its own decision thresholds. τ_d: None = calibrate from validation so `known_coverage` of known
    # windows are accepted; a number = use it as given. τ_c: None = the Training form's value.
    classifier_confidence_threshold: Optional[float] = None
    ood_distance_threshold: Optional[float] = None
    known_coverage: float = 0.95
    embedding_dim: Optional[int] = None  # None = the Training form's embedding size
    dataset: Optional[str] = None        # name of the dataset to train on; None = the active one
    # Cross-validation: test_pct % of the tracks of every family are set aside as the TEST split; the rest is cut into
    # cv_folds buckets, each fold validating on one and training on the others. cross_validation=False trains fold 1 only.
    cross_validation: bool = True
    cv_folds: int = 4
    test_pct: float = 20.0
    include_test: bool = False          # also train on the test split (then it is no independent test)
    seed: Optional[int] = None          # training seed (weights, batch order, crops); None = a random one, recorded with the run


def _cv_report(seq_model, spec, df, test_ids, fold_results, best_fold, req, run_folds, centroids, distance_fn, hp,
               kept_split) -> dict:
    """Cross-validation summary stored with the run: per-fold validation results, the kept fold, and the
    held-out TEST split (its track ids - the Testing page re-uses them - and the kept model's result on it)."""
    accs = np.array([r["closed_set_accuracy"] for r in fold_results], dtype=float)
    kept = fold_results[best_fold] if len(run_folds) > 1 else fold_results[0]
    out = {
        "folds": len(run_folds), "cross_validation": bool(req.cross_validation), "test_pct": req.test_pct,
        "include_test": bool(req.include_test), "best_fold": best_fold + 1,
        "per_fold": fold_results, "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "auroc_mean": _mean_or_none([r["open_set_auroc"] for r in fold_results]),
        # The kept fold was chosen BECAUSE its validation score was highest, so its own validation numbers
        # are optimistic; the run's headline metrics are the means over the folds (and the test result below).
        "kept_fold_validation": {k: kept.get(k) for k in ("closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl")},
        "headline_metrics": "mean over the cross-validation folds" if len(run_folds) > 1 else "the single split's validation",
        "test_instances": sorted(str(t) for t in test_ids),
        "kept_fold_split": {k: sorted(str(t) for t in ids) for k, ids in zip(("train", "validation", "calibration"), kept_split[:3])},
        "kept_model": f"fold {best_fold + 1} (highest validation accuracy)" if len(run_folds) > 1 else "the single split",
    }
    test_df = df[df["instance"].isin(test_ids)]
    if len(test_df):
        try:
            res = evaluate_tracks(seq_model, spec, test_df, centroids, distance_fn,
                                  hp["classifier_confidence_threshold"], hp["ood_distance_threshold"])
            out["test_result"] = {k: res.get(k) for k in ("closed_set_accuracy", "open_set_auroc", "track_accuracy",
                                                           "num_windows_known", "num_windows_unknown")}
        except Exception:
            log.exception("test split evaluation failed")
    return out


def _mean_or_none(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


async def _run_sequence_training_job(req: TrainSequenceRequest, names: List[str]):
    training_status["is_training"] = True
    training_status["progress"] = {"epoch": 0, "epochs": req.epochs, "loss": None, "dataset": req.dataset}
    training_status["error"] = None
    _set_phase("Loading the dataset")
    start_time = time.time()
    try:
        df, meta = await asyncio.to_thread(_load_sequence_dataset, req.dataset)
        if df is None:
            raise RuntimeError("No sequence dataset — generate or upload one on the Dataset page first.")
        training_status["progress"]["dataset"] = meta["name"]
        dataset_sha256 = await asyncio.to_thread(datasets.content_hash, meta["name"])  # recorded with the run
        missing = [c for c in required_seq_columns(names) if c not in df.columns]
        if missing:
            raise RuntimeError(f"The sequence dataset has no column(s) {missing} needed by the selected features.")
        num_classes = meta["num_classes"]
        cfg = copy.deepcopy(current_config)  # this run's own copy of the shared settings
        cfg["model"]["num_classes"] = num_classes
        if req.embedding_dim is not None:
            cfg["model"]["embedding_dim"] = int(req.embedding_dim)
        _set_phase("Splitting tracks into train / validation / test")
        test_ids, fold_splits = await asyncio.to_thread(
            cv_split, df, meta["dataset_seed"], req.test_pct, req.cv_folds, 0.1, req.include_test)
        run_folds = list(range(req.cv_folds)) if req.cross_validation else [0]
        train_seed = int(req.seed) if req.seed is not None else int(np.random.randint(2 ** 31 - 1))
        fold_results, best = [], None
        for n_done, f in enumerate(run_folds):
            _set_phase(f"Building windows (fold {n_done + 1}/{len(run_folds)})")
            cancel.check()
            fold_dataset = await asyncio.to_thread(load_sequence_dataset_for_training, df, names, req.window,
                                                   meta["dataset_seed"], 0.2, None, 0.1, fold_splits[f], req.window_mode)
            fold_seed = train_seed + f  # each fold repeatable on its own
            seed_training(fold_seed)    # before the model is built: its initial weights are part of the run
            fold_model = SequenceDualHeadModel(req.backbone_type, fold_dataset["feature_spec"].dim,
                                               cfg["model"]["embedding_dim"], num_classes)

            _set_phase("Training")

            def progress_cb(epoch, epochs, loss, n_done=n_done):
                training_status["progress"] = {"epoch": epoch, "epochs": epochs, "loss": loss, "dataset": meta["name"],
                                               "fold": n_done + 1, "folds": len(run_folds)}

            fold_metrics = await asyncio.to_thread(train_model, fold_model, fold_dataset, cfg, req.epochs, 64, progress_cb,
                                                   fold_seed)
            fold_results.append({
                "fold": f + 1, "seed": fold_seed, "closed_set_accuracy": fold_metrics["closed_set_accuracy"],
                "open_set_auroc": fold_metrics["open_set_auroc"], "open_set_auroc_rpl": fold_metrics.get("open_set_auroc_rpl"),
                "final_loss": fold_metrics["loss_curve"][-1],
                "train_tracks": len(fold_splits[f][0]), "validation_tracks": len(fold_splits[f][1]),
                "calibration_tracks": len(fold_splits[f][2]),
            })
            auroc = fold_metrics["open_set_auroc"]
            score = (fold_metrics["closed_set_accuracy"], -1.0 if auroc is None else auroc)
            if best is None or score > best[0]:
                best = (score, f, fold_model, fold_dataset, fold_metrics)
        cancel.check()  # last point a stop discards the run - after this it is saved
        _set_phase("Calibrating thresholds and saving the run")
        _, best_fold, seq_model, dataset, metrics = best
        spec = dataset["feature_spec"]
        accs = np.array([r["closed_set_accuracy"] for r in fold_results], dtype=float)
        centroids = metrics.pop("centroids")
        hp = cfg["hyperparameters"]
        if req.classifier_confidence_threshold is not None:
            hp["classifier_confidence_threshold"] = req.classifier_confidence_threshold
        if req.ood_distance_threshold is None:
            use_cal = len(dataset.get("cal_x", [])) >= 5  # held-out calibration tracks; else validation windows
            calibration = await asyncio.to_thread(
                calibrate_distance_threshold, seq_model, dataset["cal_x"] if use_cal else dataset["val_x"], centroids,
                build_distance_fn(hp["distance_metric"]), req.known_coverage, dataset.get("cal_short_x"))
            if calibration.get("by_length"):
                calibration["by_length_calibrated_on"] = f"the first L pulses of each {dataset.get('cal_short_source', 'calibration')} track"
            calibration["calibrated_on"] = (f"{dataset['num_cal_tracks']} held-out calibration tracks" if use_cal
                                            else "validation windows (too few tracks for a calibration split)")
            hp["ood_distance_threshold"] = calibration["ood_distance_threshold"]
            calibration["mode"] = "auto"
        else:
            hp["ood_distance_threshold"] = req.ood_distance_threshold
            calibration = {"mode": "manual", "ood_distance_threshold": req.ood_distance_threshold}
        calibration["classifier_confidence_threshold"] = hp["classifier_confidence_threshold"]
        freq_by_class = df[df["true_class_id"] >= 0].groupby("true_class_id")["Frequency"].mean().to_dict() \
            if "Frequency" in df.columns else {}
        metadata = {c: {"freq_band": frequency_to_band(freq_by_class[c]) if c in freq_by_class else None} for c in centroids}
        save_sequence_checkpoint(SEQUENCE_TRAIN_CHECKPOINT, seq_model, spec)

        hp = cfg["hyperparameters"]
        run_id = insert_run({
            "backbone_type": req.backbone_type, "distance_metric": hp["distance_metric"],
            "num_classes": num_classes, "num_holdout_classes": meta["num_holdout_classes"],
            "samples_per_class": meta.get("tracks_per_class"), "noise_pct": meta.get("noise_track_pct") or 0.0,
            "epochs": req.epochs, "learning_rate": hp["learning_rate"], "triplet_margin": hp["triplet_margin"],
            "classifier_confidence_threshold": hp["classifier_confidence_threshold"],
            "ood_distance_threshold": hp["ood_distance_threshold"],
            "rpl_loss_weight": hp.get("rpl_loss_weight", 0.1), "rpl_adversarial": metrics.get("rpl_adversarial", False),
            # Headline = mean over the folds: the kept fold's own validation score is the maximum of several,
            # so reporting it would overstate the model (selection bias). Kept-fold values are in the report.
            "closed_set_accuracy": round(float(accs.mean()), 4),
            "open_set_auroc": _mean_or_none([r["open_set_auroc"] for r in fold_results]),
            "open_set_auroc_rpl": _mean_or_none([r["open_set_auroc_rpl"] for r in fold_results]),
            "final_loss": metrics["loss_curve"][-1], "train_seed": train_seed,
            "training_duration_seconds": round(time.time() - start_time, 2), "loss_curve": metrics["loss_curve"],
            "dataset_seed": meta["dataset_seed"], "features": ",".join(names),
            "model_kind": "sequence", "window_k": spec.window, "window_mode": spec.mode, "pulses_per_track": meta.get("pulses_per_track"),
            "embedding_dim": seq_model.classifier_head.in_features,
            "missed_pct": meta.get("missed_pct"), "spurious_pct": meta.get("spurious_pct"),
            **meta["ranges"],
            "include_pgri": bool(meta.get("include_pgri", False)),
            "balanced_patterns": bool(meta.get("balanced_patterns", False)),
            "dataset_name": meta["name"], "dataset_sha256": dataset_sha256, "app_version": APP_VERSION,
            "cv_folds": len(run_folds), "test_pct": req.test_pct, "include_test": bool(req.include_test),
            "cv_accuracy_mean": float(accs.mean()), "cv_accuracy_std": float(accs.std()),
        })
        _set_phase("Building the report")
        try:
            distance_fn = build_distance_fn(hp["distance_metric"])
            # The kept fold's validation tracks + its open-set pool (the test split's unknown tracks stay out).
            open_pool = set(fold_splits[best_fold][3])
            eval_df = df[df["instance"].isin(set(dataset["val_track_ids"])) | df["instance"].isin(open_pool)]
            report = await asyncio.to_thread(lambda: {
                "architecture": sequence_architecture_report(seq_model, spec, cfg),
                "feature_importance": sequence_feature_importance(seq_model, dataset, spec),
                "evaluation": evaluation_report(seq_model, dataset, distance_fn, centroids,
                                                hp["ood_distance_threshold"], hp["classifier_confidence_threshold"]),
                "sequence": evaluate_tracks(seq_model, spec, eval_df, centroids, distance_fn,
                                            hp["classifier_confidence_threshold"], hp["ood_distance_threshold"]),
                "thresholds": calibration,
                "window_input": {"mode": spec.mode, "window": spec.window,
                                 "known_tracks_without_input": dataset.get("known_tracks_without_input", 0)},
                "class_labels": meta.get("class_labels") or [f"RADAR_FAMILY_{c:02d}" for c in range(num_classes)],
                "branch_ablation": branch_ablation(seq_model, dataset, distance_fn, centroids),  # seq_hybrid only
                "cross_validation": _cv_report(seq_model, spec, df, test_ids, fold_results, best_fold, req, run_folds,
                                               centroids, distance_fn, hp, fold_splits[best_fold]),
            })
        except Exception:  # a report failure must never lose a completed training run
            log.exception("training report failed")
            report = {"error": "Report could not be generated (see the server log)."}
        registry.save_artifacts(run_id, SEQUENCE_TRAIN_CHECKPOINT, None, centroids, metadata, report)
        training_status["last_metrics"] = metrics
        training_status["last_run_id"] = run_id
        registry.prune([r["id"] for r in get_all_runs(100_000)], [r["id"] for r in get_saved_runs()])
        _end_job("succeeded")
    except TrainingCancelled:
        log.info("training stopped by the user")
        _end_job("cancelled", message="Training stopped. Nothing was saved; the model in use is unchanged.")
    except Exception as e:
        log.exception("training job failed")
        _end_job("failed", _explain_error(e))
    finally:
        if training_status["is_training"]:
            _end_job("failed", "The training job ended unexpectedly. See the server log.")


@router.post("/api/sequence/train")
async def train_sequence(req: TrainSequenceRequest):
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "Training already in progress"}, status_code=409)
    if req.backbone_type not in SEQ_BACKBONES:
        return JSONResponse({"status": "error", "message": f"Unknown sequence backbone; choose from {list(SEQ_BACKBONES)}."}, status_code=400)
    if req.window_mode not in WINDOW_MODES:
        return JSONResponse({"status": "error", "message": f"Window mode must be one of {list(WINDOW_MODES)}."}, status_code=400)
    if req.window_mode == "track" and req.window == 0:
        pass  # whole track, no cap
    elif not 2 <= req.window <= 512:
        return JSONResponse({"status": "error", "message": "Window length must be 2–512 pulses."}, status_code=400)
    if not 1 <= req.epochs <= 500:
        return JSONResponse({"status": "error", "message": "Epochs must be 1–500."}, status_code=400)
    if not 2 <= req.cv_folds <= 10:
        return JSONResponse({"status": "error", "message": "Cross-validation folds must be 2–10."}, status_code=400)
    if not 0 <= req.test_pct <= 50:
        return JSONResponse({"status": "error", "message": "The test share must be 0–50 %."}, status_code=400)
    if req.embedding_dim is not None and not EMBEDDING_DIM_RANGE[0] <= req.embedding_dim <= EMBEDDING_DIM_RANGE[1]:
        return JSONResponse({"status": "error", "message": f"Embedding size must be {EMBEDDING_DIM_RANGE[0]}–{EMBEDDING_DIM_RANGE[1]}."}, status_code=400)
    if not 0.5 <= req.known_coverage <= 0.999:
        return JSONResponse({"status": "error", "message": "Known-window acceptance must be between 50% and 99.9%."}, status_code=400)
    if req.classifier_confidence_threshold is not None and not 0 < req.classifier_confidence_threshold < 1:
        return JSONResponse({"status": "error", "message": "Min classifier confidence must be between 0 and 1."}, status_code=400)
    if req.ood_distance_threshold is not None and not req.ood_distance_threshold > 0:
        return JSONResponse({"status": "error", "message": "The distance threshold must be greater than 0."}, status_code=400)
    if req.seed is not None and not 0 <= req.seed <= 2 ** 31 - 100:
        return JSONResponse({"status": "error", "message": "The training seed must be between 0 and 2147483547."}, status_code=400)
    job_id = _begin_job("Checking the dataset")  # claim the slot before any slow work (no double start)
    try:
        df_cols, _ = await asyncio.to_thread(_load_sequence_dataset, req.dataset)
        if req.dataset and df_cols is None:
            _end_job("failed", f"Dataset '{req.dataset}' not found.")
            return JSONResponse({"status": "error", "message": f"Dataset '{req.dataset}' not found."}, status_code=404)
        available = (sequence_dataset_columns(df_cols) + sequence_categorical_columns(df_cols)) if df_cols is not None else []
        wanted = req.features or ([n for n in DEFAULT_SEQ_FEATURES if all(c in df_cols.columns for c in SEQ_FEATURES[n].columns)]
                                  if df_cols is not None else DEFAULT_SEQ_FEATURES)
        names = validate_seq_features(wanted, available)
    except ValueError as e:
        _end_job("failed", _public_error(e))
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    except Exception as e:  # e.g. an unreadable / corrupt CSV - say so instead of a silent hang
        log.exception("could not read the dataset")
        msg = f"Could not read the dataset: {_explain_error(e)}"
        _end_job("failed", msg)
        return JSONResponse({"status": "error", "message": msg}, status_code=400)
    jobs.training_task = asyncio.create_task(_run_sequence_training_job(req, names))
    return JSONResponse({"status": "started", "job_id": job_id})


class SequenceTestRequest(BaseModel):
    run_id: int
    source: str = "auto"                 # "auto" (fresh tracks of the model's families) | "dataset" (a stored dataset)
    filename: Optional[str] = None       # the dataset's name (blank = the dataset the model was trained on)
    tracks_per_class: int = 20
    pulses_per_track: Optional[int] = None
    missed_pct: Optional[float] = None   # None = the rate the model was trained with
    spurious_pct: Optional[float] = None
    window: Optional[int] = None         # None = the model's training window
    classifier_confidence_threshold: Optional[float] = None  # None = the model's own thresholds
    ood_distance_threshold: Optional[float] = None


@router.post("/api/sequence/test")
def test_sequence_model(req: SequenceTestRequest):
    r = get_run(req.run_id)
    if r is None or r.get("model_kind") != "sequence" or not registry.has_artifacts(req.run_id):
        return JSONResponse({"status": "error", "message": "Choose a kept sequence model."}, status_code=404)
    seq_model, spec, msg = load_sequence_checkpoint(registry.checkpoint_path(req.run_id))
    if seq_model is None:
        return JSONResponse({"status": "error", "message": msg}, status_code=400)
    centroids, _ = registry.load_prototypes(req.run_id)

    if req.window is not None and not 2 <= req.window <= 512:
        return JSONResponse({"status": "error", "message": "Window length must be 2–512 pulses."}, status_code=400)
    model_meta = datasets.meta(r["dataset_name"]) if r.get("dataset_name") and datasets.exists(r["dataset_name"]) else {}
    extras = (model_meta or {}).get("extra_features") or []
    if req.source == "auto" and _simulator_gap(spec, extras):
        return JSONResponse({"status": "error", "message": (
            f"This model uses dataset column(s) {_simulator_gap(spec, extras)} that the track simulator can't produce — "
            f"test it on the sequence dataset or an uploaded track file that has them.")}, status_code=400)
    if req.source == "auto":
        seed, ranges, noise = _reused_seed_and_ranges(r)
        missed = r.get("missed_pct") if req.missed_pct is None else req.missed_pct
        spurious = r.get("spurious_pct") if req.spurious_pct is None else req.spurious_pct
        for v in (missed, spurious):
            if v is not None and not 0 <= v <= 0.5:
                return JSONResponse({"status": "error", "message": "Missed/spurious rates must be between 0 and 0.5."}, status_code=400)
        if not 2 <= req.tracks_per_class <= 500:
            return JSONResponse({"status": "error", "message": "Tracks per class must be 2–500."}, status_code=400)
        pulses = req.pulses_per_track or r.get("pulses_per_track") or 128
        holdout, revealed = _population_split(r)
        rows = build_sequence_dataset_rows(r["num_classes"], holdout, req.tracks_per_class, int(pulses),
                                       seed, ranges, missed or 0.0, spurious or 0.0, noise, int(np.random.randint(1_000_000_000)),
                                       bool(r.get("include_pgri")), bool(r.get("balanced_patterns")), revealed, extras)
        df = pd.DataFrame(rows)
        dataset_name = None
        label = (f"sequence: fresh tracks of the model's families (seed {seed}), missed {(missed or 0) * 100:.0f}%, "
                 f"spurious {(spurious or 0) * 100:.0f}%")
    elif req.source == "test_split":
        cv = (registry.load_report(req.run_id) or {}).get("cross_validation") or {}
        dataset_name = r.get("dataset_name")
        if not cv.get("test_instances"):
            return JSONResponse({"status": "error", "message": "This model was trained without a held-out test split (older run, or test share 0 %)."}, status_code=400)
        try:
            _check_run_dataset(r)
        except ValueError as e:
            return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=409)
        full, _meta = _load_sequence_dataset(dataset_name)
        if full is None:
            return JSONResponse({"status": "error", "message": f"The model's dataset '{dataset_name}' no longer exists."}, status_code=404)
        df = full[full["instance"].astype(str).isin(set(cv["test_instances"]))]
        label = (f"sequence: held-out test split of dataset {dataset_name} ({cv.get('test_pct'):g} % of the tracks)"
                 + (" - ALSO USED IN TRAINING, not independent" if cv.get("include_test") else ""))
    else:
        dataset_name = req.filename or r.get("dataset_name") or datasets.get_active()
        df, _meta = _load_sequence_dataset(dataset_name)
        if df is None:
            return JSONResponse({"status": "error", "message": "Test dataset not found."}, status_code=404)
        error = _check_sequence_file(df)
        if error:
            return JSONResponse({"status": "error", "message": error}, status_code=400)
        label = f"sequence: dataset {dataset_name}" + (" (includes the training tracks)" if dataset_name == r.get("dataset_name") else "")
    missing = [c for c in required_seq_columns(spec.names) if c not in df.columns]
    if missing:
        return JSONResponse({"status": "error", "message": f"The test tracks lack column(s) {missing} this model needs."}, status_code=400)
    if req.source not in ("auto", "test_split"):  # a file's own class ids needn't match the model's — match by label name
        df = relabel_by_name(df, (registry.load_report(req.run_id) or {}).get("class_labels"))
    tau_c = r["classifier_confidence_threshold"] if req.classifier_confidence_threshold is None else req.classifier_confidence_threshold
    tau_d = r["ood_distance_threshold"] if req.ood_distance_threshold is None else req.ood_distance_threshold
    class_labels = (registry.load_report(req.run_id) or {}).get("class_labels")
    try:
        result = evaluate_tracks(seq_model, spec, df, centroids, build_distance_fn(r["distance_metric"]),
                                 tau_c, tau_d, req.window, detail=True, class_labels=class_labels)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)

    record = {
        "backbone_type": r["backbone_type"], "distance_metric": r["distance_metric"],
        "dataset_source": label, "num_known_classes": len(centroids), "num_holdout_classes": r["num_holdout_classes"],
        "closed_set_accuracy": result["closed_set_accuracy"], "open_set_auroc": result["open_set_auroc"],
        "num_known_pulses": result["num_windows_known"], "num_openset_pulses": result["num_windows_unknown"],
        "total_pulses": result["num_windows_known"] + result["num_windows_unknown"],
        "dataset_seed": r.get("dataset_seed") if req.source == "auto" else None, "dataset_name": dataset_name,
        "dataset_sha256": datasets.content_hash(dataset_name) if dataset_name else None,
        "features": r.get("features"), "run_id": req.run_id, "model_name": r.get("saved_name"),
        "model_kind": "sequence", "track_accuracy": result["track_accuracy"],
    }
    test_id = insert_test_run(record)
    result["thresholds"] = {"classifier_confidence_threshold": tau_c, "ood_distance_threshold": tau_d,
                            "overridden": req.classifier_confidence_threshold is not None or req.ood_distance_threshold is not None}
    try:  # the full result (per-emitter tables, per-track rows) is kept so a history row can be reopened
        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
        with open(os.path.join(TEST_RESULTS_DIR, f"test_{test_id}.json"), "w", encoding="utf-8") as f:
            json.dump({"run": dict(record, id=test_id), "result": result}, f)
    except Exception:
        log.exception("could not store the detailed test result")
    return JSONResponse({"status": "success", "test_id": test_id, "result": result, "run": record})


@router.post("/api/sequence/test/upload")
async def upload_sequence_test_file(file: UploadFile = File(...)):
    """A track file for testing becomes a stored dataset too (it is not made the training default)."""
    original = os.path.basename(file.filename or "sequence_test.csv")
    if not original.lower().endswith(".csv"):
        return JSONResponse({"status": "error", "message": "Upload a .csv file."}, status_code=400)
    try:
        contents = await _read_upload(file)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=413)
    return await asyncio.to_thread(_ingest_test_upload, contents, original)


def _ingest_test_upload(contents: bytes, original: str) -> JSONResponse:
    try:
        df = pd.read_csv(io.BytesIO(contents))
        error = _check_sequence_file(df)
    except Exception:
        error = "Could not parse file as CSV."
    if error:
        return JSONResponse({"status": "error", "message": error}, status_code=400)
    name = datasets.save(order_columns(_ensure_class_ids(upgrade_columns(df))), {"source": "uploaded", "purpose": "test", "original_filename": original,
                                               "original_sha256": hashlib.sha256(contents).hexdigest(),
                                               "created_at_utc": datetime.now(timezone.utc).isoformat()},
                         f"test_{os.path.splitext(original)[0]}", activate=False)
    return JSONResponse({"status": "success", "name": name, "size_bytes": len(contents)})

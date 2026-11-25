"""Incremental training: new-family datasets and the methods A-D experiments."""

import asyncio
import copy
import json
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.data.generator import frequency_to_band
from src.data.sequence_features import split_track_ids, upgrade_columns
from src.data.sequence_generator import build_sequence_dataset_rows
from src.models.checkpoint import load_sequence_checkpoint, save_sequence_checkpoint
from src.models.explain import branch_ablation, evaluation_report
from src.models.sequence_eval import sequence_architecture_report, sequence_feature_importance
from src.storage.training_store import _COLUMNS as RUN_COLUMNS, get_all_runs, get_run, get_saved_runs, insert_run
from src.training.cancel import TrainingCancelled
from src.training.incremental import (
    map_labels, METHODS as INCREMENTAL_METHODS, prefix_tracks, run_method as run_incremental_method,
    score as incremental_score,
)
from src.training.trainer import build_distance_fn
from src.api.core import (
    _begin_job, _end_job, _explain_error, _public_error, _set_phase, APP_VERSION, current_config, DATA_DIR,
    DatasetRanges, datasets, jobs, log, MAX_HISTORY_ROWS, registry, training_status,
)
from src.api.common import (
    _ist, _load_sequence_dataset, _model_dataset, _model_extras, _population_split, _reused_seed_and_ranges,
    SEQUENCE_TRAIN_CHECKPOINT,
)
router = APIRouter()


# ---------------------------------------------------------------- incremental training (new families)
INCREMENTAL_DIR = os.path.join(DATA_DIR, "incremental_experiments")


class NewFamiliesRequest(BaseModel):
    run_id: int
    class_ids: Optional[List[int]] = None   # which of the model's held-out families appear; None = the first one
    tracks_per_class: int = 30
    pulses_per_track: Optional[int] = None  # None = as the model's dataset
    name: Optional[str] = None


@router.post("/api/sequence/dataset/new_families")
def generate_new_family_dataset(req: NewFamiliesRequest):
    """'After deployment a new emitter turns up': fresh tracks of families the model was trained
    WITHOUT (its held-out families), now labelled with their own names, plus the remaining held-out
    families and noise as unknown tracks — the new data for incremental training."""
    r = get_run(req.run_id)
    if r is None or r.get("model_kind") != "sequence" or r.get("dataset_seed") is None or r.get("missed_pct") is None:
        return JSONResponse({"status": "error", "message": "Choose a model trained on a GENERATED dataset (its population can be simulated)."}, status_code=400)
    holdout, revealed = _population_split(r)
    candidates = [c for c in holdout if c not in revealed]
    if not candidates:
        return JSONResponse({"status": "error", "message": "This model has no held-out family left to add — generate a dataset with more holdout classes."}, status_code=400)
    chosen = req.class_ids if req.class_ids else candidates[:1]
    if any(c not in candidates for c in chosen):
        return JSONResponse({"status": "error", "message": f"Choose among the model's still-unknown families {candidates}."}, status_code=400)
    if not 4 <= req.tracks_per_class <= 1000:
        return JSONResponse({"status": "error", "message": "Tracks per family must be 4–1000."}, status_code=400)
    seed, ranges, noise = _reused_seed_and_ranges(r)
    pulses = int(req.pulses_per_track or r.get("pulses_per_track") or 128)
    rows = build_sequence_dataset_rows(r["num_classes"], holdout, req.tracks_per_class, pulses, seed, ranges,
        r.get("missed_pct") or 0.0, r.get("spurious_pct") or 0.0, noise, int(np.random.randint(1_000_000_000)),
        bool(r.get("include_pgri")), bool(r.get("balanced_patterns")), revealed + chosen, _model_extras(r))
    df = pd.DataFrame(rows)
    # Keep the new families and the unknown tracks; the families the model already knows are not "new data".
    df = df[df["true_class_id"].isin(chosen) | (df["true_class_id"] < 0)].copy()
    meta = {"source": "generated", "purpose": "new_families", "dataset_seed": seed, "num_classes": r["num_classes"],
            "num_holdout_classes": len([c for c in candidates if c not in chosen]), "holdout_class_ids": holdout,
            "new_family_ids": chosen, "for_run_id": r["id"], "tracks_per_class": req.tracks_per_class,
            "pulses_per_track": pulses, "missed_pct": r.get("missed_pct"), "spurious_pct": r.get("spurious_pct"),
            "noise_track_pct": noise, "ranges": {k: r[k] for k in DatasetRanges().dict() if r.get(k) is not None},
            "include_pgri": bool(r.get("include_pgri")), "balanced_patterns": bool(r.get("balanced_patterns")),
            "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    wanted = (req.name or "").strip() or f"new_families_run{r['id']}_" + "_".join(f"{c:02d}" for c in chosen)
    name = datasets.save(df, meta, wanted, activate=False)
    return JSONResponse({"status": "success", "name": name,
                         "message": f"Dataset '{name}': {len(chosen)} new famil{'y' if len(chosen) == 1 else 'ies'} "
                                    f"({', '.join(f'RADAR_FAMILY_{c:02d}' for c in chosen)}), {df['instance'].nunique()} tracks incl. unknown ones."})


@router.get("/api/sequence/incremental/options")
def incremental_options(run_id: int):
    r = get_run(run_id)
    if r is None or r.get("model_kind") != "sequence":
        return JSONResponse({"status": "error", "message": "Model not found."}, status_code=404)
    holdout, revealed = _population_split(r)
    generated = r.get("dataset_seed") is not None and r.get("missed_pct") is not None
    return JSONResponse({"status": "success", "methods": [{"value": k, "label": v} for k, v in INCREMENTAL_METHODS.items()],
                         "unknown_families": [c for c in holdout if c not in revealed] if generated else [],
                         "class_labels": (registry.load_report(run_id) or {}).get("class_labels"),
                         "epochs": r.get("epochs"), "dataset_name": r.get("dataset_name")})


class IncrementalRequest(BaseModel):
    parent_run_id: int
    dataset: str                              # the new data (labelled tracks of the new family/families)
    methods: List[str] = ["A", "B", "C", "D"]
    finetune_epochs: int = 20                 # B and C (the new data is small: few steps per epoch)
    full_epochs: Optional[int] = None         # D; None = the base model's epochs
    replay_tracks_per_class: int = 10         # B: stored tracks per old family
    lr_scale: float = 0.5                     # B and C: learning rate = base learning rate × this
    eval_fraction: float = 0.3                # share of each new family's tracks kept for the comparison


def _incremental_inputs(req: IncrementalRequest) -> dict:
    """Everything the four methods share, built once so they are compared on identical data."""
    r = get_run(req.parent_run_id)
    if r is None or r.get("model_kind") != "sequence" or not registry.has_artifacts(r["id"]):
        raise ValueError("Choose a kept sequence model as the base.")
    base, spec, msg = load_sequence_checkpoint(registry.checkpoint_path(r["id"]))
    if base is None:
        raise ValueError(msg)
    base_centroids, _ = registry.load_prototypes(r["id"])
    report = registry.load_report(r["id"]) or {}
    labels = list(report.get("class_labels") or [f"RADAR_FAMILY_{c:02d}" for c in range(base.classifier_head.out_features)])
    old_ids = sorted(int(c) for c in base_centroids)
    old_label_ids = {labels[c]: c for c in old_ids}

    base_df, _ = _model_dataset(r)
    if base_df is None:
        raise ValueError("The base model's training dataset is not stored (and can't be regenerated) — "
                         "it is needed for the old-family evaluation and for methods B and D.")
    base_df = upgrade_columns(base_df)
    if "class" not in base_df.columns:
        raise ValueError("The base model's dataset has no class column.")
    kept = ((registry.load_report(r["id"]) or {}).get("cross_validation") or {}).get("kept_fold_split")
    if kept:  # the split of the cross-validation fold whose model was kept
        lookup = {str(i): i for i in base_df["instance"].unique()}
        train_ids, val_ids, cal_ids = ({lookup[t] for t in kept[k] if t in lookup} for k in ("train", "validation", "calibration"))
    else:     # models trained before cross-validation: the single split rebuilt from the dataset seed
        train_ids, val_ids, cal_ids, _ = split_track_ids(base_df, r["dataset_seed"] or 0)
    base_known = map_labels(base_df, old_label_ids)
    base_known = base_known[base_known["true_class_id"] >= 0]

    new_df, new_meta = _load_sequence_dataset(req.dataset)
    if new_df is None:
        raise ValueError(f"Dataset '{req.dataset}' not found.")
    if "class" not in new_df.columns:
        raise ValueError("The new dataset needs a class column (the family names).")
    known_new = new_df[pd.to_numeric(new_df["true_class_id"], errors="coerce").fillna(-1) >= 0]
    new_labels = sorted(set(known_new["class"].astype(str)) - set(old_label_ids))
    if not new_labels:
        raise ValueError("The new dataset contains no family this model doesn't already know.")
    for lab in new_labels:
        if lab not in labels:
            labels.append(lab)
    label_ids = {lab: i for i, lab in enumerate(labels) if i in old_ids or lab in new_labels}
    new_ids = sorted(label_ids[lab] for lab in new_labels)
    missing = [c for c in spec.columns if c not in new_df.columns]
    if missing:
        raise ValueError(f"The new dataset lacks column(s) {missing} that the base model's features need.")

    # The new data: per family a share of tracks kept aside for the comparison, the rest to learn from.
    new_m = prefix_tracks(map_labels(known_new, label_ids), "new:")
    rng = np.random.RandomState(12345)
    eval_ids, pool_ids = set(), set()
    for c, g in new_m.groupby("true_class_id"):
        ids = g["instance"].unique()
        ids = ids[rng.permutation(len(ids))]
        n_eval = max(1, int(round(len(ids) * req.eval_fraction))) if len(ids) >= 2 else 0
        eval_ids.update(ids[:n_eval]); pool_ids.update(ids[n_eval:])
    new_pool = new_m[new_m["instance"].isin(pool_ids)]
    eval_new = new_m[new_m["instance"].isin(eval_ids)]
    unknown = new_df[pd.to_numeric(new_df["true_class_id"], errors="coerce").fillna(-1) < 0].copy()
    unknown["true_class_id"] = -1
    eval_unknown = prefix_tracks(unknown, "unk:")

    old_m = prefix_tracks(base_known, "old:")
    pref = lambda ids: {"old:" + str(t) for t in ids}
    eval_old = old_m[old_m["instance"].isin(pref(val_ids))]
    old_train = old_m[old_m["instance"].isin(pref(train_ids) | pref(cal_ids))]
    replay_ids = set()
    for c, g in old_m[old_m["instance"].isin(pref(train_ids))].groupby("true_class_id"):
        ids = np.sort(g["instance"].unique())
        replay_ids.update(rng.permutation(ids)[:req.replay_tracks_per_class])
    replay = old_m[old_m["instance"].isin(replay_ids)]

    holdout, revealed = _population_split(r)
    if new_meta.get("source") == "generated" and new_meta.get("dataset_seed") == r.get("dataset_seed"):
        revealed = sorted(set(revealed) | {int(l[-2:]) for l in new_labels if re.fullmatch(r"RADAR_FAMILY_\d\d", l) and int(l[-2:]) in holdout})
    cfg = copy.deepcopy(current_config)
    cfg["hyperparameters"]["learning_rate"] = r["learning_rate"]
    cfg["hyperparameters"]["triplet_margin"] = r["triplet_margin"]
    cfg["hyperparameters"]["distance_metric"] = r["distance_metric"]
    if r.get("rpl_loss_weight") is not None:
        cfg["hyperparameters"]["rpl_loss_weight"] = r["rpl_loss_weight"]
    cfg["hyperparameters"]["rpl_adversarial"] = bool(r.get("rpl_adversarial"))
    thresholds = report.get("thresholds") or {"ood_distance_threshold": r["ood_distance_threshold"]}
    return {"run": r, "base": base, "spec": spec, "base_centroids": base_centroids, "base_thresholds": thresholds,
            "labels": labels, "old_ids": old_ids, "new_ids": new_ids, "new_labels": new_labels, "cfg": cfg,
            "new_pool": new_pool, "replay": replay, "old_train": old_train,
            "eval_old": eval_old, "eval_new": eval_new, "eval_unknown": eval_unknown,
            "population": {"holdout_class_ids": holdout, "revealed_class_ids": revealed},
            "frames": [base_df, new_df], "coverage": thresholds.get("coverage", 0.95)}


def _save_incremental_run(inp: dict, req: IncrementalRequest, res: dict, scores: dict, epochs: int) -> int:
    r, cfg = inp["run"], inp["cfg"]
    model, spec, centroids, thr = res["model"], res["spec"], res["centroids"], res["thresholds"]
    record = {c: r.get(c) for c in RUN_COLUMNS}
    tau_c = r["classifier_confidence_threshold"]
    overall = scores["overall"] or {}
    record.update({
        "num_classes": model.classifier_head.out_features, "epochs": epochs,
        "closed_set_accuracy": overall.get("closed_set_accuracy"), "open_set_auroc": scores["open_set_auroc"],
        "open_set_auroc_rpl": None, "final_loss": res["loss_curve"][-1] if res["loss_curve"] else None,
        "training_duration_seconds": res["seconds"], "loss_curve": res["loss_curve"],
        "ood_distance_threshold": thr["ood_distance_threshold"], "classifier_confidence_threshold": tau_c,
        "dataset_name": req.dataset, "dataset_sha256": datasets.content_hash(req.dataset), "app_version": APP_VERSION,
        "parent_run_id": r["id"], "incremental_method": res["method"],
        "saved_name": None, "saved_at_utc": None, "train_seed": r["dataset_seed"] or 0,
    })
    run_id = insert_run(record)
    freq = {}
    for df in inp["frames"]:
        known = map_labels(upgrade_columns(df), {lab: i for i, lab in enumerate(inp["labels"])})
        known = known[known["true_class_id"] >= 0]
        if "Frequency" in known.columns:
            freq.update(known.groupby("true_class_id")["Frequency"].mean().to_dict())
    metadata = {c: {"freq_band": frequency_to_band(freq[c]) if c in freq else None} for c in centroids}
    distance_fn = build_distance_fn(r["distance_metric"])
    report = {"architecture": None, "class_labels": inp["labels"], "thresholds": thr, "population": inp["population"],
              "sequence": overall,
              "incremental": {"method": res["method"], "method_name": INCREMENTAL_METHODS[res["method"]], "parent_run_id": r["id"],
                              "dataset": req.dataset, "new_class_labels": inp["new_labels"], "old_class_ids": inp["old_ids"],
                              "new_class_ids": inp["new_ids"], "data_used": res["data_used"],
                              "stale_prototypes": [inp["labels"][c] for c in res["stale_prototypes"]],
                              "scores": {k: v for k, v in scores.items() if k != "overall"}}}
    try:
        report["architecture"] = sequence_architecture_report(model, spec, dict(cfg, model=dict(cfg["model"], embedding_dim=model.classifier_head.in_features)))
        report["feature_importance"] = sequence_feature_importance(model, res["train_data"], spec)
        report["evaluation"] = evaluation_report(model, res["train_data"], distance_fn, centroids, thr["ood_distance_threshold"], tau_c)
        report["branch_ablation"] = branch_ablation(model, res["train_data"], distance_fn, centroids)
    except Exception:  # a report failure must never lose a trained model
        log.exception("incremental report failed")
        report["error"] = "Part of the report could not be generated (see the server log)."
    save_sequence_checkpoint(SEQUENCE_TRAIN_CHECKPOINT, model, spec)
    registry.save_artifacts(run_id, SEQUENCE_TRAIN_CHECKPOINT, None, centroids, metadata, report)
    return run_id


async def _run_incremental_job(req: IncrementalRequest):
    training_status.update(is_training=True, error=None, incremental=None,
                           progress={"phase": "incremental", "method": None, "epoch": 0, "epochs": 0, "loss": None, "dataset": req.dataset})
    _set_phase("Scoring the base model")
    try:
        inp = await asyncio.to_thread(_incremental_inputs, req)
        r = inp["run"]
        metric, tau_c = r["distance_metric"], r["classifier_confidence_threshold"]
        base_eval_new = inp["eval_new"].assign(true_class_id=-1)  # to the base model the new family is unknown
        base_scores = await asyncio.to_thread(incremental_score, inp["base"], inp["spec"], inp["base_centroids"], metric, tau_c,
                                              r["ood_distance_threshold"], inp["eval_old"], base_eval_new, inp["eval_unknown"])
        rows = [{"method": "base", "method_name": f"Base model #{r['id']} (before)", "run_id": r["id"],
                 "data_used": "—", "seconds": None, **{k: v for k, v in base_scores.items() if k != "overall"}}]
        full_epochs = int(req.full_epochs or r.get("epochs") or 20)
        for m in req.methods:
            epochs = {"A": 0, "B": req.finetune_epochs, "C": req.finetune_epochs, "D": full_epochs}[m]
            training_status["progress"] = {"phase": "incremental", "method": m, "epoch": 0, "epochs": epochs, "loss": None, "dataset": req.dataset}
            _set_phase(f"Method {m}: {INCREMENTAL_METHODS[m]}")

            def cb(epoch, total, loss, m=m):
                training_status["progress"] = {"phase": "incremental", "method": m, "epoch": epoch, "epochs": total, "loss": loss, "dataset": req.dataset}

            res = await asyncio.to_thread(
                run_incremental_method, m, inp["base"], inp["spec"], inp["base_centroids"], inp["base_thresholds"], inp["cfg"],
                len(inp["labels"]), inp["old_ids"], inp["new_ids"], inp["new_pool"], inp["replay"], inp["old_train"],
                r["dataset_seed"] or 0, req.finetune_epochs, full_epochs, req.lr_scale, inp["coverage"], cb)
            scores = await asyncio.to_thread(incremental_score, res["model"], res["spec"], res["centroids"], metric, tau_c,
                                             res["thresholds"]["ood_distance_threshold"], inp["eval_old"], inp["eval_new"], inp["eval_unknown"])
            run_id = await asyncio.to_thread(_save_incremental_run, inp, req, res, scores, epochs)
            rows.append({"method": m, "method_name": INCREMENTAL_METHODS[m], "run_id": run_id, "data_used": res["data_used"],
                         "seconds": res["seconds"], **{k: v for k, v in scores.items() if k != "overall"}})
            training_status["incremental"] = {"rows": rows}
        experiment = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_run_id": r["id"],
                      "dataset": req.dataset, "new_class_labels": inp["new_labels"], "settings": req.dict(),
                      "eval_tracks": {"old": int(inp["eval_old"]["instance"].nunique()), "new": int(inp["eval_new"]["instance"].nunique()),
                                      "unknown": int(inp["eval_unknown"]["instance"].nunique())},
                      "rows": rows}
        os.makedirs(INCREMENTAL_DIR, exist_ok=True)
        with open(os.path.join(INCREMENTAL_DIR, f"experiment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(experiment, f, indent=2)
        training_status["incremental"] = experiment
        training_status["last_run_id"] = rows[-1]["run_id"]
        registry.prune([x["id"] for x in get_all_runs(100_000)], [x["id"] for x in get_saved_runs()] + [r["id"]])
        _end_job("succeeded")
    except TrainingCancelled:
        log.info("incremental training stopped by the user")
        _end_job("cancelled", message="Stopped. Methods that had already finished stay in Run History; the running one was discarded.")
    except Exception as e:
        log.exception("training job failed")
        _end_job("failed", _explain_error(e))
    finally:
        if training_status["is_training"]:
            _end_job("failed", "The training job ended unexpectedly. See the server log.")


@router.post("/api/sequence/incremental")
async def start_incremental(req: IncrementalRequest):
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "Training already in progress"}, status_code=409)
    if not req.methods or any(m not in INCREMENTAL_METHODS for m in req.methods):
        return JSONResponse({"status": "error", "message": f"Choose methods among {list(INCREMENTAL_METHODS)}."}, status_code=400)
    if not 1 <= req.finetune_epochs <= 200 or (req.full_epochs is not None and not 1 <= req.full_epochs <= 500):
        return JSONResponse({"status": "error", "message": "Epochs: fine-tuning 1–200, full retrain 1–500."}, status_code=400)
    if not 0 <= req.replay_tracks_per_class <= 1000 or not 0.01 <= req.lr_scale <= 2 or not 0.1 <= req.eval_fraction <= 0.6:
        return JSONResponse({"status": "error", "message": "Replay tracks 0–1000, learning-rate scale 0.01–2, evaluation share 10–60%."}, status_code=400)
    try:  # fail fast on bad input instead of in the background
        await asyncio.to_thread(_incremental_inputs, req)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    req.methods = [m for m in INCREMENTAL_METHODS if m in req.methods]
    job_id = _begin_job("Preparing")
    jobs.training_task = asyncio.create_task(_run_incremental_job(req))
    return JSONResponse({"status": "started", "job_id": job_id})


@router.get("/api/sequence/incremental/experiments")
def incremental_experiments(limit: int = Query(20, ge=1, le=MAX_HISTORY_ROWS)):
    out = []
    if os.path.isdir(INCREMENTAL_DIR):
        for f in sorted(os.listdir(INCREMENTAL_DIR), reverse=True)[:limit]:
            try:
                with open(os.path.join(INCREMENTAL_DIR, f)) as fh:
                    e = json.load(fh)
                e["created"] = _ist(e.get("created_at_utc"))
                out.append(e)
            except (OSError, ValueError):
                continue
    return JSONResponse({"status": "success", "experiments": out})

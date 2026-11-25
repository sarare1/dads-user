"""Helpers used by more than one route module (dataset lookup, run lineage, the model cache ...)."""

import os
import threading
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from fastapi import UploadFile
from src.data.sequence_features import DEFAULT_SEQ_FEATURES, required_seq_columns, upgrade_columns
from src.data.sequence_generator import build_sequence_dataset_rows
from src.models.checkpoint import load_sequence_checkpoint
from src.models.sequence_eval import SequenceClassifier
from src.storage.training_store import get_all_runs, get_run
from src.training.trainer import _select_holdout_ids, build_distance_fn
from src.api.core import DATA_DIR, DatasetRanges, datasets, MAX_UPLOAD_BYTES, MAX_UPLOAD_PULSES, MODELS_DIR, registry


def _reused_seed_and_ranges(latest_run: Optional[dict]):
    """Extracts the exact dataset_seed + physical ranges + noise level a training run used, so
    Testing reconstructs the identical population it was trained on — including how much
    unstructured OOD noise was mixed in, not just the seed/ranges. Falls back to the original
    fixed defaults if there's no run yet, or it predates these columns (pre-migration)."""
    defaults = DatasetRanges()
    if latest_run is None:
        return None, defaults.to_ranges_dict(), 0.0
    seed = latest_run.get("dataset_seed")
    range_fields = defaults.dict()
    for key in range_fields:
        if latest_run.get(key) is not None:
            range_fields[key] = latest_run[key]
    ranges = DatasetRanges(**range_fields).to_ranges_dict()
    noise_pct = latest_run.get("noise_pct") if latest_run.get("noise_pct") is not None else 0.0
    return seed, ranges, noise_pct


TEST_RESULTS_DIR = os.path.join(DATA_DIR, "test_results")





_IST = timezone(timedelta(hours=5, minutes=30))


def _ist(iso: Optional[str]) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(_IST).strftime("%Y-%m-%d %H:%M IST")
    except (TypeError, ValueError):
        return ""


def _dataset_kind(meta: dict) -> str:
    if meta.get("source") == "uploaded":
        return "uploaded (test file)" if meta.get("purpose") == "test" else "uploaded"
    return "generated"


def _kept_sequence_runs() -> List[dict]:
    return [r for r in get_all_runs(500) if r.get("model_kind") == "sequence" and registry.has_artifacts(r["id"])]


def _check_run_dataset(r: dict) -> None:
    """Raises ValueError when the run's stored dataset is no longer byte-for-byte the file it was trained on."""
    name = r.get("dataset_name")
    if name and datasets.exists(name) and not datasets.verify(name, r.get("dataset_sha256")):
        raise ValueError(f"Dataset '{name}' has changed since run #{r['id']} was trained on it (its SHA-256 no longer "
                         f"matches), so it can't stand in for that model's training data.")


def _model_dataset(r: dict) -> Tuple[Optional[pd.DataFrame], dict]:
    """The dataset a kept model was trained on: its named dataset when that is still stored, else
    (runs made before datasets had names) rebuilt from the run's own seed and settings — exact,
    because generation is deterministic. A run on an uploaded file cannot be rebuilt (None)."""
    name = r.get("dataset_name")
    if name and datasets.exists(name):
        _check_run_dataset(r)
        df, meta = datasets.load(name)
        return df, dict(meta, reconstructed=False)
    if r.get("dataset_seed") is None or r.get("missed_pct") is None or r.get("parent_run_id"):
        return None, {}  # an uploaded file has no generation settings; an incremental run's data isn't one generation
    seed, ranges, noise = _reused_seed_and_ranges(r)
    holdout = _select_holdout_ids(r["num_classes"], r["num_holdout_classes"], seed)
    rows = build_sequence_dataset_rows(
        r["num_classes"], holdout, int(r.get("samples_per_class") or 40), int(r.get("pulses_per_track") or 128), seed, ranges,
        r.get("missed_pct") or 0.0, r.get("spurious_pct") or 0.0, noise, None, bool(r.get("include_pgri")),
        bool(r.get("balanced_patterns")))
    meta = {"name": name or f"run{r['id']}_regenerated", "source": "generated", "dataset_seed": seed, "reconstructed": True,
            "num_classes": r["num_classes"], "ranges": {k: r[k] for k in DatasetRanges().dict() if r.get(k) is not None}}
    return pd.DataFrame(rows), meta


# ============================================================================ pulse-sequence models
# Classify a window of consecutive pulses of ONE emitter (a de-interleaved track) — see
# src/models/sequence_models.py: dataset, training job, testing, inference and live simulation.
SEQUENCE_TRAIN_CHECKPOINT = os.path.join(MODELS_DIR, "sequence_training.pt")  # staging file for the registry


def _load_sequence_dataset(name: Optional[str] = None):
    """(dataframe, meta) of the named dataset — the active one when no name is given."""
    df, meta = datasets.load(name)
    return (None, None) if df is None else (upgrade_columns(df), meta)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB).")
    return data


def _ensure_class_ids(df: pd.DataFrame) -> pd.DataFrame:
    """A file with a `class` column but no true_class_id gets ids from the class names (sorted; a class
    named UNKNOWN... or ...NOISE... is unknown, -1)."""
    if "true_class_id" in df.columns or "class" not in df.columns:
        return df
    names = df["class"].astype(str)
    unknown = names.str.contains("UNKNOWN|NOISE", case=False)
    ids = {n: i for i, n in enumerate(sorted(set(names[~unknown])))}
    df = df.copy()
    df["true_class_id"] = names.map(lambda n: ids.get(n, -1)).astype(int)
    return df


def _check_sequence_file(df: pd.DataFrame) -> Optional[str]:
    df = _ensure_class_ids(upgrade_columns(df))
    missing = [c for c in ("instance", "true_class_id") if c not in df.columns]
    if missing:
        return (f"Missing required column(s) {missing}. Each row is one pulse: the measurements (Frequency, PW, PRI, Modlntype, Modlnval, snr, rise, fall, timeBW), "
                f"instance (which de-interleaved track), class (the emitter name; or true_class_id, -1 = unknown) and optionally pulse_idx (order).")
    if len(df) > MAX_UPLOAD_PULSES:
        return f"Too many pulses ({len(df):,}); the limit is {MAX_UPLOAD_PULSES:,}."
    if not any(c in df.columns for c in required_seq_columns(DEFAULT_SEQ_FEATURES)):
        return f"No measurement columns found; expected some of {required_seq_columns(DEFAULT_SEQ_FEATURES)}."
    return None


# Columns the track simulator produces — a model using any other (dataset) column can't be fed by
# Live Simulation tracks, example tracks or auto-generated test tracks.
SIMULATED_SEQUENCE_COLUMNS = {"Frequency", "PW", "PRI", "Modlntype", "Modlnval", "snr", "rise", "fall", "timeBW"}


def _model_extras(r: dict) -> List[dict]:
    """User-defined extra features of the dataset the model was trained on (the simulator can regenerate them)."""
    name = r.get("dataset_name")
    return ((datasets.meta(name) or {}).get("extra_features") or []) if name and datasets.exists(name) else []


def _simulator_gap(spec, extras: Optional[List[dict]] = None) -> List[str]:
    """Columns the model needs that the simulator can't produce (user-defined extra features of the
    model's dataset can be, for the Testing page's fresh tracks)."""
    known = SIMULATED_SEQUENCE_COLUMNS | {e["name"] for e in (extras or [])}
    return [c for c in required_seq_columns(spec.names) if c not in known]


# ---------------------------------------------------------------- sequence models: phase 2
# The sequence model "in use" (Inference + Live Simulation) is recorded by the registry.
_seq_cache: dict = {}
_seq_cache_lock = threading.Lock()  # endpoints run in worker threads: one loads/swaps the cached model at a time


def _sequence_classifier(run_id: int):
    """(SequenceClassifier, run row) for a kept sequence run — loaded once, then cached."""
    r = get_run(run_id)
    if r is None or r.get("model_kind") != "sequence" or not registry.has_artifacts(run_id):
        raise ValueError(f"Run #{run_id} is not a kept pulse-sequence model.")
    key = (run_id, r["classifier_confidence_threshold"], r["ood_distance_threshold"])
    with _seq_cache_lock:
        clf = _seq_cache.get(key)
        if clf is None:
            m, spec, msg = load_sequence_checkpoint(registry.checkpoint_path(run_id))
            if m is None:
                raise ValueError(msg)
            centroids, _ = registry.load_prototypes(run_id)
            report = registry.load_report(run_id) or {}
            thr = report.get("thresholds") or {}
            clf = SequenceClassifier(m, spec, centroids, build_distance_fn(r["distance_metric"]),
                                     r["classifier_confidence_threshold"], r["ood_distance_threshold"],
                                     thr.get("by_length"), thr.get("ood_distance_threshold"), report.get("class_labels"))
            _seq_cache.clear()  # keep one model in memory
            _seq_cache[key] = clf
    return clf, r


def _population_split(r: dict) -> Tuple[List[int], List[int]]:
    """(the families held out when the model's population was generated, those of them the model
    has since been taught incrementally). The first fixes the families themselves; held-out minus
    revealed is what is still unknown to this model."""
    holdout = _select_holdout_ids(r["num_classes"], r["num_holdout_classes"], r.get("dataset_seed") or 0)
    pop = (registry.load_report(r["id"]) or {}).get("population") or {}
    return holdout, [int(c) for c in pop.get("revealed_class_ids", []) if int(c) in holdout]

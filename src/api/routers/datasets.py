"""Named datasets: generate, upload, expand, list, activate, delete, download; feature options."""

import asyncio
import hashlib
import io
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from src.data.dataset_expander import expand_dataset
from src.data.sequence_features import (
    DATASET_CATEGORY, DATASET_DELTA, DATASET_VALUE, DEFAULT_SEQ_FEATURES, DEFAULT_TRACK_CAP, DEFAULT_WINDOW,
    MAX_TRACK_PULSES, order_columns, SEQ_FEATURES, sequence_categorical_columns, sequence_dataset_columns,
    upgrade_columns,
)
from src.data.sequence_generator import build_sequence_dataset_rows, validate_extra_features
from src.models.sequence_models import SEQ_BACKBONES
from src.training.trainer import _select_holdout_ids
from src.api.core import _public_error, DatasetRanges, datasets, log, training_status
from src.api.common import (
    _check_sequence_file, _dataset_kind, _ensure_class_ids, _IST, _ist, _kept_sequence_runs,
    _load_sequence_dataset, _read_upload,
)
router = APIRouter()


MAX_SEQUENCE_PULSES = 3_000_000


class ExtraFeature(BaseModel):
    """A user-defined column added to every generated pulse. numeric: each family gets its own value in
    [min, max] and pulses scatter around it by noise_pct %; categorical: each family gets one of `values`."""
    name: str
    kind: str = "numeric"                # numeric | categorical
    min: Optional[float] = None
    max: Optional[float] = None
    noise_pct: float = 0.0
    values: List[str] = []


class GenerateSequenceRequest(BaseModel):
    num_classes: int = 20
    num_holdout_classes: int = 3
    tracks_per_class: int = 40
    pulses_per_track: int = 128
    missed_pct: float = 0.02
    spurious_pct: float = 0.02
    noise_track_pct: float = 0.10
    ranges: DatasetRanges = DatasetRanges()
    include_pgri: bool = True
    extra_features: List[ExtraFeature] = []   # user-defined columns, usable as model inputs like any other
    capture_date: Optional[str] = None        # YYYY-MM-DD of the first captures (part of every instance name); blank = today
    name: Optional[str] = None  # shown everywhere the dataset is used; blank = generated_<date>_seed<n>


def _validate_sequence_generation(req: GenerateSequenceRequest) -> Optional[str]:
    if not 2 <= req.num_classes <= 100:
        return "Number of classes must be 2–100."
    if not 0 <= req.num_holdout_classes < req.num_classes:
        return "Holdout classes must be fewer than the number of classes."
    if not 2 <= req.tracks_per_class <= 1000:
        return "Tracks per class must be 2–1000."
    if not 8 <= req.pulses_per_track <= 4000:
        return "Pulses per track must be 8–4000."
    for name in ("missed_pct", "spurious_pct", "noise_track_pct"):
        if not 0 <= getattr(req, name) <= 0.5:
            return f"{name} must be between 0 and 0.5."
    ranges = req.ranges.dict()
    if not all(math.isfinite(v) and v >= 0 for v in ranges.values()):
        return "Population ranges must be finite, non-negative numbers."
    for lo, hi in (("freq_min_mhz", "freq_max_mhz"), ("pw_min_us", "pw_max_us"), ("pri_min_us", "pri_max_us"),
                   ("rise_min_ns", "rise_max_ns"), ("bw_min_mhz", "bw_max_mhz"), ("fall_min_ns", "fall_max_ns"),
                   ("duty_min_pct", "duty_max_pct")):
        if ranges[lo] > ranges[hi]:
            return f"Range minimum {lo} is above its maximum {hi}."
    if ranges["pri_min_us"] <= 0 or ranges["duty_max_pct"] > 100:
        return "PRI must be positive and the duty cycle at most 100%."
    if req.num_classes * req.tracks_per_class * req.pulses_per_track > MAX_SEQUENCE_PULSES:
        return f"Dataset too large (classes × tracks × pulses must stay ≤ {MAX_SEQUENCE_PULSES:,})."
    if len(req.extra_features) > 20:
        return "At most 20 extra features."
    error = validate_extra_features([e.dict() for e in req.extra_features])
    if error:
        return error
    if req.capture_date:
        try:
            datetime.strptime(req.capture_date, "%Y-%m-%d")
        except ValueError:
            return "Capture date must be YYYY-MM-DD."
    return None


def _sequence_summary(df: pd.DataFrame, meta: dict) -> dict:
    """Composition of a sequence dataset, plus a few example tracks for the PRI/frequency preview."""
    first = df.groupby("instance", sort=False).first()
    lengths = df.groupby("instance", sort=False).size()
    labels = first["true_class_id"].astype(int)
    per_track = lambda col: {str(k): int(v) for k, v in first[col].value_counts().items()} if col in first.columns else {}
    examples = []
    if "pri_pattern" in first.columns:
        for pattern, grp in first[labels >= 0].groupby("pri_pattern"):
            tid = grp.index[0]
            t = df[df["instance"] == tid]
            if "pulse_idx" in t.columns:
                t = t.sort_values("pulse_idx")
            t = t.head(80)
            examples.append({
                "track_id": str(tid), "label": str(t["class"].iloc[0]) if "class" in t else str(t["true_class_id"].iloc[0]),
                "pri_pattern": str(pattern), "freq_agility": str(t["freq_agility"].iloc[0]) if "freq_agility" in t else None,
                "pri_us": [None if pd.isna(v) else float(v) for v in t["PRI"]] if "PRI" in t else [],
                "freq_mhz": [None if pd.isna(v) else float(v) for v in t["Frequency"]] if "Frequency" in t else [],
            })
    return {
        "total_pulses": int(len(df)), "total_tracks": int(len(first)),
        "known_tracks": int((labels >= 0).sum()), "unknown_tracks": int((labels < 0).sum()),
        "known_classes": int(labels[labels >= 0].nunique()),
        "pulses_per_track": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
        "pri_patterns": per_track("pri_pattern"), "freq_agility": per_track("freq_agility"),
        "labels": per_track("class"),
        "spurious_pulses": int(df["is_spurious"].sum()) if "is_spurious" in df.columns else None,
        "columns": list(df.columns), "meta": meta, "examples": examples[:6],
        "extra_feature_columns": sequence_dataset_columns(df),
    }


@router.post("/api/sequence/dataset/generate")
def generate_sequence_dataset(req: GenerateSequenceRequest):
    error = _validate_sequence_generation(req)
    if error:
        return JSONResponse({"status": "error", "message": error}, status_code=400)
    seed = int(np.random.randint(1_000_000))
    holdout = _select_holdout_ids(req.num_classes, req.num_holdout_classes, seed)
    extras = [dict(e.dict(), name=e.name.strip()) for e in req.extra_features]
    capture = datetime.strptime(req.capture_date, "%Y-%m-%d").date() if req.capture_date else None
    rows = build_sequence_dataset_rows(req.num_classes, holdout, req.tracks_per_class, req.pulses_per_track, seed,
        req.ranges.to_ranges_dict(), req.missed_pct, req.spurious_pct, req.noise_track_pct, None, req.include_pgri, True,
        None, extras, capture)
    df = pd.DataFrame(rows)
    meta = {
        "dataset_seed": seed, "num_classes": req.num_classes, "num_holdout_classes": req.num_holdout_classes,
        "holdout_class_ids": holdout, "known_class_ids": [c for c in range(req.num_classes) if c not in holdout],
        "tracks_per_class": req.tracks_per_class, "pulses_per_track": req.pulses_per_track,
        "missed_pct": req.missed_pct, "spurious_pct": req.spurious_pct, "noise_track_pct": req.noise_track_pct,
        "ranges": req.ranges.dict(), "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source": "generated",
        "include_pgri": req.include_pgri, "balanced_patterns": True,
        "extra_features": extras, "capture_date": req.capture_date,
    }
    wanted = (req.name or "").strip() or f"generated_{datetime.now(_IST).strftime('%Y%m%d_%H%M')}_seed{seed}"
    name = datasets.save(df, meta, wanted)
    meta = datasets.meta(name)
    return JSONResponse({"status": "success", "name": name,
                         "message": f"Generated dataset '{name}': {df['instance'].nunique()} tracks ({len(df):,} pulses).",
                         **_sequence_summary(df, meta)})


@router.get("/api/sequence/dataset")
def sequence_dataset_info(name: Optional[str] = None):
    df, meta = _load_sequence_dataset(name)
    if df is None:
        return JSONResponse({"status": "error", "message": "No sequence dataset yet — generate or upload one."}, status_code=404)
    return JSONResponse({"status": "success", "name": meta["name"], **_sequence_summary(df, meta)})


@router.get("/api/datasets")
def list_datasets():
    """Every stored dataset with the models trained on it."""
    runs = _kept_sequence_runs()
    out = []
    for m in datasets.list():
        used = [r["id"] for r in runs if r.get("dataset_name") == m["name"]]
        out.append({"name": m["name"], "kind": _dataset_kind(m), "source": m.get("source"), "purpose": m.get("purpose"),
                    "created": _ist(m.get("created_at_utc") or m.get("generated_at_utc")), "is_active": m["is_active"],
                    "tracks": m.get("total_tracks"), "pulses": m.get("total_pulses"), "known_classes": m.get("num_classes"),
                    "size_bytes": m["size_bytes"], "seed": m.get("dataset_seed"), "original_filename": m.get("original_filename"),
                    "used_by_models": used, "sha256": m.get("sha256")})
    return JSONResponse({"status": "success", "active": datasets.get_active(), "datasets": out})


@router.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    if not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "Dataset not found."}, status_code=404)
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "A model is training — delete datasets when it has finished."}, status_code=409)
    datasets.delete(name)
    return JSONResponse({"status": "success", "message": f"Deleted dataset '{name}'. Its name is retired and won't be given to another dataset.", "active": datasets.get_active()})


@router.post("/api/datasets/{name}/activate")
def activate_dataset(name: str):
    if not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "Dataset not found."}, status_code=404)
    datasets.set_active(name)
    return JSONResponse({"status": "success", "active": name, "message": f"'{name}' is now the default training dataset."})


@router.get("/api/sequence/dataset/download")
def download_sequence_dataset(name: Optional[str] = None):
    name = name or datasets.get_active()
    if not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "No sequence dataset yet."}, status_code=404)
    df, _ = datasets.load(name)
    return Response(content=_csv_safe(df).to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Text cells that a spreadsheet would run as a formula (=, +, -, @ ...) get a leading apostrophe."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in _FORMULA_STARTS else v)
    return out


@router.post("/api/sequence/dataset/upload")
async def upload_sequence_dataset(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Your own de-interleaved tracks become the sequence training dataset. Class ids are
    remapped to 0..N−1 (−1 stays unknown)."""
    try:
        data = await _read_upload(file)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=413)
    # Parsing and saving a 100 MB CSV takes seconds: done in a worker thread so the server keeps answering.
    return await asyncio.to_thread(_ingest_training_upload, data, file.filename, name)


def _ingest_training_upload(data: bytes, filename: Optional[str], name: Optional[str]) -> JSONResponse:
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        return JSONResponse({"status": "error", "message": "Could not parse file as CSV."}, status_code=400)
    df = _ensure_class_ids(upgrade_columns(df))
    error = _check_sequence_file(df)
    if error:
        return JSONResponse({"status": "error", "message": error}, status_code=400)
    try:
        labels = pd.to_numeric(df["true_class_id"], errors="coerce").fillna(-1).astype(int)
    except Exception:
        return JSONResponse({"status": "error", "message": "true_class_id must be an integer (−1 = unknown)."}, status_code=400)
    known_raw = sorted(set(labels[labels >= 0].tolist()))
    if not known_raw:
        return JSONResponse({"status": "error", "message": "No known-class tracks — every true_class_id is −1."}, status_code=400)
    remap = {old: new for new, old in enumerate(known_raw)}
    df["true_class_id"] = labels.map(lambda x: remap.get(int(x), -1))
    if "class" not in df.columns:
        df["class"] = df["true_class_id"].map(lambda c: f"class_{c}" if c >= 0 else "unknown")
    # The file's own name for each renumbered class — the model reports these names, not RADAR_FAMILY_<new id>.
    known_rows = df[df["true_class_id"] >= 0]
    class_labels = [str(known_rows.loc[known_rows["true_class_id"] == c, "class"].astype(str).mode().iloc[0])
                    for c in range(len(known_raw))]
    lengths = df.groupby("instance").size()
    meta = {
        "dataset_seed": int(np.random.randint(1_000_000)), "num_classes": len(known_raw),
        "num_holdout_classes": 0, "holdout_class_ids": [], "known_class_ids": list(range(len(known_raw))),
        "tracks_per_class": int(df[df["true_class_id"] >= 0].groupby("true_class_id")["instance"].nunique().median()),
        "pulses_per_track": int(lengths.median()), "missed_pct": None, "spurious_pct": None, "noise_track_pct": None,
        "ranges": DatasetRanges().dict(), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "uploaded", "original_filename": os.path.basename(filename or ""),
        "original_sha256": hashlib.sha256(data).hexdigest(),  # the file exactly as uploaded
        "class_labels": class_labels,
    }
    wanted = (name or "").strip() or f"uploaded_{os.path.splitext(meta['original_filename'])[0] or 'tracks'}"
    df = order_columns(df)
    saved = datasets.save(df, meta, wanted)
    meta = datasets.meta(saved)
    try:
        summary = _sequence_summary(df, meta)
        return JSONResponse({"status": "success", "name": saved,
                             "message": f"Uploaded dataset '{saved}': {df['instance'].nunique()} tracks ({len(df):,} pulses).",
                             **summary})
    except Exception:
        log.exception("dataset summary failed")
        return JSONResponse({"status": "error", "message": "File saved but could not be summarised (see the server log)."},
                            status_code=500)


class ExpandDatasetRequest(BaseModel):
    source: str                                # name of the stored (small) dataset
    tracks_per_class: int = 200                # every class is grown to this many tracks
    pulses_per_track: Optional[int] = None     # None = the length of the source tracks
    name: Optional[str] = None


@router.post("/api/sequence/dataset/expand")
def expand_sequence_dataset(req: ExpandDatasetRequest):
    """A large dataset from a small one: new tracks per class that follow the class's own patterns and
    between-track spread (see src/data/dataset_expander.py). The source dataset is left unchanged."""
    df, src_meta = _load_sequence_dataset(req.source)
    if df is None:
        return JSONResponse({"status": "error", "message": "Source dataset not found."}, status_code=404)
    if not 2 <= req.tracks_per_class <= 5000:
        return JSONResponse({"status": "error", "message": "Tracks per class must be 2-5000."}, status_code=400)
    if req.pulses_per_track is not None and not 8 <= req.pulses_per_track <= 4000:
        return JSONResponse({"status": "error", "message": "Pulses per track must be 8-4000."}, status_code=400)
    df = _ensure_class_ids(df)
    if "class" not in df.columns or "instance" not in df.columns:
        return JSONResponse({"status": "error", "message": "The source needs instance and class columns."}, status_code=400)
    n_classes = df["class"].nunique()
    n_pulses = req.pulses_per_track or int(df.groupby("instance").size().median())
    if n_classes * req.tracks_per_class * n_pulses > MAX_SEQUENCE_PULSES:
        return JSONResponse({"status": "error", "message": f"Result too large (classes x tracks x pulses must stay <= {MAX_SEQUENCE_PULSES:,})."},
                            status_code=400)
    big = expand_dataset(df, req.tracks_per_class, req.pulses_per_track,
                                  int(np.random.randint(1_000_000)))
    meta = {k: v for k, v in src_meta.items() if k not in ("name", "created_at_utc", "total_tracks", "total_pulses",
                                                          "known_tracks", "unknown_tracks")}
    meta.update({"source": "expanded", "expanded_from": req.source, "tracks_per_class": req.tracks_per_class,
                 "pulses_per_track": n_pulses, "generated_at_utc": datetime.now(timezone.utc).isoformat()})
    name = datasets.save(big, meta, (req.name or "").strip() or f"{req.source}_x{req.tracks_per_class}")
    meta = datasets.meta(name)
    return JSONResponse({"status": "success", "name": name,
                         "message": f"Expanded '{req.source}' ({df['instance'].nunique()} tracks) into '{name}': "
                                    f"{big['instance'].nunique()} tracks ({len(big):,} pulses).",
                         **_sequence_summary(big, meta)})


@router.get("/api/sequence/options")
def sequence_options(dataset: Optional[str] = None):
    df, meta = _load_sequence_dataset(dataset)
    columns = set(df.columns) if df is not None else None
    features = [{"name": n, "label": f.label, "unit": f.unit, "description": f.description, "source": "builtin",
                 "available": columns is None or all(c in columns for c in f.columns)}
                for n, f in SEQ_FEATURES.items()]
    for col in (sequence_categorical_columns(df) if df is not None else []):
        features.append({"name": DATASET_CATEGORY + col, "label": col, "unit": "", "source": "dataset", "available": True,
                         "description": f"Text column '{col}' of the track file, fed as a number (each category gets its own value)."})
    for col in (sequence_dataset_columns(df) if df is not None else []):
        features.append({"name": DATASET_VALUE + col, "label": col, "unit": "", "source": "dataset", "available": True,
                         "description": f"Column '{col}' of the uploaded track file, per pulse."})
        features.append({"name": DATASET_DELTA + col, "label": f"Δ{col}", "unit": "", "source": "dataset", "available": True,
                         "description": f"Change of '{col}' from the previous pulse of the same track."})
    return JSONResponse({
        "status": "success",
        "features": features,
        "defaults": [n for n in DEFAULT_SEQ_FEATURES if columns is None or all(c in columns for c in SEQ_FEATURES[n].columns)],
        "default_window": DEFAULT_WINDOW,
        "window_modes": [{"value": "windows", "label": "Windows of k pulses (voted per track)"},
                         {"value": "track", "label": f"Whole track (first N pulses, N ≤ {MAX_TRACK_PULSES}; 0 = no cap)"}],
        "default_track_cap": DEFAULT_TRACK_CAP,
        "backbones": [{"value": k, "label": v} for k, v in SEQ_BACKBONES.items()],
        "dataset": None if df is None else {"name": meta["name"], "tracks": int(df["instance"].nunique()), "pulses": int(len(df)),
                                            "source": meta.get("source"), "seed": meta.get("dataset_seed")},
        "active_dataset": datasets.get_active(),
    })

"""The model in use: activation, inference on pasted / uploaded tracks, example tracks, the live feed."""

import asyncio
import io
import json
import time
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.data.sequence_features import (
    has_pri, ordered_tracks, required_seq_columns, track_feature_matrix, upgrade_columns,
)
from src.data.sequence_generator import (
    add_extra_columns, add_side_columns, build_sequence_library, family_modulation, generate_noise_track,
    generate_track,
)
from src.storage.training_store import get_all_runs
from src.api.core import (
    _client_ip, _public_error, _same_origin, _session_user, audit, datasets, jobs, log, MAX_LIVE_SESSIONS,
    registry,
)
from src.api.common import (
    _check_run_dataset, _load_sequence_dataset, _model_extras, _population_split, _reused_seed_and_ranges,
    _sequence_classifier, _simulator_gap,
)
router = APIRouter()


MAX_LIVE_FEED_SECONDS = 900  # hard server-side cap (15 min) regardless of what the client requests


@router.get("/api/sequence/models")
def sequence_models():
    """Kept sequence models (newest first) — what the Testing page's sequence section offers."""
    runs = [r for r in get_all_runs(500) if r.get("model_kind") == "sequence" and registry.has_artifacts(r["id"])]
    active = registry.get_active_sequence()
    if active not in {r["id"] for r in runs}:
        active = runs[0]["id"] if runs else None
    return JSONResponse({"status": "success", "active_run_id": active, "models": [{
        "run_id": r["id"], "name": r.get("saved_name"), "timestamp_ist": r["timestamp_ist"],
        "backbone_type": r["backbone_type"], "window_k": r.get("window_k"), "window_mode": r.get("window_mode") or "windows", "features": r.get("features"),
        "closed_set_accuracy": r["closed_set_accuracy"], "open_set_auroc": r["open_set_auroc"],
        "dataset_seed": r.get("dataset_seed"), "dataset_name": r.get("dataset_name"), "is_active": r["id"] == active,
        "parent_run_id": r.get("parent_run_id"), "incremental_method": r.get("incremental_method"),
        "tau_c": r["classifier_confidence_threshold"], "tau_d": r["ood_distance_threshold"],
        "test_pct": r.get("test_pct"), "include_test": r.get("include_test"), "cv_folds": r.get("cv_folds"),
    } for r in runs]})


def _resolve_sequence_run(run_id: Optional[int]) -> int:
    rid = run_id if run_id is not None else registry.get_active_sequence()
    if rid is None:
        kept = [r for r in get_all_runs(500) if r.get("model_kind") == "sequence" and registry.has_artifacts(r["id"])]
        if not kept:
            raise ValueError("No pulse-sequence model yet — train one on the Training page (Sequence Model).")
        rid = kept[0]["id"]
    return int(rid)


class ActivateSequenceRequest(BaseModel):
    run_id: int


@router.post("/api/sequence/models/activate")
def activate_sequence_model(req: ActivateSequenceRequest):
    try:
        _, r = _sequence_classifier(req.run_id)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    registry.set_active_sequence(req.run_id)
    label = f"'{r['saved_name']}'" if r.get("saved_name") else f"run #{req.run_id}"
    return JSONResponse({"status": "success", "active_run_id": req.run_id,
                         "message": f"Sequence Inference and Live Simulation (tracks) now use {label}."})


class SequenceInferRequest(BaseModel):
    run_id: Optional[int] = None          # None = the sequence model in use
    csv_text: Optional[str] = None        # one row per pulse (track_id optional)
    pulses: Optional[List[dict]] = None   # or the same rows as JSON


MAX_INFER_PULSES = 20_000
MAX_INFER_CHARS = 5_000_000  # pasted CSV text


@router.post("/api/sequence/infer")
def sequence_infer(req: SequenceInferRequest):
    """Classify one or more tracks given as pulses: per track the vote over its windows, every
    window's decision, and how the decision develops as pulses arrive."""
    try:
        clf, r = _sequence_classifier(_resolve_sequence_run(req.run_id))
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    if (req.csv_text and len(req.csv_text) > MAX_INFER_CHARS) or (req.pulses and len(req.pulses) > MAX_INFER_PULSES):
        return JSONResponse({"status": "error", "message": f"Give 1–{MAX_INFER_PULSES:,} pulses."}, status_code=400)
    try:
        if req.csv_text:
            df = pd.read_csv(io.StringIO(req.csv_text))
        elif req.pulses:
            df = pd.DataFrame(req.pulses)
        else:
            return JSONResponse({"status": "error", "message": "Give the track's pulses (CSV text or rows)."}, status_code=400)
    except Exception:
        return JSONResponse({"status": "error", "message": "Could not parse the pulses as CSV."}, status_code=400)
    if len(df) == 0 or len(df) > MAX_INFER_PULSES:
        return JSONResponse({"status": "error", "message": f"Give 1–{MAX_INFER_PULSES:,} pulses."}, status_code=400)
    df = upgrade_columns(df)
    missing = [c for c in required_seq_columns(clf.spec.names) if c not in df.columns]
    if missing:
        return JSONResponse({"status": "error", "message": f"Missing column(s) {missing} needed by this model (features {clf.spec.names})."}, status_code=400)
    if "instance" not in df.columns:
        df["instance"] = 0
    try:
        tracks = []
        for tid, t in ordered_tracks(df):
            try:
                res = clf.classify_track(t)
            except ValueError as e:
                raise ValueError(f"track {tid}: {e}")
            res["track_id"] = str(tid)
            res["pri_us"] = [None if pd.isna(v) else float(v) for v in t["PRI"]] if "PRI" in t else []
            res["freq_mhz"] = [None if pd.isna(v) else float(v) for v in t["Frequency"]] if "Frequency" in t else []
            if "class" in t.columns:
                res["true_label"] = str(t["class"].iloc[0])
            tracks.append(res)
    except Exception:
        log.exception("classification failed")
        return JSONResponse({"status": "error", "message": "Could not classify these pulses — check the columns and values."}, status_code=400)
    return JSONResponse({"status": "success", "run_id": r["id"], "model_name": r.get("saved_name"),
                         "backbone_type": r["backbone_type"], "window": clf.spec.window,
                         "thresholds": {"classifier_confidence_threshold": clf.tau_c, "ood_distance_threshold": clf.tau_d},
                         "tracks": tracks})


class ExampleTrackRequest(BaseModel):
    run_id: Optional[int] = None
    kind: str = "known"                   # known | unknown (held-out family) | noise
    class_id: Optional[int] = None
    pulses: int = 96
    missed_pct: float = 0.02
    spurious_pct: float = 0.02
    source: str = "test"                  # test = the model's held-out test tracks when it has any | any = every track of its dataset


def _model_population(r: dict):
    seed, ranges, _ = _reused_seed_and_ranges(r)
    holdout, revealed = _population_split(r)
    library = build_sequence_library(r["num_classes"], seed, ranges, bool(r.get("include_pgri")),
                                     holdout if r.get("balanced_patterns") else None)
    return seed, ranges, [c for c in holdout if c not in revealed], library


def _dataset_example_track(req: "ExampleTrackRequest", clf, r: dict, dataset_name: str) -> Optional[dict]:
    """An example taken from the dataset the model was trained on — its real emitters, under the
    names the model reports — so it matches the model whether the dataset was generated or uploaded.
    None when the dataset is gone (the caller falls back to the simulator)."""
    from src.data.sequence_features import SIMULATOR_INFO
    _check_run_dataset(r)  # a changed file would show tracks the model never saw under its names
    df, _meta = _load_sequence_dataset(dataset_name)
    if df is None or "instance" not in df.columns:
        return None
    known_ids = sorted(set(clf.proto_labels))
    known_labels = [{"id": c, "label": clf.name(c)} for c in known_ids]
    first = df.groupby("instance", sort=False).first()
    names = first["class"].astype(str) if "class" in first.columns else None
    if clf.class_labels and names is not None:        # match by the emitter's name (class ids can differ between files)
        index = {str(l): i for i, l in enumerate(clf.class_labels)}
        model_ids = names.map(lambda l: index.get(l, -1))
    elif "true_class_id" in first.columns:
        model_ids = first["true_class_id"].astype(int)
    else:
        return None
    if names is None:
        names = model_ids.map(lambda c: clf.name(int(c)) if c >= 0 else "UNKNOWN")
    noise = names.str.contains("NOISE", case=False)
    if req.kind == "noise":
        pool = names.index[noise]
    elif req.kind == "unknown":
        pool = names.index[(model_ids < 0) & ~noise]
        if not len(pool):
            pool = names.index[noise]
    else:
        pool = names.index[model_ids.isin(known_ids) & (True if req.class_id is None else (model_ids == req.class_id))]
    note = None
    if req.source == "test":
        test_ids = {str(t) for t in ((registry.load_report(r["id"]) or {}).get("cross_validation") or {}).get("test_instances", [])}
        narrowed = [t for t in pool if str(t) in test_ids]
        if narrowed:
            pool = narrowed
        else:
            note = "The model has no held-out test track of this kind, so the example comes from its whole dataset (it may include training tracks)."
    if not len(pool):
        what = {"known": f"known emitter {clf.name(req.class_id)}" if req.class_id is not None else "known emitter",
                "unknown": "unknown (held-out) emitter", "noise": "noise"}.get(req.kind, req.kind)
        raise ValueError(f"Dataset '{dataset_name}' has no {what} tracks to take an example from.")
    rng = np.random.RandomState()
    tid = pool[int(rng.randint(len(pool)))]
    track = df[df["instance"] == tid]
    if "pulse_idx" in track.columns:
        track = track.sort_values("pulse_idx", kind="stable")
    n = min(req.pulses, len(track))
    start = int(rng.randint(0, len(track) - n + 1))
    seg = track.iloc[start:start + n].copy()
    if n < req.pulses:
        note = (note + " " if note else "") + f"That track has only {n} pulses (you asked for {req.pulses})."
    seg["instance"] = "example"
    seg["pulse_idx"] = range(n)
    cols = [c for c in seg.columns if c not in SIMULATOR_INFO and c != "true_class_id"]
    label = str(names[tid])
    cid = int(model_ids[tid])
    info = lambda col: str(seg[col].iloc[0]) if col in seg.columns and pd.notna(seg[col].iloc[0]) else "n/a"
    return {"status": "success", "csv_text": seg[cols].to_csv(index=False), "true_label": label, "true_class_id": cid,
            "pri_pattern": info("pri_pattern"), "freq_agility": info("freq_agility"),
            "known_class_ids": known_ids, "known_classes": known_labels,
            "source": f"dataset '{dataset_name}' (track {tid}, pulses {start + 1}–{start + n})", "from_dataset": True, "note": note}


@router.post("/api/sequence/example_track")
def sequence_example_track(req: ExampleTrackRequest):
    """An example track for trying sequence Inference without your own data. Taken from the dataset
    the model was trained on (its real emitters and names); only when that dataset no longer exists
    is one simulated from the model's recorded population."""
    try:
        clf, r = _sequence_classifier(_resolve_sequence_run(req.run_id))
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    if req.kind not in ("known", "unknown", "noise"):
        return JSONResponse({"status": "error", "message": "Kind must be known, unknown or noise."}, status_code=400)
    if not 4 <= req.pulses <= 2000:
        return JSONResponse({"status": "error", "message": "Pulses 4–2000."}, status_code=400)
    dataset_name = r.get("dataset_name")
    if dataset_name and datasets.exists(dataset_name):
        try:
            out = _dataset_example_track(req, clf, r, dataset_name)
        except ValueError as e:
            return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
        except Exception:
            log.exception("example track from dataset failed")
            return JSONResponse({"status": "error", "message": f"Could not read an example from dataset '{dataset_name}' — see the server log."}, status_code=400)
        if out is not None:
            return JSONResponse(out)
    if r.get("dataset_seed") is None or r.get("missed_pct") is None:
        return JSONResponse({"status": "error", "message": (
            f"This model's dataset ({dataset_name or 'unnamed'}) is no longer stored and it was not a generated population, "
            f"so no example can be made — paste or load a track file.")}, status_code=400)
    extras = _model_extras(r)
    if _simulator_gap(clf.spec, extras):
        return JSONResponse({"status": "error", "message": (
            f"This model uses dataset column(s) {_simulator_gap(clf.spec, extras)} that the track simulator can't produce — "
            f"paste or load a track file that has them.")}, status_code=400)
    if not 4 <= req.pulses <= 2000 or not 0 <= req.missed_pct <= 0.5 or not 0 <= req.spurious_pct <= 0.5:
        return JSONResponse({"status": "error", "message": "Pulses 4–2000; missed/spurious 0–50%."}, status_code=400)
    seed, ranges, holdout, library = _model_population(r)
    rng = np.random.RandomState()
    known = [c for c in range(r["num_classes"]) if c not in holdout]
    if req.kind == "noise" or (req.kind == "unknown" and not holdout):
        pulses, label, cid, pattern, agility = generate_noise_track(req.pulses, rng), "UNKNOWN_NOISE_OOD", -1, "n/a", "n/a"
    else:
        if req.kind == "unknown":
            c = int(rng.choice(holdout))
        else:
            c = req.class_id if req.class_id in known else int(rng.choice(known))
        pulses = generate_track(library[c], req.pulses, rng, req.missed_pct, req.spurious_pct, ranges)
        label = "UNKNOWN_HOLDOUT" if c in holdout else f"RADAR_FAMILY_{c:02d}"
        cid, pattern, agility = (-1 if c in holdout else c), library[c]["pri_pattern"], library[c]["seq"]["freq_agility"]
    add_side_columns(pulses, None if cid < 0 and label == "UNKNOWN_NOISE_OOD" else family_modulation(r["num_classes"], seed)[c], rng)
    add_extra_columns(pulses, extras, r["num_classes"], seed, None if cid < 0 and label == "UNKNOWN_NOISE_OOD" else c, rng)
    rows = [{"Frequency": round(p["freq_mhz"], 6), "PW": round(p["pw"], 4),
             "PRI": round(p["interval"], 3) if i > 0 else None, "Modlntype": p["mod_type"], "Modlnval": round(p["mod_val"], 4),
             "snr": round(p["snr"], 2), "rise": round(p["rise"], 1), "fall": round(p["fall"], 1),
             "timeBW": round(p["bw"] * 1e6), **{e["name"]: p[e["name"]] for e in extras}, "instance": "example", "pulse_idx": i}
            for i, p in enumerate(pulses)]
    csv_text = pd.DataFrame(rows).to_csv(index=False)
    return JSONResponse({"status": "success", "csv_text": csv_text, "true_label": label, "true_class_id": cid,
                         "pri_pattern": str(pattern), "freq_agility": str(agility), "known_class_ids": known,
                         "known_classes": [{"id": c, "label": clf.name(c)} for c in known], "from_dataset": False,
                         "source": "simulated (the model's dataset is no longer stored)",
                         "note": f"Dataset '{dataset_name}' is no longer stored, so this track was simulated from the model's recorded population."})


@router.websocket("/ws/sequence_live")
async def sequence_live(websocket: WebSocket, run_id: Optional[int] = None, emitters: int = 5, unknown_pct: float = 0.3,
                        missed_pct: float = 0.02, spurious_pct: float = 0.02, duration: int = 120, rate: float = 20.0):
    """Live Simulation of de-interleaved tracks: several simulated emitters transmit at once;
    each pulse arrives already tagged with its track (de-interleaver output). Every time a track
    gets a pulse, its last ≤ k pulses are classified, so you watch the decision firm up as the
    track grows. Tracks end and new emitters appear. Ground truth is sent for scoring."""
    user = _session_user(websocket.cookies)
    if user is None or not _same_origin(websocket.headers):
        await websocket.close(code=1008)
        return
    if jobs.live_sessions >= MAX_LIVE_SESSIONS:
        await websocket.close(code=1013)  # try again later
        return
    jobs.live_sessions += 1
    audit.record("live_session", user, _client_ip(websocket), run_id=run_id)
    try:
        await _run_live_session(websocket, run_id, emitters, unknown_pct, missed_pct, spurious_pct, duration, rate)
    finally:
        jobs.live_sessions -= 1


async def _run_live_session(websocket: WebSocket, run_id, emitters, unknown_pct, missed_pct, spurious_pct, duration, rate):
    await websocket.accept()
    try:
        clf, r = await asyncio.to_thread(lambda: _sequence_classifier(_resolve_sequence_run(run_id)))
        live_extras = _model_extras(r)
        if _simulator_gap(clf.spec, live_extras):
            raise ValueError(f"This sequence model uses dataset column(s) {_simulator_gap(clf.spec, live_extras)} that the track "
                             f"simulator can't produce, so it can't run on simulated live tracks. Choose another sequence model.")
    except ValueError as e:
        await websocket.send_text(json.dumps({"type": "error", "error": _public_error(e)}))
        await websocket.close()
        return
    emitters = max(1, min(int(emitters), 12))
    unknown_pct = min(max(float(unknown_pct), 0.0), 1.0)
    missed_pct = min(max(float(missed_pct), 0.0), 0.5)
    spurious_pct = min(max(float(spurious_pct), 0.0), 0.5)
    duration = max(5, min(int(duration), MAX_LIVE_FEED_SECONDS))
    rate = max(1.0, min(float(rate), 100.0))
    seed, ranges, holdout, library = _model_population(r)
    known = [c for c in range(r["num_classes"]) if c not in holdout]
    rng = np.random.RandomState()
    k = clf.spec.window
    next_id = [1]

    def spawn():
        unknown = rng.random() < unknown_pct
        n = int(rng.randint(40, 201))
        if unknown and holdout and rng.random() < 0.7:
            c = int(rng.choice(holdout))
            pulses, truth, cid = generate_track(library[c], n, rng, missed_pct, spurious_pct, ranges), "UNKNOWN_HOLDOUT", -1
        elif unknown:
            pulses, truth, cid = generate_noise_track(n, rng), "UNKNOWN_NOISE_OOD", -1
        else:
            c = int(rng.choice(known))
            pulses, truth, cid = generate_track(library[c], n, rng, missed_pct, spurious_pct, ranges), f"RADAR_FAMILY_{c:02d}", c
        fam = c if cid >= 0 or truth == "UNKNOWN_HOLDOUT" else None
        add_side_columns(pulses, family_modulation(r["num_classes"], seed)[fam] if fam is not None else None, rng)
        add_extra_columns(pulses, live_extras, r["num_classes"], seed, fam, rng)
        tid = next_id[0]; next_id[0] += 1
        return {"id": tid, "pulses": pulses, "pos": 0, "rows": [], "truth": truth, "truth_id": cid}

    active = [spawn() for _ in range(emitters)]
    await websocket.send_text(json.dumps({"type": "start", "run_id": r["id"], "model_name": r.get("saved_name"),
                                          "backbone_type": r["backbone_type"], "window": k,
                                          "thresholds": {"tau_c": clf.tau_c, "tau_d": clf.tau_d}}))
    start = time.time()
    try:
        while time.time() - start < duration:
            em = active[int(rng.randint(len(active)))]
            p = em["pulses"][em["pos"]]
            em["rows"].append({"pulse_idx": em["pos"], "Frequency": p["freq_mhz"], "PW": p["pw"],
                               "PRI": p["interval"] if em["pos"] > 0 else None, "rise": p["rise"],
                               "fall": p["fall"], "timeBW": p["bw"] * 1e6, "Modlntype": p["mod_type"],
                               "Modlnval": p["mod_val"], "snr": p["snr"],
                               **{e["name"]: p[e["name"]] for e in live_extras}})
            em["pos"] += 1
            recent = pd.DataFrame(em["rows"][-(k + 1):])
            if clf.needs_pri and not has_pri(recent):
                # first pulse of a track: no PRI yet, so no decision (the track appears with its 2nd pulse)
                await asyncio.sleep(1.0 / rate)
                continue
            # Feature building + the model run happen in a worker thread, so other users' requests and
            # WebSockets aren't stalled by every simulated pulse.
            decision = await asyncio.to_thread(_live_decision, clf, em["rows"], recent, k)
            known_truth = em["truth_id"] >= 0
            correct = (decision["class_id"] == em["truth_id"] and decision["verdict"] == "VERIFIED_KNOWN_TARGET") if known_truth \
                else decision["verdict"] != "VERIFIED_KNOWN_TARGET"
            ended = em["pos"] >= len(em["pulses"])
            await websocket.send_text(json.dumps({
                "type": "pulse", "track_id": em["id"], "pulses_seen": em["pos"], "track_length": len(em["pulses"]),
                "pulse": {"pri_us": em["rows"][-1]["PRI"], "freq_mhz": em["rows"][-1]["Frequency"],
                          "pulse_width_us": p["pw"], "bandwidth_hz": p["bw"] * 1e6, "spurious": bool(p["spurious"])},
                "decision": decision, "truth": em["truth"], "known": known_truth, "correct": bool(correct), "ended": ended,
                "elapsed": round(time.time() - start, 1),
            }))
            if ended:
                active[active.index(em)] = spawn()
            await asyncio.sleep(1.0 / rate)
        await websocket.send_text(json.dumps({"type": "complete", "elapsed": round(time.time() - start, 1)}))
        await websocket.close()
    except WebSocketDisconnect:
        pass


def _live_decision(clf, rows: List[dict], recent: pd.DataFrame, k: int) -> dict:
    if clf.spec.whole:  # track mode: the decision is on the track's first ≤ k pulses
        matrix = track_feature_matrix(pd.DataFrame(rows[:k]), clf.spec.names)
    else:
        matrix = track_feature_matrix(recent, clf.spec.names)[-k:]
    return clf.classify([matrix])[0]

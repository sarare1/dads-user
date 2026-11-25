"""Training-job status and stop, run history, saving / deleting runs, model and weight downloads, test history."""

import copy
import hashlib
import io
import json
import os
import re
import torch
from datetime import datetime, timezone
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from src.data.sequence_features import (
    DATASET_CATEGORY, DATASET_DELTA, dataset_feature_column, MODULATION_TYPES, seq_feature_label, SEQ_FEATURES,
    ZERO_IS_MISSING_SEQ,
)
from src.models.checkpoint import load_sequence_checkpoint
from src.models.explain import weights_export
from src.storage.training_store import (
    delete_run, get_all_runs, get_all_test_runs, get_latest_run, get_run, set_saved_name,
)
from src.training import cancel
from src.api.core import (
    _client_ip, _end_job, _public_error, APP_VERSION, audit, DEPLOY_SEQUENCE_SCRIPT, DEPLOY_VERIFY_SCRIPT, jobs,
    MAX_HISTORY_ROWS, package_signer, registry, training_status,
)
from src.api.common import _seq_cache, _sequence_classifier, TEST_RESULTS_DIR
router = APIRouter()


@router.get("/api/train/status")
async def train_status():
    if training_status["is_training"] and training_status["state"] != "starting" and (jobs.training_task is None or jobs.training_task.done()):
        # The job's task is gone but the flag is still set (it died without cleaning up): heal, say why.
        _end_job("failed", "The training job ended unexpectedly. See the server log; you can start training again.")
    status = dict(training_status)
    if status["last_metrics"] is None and not status["is_training"]:
        # Survives process restarts/page navigation — the in-memory dict is empty right after
        # a fresh start, but the last real run is still on disk.
        latest = get_latest_run()
        if latest is not None:
            status["last_metrics"] = latest
    return JSONResponse(status)

@router.post("/api/train/stop")
async def stop_training():
    """Asks the running training / incremental job to stop. It unwinds within one batch; nothing is
    saved for a stopped run and the model in use stays as it was."""
    if not training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "No training is running."}, status_code=409)
    cancel.request()
    training_status["state"] = "stopping"
    return JSONResponse({"status": "success", "message": "Stopping… the current batch finishes first."})


@router.get("/api/train/history")
def train_history(limit: int = Query(50, ge=1, le=MAX_HISTORY_ROWS)):
    runs = get_all_runs(limit)
    active = registry.get_active_sequence()
    for r in runs:
        r["has_model"] = registry.has_artifacts(r["id"])
        r["is_active"] = r["id"] == active
    return JSONResponse({"status": "success", "runs": runs, "active_run_id": active})


@router.get("/api/runs/{run_id}")
def run_details(run_id: int):
    r = get_run(run_id)
    if r is None:
        return JSONResponse({"status": "error", "message": f"No training run #{run_id}."}, status_code=404)
    return JSONResponse({
        "status": "success",
        "run": r,
        "has_model": registry.has_artifacts(run_id),
        "is_active": run_id == registry.get_active_sequence(),
        "report": registry.load_report(run_id),
    })



class SaveModelRequest(BaseModel):
    name: str


_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,59}$")


@router.post("/api/runs/{run_id}/save")
async def save_model(run_id: int, req: SaveModelRequest):
    name = req.name.strip()
    if not _MODEL_NAME.match(name):
        return JSONResponse({"status": "error", "message": "Name must be 1–60 characters: letters, digits, spaces, '_', '-' or '.', starting with a letter or digit."}, status_code=400)
    if get_run(run_id) is None:
        return JSONResponse({"status": "error", "message": f"No training run #{run_id}."}, status_code=404)
    if not registry.has_artifacts(run_id):
        return JSONResponse({"status": "error", "message": "This run's trained model wasn't kept (it predates saved models, or was pruned), so it can't be saved."}, status_code=400)
    try:
        set_saved_name(run_id, name)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=409)
    return JSONResponse({"status": "success", "message": f"Saved run #{run_id} as '{name}'."})


@router.delete("/api/runs/{run_id}/save")
async def unsave_model(run_id: int):
    if get_run(run_id) is None:
        return JSONResponse({"status": "error", "message": f"No training run #{run_id}."}, status_code=404)
    set_saved_name(run_id, None)
    return JSONResponse({"status": "success", "message": f"Run #{run_id} is no longer saved (its model may be pruned once more than 20 newer unsaved runs exist, unless it's active)."})


@router.delete("/api/runs/{run_id}")
async def delete_training_run(run_id: int):
    """Permanently deletes a training run: its Run History row, its kept model files and the
    Testing-page results of its model."""
    r = get_run(run_id)
    if r is None:
        return JSONResponse({"status": "error", "message": f"No training run #{run_id}."}, status_code=404)
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "A training run is in progress — delete runs after it finishes."}, status_code=409)
    if registry.get_active_sequence() == run_id:
        registry.set_active_sequence(None)
    _seq_cache.clear()
    registry.delete_artifacts(run_id)
    tests_removed = delete_run(run_id)
    label = f"'{r['saved_name']}' (run #{run_id})" if r.get("saved_name") else f"Run #{run_id}"
    tests_note = f", along with {tests_removed} test result(s) of its model" if tests_removed else ""
    return JSONResponse({"status": "success", "message": f"{label} deleted{tests_note}.",
                         "tests_removed": tests_removed})



def _download_name(r: dict) -> str:
    base = r.get("saved_name") or f"run_{r['id']}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") or f"run_{r['id']}"


@router.get("/api/runs/{run_id}/download")
def download_model(run_id: int, request: Request):
    """Deployable package: model.pt, model.onnx, torchscript, prototypes, model card, report, README."""
    r = get_run(run_id)
    if r is None or not registry.has_artifacts(run_id):
        return JSONResponse({"status": "error", "message": "No kept model for this run."}, status_code=404)
    try:
        data = _sequence_package(r)
    except ValueError as e:
        return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=400)
    audit.record("package_export", getattr(request.state, "user", None), _client_ip(request), run_id=run_id,
                 sha256=hashlib.sha256(data).hexdigest(), signed=package_signer.available)
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{_download_name(r)}.zip"'})


@router.get("/api/runs/{run_id}/weights")
def download_weights(run_id: int):
    """Every trained weight of this run's model as JSON (layer name -> shape + values)."""
    r = get_run(run_id)
    if r is None or not registry.has_artifacts(run_id):
        return JSONResponse({"status": "error", "message": "No kept model for this run."}, status_code=404)
    m, _, _ = load_sequence_checkpoint(registry.checkpoint_path(run_id))
    return Response(content=json.dumps(weights_export(m)), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{_download_name(r)}_weights.json"'})

@router.get("/api/test/history")
def test_history(limit: int = Query(50, ge=1, le=MAX_HISTORY_ROWS)):
    return JSONResponse({"status": "success", "runs": get_all_test_runs(limit)})


@router.get("/api/test/history/{test_id}")
def test_history_detail(test_id: int):
    """The stored full result of an earlier sequence test (older tests, from before details were kept, have none)."""
    path = os.path.join(TEST_RESULTS_DIR, f"test_{int(test_id)}.json")
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "No detailed result was stored for this test (it was run before details were kept)."}, status_code=404)
    with open(path, encoding="utf-8") as f:
        return JSONResponse({"status": "success", **json.load(f)})


def _sequence_model_card(r: dict, clf) -> dict:
    defs = []
    for n in clf.spec.names:
        col = dataset_feature_column(n)
        if col is not None:
            d = {"name": n, "label": seq_feature_label(n), "unit": "", "source": "dataset",
                 "op": "delta" if n.startswith(DATASET_DELTA) else "value", "columns": [col]}
            if n.startswith(DATASET_CATEGORY):
                d["categorical"] = "hash"
            defs.append(d)
            continue
        f = SEQ_FEATURES[n]
        op = "delta" if n.startswith("d_") else "ratio" if n == "duty_cycle" else "product" if n == "pw_x_pri" else "value"
        d = {"name": n, "label": f.label, "unit": f.unit, "op": op, "columns": list(f.columns), "scale": f.scale}
        if n == "modtype":
            d["categorical"] = "modulation"
        defs.append(d)
    return {
        "kind": "sequence", "name": r.get("saved_name"), "run_id": r["id"], "trained_at": r["timestamp_ist"],
        # provenance: the code release that trained it, the exact training data and the training seed
        "dads_version": r.get("app_version"), "packaged_by_dads_version": APP_VERSION,
        "dataset_name": r.get("dataset_name"), "dataset_sha256": r.get("dataset_sha256"), "train_seed": r.get("train_seed"),
        "backbone_type": r["backbone_type"], "num_classes": r["num_classes"],
        "class_labels": [clf.name(c) for c in range(r["num_classes"])],
        "modulation_types": MODULATION_TYPES,
        "features": clf.spec.names, "feature_definitions": defs, "input_columns": clf.spec.columns,
        "zero_is_missing_columns": sorted(ZERO_IS_MISSING_SEQ), "window": clf.spec.window, "window_mode": clf.spec.mode,
        "scaling": {"mean": clf.spec.mean.tolist(), "std": clf.spec.std.tolist(),
                    "delta_log_scale": clf.spec.delta_scale.tolist() if clf.spec.delta_scale is not None else None},
        "missing_value_rule": ("blank (or 0 rise/fall time) → leading gaps (the first pulse's PRI) take the first measured value, "
                               "other gaps that track's median; first pulse: Δ = 0; a track needs ≥ 2 pulses (a PRI)"),
        "scaling_rule": "Δ features: x → sign(x)·ln(1 + |x|/delta_log_scale) first (when delta_log_scale > 0); then (x − mean) / std",
        "track_rule": ("windows of `window` consecutive pulses (+ a final window ending at the last pulse); class = most frequent window class, "
                       "verdict = most frequent window verdict, ties → the one seen first"),
        "distance_metric": r["distance_metric"],
        "decision_rule": {"classifier_confidence_threshold": clf.tau_c, "ood_distance_threshold": clf.tau_d,
                          # windows shorter than `window` (short tracks): τ_d of the largest listed length ≤ n
                          "ood_distance_threshold_by_length": clf.tau_d_by_length},
        "metrics": {k: r.get(k) for k in ("closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss")},
    }


def _sequence_package(r: dict) -> bytes:
    clf, _ = _sequence_classifier(r["id"])
    model_ = copy.deepcopy(clf.model).eval()  # export a private copy: Inference / Live may be running the cached one
    onnx_buf = io.BytesIO()
    dummy = torch.randn(1, clf.spec.window, clf.spec.dim)
    torch.onnx.export(model_, dummy, onnx_buf, input_names=["pulses"], output_names=["probs", "embeddings"],
                      dynamic_axes={"pulses": {0: "batch", 1: "n_pulses"}, "probs": {0: "batch"}, "embeddings": {0: "batch"}},
                      opset_version=17, dynamo=False)
    ts = io.BytesIO()
    torch.jit.save(torch.jit.script(model_), ts)
    with open(DEPLOY_SEQUENCE_SCRIPT, "rb") as f:
        script = f.read()
    with open(DEPLOY_VERIFY_SCRIPT, "rb") as f:
        verifier = f.read()
    provenance = {"dads_version": APP_VERSION, "dataset": r.get("dataset_source"),
                  "dataset_sha256": r.get("dataset_sha256"), "trained_by_version": r.get("app_version"),
                  "train_seed": r.get("train_seed"), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return registry.build_sequence_package(r["id"], _sequence_model_card(r, clf), {
        "model.onnx": onnx_buf.getvalue(), "model.torchscript.pt": ts.getvalue(), "sequence_inference.py": script,
        "verify_package.py": verifier}, provenance=provenance, signer=package_signer)

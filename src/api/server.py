import os
import torch
import numpy as np
import pandas as pd
import yaml
from typing import Optional, List, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
import asyncio
import copy
import json
import time
import hashlib
import io
import logging
import math
import re
import secrets
import shutil
import threading
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

from src.data.generator import frequency_to_band
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.analytics.dataset_analysis import (
    analysis_frame, row_groups, feature_distributions, feature_ranking, rf_spectrum,
    mode_by_class, data_quality, dataset_overview,
)
from src.models.checkpoint import save_sequence_checkpoint, load_sequence_checkpoint
from src.data.sequence_generator import (
    build_sequence_dataset_rows, build_sequence_library, generate_track, generate_noise_track,
    validate_extra_features, add_side_columns, family_modulation, add_extra_columns,
)
from src.data.dataset_expander import expand_dataset
from src.data.sequence_features import (
    SEQ_FEATURES, DEFAULT_SEQ_FEATURES, DEFAULT_WINDOW, WINDOW_MODES, DEFAULT_TRACK_CAP, MAX_TRACK_PULSES, MODULATION_TYPES, validate_seq_features, required_seq_columns,
    load_sequence_dataset_for_training, track_feature_matrix, ordered_tracks, ZERO_IS_MISSING_SEQ,
    sequence_dataset_columns, dataset_feature_column, seq_feature_label, DATASET_VALUE, DATASET_DELTA, has_pri,
    upgrade_columns, split_track_ids, cv_split, sequence_categorical_columns, DATASET_CATEGORY, order_columns,
)
from src.models.sequence_models import SequenceDualHeadModel, SEQ_BACKBONES
from src.models.sequence_eval import (
    evaluate_tracks, sequence_architecture_report, sequence_feature_importance, calibrate_distance_threshold, relabel_by_name,
    SequenceClassifier,
)
from src.training import cancel
from src.training.cancel import TrainingCancelled
from src.training.trainer import train_model, build_distance_fn, _select_holdout_ids, seed_training
from src.storage.training_store import (
    _COLUMNS as RUN_COLUMNS, init_db, insert_run, get_latest_run, get_all_runs, insert_test_run, get_all_test_runs,
    get_run, set_saved_name, get_saved_runs, delete_run, referenced_dataset_names,
)
from src.models.registry import ModelRegistry
from src.models.package_signing import PackageSigner
from src.training.incremental import (
    METHODS as INCREMENTAL_METHODS, run_method as run_incremental_method, score as incremental_score, map_labels, prefix_tracks,
)
from src.storage.dataset_store import DatasetStore
from src.storage.auth_store import AuthStore, LoginLimiter, AuditLog
from src.models.explain import evaluation_report, weights_export, branch_ablation


_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")
APP_VERSION = open(_VERSION_FILE).read().strip() if os.path.exists(_VERSION_FILE) else "unknown"  # release = git tag v<VERSION>

app = FastAPI(title="DADS - PhD Research Project", version=APP_VERSION)

UI_DIR = os.path.join(os.path.dirname(__file__), "../../ui")
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# --- Auth gate -------------------------------------------------------------
# Named accounts with PBKDF2-hashed passwords, random server-side sessions, sign-in throttling and a
# hash-chained audit trail: see src/storage/auth_store.py. No credential is in the source code; accounts are
# managed with scripts/manage_users.py.
log = logging.getLogger("dads")
DATA_DIR = os.environ.get("DADS_DATA_DIR") or os.path.join(os.path.dirname(__file__), "../../data")  # override: tests / ops
os.makedirs(DATA_DIR, exist_ok=True)


def _setup_logging() -> None:
    """DADS's own messages (and errors with their tracebacks) go to data/logs/dads.log, rotated at 5 MB x 5,
    as well as to the console. Before, INFO messages were dropped. DADS_LOG_LEVEL changes the level."""
    if any(getattr(h, "_dads", False) for h in log.handlers):
        return  # module reloaded (tests, --reload)
    from logging.handlers import RotatingFileHandler
    os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in (RotatingFileHandler(os.path.join(DATA_DIR, "logs", "dads.log"), maxBytes=5_000_000, backupCount=5,
                                  encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        h._dads = True
        log.addHandler(h)
    log.setLevel(os.environ.get("DADS_LOG_LEVEL", "INFO").upper())
    log.propagate = False


_setup_logging()
SESSION_COOKIE_NAME = "dads_session"
auth = AuthStore(os.path.join(DATA_DIR, "auth.db"),
                 idle_minutes=int(os.environ.get("DADS_SESSION_IDLE_MINUTES", "120")),
                 absolute_hours=int(os.environ.get("DADS_SESSION_MAX_HOURS", "12")))
login_limiter = LoginLimiter()
audit = AuditLog(os.path.join(DATA_DIR, "audit.log"))
_first_start_note = auth.ensure_initial_admin(DATA_DIR)
if _first_start_note:
    log.warning(_first_start_note)
    print("\n*** " + _first_start_note + " ***\n", flush=True)
    audit.record("initial_admin_created", None, None)
_COOKIE_SECURE_ALWAYS = os.environ.get("DADS_COOKIE_SECURE") == "1"
_PUBLIC_PATHS = {"/login", "/api/login", "/static/shared.css", "/api/health"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_AUDITED_GETS = re.compile(r"^/api/(runs/\d+/(download|weights)|sequence/dataset/download)$")
MAX_BODY_BYTES = 110 * 1024 * 1024  # largest request body accepted (uploads are capped at MAX_UPLOAD_BYTES)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_PULSES = 5_000_000
MAX_LIVE_SESSIONS = 4
MAX_HISTORY_ROWS = 500  # most rows one history request may return (SQLite treats LIMIT -1 as "all")
_live_sessions = 0
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; "
                                "form-action 'self'; frame-ancestors 'none'"),
}


def _client_ip(conn) -> str:
    # Behind Nginx, uvicorn's proxy-headers support (trusted from 127.0.0.1) puts X-Forwarded-For here.
    return conn.client.host if conn.client else "unknown"


def _same_origin(headers) -> bool:
    """Browsers always send Origin on cross-site POST/DELETE and WebSocket requests; refuse a foreign one."""
    origin = headers.get("origin")
    return not origin or urlparse(origin).netloc == headers.get("host")


def _session_user(cookies) -> Optional[str]:
    return auth.session_user(cookies.get(SESSION_COOKIE_NAME))


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path, method = request.url.path, request.method
    response = None
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        response = JSONResponse({"status": "error", "message": "Request too large."}, status_code=413)
    else:
        public = path in _PUBLIC_PATHS
        user = None if public else _session_user(request.cookies)
        if not public and user is None:
            if path.startswith("/api/"):
                response = JSONResponse({"status": "error", "message": "Not authenticated — please sign in."}, status_code=401)
            else:
                response = RedirectResponse(url="/login")
        elif method not in _SAFE_METHODS and not _same_origin(request.headers):
            audit.record("cross_origin_refused", user, _client_ip(request), method=method, path=path,
                         origin=request.headers.get("origin", "")[:200])
            response = JSONResponse({"status": "error", "message": "Cross-site request refused."}, status_code=403)
        else:
            request.state.user = user
            response = await call_next(request)
            if user and path.startswith("/api/") and (method not in _SAFE_METHODS or _AUDITED_GETS.match(path)):
                audit.record("request", user, _client_ip(request), method=method, path=path,
                             query=request.url.query[:300], status=response.status_code)
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response

MODELS_DIR = os.path.join(DATA_DIR, "models")
SEQUENCE_DATASET_DIR = os.path.join(DATA_DIR, "sequence_datasets")  # pulse-sequence (track) pipeline
os.makedirs(MODELS_DIR, exist_ok=True)
# Every dataset (generated, uploaded for training, uploaded for testing) is a named pair of files here.
init_db()
# A dataset name is never reused, including names that training/test runs still refer to (lineage).
datasets = DatasetStore(SEQUENCE_DATASET_DIR, reserved_names=referenced_dataset_names)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

current_config = load_config()
current_config["hyperparameters"].setdefault("rpl_loss_weight", 0.1)

DEPLOY_SEQUENCE_SCRIPT = os.path.join(os.path.dirname(__file__), "../models/deploy/sequence_inference.py")
DEPLOY_VERIFY_SCRIPT = os.path.join(os.path.dirname(__file__), "../models/package_signing.py")

registry = ModelRegistry(MODELS_DIR)
package_signer = PackageSigner(DATA_DIR)  # signs downloaded model packages (key in data/signing/, created on first use)


# Resuming a previous session: restore the settings of the sequence model in use (or, failing
# that, the latest sequence run), so the Training form agrees with what is already persisted
# instead of silently falling back to config.yaml's defaults.
_active_seq = registry.get_active_sequence()
_last_run = (get_run(_active_seq) if _active_seq is not None else None) or get_latest_run()
if _last_run is not None:
    if _last_run.get("embedding_dim"):
        current_config["model"]["embedding_dim"] = _last_run["embedding_dim"]
    current_config["model"]["num_classes"] = _last_run["num_classes"]
    current_config["hyperparameters"]["distance_metric"] = "cosine" if _last_run["distance_metric"] == "dot" else _last_run["distance_metric"]
    for _key in ("classifier_confidence_threshold", "ood_distance_threshold"):
        if _last_run.get(_key) is not None:
            current_config["hyperparameters"][_key] = _last_run[_key]
    if _last_run.get("rpl_loss_weight") is not None:
        current_config["hyperparameters"]["rpl_loss_weight"] = _last_run["rpl_loss_weight"]
    if _last_run.get("rpl_adversarial") is not None:
        current_config["hyperparameters"]["rpl_adversarial"] = _last_run["rpl_adversarial"]

# state: idle | starting | running | stopping | succeeded | failed | cancelled   (job_id counts starts, so a
# page can tell "my job finished" from "nothing started yet" even if the job ends within a second)
training_status = {"is_training": False, "progress": None, "last_metrics": None, "error": None,
                   "state": "idle", "phase": None, "job_id": 0, "message": None}
_training_task = None


def _begin_job(phase: str) -> int:
    """Marks a training/incremental job as started (before any slow work) and clears a pending stop."""
    cancel.reset()
    training_status.update(is_training=True, state="starting", phase=phase, error=None, message=None,
                           job_id=training_status["job_id"] + 1)
    return training_status["job_id"]


def _set_phase(phase: str, state: str = "running") -> None:
    training_status["phase"] = phase
    if training_status["state"] != "stopping":
        training_status["state"] = state


def _end_job(state: str, error: Optional[str] = None, message: Optional[str] = None) -> None:
    training_status.update(is_training=False, state=state, phase=None, error=error, message=message)


# A Windows path (C:\... or \\server\...) or a POSIX path with at least two parts (/root/dads/...).
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?=[\w.-]+/))[^\s'\"<>|,;)]*")


def _public_error(e: BaseException, limit: int = 300) -> str:
    """An exception's message as it may be shown in the browser: file-system paths are cut down to the file
    name (they reveal the server's layout) and the length is capped. The full text stays in the server log."""
    text = _PATH_RE.sub(lambda m: "…/" + re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1], str(e))
    return text if len(text) <= limit else text[:limit] + "…"


def _error_reference(e: BaseException, what: str) -> str:
    """For an unexpected error: log the full traceback under a short reference, and return a message with only
    that reference. The user quotes it, the operator finds it in data/logs/dads.log."""
    ref = secrets.token_hex(4)
    log.error("%s failed [ref %s]", what, ref, exc_info=e)
    return f"An internal error occurred (reference {ref}). The administrator can find the details in the server log."


def _explain_error(e: BaseException) -> str:
    """The reason a job failed, in words — with a hint for the usual causes. Anything else is an internal error:
    the browser gets a reference, never the exception text (it can contain paths and internal details)."""
    if isinstance(e, MemoryError):
        return "Out of memory while training. Use a smaller window, fewer epochs or fewer/shorter tracks, or free memory and retry."
    if isinstance(e, KeyError):
        return f"The dataset is missing a column the model needs ({_public_error(e, 80)}). Check the column names on the Dataset page."
    if isinstance(e, ValueError) and "nan" in str(e).lower():
        return f"The data contains missing/NaN values the features can't use ({_public_error(e, 200)}). Clean those columns or deselect the features that use them."
    if isinstance(e, ValueError) and ("empty" in str(e).lower() or "no window" in str(e).lower() or "shorter" in str(e).lower()):
        return f"Not enough usable data: {_public_error(e, 200)}. Try a smaller window, or a dataset with longer tracks / more tracks per class."
    if isinstance(e, ValueError):  # DADS's own checks raise ValueError with a message written for the user
        return _public_error(e)
    return _error_reference(e, "job")


class ConfigUpdateModel(BaseModel):
    learning_rate: float
    triplet_margin: float
    classifier_confidence_threshold: float
    ood_distance_threshold: float
    distance_metric: str
    rpl_loss_weight: float = 0.1
    rpl_adversarial: bool = False
    embedding_dim: Optional[int] = None      # None = keep; size of the feature vector f and the embedding e


EMBEDDING_DIM_RANGE = (2, 512)



class DatasetRanges(BaseModel):
    """Physical parameter ranges the 20 radar-family archetypes are drawn from. Defaults
    match the original hardcoded constants exactly, so leaving these untouched reproduces
    prior behavior — only the dataset_seed (drawn fresh each run) changes by default."""
    freq_min_mhz: float = 2000.0
    freq_max_mhz: float = 18000.0
    pw_min_us: float = 0.5
    pw_max_us: float = 50.0
    pri_min_us: float = 10.0
    pri_max_us: float = 1000.0
    rise_min_ns: float = 5.0
    rise_max_ns: float = 150.0
    bw_min_mhz: float = 1.0
    bw_max_mhz: float = 100.0
    fall_min_ns: float = 5.0
    fall_max_ns: float = 200.0
    duty_min_pct: float = 0.1    # family duty cycle (PW / PRI) range, % — PW = duty × PRI
    duty_max_pct: float = 10.0

    def to_ranges_dict(self) -> dict:
        return {
            "freq_range": (self.freq_min_mhz, self.freq_max_mhz),
            "pw_range": (self.pw_min_us, self.pw_max_us),
            "pri_range": (self.pri_min_us, self.pri_max_us),
            "rise_range": (self.rise_min_ns, self.rise_max_ns),
            "bandwidth_range": (self.bw_min_mhz, self.bw_max_mhz),
            "fall_range": (self.fall_min_ns, self.fall_max_ns),
            "duty_range": (self.duty_min_pct / 100.0, self.duty_max_pct / 100.0),
        }

def serve_page(filename: str):
    with open(os.path.join(UI_DIR, filename), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=1024)

@app.get("/login")
def get_login_page():
    return serve_page("login.html")

@app.post("/api/login")
async def login(req: LoginRequest, request: Request):
    ip, name = _client_ip(request), req.username.strip()
    wait = login_limiter.retry_after(ip, name)
    if wait:
        audit.record("login_blocked", name, ip, retry_after_s=wait)
        return JSONResponse({"status": "error", "message": f"Too many failed sign-ins. Try again in {max(1, round(wait / 60))} min."},
                            status_code=429, headers={"Retry-After": str(wait)})
    user = await asyncio.to_thread(auth.authenticate, name, req.password)  # PBKDF2 is slow on purpose
    if user is None:
        login_limiter.failed(ip, name)
        audit.record("login_failed", name, ip)
        return JSONResponse({"status": "error", "message": "Invalid User ID or password."}, status_code=401)
    login_limiter.succeeded(ip, name)
    auth.revoke(request.cookies.get(SESSION_COOKIE_NAME))  # never keep a pre-login session (fixation)
    token = auth.create_session(user, ip)
    audit.record("login", user, ip)
    resp = JSONResponse({"status": "success", "user": user})
    resp.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, httponly=True, samesite="strict", path="/",
        max_age=auth.absolute_s, secure=_COOKIE_SECURE_ALWAYS or request.url.scheme == "https",
    )
    resp.delete_cookie("dads_auth")  # cookie of the old shared-secret scheme
    return resp

@app.post("/api/logout")
async def logout(request: Request):
    auth.revoke(request.cookies.get(SESSION_COOKIE_NAME))
    audit.record("logout", getattr(request.state, "user", None), _client_ip(request))
    resp = JSONResponse({"status": "success"})
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp

@app.get("/api/me")
def whoami(request: Request):
    return JSONResponse({"status": "success", "user": request.state.user})

@app.get("/")
def get_training_page():
    return serve_page("index.html")

@app.get("/validation")
def get_validation_page():
    return serve_page("validation.html")

@app.get("/inference")
def get_inference_page():
    return serve_page("inference.html")

@app.get("/dataset")
def get_dataset_page():
    return serve_page("dataset.html")

@app.get("/analytics")
def get_analytics_page():
    return serve_page("analytics.html")

@app.get("/live")
def get_live_page():
    return serve_page("live.html")

@app.get("/help")
def get_help_page():
    return serve_page("help.html")


DISTANCE_METRICS = ("cosine", "euclidean")  # "dot" was always computed as cosine; old runs keep it as a label


def _validate_config(cfg: "ConfigUpdateModel") -> Optional[str]:
    checks = (("Learning rate", cfg.learning_rate, 0, 1), ("Triplet margin", cfg.triplet_margin, 0, 10),
              ("Min classifier confidence", cfg.classifier_confidence_threshold, 0, 1),
              ("Distance threshold", cfg.ood_distance_threshold, 0, 100), ("RPL loss weight", cfg.rpl_loss_weight, 0, 10))
    for label, value, lo, hi in checks:
        if not math.isfinite(value) or not lo <= value <= hi:
            return f"{label} must be a number between {lo} and {hi}."
    if cfg.distance_metric not in DISTANCE_METRICS:
        return f"Distance metric must be one of {list(DISTANCE_METRICS)}."
    return None


@app.get("/api/config")
async def get_current_config():
    return JSONResponse(content=current_config)

@app.post("/api/config")
async def update_config(cfg: ConfigUpdateModel):
    """The shared training settings (hyperparameters, distance metric, RPL)
    the next sequence training run starts from. Never touches a trained model or its stored
    prototypes."""
    error = _validate_config(cfg)
    if error:
        return JSONResponse({"status": "error", "message": error}, status_code=400)
    current_config["hyperparameters"]["learning_rate"] = cfg.learning_rate
    current_config["hyperparameters"]["triplet_margin"] = cfg.triplet_margin
    current_config["hyperparameters"]["classifier_confidence_threshold"] = cfg.classifier_confidence_threshold
    current_config["hyperparameters"]["ood_distance_threshold"] = cfg.ood_distance_threshold
    current_config["hyperparameters"]["distance_metric"] = cfg.distance_metric
    current_config["hyperparameters"]["rpl_loss_weight"] = cfg.rpl_loss_weight
    current_config["hyperparameters"]["rpl_adversarial"] = cfg.rpl_adversarial
    if cfg.embedding_dim is not None:
        if not EMBEDDING_DIM_RANGE[0] <= cfg.embedding_dim <= EMBEDDING_DIM_RANGE[1]:
            return JSONResponse({"status": "error", "message": f"Embedding size must be {EMBEDDING_DIM_RANGE[0]}–{EMBEDDING_DIM_RANGE[1]}."}, status_code=400)
        current_config["model"]["embedding_dim"] = int(cfg.embedding_dim)
    return JSONResponse({"status": "success", "message": "Configuration saved. It applies to the next training run; trained models are unaffected."})


@app.get("/api/version")
def app_version():
    return JSONResponse({"status": "success", "version": APP_VERSION})


@app.get("/api/health")
def health():
    """For monitoring (no sign-in needed, so it reveals nothing but up/down): the data folder is writable, the
    account and history databases answer and there is free disk. 503 when something is wrong."""
    problems = []
    try:
        auth.has_users()
        get_latest_run()
    except Exception:
        log.exception("health check: database")
        problems.append("database")
    if not os.access(DATA_DIR, os.W_OK):
        problems.append("data folder not writable")
    if shutil.disk_usage(DATA_DIR).free < 1_000_000_000:
        problems.append("less than 1 GB free disk")
    if problems:
        return JSONResponse({"status": "degraded", "problems": problems}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    """Any error no endpoint handled: logged with its traceback under a reference; the browser sees only that."""
    return JSONResponse({"status": "error", "message": _error_reference(exc, f"{request.method} {request.url.path}")},
                        status_code=500)


@app.get("/api/package-key")
def package_key():
    """The public key that signs downloaded model packages. Receivers pin it to check packages
    (verify_package.py --trusted-key <public_key_hex>)."""
    key = package_signer.public_key() if package_signer.available else None
    if key is None:
        return JSONResponse({"status": "success", "signing": False,
                             "message": "Packages are hashed but not signed: install the 'cryptography' package."})
    return JSONResponse({"status": "success", "signing": True, **key})


@app.get("/api/train/status")
async def train_status():
    if training_status["is_training"] and training_status["state"] != "starting" and (_training_task is None or _training_task.done()):
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

@app.post("/api/train/stop")
async def stop_training():
    """Asks the running training / incremental job to stop. It unwinds within one batch; nothing is
    saved for a stopped run and the model in use stays as it was."""
    if not training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "No training is running."}, status_code=409)
    cancel.request()
    training_status["state"] = "stopping"
    return JSONResponse({"status": "success", "message": "Stopping… the current batch finishes first."})


@app.get("/api/train/history")
def train_history(limit: int = Query(50, ge=1, le=MAX_HISTORY_ROWS)):
    runs = get_all_runs(limit)
    active = registry.get_active_sequence()
    for r in runs:
        r["has_model"] = registry.has_artifacts(r["id"])
        r["is_active"] = r["id"] == active
    return JSONResponse({"status": "success", "runs": runs, "active_run_id": active})


@app.get("/api/runs/{run_id}")
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


@app.post("/api/runs/{run_id}/save")
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


@app.delete("/api/runs/{run_id}/save")
async def unsave_model(run_id: int):
    if get_run(run_id) is None:
        return JSONResponse({"status": "error", "message": f"No training run #{run_id}."}, status_code=404)
    set_saved_name(run_id, None)
    return JSONResponse({"status": "success", "message": f"Run #{run_id} is no longer saved (its model may be pruned once more than 20 newer unsaved runs exist, unless it's active)."})


@app.delete("/api/runs/{run_id}")
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


@app.get("/api/runs/{run_id}/download")
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


@app.get("/api/runs/{run_id}/weights")
def download_weights(run_id: int):
    """Every trained weight of this run's model as JSON (layer name -> shape + values)."""
    r = get_run(run_id)
    if r is None or not registry.has_artifacts(run_id):
        return JSONResponse({"status": "error", "message": "No kept model for this run."}, status_code=404)
    m, _, _ = load_sequence_checkpoint(registry.checkpoint_path(run_id))
    return Response(content=json.dumps(weights_export(m)), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{_download_name(r)}_weights.json"'})


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

@app.get("/api/test/history")
def test_history(limit: int = Query(50, ge=1, le=MAX_HISTORY_ROWS)):
    return JSONResponse({"status": "success", "runs": get_all_test_runs(limit)})


TEST_RESULTS_DIR = os.path.join(DATA_DIR, "test_results")


@app.get("/api/test/history/{test_id}")
def test_history_detail(test_id: int):
    """The stored full result of an earlier sequence test (older tests, from before details were kept, have none)."""
    path = os.path.join(TEST_RESULTS_DIR, f"test_{int(test_id)}.json")
    if not os.path.exists(path):
        return JSONResponse({"status": "error", "message": "No detailed result was stored for this test (it was run before details were kept)."}, status_code=404)
    with open(path, encoding="utf-8") as f:
        return JSONResponse({"status": "success", **json.load(f)})


_LABEL_COLUMN_CANDIDATES = ["class", "class_id"]
CORRELATION_HIDDEN = {"prf_hz", "pw_x_pri", "tb_product"}

def _pri_pattern_check(df: pd.DataFrame) -> Optional[dict]:
    """Does each family's PRI behave like its pri_pattern label? Uses the generator's ground-truth
    pri_deviation_pct (PRI relative to the operating mode's base PRI), on clean pulses of labelled
    families only (no missed / spurious pulse involved). None for files without these columns."""
    if not {"pri_pattern", "pri_deviation_pct"} <= set(df.columns):
        return None
    clean = pd.to_numeric(df["pri_deviation_pct"], errors="coerce").notna()
    for col in ("missed_before", "spurious_before", "is_spurious"):
        if col in df.columns:
            clean &= pd.to_numeric(df[col], errors="coerce").fillna(0) == 0
    if "class" in df.columns:
        clean &= ~df["class"].astype(str).str.contains("NOISE", case=False)
    sub = df[clean]
    if sub.empty:
        return None
    dev = pd.to_numeric(sub["pri_deviation_pct"])
    edges = np.arange(-40, 42, 2.0)
    patterns = []
    for pat, g in dev.groupby(sub["pri_pattern"].astype(str)):
        counts, _ = np.histogram(g.clip(-40, 40), bins=edges)
        patterns.append({"pattern": pat, "pulses": int(len(g)), "std": round(float(g.std(ddof=0)), 2),
                         "min": round(float(g.min()), 2), "max": round(float(g.max()), 2),
                         "share": [round(float(c / len(g)), 4) for c in counts]})
    label_col = "class" if "class" in sub.columns else None
    families = []
    if label_col:
        # by label AND pattern: held-out rows share one label but come from several families
        for (lab, pat), g in sub.groupby([sub[label_col].astype(str), sub["pri_pattern"].astype(str)]):
            d = pd.to_numeric(g["pri_deviation_pct"])
            families.append({"label": lab, "pattern": pat, "pulses": int(len(g)),
                             "std": round(float(d.std(ddof=0)), 2), "min": round(float(d.min()), 2),
                             "max": round(float(d.max()), 2), "levels": int(d.round(0).nunique())})
    return {"bins": [round(float((a + b) / 2), 1) for a, b in zip(edges[:-1], edges[1:])],
            "patterns": patterns, "families": families,
            "excluded_disturbed": int((~clean).sum())}


def _pri_traces(df: pd.DataFrame, max_groups: int = 12, max_pulses: int = 160) -> Optional[dict]:
    """PRI against pulse number for one example track per PRI pattern (per class label when the file has
    no pattern column): what the pattern looks like in time, and where missed / spurious pulses make
    the PRI jump (flagged). Labelled families only — noise tracks have no pattern."""
    if not {"instance", "PRI"} <= set(df.columns):
        return None
    key = "pri_pattern" if "pri_pattern" in df.columns else ("class" if "class" in df.columns else None)
    if key is None:
        return None
    first = df.groupby("instance", sort=False).first()
    if "true_class_id" in first.columns:
        first = first[pd.to_numeric(first["true_class_id"], errors="coerce").fillna(-1) >= 0]
    traces = []
    for group, tracks in first.groupby(first[key].astype(str), sort=True):
        if len(traces) >= max_groups:
            break
        tid = tracks.index[0]
        t = df[df["instance"] == tid]
        t = (t.sort_values("pulse_idx") if "pulse_idx" in t.columns else t).head(max_pulses)
        flags = np.zeros(len(t), dtype=bool)
        for col in ("missed_before", "spurious_before", "is_spurious"):
            if col in t.columns:
                flags |= pd.to_numeric(t[col], errors="coerce").fillna(0).to_numpy() > 0
        label = str(tracks["class"].iloc[0]) if "class" in tracks.columns else str(group)
        pri = pd.to_numeric(t["PRI"], errors="coerce")
        dev = pd.to_numeric(t["pri_deviation_pct"], errors="coerce") if "pri_deviation_pct" in t.columns else None
        traces.append({"group": group, "label": label, "track_id": str(tid),
                       "pri_us": [None if pd.isna(v) else round(float(v), 3) for v in pri],
                       "deviation_pct": None if dev is None else [None if pd.isna(v) else round(float(v), 2) for v in dev],
                       "disturbed": [int(i) for i in np.flatnonzero(flags)]})
    return {"by": key, "traces": traces} if traces else None





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


def _analysis_source(source: str, filename: Optional[str]):
    """(dataframe, display name, meta) for an Analytics/Dataset-page selection, or a JSON error response.
    source "dataset": a named dataset (blank = the active one); source "model": the dataset of kept run <filename>.
    The old source names "sequence" / "sequence_uploaded" still mean "a named dataset"."""
    if source == "model":
        try:
            r = get_run(int(filename))
        except (TypeError, ValueError):
            r = None
        if r is None or r.get("model_kind") != "sequence":
            return JSONResponse({"status": "error", "message": "Model not found."}, status_code=404)
        try:
            df, meta = _model_dataset(r)
        except ValueError as e:
            return JSONResponse({"status": "error", "message": _public_error(e)}, status_code=409)
        if df is None:
            return JSONResponse({"status": "error", "message": f"Run #{r['id']} was trained on a dataset that is no longer stored."}, status_code=404)
        return df, meta.get("name") or f"run #{r['id']}", meta
    name = filename or datasets.get_active()
    if not name or not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "No dataset found — generate or upload one on the Dataset page."}, status_code=404)
    df, meta = datasets.load(name)
    return df, name, meta


@app.get("/api/analytics/sources")
def analytics_sources():
    """Everything Analytics can show: every stored dataset (generated or uploaded) and the dataset behind each kept model."""
    sources = []
    for m in datasets.list():
        sources.append({"source": "dataset", "filename": m["name"], "group": "Datasets",
                        "label": f"{m['name']} — {_dataset_kind(m)}, {m.get('total_tracks', '?')} tracks",
                        "modified": _ist(m.get("created_at_utc"))})
    for r in _kept_sequence_runs():
        name = r.get("dataset_name") or f"regenerated from seed {r.get('dataset_seed')}"
        title = r.get("saved_name") or f"run #{r['id']}"
        sources.append({"source": "model", "filename": str(r["id"]), "group": "Dataset of a trained model",
                        "label": f"{title} ({r['backbone_type']}) → {name}", "modified": r.get("timestamp_ist", "")})
    return JSONResponse({"status": "success", "sources": sources, "active": datasets.get_active()})


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Single shared candidate list for the ground-truth/class label column — shared by
    inspect_dataset and dataset_analytics."""
    return next((c for c in _LABEL_COLUMN_CANDIDATES if c in df.columns), None)


@app.get("/api/dataset/inspect")
def inspect_dataset(source: str = "dataset", filename: str = None):
    found = _analysis_source(source, filename)
    if isinstance(found, JSONResponse):
        return found
    df, name, meta = found
    df = upgrade_columns(df)  # older track files: Hz frequency column -> MHz
    label_col = _detect_label_column(df)
    class_counts = df[label_col].value_counts().to_dict() if label_col else {}
    # A generated dataset's meta carries the configured generation ranges, shown next to each
    # feature's actual spread so a wrong range setting is obvious at a glance. An uploaded
    # file was not generated from ranges, so it has none to show.
    ranges_meta = meta if meta.get("source") == "generated" else None
    return JSONResponse({
        "status": "success",
        "filename": name,
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        **dataset_overview(df, label_col, ranges_meta),
    })


def _tukey_box_stats(values: np.ndarray) -> Optional[dict]:
    """Standard Tukey boxplot: box at Q1/median/Q3, whiskers extended to the furthest point
    still within 1.5*IQR of the box, everything beyond flagged as an outlier."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = values[(values >= lo_fence) & (values <= hi_fence)]
    whisker_lo = float(inliers.min()) if len(inliers) else float(values.min())
    whisker_hi = float(inliers.max()) if len(inliers) else float(values.max())
    outliers = sorted(values[(values < whisker_lo) | (values > whisker_hi)].tolist())
    return {
        "min": whisker_lo, "q1": float(q1), "median": float(median), "q3": float(q3), "max": whisker_hi,
        "outliers": outliers[:25],
    }


def _pca_2d(matrix: np.ndarray, labels: Optional[List[str]] = None, max_points: int = 800) -> Optional[dict]:
    """Standardizes and projects `matrix` to 2D via PCA — the standard first-look for whether
    classes are actually separable in a given feature space, rather than assuming it from
    summary statistics alone. Subsamples for plotting if there are many more rows than
    max_points (PCA itself still fits on the full matrix)."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid]
    if labels is not None:
        labels = [l for l, v in zip(labels, valid) if v]
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / std
    n_components = min(2, matrix.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(standardized)
    if n_components == 1:
        projected = np.concatenate([projected, np.zeros_like(projected)], axis=1)

    idx = np.arange(len(projected))
    if len(idx) > max_points:
        idx = np.random.RandomState(0).choice(idx, size=max_points, replace=False)

    points = [{"x": float(projected[i, 0]), "y": float(projected[i, 1]),
               "label": str(labels[i]) if labels is not None else None} for i in idx]
    explained = pca.explained_variance_ratio_.tolist()
    return {"points": points, "explained_variance_ratio": [round(float(v), 4) for v in explained]}


SILHOUETTE_MAX_ROWS = 5000


def _silhouette(matrix: np.ndarray, labels: List[str]) -> Optional[float]:
    """Single-number complement to the PCA scatter: how well-separated classes actually are
    in a feature space (+1 = tight, well-separated clusters; 0 = overlapping; negative =
    likely mislabeled/worse than random). Needs at least 2 classes with 2+ samples each."""
    valid = ~np.isnan(matrix).any(axis=1)
    matrix, labels = matrix[valid], [l for l, v in zip(labels, valid) if v]
    unique = set(labels)
    if len(unique) < 2 or matrix.shape[0] < 4:
        return None
    counts = {u: labels.count(u) for u in unique}
    if any(c < 2 for c in counts.values()):
        return None
    # Exact silhouette is O(n²): fine for a few thousand rows, minutes for a 100k-pulse sequence
    # dataset. Above SILHOUETTE_MAX_ROWS it's estimated on a fixed random sample (same seed each
    # time, so the number is reproducible).
    sample = {"sample_size": SILHOUETTE_MAX_ROWS, "random_state": 0} if matrix.shape[0] > SILHOUETTE_MAX_ROWS else {}
    try:
        return round(float(silhouette_score(matrix, labels, **sample)), 4)
    except Exception:
        return None


@app.get("/api/analytics")
def dataset_analytics(source: str = "dataset", filename: str = None,
                      max_boxplot_classes: int = Query(15, ge=1, le=50), max_scatter_rows: int = Query(600, ge=1, le=5000)):
    # plain def (not async): FastAPI runs it in a worker thread, so a big file doesn't stall the server
    """One-shot EDA payload for the Analytics page: summary statistics, a Pearson correlation
    matrix, per-feature histograms, per-class Tukey boxplots, and a row sample for the scatter
    explorer — computed once per dataset selection so the page's plot buttons just toggle
    which already-fetched view is shown, rather than re-hitting the server per click."""
    found = _analysis_source(source, filename)
    if isinstance(found, JSONResponse):
        return found
    df, filename, _meta = found
    df = upgrade_columns(df)  # older track files: Hz frequency column -> MHz
    label_col = _detect_label_column(df)
    class_counts_full = df[label_col].value_counts() if label_col else pd.Series(dtype=int)

    # Every measured, derived (duty cycle, PRF, PW×PRI, time-bandwidth product) and extra
    # (dataset/synthetic) feature; rise/fall zeros are "not measured" and excluded here —
    # they're reported in data_quality instead.
    sub = analysis_frame(df)
    if sub.empty:
        return JSONResponse({"status": "error", "message": "No recognized numeric PDW columns in this file."}, status_code=400)
    labels, groups = row_groups(df, label_col)

    describe = {}
    for col in sub.columns:
        vals = sub[col].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        describe[col] = {
            "count": int(len(valid)), "missing": int(len(vals) - len(valid)),
            "mean": float(np.mean(valid)), "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)), "p25": float(np.percentile(valid, 25)),
            "median": float(np.median(valid)), "p75": float(np.percentile(valid, 75)),
            "max": float(np.max(valid)),
        }

    # The derived columns that only restate PRI / PW (PRF = 1/PRI, PW×PRI, time-bandwidth product) are left
    # out of the heatmap: they would show up as near-perfect correlations with their own inputs.
    corr_cols = [c for c in sub.columns if c not in CORRELATION_HIDDEN]
    corr_df = sub[corr_cols].corr(numeric_only=True).round(4)
    correlation = {
        "columns": list(corr_df.columns),
        "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in corr_df.to_numpy()],
    }

    distributions = feature_distributions(sub, labels, groups)
    ranking = feature_ranking(sub, labels, groups) if label_col else []
    quality = data_quality(df, labels, groups)
    spectrum = rf_spectrum(sub)
    modes = mode_by_class(df, labels, groups) if label_col else None
    boxplot = {}
    classes_truncated = False
    if label_col:
        top_classes = class_counts_full.index.tolist()
        if len(top_classes) > max_boxplot_classes:
            top_classes = top_classes[:max_boxplot_classes]
            classes_truncated = True
        label_values = df[label_col].to_numpy()
        for col in sub.columns:
            col_values = sub[col].to_numpy(dtype=float)
            classes_here, stats_here = [], []
            for cls in top_classes:
                stats = _tukey_box_stats(col_values[label_values == cls])
                if stats:
                    classes_here.append(str(cls))
                    stats_here.append(stats)
            if stats_here:
                boxplot[col] = {"classes": classes_here, "stats": stats_here}

    sample_df = sub.copy()
    if label_col:
        sample_df["_label"] = df[label_col].astype(str).to_numpy()
    if len(sample_df) > max_scatter_rows:
        sample_df = sample_df.sample(n=max_scatter_rows, random_state=0)
    # object dtype first: on float columns pandas turns None back into NaN, which isn't valid JSON
    scatter_sample = sample_df.astype(object).where(pd.notnull(sample_df), None).to_dict("records")

    # Frequency band breakdown — the classic first-pass ESM/EW analysis (how much of the
    # intercepted spectrum falls in which band), using the same L/S/C/X/Ku/K/Ka mapping the
    # training pipeline uses for its prototype metadata.
    band_distribution = None
    if "frequency_mhz" in sub.columns:
        bands = sub["frequency_mhz"].dropna().apply(frequency_to_band)
        band_distribution = {str(k): int(v) for k, v in bands.value_counts().items()}

    # PRI modulation pattern breakdown — only meaningful now that the generator actually
    # produces more than one pattern (see src/data/pri_patterns.py); shows which of the six
    # standard types (constant/jittered/staggered/sliding/wobulated/dwell_switch) the current
    # dataset's families/emitters are using.
    pri_pattern_distribution = None
    if "pri_pattern" in df.columns:
        pri_pattern_distribution = {str(k): int(v) for k, v in df["pri_pattern"].value_counts().items()}
    pri_pattern_check = _pri_pattern_check(df)
    pri_traces = _pri_traces(df)

    # Class balance as a single number, not just the bar chart — Shannon entropy (normalized
    # to [0,1], 1.0 = perfectly even) and the max/min count ratio, standard dataset-health
    # metrics for any classification task.
    class_diversity = None
    if label_col and len(class_counts_full) > 1:
        counts = class_counts_full.to_numpy(dtype=float)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        max_entropy = float(np.log(len(counts)))
        class_diversity = {
            "shannon_entropy_normalized": round(entropy / max_entropy, 4) if max_entropy > 0 else None,
            "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
            "num_classes": int(len(counts)),
        }

    # PCA + silhouette in the RAW physical feature space — directly visualizes/quantifies
    # whether classes are actually separable before any model gets involved, the standard
    # first question in any classification EDA.
    # Features missing for more than 20% of rows (e.g. rise time recorded as 0) are left out
    # here, since PCA needs complete rows and would otherwise drop most of the dataset.
    label_values_list = df[label_col].astype(str).tolist() if label_col else None
    pca_columns = [c for c in sub.columns if sub[c].isna().mean() <= 0.2]
    pca_matrix = sub[pca_columns].to_numpy(dtype=float)
    pca_raw = _pca_2d(pca_matrix, label_values_list)
    if pca_raw is not None:
        pca_raw["columns"] = pca_columns
    silhouette_raw = _silhouette(pca_matrix, label_values_list) if label_values_list else None

    return JSONResponse({
        "status": "success",
        "filename": filename,
        "total_rows": int(len(df)),
        "label_column": label_col,
        "class_counts": {str(k): int(v) for k, v in class_counts_full.items()},
        "features": list(sub.columns),
        "describe": describe,
        "correlation": correlation,
        "distributions": distributions,
        "feature_ranking": ranking,
        "data_quality": quality,
        "rf_spectrum": spectrum,
        "mode_by_class": modes,
        "boxplot": boxplot,
        "boxplot_classes_truncated": classes_truncated,
        "scatter_sample": scatter_sample,
        "band_distribution": band_distribution,
        "pri_pattern_distribution": pri_pattern_distribution,
        "pri_pattern_check": pri_pattern_check,
        "pri_traces": pri_traces,
        "class_diversity": class_diversity,
        "pca_raw": pca_raw,
        "silhouette_raw": silhouette_raw,
    })


MAX_LIVE_FEED_SECONDS = 900  # hard server-side cap (15 min) regardless of what the client requests


# ============================================================================ pulse-sequence models
# Classify a window of consecutive pulses of ONE emitter (a de-interleaved track) — see
# src/models/sequence_models.py: dataset, training job, testing, inference and live simulation.
SEQUENCE_TRAIN_CHECKPOINT = os.path.join(MODELS_DIR, "sequence_training.pt")  # staging file for the registry
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


def _load_sequence_dataset(name: Optional[str] = None):
    """(dataframe, meta) of the named dataset — the active one when no name is given."""
    df, meta = datasets.load(name)
    return (None, None) if df is None else (upgrade_columns(df), meta)


@app.post("/api/sequence/dataset/generate")
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


@app.get("/api/sequence/dataset")
def sequence_dataset_info(name: Optional[str] = None):
    df, meta = _load_sequence_dataset(name)
    if df is None:
        return JSONResponse({"status": "error", "message": "No sequence dataset yet — generate or upload one."}, status_code=404)
    return JSONResponse({"status": "success", "name": meta["name"], **_sequence_summary(df, meta)})


@app.get("/api/datasets")
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


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    if not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "Dataset not found."}, status_code=404)
    if training_status["is_training"]:
        return JSONResponse({"status": "error", "message": "A model is training — delete datasets when it has finished."}, status_code=409)
    datasets.delete(name)
    return JSONResponse({"status": "success", "message": f"Deleted dataset '{name}'. Its name is retired and won't be given to another dataset.", "active": datasets.get_active()})


@app.post("/api/datasets/{name}/activate")
def activate_dataset(name: str):
    if not datasets.exists(name):
        return JSONResponse({"status": "error", "message": "Dataset not found."}, status_code=404)
    datasets.set_active(name)
    return JSONResponse({"status": "success", "active": name, "message": f"'{name}' is now the default training dataset."})


@app.get("/api/sequence/dataset/download")
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


@app.post("/api/sequence/dataset/upload")
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


@app.post("/api/sequence/dataset/expand")
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


@app.get("/api/sequence/options")
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


@app.post("/api/sequence/train")
async def train_sequence(req: TrainSequenceRequest):
    global _training_task
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
    _training_task = asyncio.create_task(_run_sequence_training_job(req, names))
    return JSONResponse({"status": "started", "job_id": job_id})


@app.get("/api/sequence/models")
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


@app.post("/api/sequence/test")
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


@app.post("/api/sequence/test/upload")
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


@app.post("/api/sequence/models/activate")
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


@app.post("/api/sequence/infer")
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


def _population_split(r: dict) -> Tuple[List[int], List[int]]:
    """(the families held out when the model's population was generated, those of them the model
    has since been taught incrementally). The first fixes the families themselves; held-out minus
    revealed is what is still unknown to this model."""
    holdout = _select_holdout_ids(r["num_classes"], r["num_holdout_classes"], r.get("dataset_seed") or 0)
    pop = (registry.load_report(r["id"]) or {}).get("population") or {}
    return holdout, [int(c) for c in pop.get("revealed_class_ids", []) if int(c) in holdout]


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


@app.post("/api/sequence/example_track")
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


# ---------------------------------------------------------------- incremental training (new families)
INCREMENTAL_DIR = os.path.join(DATA_DIR, "incremental_experiments")


class NewFamiliesRequest(BaseModel):
    run_id: int
    class_ids: Optional[List[int]] = None   # which of the model's held-out families appear; None = the first one
    tracks_per_class: int = 30
    pulses_per_track: Optional[int] = None  # None = as the model's dataset
    name: Optional[str] = None


@app.post("/api/sequence/dataset/new_families")
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


@app.get("/api/sequence/incremental/options")
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


@app.post("/api/sequence/incremental")
async def start_incremental(req: IncrementalRequest):
    global _training_task
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
    _training_task = asyncio.create_task(_run_incremental_job(req))
    return JSONResponse({"status": "started", "job_id": job_id})


@app.get("/api/sequence/incremental/experiments")
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


@app.websocket("/ws/sequence_live")
async def sequence_live(websocket: WebSocket, run_id: Optional[int] = None, emitters: int = 5, unknown_pct: float = 0.3,
                        missed_pct: float = 0.02, spurious_pct: float = 0.02, duration: int = 120, rate: float = 20.0):
    """Live Simulation of de-interleaved tracks: several simulated emitters transmit at once;
    each pulse arrives already tagged with its track (de-interleaver output). Every time a track
    gets a pulse, its last ≤ k pulses are classified, so you watch the decision firm up as the
    track grows. Tracks end and new emitters appear. Ground truth is sent for scoring."""
    global _live_sessions
    user = _session_user(websocket.cookies)
    if user is None or not _same_origin(websocket.headers):
        await websocket.close(code=1008)
        return
    if _live_sessions >= MAX_LIVE_SESSIONS:
        await websocket.close(code=1013)  # try again later
        return
    _live_sessions += 1
    audit.record("live_session", user, _client_ip(websocket), run_id=run_id)
    try:
        await _run_live_session(websocket, run_id, emitters, unknown_pct, missed_pct, spurious_pct, duration, rate)
    finally:
        _live_sessions -= 1


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

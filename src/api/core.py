"""Application state shared by every route module: data paths, stores (auth, datasets, models, history),
the loaded configuration, limits, the training-job status and the error helpers."""

import logging
import os
import re
import secrets
import yaml
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlparse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.models.package_signing import PackageSigner
from src.models.registry import ModelRegistry
from src.storage.auth_store import AuditLog, AuthStore, LoginLimiter
from src.storage.dataset_store import DatasetStore
from src.storage.training_store import get_latest_run, get_run, init_db, referenced_dataset_names
from src.training import cancel


_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")
APP_VERSION = open(_VERSION_FILE).read().strip() if os.path.exists(_VERSION_FILE) else "unknown"  # release = git tag v<VERSION>

UI_DIR = os.path.join(os.path.dirname(__file__), "../../ui")

# --- Accounts and sessions (the gate that checks them is in server.py) -------
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
# Running background task and open live sessions, shared by the route modules.
jobs = SimpleNamespace(training_task=None, live_sessions=0)


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

"""Sign-in, the HTML pages, settings, version / health and the package-signing key."""

import asyncio
import math
import os
import shutil
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.storage.training_store import get_latest_run
from src.api.core import (
    _client_ip, _COOKIE_SECURE_ALWAYS, APP_VERSION, audit, auth, ConfigUpdateModel, current_config, DATA_DIR,
    EMBEDDING_DIM_RANGE, log, login_limiter, package_signer, serve_page, SESSION_COOKIE_NAME,
)
router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=1024)

@router.get("/login")
def get_login_page():
    return serve_page("login.html")

@router.post("/api/login")
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

@router.post("/api/logout")
async def logout(request: Request):
    auth.revoke(request.cookies.get(SESSION_COOKIE_NAME))
    audit.record("logout", getattr(request.state, "user", None), _client_ip(request))
    resp = JSONResponse({"status": "success"})
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp

@router.get("/api/me")
def whoami(request: Request):
    return JSONResponse({"status": "success", "user": request.state.user})

@router.get("/")
def get_training_page():
    return serve_page("index.html")

@router.get("/validation")
def get_validation_page():
    return serve_page("validation.html")

@router.get("/inference")
def get_inference_page():
    return serve_page("inference.html")

@router.get("/dataset")
def get_dataset_page():
    return serve_page("dataset.html")

@router.get("/analytics")
def get_analytics_page():
    return serve_page("analytics.html")

@router.get("/live")
def get_live_page():
    return serve_page("live.html")

@router.get("/help")
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


@router.get("/api/config")
async def get_current_config():
    return JSONResponse(content=current_config)

@router.post("/api/config")
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


@router.get("/api/version")
def app_version():
    return JSONResponse({"status": "success", "version": APP_VERSION})


@router.get("/api/health")
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


@router.get("/api/package-key")
def package_key():
    """The public key that signs downloaded model packages. Receivers pin it to check packages
    (verify_package.py --trusted-key <public_key_hex>)."""
    key = package_signer.public_key() if package_signer.available else None
    if key is None:
        return JSONResponse({"status": "success", "signing": False,
                             "message": "Packages are hashed but not signed: install the 'cryptography' package."})
    return JSONResponse({"status": "success", "signing": True, **key})

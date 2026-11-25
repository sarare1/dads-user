"""DADS web application: creates the FastAPI app, guards every request (sign-in, same-origin, size limit,
security headers, audit), and mounts the route modules in src/api/routers/. Run: uvicorn src.api.server:app

Layout: core.py (state, stores, config, errors) <- common.py (shared helpers) <- routers/*.py (endpoints)."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from src.api.core import (
    _AUDITED_GETS, _client_ip, _error_reference, _PUBLIC_PATHS, _SAFE_METHODS, _same_origin, _SECURITY_HEADERS,
    _session_user, APP_VERSION, audit, MAX_BODY_BYTES, UI_DIR,
)
from src.api.routers import pages, runs, analytics, inference, datasets, training, incremental


app = FastAPI(title="DADS - PhD Research Project", version=APP_VERSION)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


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


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    """Any error no endpoint handled: logged with its traceback under a reference; the browser sees only that."""
    return JSONResponse({"status": "error", "message": _error_reference(exc, f"{request.method} {request.url.path}")},
                        status_code=500)


for _module in (pages, runs, analytics, inference, datasets, training, incremental):
    app.include_router(_module.router)

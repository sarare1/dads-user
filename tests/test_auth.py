"""Sign-in hardening: no built-in credential, random revocable sessions, sign-in throttling, same-origin
checks, private static files and a tamper-evident audit trail.
Uses its own temporary data folder. Run: python tests/test_auth.py  (or pytest)"""
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DADS_DATA_DIR", tempfile.mkdtemp(prefix="dads_test_data_"))  # before importing the server
os.environ.setdefault("DADS_ADMIN_USER", "admin")
os.environ.setdefault("DADS_ADMIN_PASSWORD", "test-only-password-123")

from fastapi.testclient import TestClient

from src.api import server
from src.storage import auth_store
from src.storage.auth_store import AuditLog, AuthStore, LoginLimiter, hash_password, verify_password

USER, PASSWORD = os.environ["DADS_ADMIN_USER"], os.environ["DADS_ADMIN_PASSWORD"]


def _login(c: TestClient, user=USER, password=PASSWORD):
    return c.post("/api/login", json={"username": user, "password": password})


def test_no_credential_in_source_and_passwords_are_hashed():
    src = open(server.__file__, encoding="utf-8").read()
    # no credential literal of any kind (the pre-1.1.0 code had ADMIN_PASSWORD = "..." and a static session secret)
    assert not re.search(r"""(PASSWORD|SECRET)\w*\s*=\s*["'][^"']+["']""", src)
    h = hash_password("correct horse battery")
    assert h.startswith("pbkdf2_sha256$600000$") and "correct horse" not in h
    assert verify_password("correct horse battery", h) and not verify_password("wrong", h)
    with open(server.auth.db_path, "rb") as f:
        assert PASSWORD.encode() not in f.read()


def test_first_start_without_password_writes_a_random_one_to_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DADS_ADMIN_PASSWORD", raising=False)
    store = AuthStore(str(tmp_path / "auth.db"))
    note = store.ensure_initial_admin(str(tmp_path))
    text = (tmp_path / auth_store.INITIAL_PASSWORD_FILE).read_text()
    password = text.split("Password: ")[1].split()[0]
    assert len(password) >= 20 and password not in note  # the password is in the file, never in the log line
    assert store.authenticate("admin", password) == "admin"
    assert store.ensure_initial_admin(str(tmp_path)) is None  # only on the very first start


def test_login_session_and_logout_revokes_the_cookie():
    c = TestClient(server.app)
    assert c.get("/api/me").status_code == 401
    assert c.get("/", follow_redirects=False).status_code in (302, 307)
    assert _login(c, password="wrong-password-xyz").status_code == 401
    r = _login(c)
    assert r.status_code == 200 and r.json()["user"] == USER
    cookie = r.cookies.get(server.SESSION_COOKIE_NAME)
    set_cookie = r.headers["set-cookie"].lower()
    assert len(cookie) >= 40 and "httponly" in set_cookie and "samesite=strict" in set_cookie
    assert c.get("/api/me").json()["user"] == USER

    other = TestClient(server.app)
    assert _login(other).cookies.get(server.SESSION_COOKIE_NAME) != cookie  # a fresh token per sign-in

    assert c.post("/api/logout").status_code == 200
    replay = TestClient(server.app)
    replay.cookies.set(server.SESSION_COOKIE_NAME, cookie)
    assert replay.get("/api/me").status_code == 401  # the old cookie is dead on the server, not just in the browser
    assert other.get("/api/me").status_code == 200   # other sessions are untouched


def test_forged_or_old_style_cookie_is_rejected():
    c = TestClient(server.app)
    c.cookies.set("dads_auth", "0" * 64)
    c.cookies.set(server.SESSION_COOKIE_NAME, "made-up-token")
    assert c.get("/api/me").status_code == 401


def test_password_change_and_disable_end_sessions(tmp_path):
    store = AuthStore(str(tmp_path / "auth.db"))
    store.set_password("analyst", "first-password-123", create=True)
    tok = store.create_session("analyst", "1.2.3.4")
    assert store.session_user(tok) == "analyst"
    store.set_password("analyst", "second-password-456")
    assert store.session_user(tok) is None
    tok = store.create_session("analyst", "1.2.3.4")
    store.set_disabled("analyst", True)
    assert store.session_user(tok) is None and store.authenticate("analyst", "second-password-456") is None
    try:
        store.set_password("analyst", "short")
        raise AssertionError("short password accepted")
    except ValueError:
        pass


def test_sessions_expire(tmp_path, monkeypatch):
    store = AuthStore(str(tmp_path / "auth.db"), idle_minutes=10, absolute_hours=1)
    store.set_password("analyst", "first-password-123", create=True)
    t = [1_000_000.0]
    monkeypatch.setattr(auth_store, "_now", lambda: t[0])
    tok = store.create_session("analyst", None)
    t[0] += 9 * 60
    assert store.session_user(tok) == "analyst"
    t[0] += 11 * 60  # idle too long
    assert store.session_user(tok) is None
    tok = store.create_session("analyst", None)
    for _ in range(8):  # kept busy, but past the absolute limit
        t[0] += 8 * 60
        store.session_user(tok)
    assert store.session_user(tok) is None


def test_repeated_failures_are_throttled():
    server.login_limiter.reset()
    c = TestClient(server.app)
    for _ in range(5):
        assert _login(c, user="intruder", password="guess-guess-guess").status_code == 401
    r = _login(c, user="intruder", password="guess-guess-guess")
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    server.login_limiter.reset()

    lim = LoginLimiter(per_user_ip=3, per_ip=5, window_s=60)
    for name in ("a", "b", "c", "d", "e"):
        lim.failed("9.9.9.9", name)
    assert lim.retry_after("9.9.9.9", "f") > 0     # too many from this address overall
    assert lim.retry_after("8.8.8.8", "a") == 0    # another address is not affected


def test_cross_origin_writes_are_refused_and_static_files_need_login():
    c = TestClient(server.app)
    assert c.get("/static/shared.css").status_code == 200  # the sign-in page needs it
    r = c.get("/static/run_details.js", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/login"
    assert _login(c).status_code == 200
    assert c.get("/static/run_details.js").status_code == 200
    r = c.post("/api/logout", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert c.get("/api/me").status_code == 200  # still signed in: the forged request did nothing


def test_audit_trail_records_actions_and_detects_tampering(tmp_path):
    c = TestClient(server.app)
    _login(c)
    c.post("/api/logout")
    path = os.path.join(server.DATA_DIR, "audit.log")
    events = [json.loads(line) for line in open(path, encoding="utf-8")]
    kinds = {e["event"] for e in events}
    assert {"login", "logout"} <= kinds and any(e["event"] == "request" and e["path"] == "/api/logout" for e in events)
    assert AuditLog.verify(path)[0]

    log = AuditLog(str(tmp_path / "audit.log"))
    for i in range(3):
        log.record("x", "u", "ip", n=i)
    assert AuditLog.verify(log.path) == (True, 3, "Audit log intact (3 entries).")
    lines = open(log.path, encoding="utf-8").read().splitlines()
    lines[1] = lines[1].replace('"n": 1', '"n": 7')
    open(log.path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    ok, n, _ = AuditLog.verify(log.path)
    assert not ok and n == 3


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

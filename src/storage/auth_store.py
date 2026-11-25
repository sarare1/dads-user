"""Sign-in for DADS: named user accounts, server-side sessions, login throttling and an audit trail.

- **Accounts** live in ``<data>/auth.db`` (SQLite). Passwords are stored only as salted PBKDF2-SHA256
  hashes (600 000 iterations, OWASP 2023). There is no password in the source code. Manage accounts with
  ``python scripts/manage_users.py``.
- **First start**: if there are no accounts, one admin account is created. Its password comes from
  ``DADS_ADMIN_PASSWORD`` (user name ``DADS_ADMIN_USER``, default ``admin``). If that is not set, a random
  password is generated and written to ``<data>/INITIAL_ADMIN_PASSWORD.txt`` (never to the log).
- **Sessions**: each sign-in gets a fresh random token (256 bits). Only its SHA-256 is stored, so a copy of
  the database cannot be replayed as a cookie. A session ends after ``idle_minutes`` without use, after
  ``absolute_hours`` in total, on sign-out, or when the user's password changes or the account is disabled.
- **Throttling** (:class:`LoginLimiter`): too many failed sign-ins from one address lock that address out
  for a while; the account itself is never locked, so a stranger cannot lock the real user out.
- **Audit trail** (:class:`AuditLog`): one JSON line per security-relevant event, each carrying the SHA-256
  of the previous line, so an edited or deleted line is detectable (``manage_users.py verify-audit``).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 12
# A file name, not a password (bandit B105 false positive).
INITIAL_PASSWORD_FILE = "INITIAL_ADMIN_PASSWORD.txt"  # nosec B105
_TOUCH_EVERY_S = 60  # write last_seen at most once a minute per session


def _now() -> float:
    return time.time()


def _utc(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(_now() if ts is None else ts, timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(iterations, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


# Compared against when the user name does not exist, so a wrong name takes as long as a wrong password.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16), iterations=PBKDF2_ITERATIONS)


def check_password_policy(password: str) -> Optional[str]:
    """None if acceptable, else the reason."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if password.strip() != password:
        return "Password must not start or end with spaces."
    return None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, db_path: str, idle_minutes: int = 120, absolute_hours: int = 12):
        self.db_path = db_path
        self.idle_s = idle_minutes * 60
        self.absolute_s = absolute_hours * 3600
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username      TEXT PRIMARY KEY COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    disabled      INTEGER NOT NULL DEFAULT 0,
                    created_utc   TEXT NOT NULL,
                    password_changed_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_sha256 TEXT PRIMARY KEY,
                    username     TEXT NOT NULL COLLATE NOCASE,
                    created      REAL NOT NULL,
                    last_seen    REAL NOT NULL,
                    ip           TEXT
                );
                """
            )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # --- accounts --------------------------------------------------------------------------------
    def has_users(self) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None

    def list_users(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT username, disabled, created_utc, password_changed_utc FROM users ORDER BY username").fetchall()
            active = dict(c.execute("SELECT username, COUNT(*) FROM sessions GROUP BY username COLLATE NOCASE").fetchall())
        return [dict(r, active_sessions=active.get(r["username"], 0)) for r in rows]

    def user_exists(self, username: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone() is not None

    def set_password(self, username: str, password: str, create: bool = False) -> None:
        """Create the user (``create=True``) or change their password. Ends all their sessions."""
        reason = check_password_policy(password)
        if reason:
            raise ValueError(reason)
        if not username or not username.replace("_", "").replace("-", "").replace(".", "").isalnum() or len(username) > 64:
            raise ValueError("User name: letters, digits, '.', '-' or '_' only (max 64).")
        encoded, now = hash_password(password), _utc()
        with self._lock, self._conn() as c:
            exists = c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone() is not None
            if create and exists:
                raise ValueError(f"User '{username}' already exists.")
            if not create and not exists:
                raise ValueError(f"No user '{username}'.")
            if exists:
                c.execute("UPDATE users SET password_hash = ?, password_changed_utc = ? WHERE username = ?", (encoded, now, username))
            else:
                c.execute("INSERT INTO users VALUES (?, ?, 0, ?, ?)", (username, encoded, now, now))
            c.execute("DELETE FROM sessions WHERE username = ?", (username,))

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self._lock, self._conn() as c:
            if c.execute("UPDATE users SET disabled = ? WHERE username = ?", (int(disabled), username)).rowcount == 0:
                raise ValueError(f"No user '{username}'.")
            if disabled:
                c.execute("DELETE FROM sessions WHERE username = ?", (username,))

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """The account's stored user name if the password is right and the account is enabled, else None.
        Slow on purpose (PBKDF2); call it from a worker thread."""
        with self._conn() as c:
            row = c.execute("SELECT username, password_hash, disabled FROM users WHERE username = ?", (username,)).fetchone()
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        return row["username"] if (row and ok and not row["disabled"]) else None

    def ensure_initial_admin(self, data_dir: str) -> Optional[str]:
        """Create the first account if there is none. Returns a note for the console, or None."""
        if self.has_users():
            return None
        user = os.environ.get("DADS_ADMIN_USER") or "admin"
        password = os.environ.get("DADS_ADMIN_PASSWORD")
        if password:
            self.set_password(user, password, create=True)
            return f"Created user '{user}' with the password from DADS_ADMIN_PASSWORD."
        password = secrets.token_urlsafe(15)  # 20 characters
        self.set_password(user, password, create=True)
        path = os.path.join(data_dir, INITIAL_PASSWORD_FILE)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"DADS first sign-in\nUser ID:  {user}\nPassword: {password}\n\n"
                    f"Change it, then delete this file:  python scripts/manage_users.py passwd {user}\n")
        return (f"No user accounts existed, so user '{user}' was created with a random password. "
                f"It is in {os.path.abspath(path)}. Change it with: python scripts/manage_users.py passwd {user}")

    # --- sessions --------------------------------------------------------------------------------
    def create_session(self, username: str, ip: Optional[str]) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM sessions WHERE created < ? OR last_seen < ?", (now - self.absolute_s, now - self.idle_s))
            c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", (_token_hash(token), username, now, now, ip))
        return token

    def session_user(self, token: Optional[str]) -> Optional[str]:
        """The signed-in user for this cookie value, or None (unknown, expired or disabled)."""
        if not token or len(token) > 128:
            return None
        th, now = _token_hash(token), _now()
        with self._conn() as c:
            row = c.execute(
                "SELECT s.username, s.created, s.last_seen, u.disabled FROM sessions s "
                "JOIN users u ON u.username = s.username WHERE s.token_sha256 = ?", (th,)).fetchone()
            if row is None:
                return None
            if row["disabled"] or now - row["created"] > self.absolute_s or now - row["last_seen"] > self.idle_s:
                c.execute("DELETE FROM sessions WHERE token_sha256 = ?", (th,))
                return None
            if now - row["last_seen"] > _TOUCH_EVERY_S:
                c.execute("UPDATE sessions SET last_seen = ? WHERE token_sha256 = ?", (now, th))
        return row["username"]

    def revoke(self, token: Optional[str]) -> None:
        if token:
            with self._lock, self._conn() as c:
                c.execute("DELETE FROM sessions WHERE token_sha256 = ?", (_token_hash(token),))

    def revoke_user(self, username: str) -> int:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE username = ?", (username,)).rowcount


class LoginLimiter:
    """Counts failed sign-ins in a sliding window. Too many from one address (for one user name, or in
    total) blocks that address until the window has passed. In memory: a restart clears it."""

    def __init__(self, per_user_ip: int = 5, per_ip: int = 20, window_s: int = 15 * 60):
        self.per_user_ip, self.per_ip, self.window_s = per_user_ip, per_ip, window_s
        self._fails: Dict[Tuple[str, ...], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: Tuple[str, ...], now: float) -> Deque[float]:
        q = self._fails[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def retry_after(self, ip: str, username: str) -> int:
        """Seconds the caller must wait, or 0 if a sign-in attempt is allowed now."""
        now = _now()
        with self._lock:
            waits = []
            for key, limit in (((ip, username.lower()), self.per_user_ip), ((ip,), self.per_ip)):
                q = self._recent(key, now)
                if len(q) >= limit:
                    waits.append(int(q[0] + self.window_s - now) + 1)
            return max(waits, default=0)

    def failed(self, ip: str, username: str) -> None:
        now = _now()
        with self._lock:
            self._fails[(ip, username.lower())].append(now)
            self._fails[(ip,)].append(now)
            if len(self._fails) > 10_000:  # forget idle keys so the table cannot grow without bound
                for k in [k for k, q in self._fails.items() if not self._recent(k, now)]:
                    del self._fails[k]

    def succeeded(self, ip: str, username: str) -> None:
        with self._lock:
            self._fails.pop((ip, username.lower()), None)

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()


class AuditLog:
    """Append-only JSON-lines audit trail with a SHA-256 hash chain."""

    GENESIS = "0" * 64

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _last_hash(self) -> str:
        """Hash of the file's last line, read from the end each time, because the server and
        scripts/manage_users.py both append to the same file."""
        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536))
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
        except FileNotFoundError:
            return self.GENESIS
        return hashlib.sha256(lines[-1]).hexdigest() if lines else self.GENESIS

    def record(self, event: str, user: Optional[str], ip: Optional[str], **details) -> None:
        with self._lock:
            entry = {"ts_utc": _utc(), "event": event, "user": user, "ip": ip, **details, "prev_sha256": self._last_hash()}
            line = json.dumps(entry, sort_keys=True, ensure_ascii=True, default=str)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    @classmethod
    def verify(cls, path: str) -> Tuple[bool, int, str]:
        """(intact, lines checked, message)."""
        prev, n = cls.GENESIS, 0
        if not os.path.exists(path):
            return True, 0, "No audit log yet."
        with open(path, "rb") as f:
            for n, raw in enumerate(f, start=1):
                line = raw.rstrip(b"\r\n")
                try:
                    if json.loads(line)["prev_sha256"] != prev:
                        return False, n, f"Line {n} does not follow line {n - 1}: a line was changed, removed or inserted."
                except (ValueError, KeyError):
                    return False, n, f"Line {n} is not a valid audit entry."
                prev = hashlib.sha256(line).hexdigest()
        return True, n, f"Audit log intact ({n} entries)."

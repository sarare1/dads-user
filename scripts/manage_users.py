"""Manage DADS sign-in accounts and check the audit trail. Works while the server is running.

    python scripts/manage_users.py list
    python scripts/manage_users.py add <user>            # asks for the password twice
    python scripts/manage_users.py passwd <user>         # change password; ends the user's sessions
    python scripts/manage_users.py disable <user>        # block sign-in; ends the user's sessions
    python scripts/manage_users.py enable <user>
    python scripts/manage_users.py logout <user>         # end all of the user's sessions
    python scripts/manage_users.py verify-audit          # check the audit log's hash chain

The password can also come from the DADS_NEW_PASSWORD environment variable (for scripted setup).
The data folder is data/ unless DADS_DATA_DIR says otherwise (same rule as the server).
"""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.storage.auth_store import (  # noqa: E402
    AuditLog, AuthStore, INITIAL_PASSWORD_FILE, MIN_PASSWORD_LENGTH, check_password_policy,
)

DATA_DIR = os.environ.get("DADS_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _ask_password() -> str:
    pw = os.environ.get("DADS_NEW_PASSWORD")
    if pw:
        return pw
    while True:
        pw = getpass.getpass(f"New password (at least {MIN_PASSWORD_LENGTH} characters): ")
        reason = check_password_policy(pw)
        if reason:
            print(reason)
            continue
        if getpass.getpass("Repeat it: ") != pw:
            print("The two entries differ, try again.")
            continue
        return pw


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage DADS sign-in accounts.")
    ap.add_argument("command", choices=["list", "add", "passwd", "disable", "enable", "logout", "verify-audit"])
    ap.add_argument("user", nargs="?")
    a = ap.parse_args()

    if a.command == "verify-audit":
        ok, _, msg = AuditLog.verify(os.path.join(DATA_DIR, "audit.log"))
        print(msg)
        return 0 if ok else 1

    os.makedirs(DATA_DIR, exist_ok=True)
    store = AuthStore(os.path.join(DATA_DIR, "auth.db"))
    audit = AuditLog(os.path.join(DATA_DIR, "audit.log"))
    actor = f"console:{getpass.getuser()}"

    if a.command == "list":
        users = store.list_users()
        if not users:
            print("No accounts yet. The server creates the first one on start, or run: manage_users.py add <user>")
        for u in users:
            print(f"{u['username']:<20} {'DISABLED' if u['disabled'] else 'active':<9} "
                  f"sessions={u['active_sessions']:<3} password changed {u['password_changed_utc']}")
        return 0

    if not a.user:
        ap.error(f"'{a.command}' needs a user name")
    try:
        if a.command in ("add", "passwd"):
            store.set_password(a.user, _ask_password(), create=(a.command == "add"))
            initial = os.path.join(DATA_DIR, INITIAL_PASSWORD_FILE)
            if a.command == "passwd" and os.path.exists(initial):
                os.remove(initial)
                print(f"Removed {initial}.")
            print(f"{'Created' if a.command == 'add' else 'Changed password for'} '{a.user}'. Their sessions were ended.")
        elif a.command in ("disable", "enable"):
            store.set_disabled(a.user, a.command == "disable")
            print(f"'{a.user}' {a.command}d.")
        elif a.command == "logout":
            if not store.user_exists(a.user):
                raise ValueError(f"No user '{a.user}'.")
            print(f"Ended {store.revoke_user(a.user)} session(s) of '{a.user}'.")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    audit.record(f"account_{a.command}", actor, "console", target=a.user)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Publish a DADS release to the private laptop repository (github.com/<owner>/dads-user).

The laptop repo gets a CLEAN snapshot of the local-machine branch, not this repo's history: one commit per
release, tagged v<VERSION>. Old history (which held the pre-1.1.0 built-in password) never leaves this repo.

    python scripts/publish_user_repo.py                 # snapshot of main_local, version from its VERSION file
    python scripts/publish_user_repo.py --ref v1.3.0    # any commit or tag of the local-machine code
    python scripts/publish_user_repo.py --dry-run       # build the snapshot and show the changes, push nothing

The first run creates the repository (private) with the GitHub sign-in that git already uses. On the laptop:
    git clone https://github.com/<owner>/dads-user.git DADS    then setup_local / run_local as in RUN_ENVIRONMENT.md
    later updates:  git pull    (data/ is never in git, so trained models, datasets and history stay)
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER, REPO = "sarare1", "dads-user"
EXCLUDE = {"cc_analysis.docx", "master_phd_system_specification.pdf"}  # thesis material, not part of the install
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, cwd=ROOT, capture=True, env=None):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=capture, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stdout or ''}{r.stderr or ''}")
    return r.stdout if capture else ""


def _token() -> str:
    """The GitHub token git's credential helper already holds (never printed or stored)."""
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, env=dict(os.environ, GIT_TERMINAL_PROMPT="0")).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    sys.exit("No GitHub sign-in found for git. Push anything to GitHub once so git stores it, then retry.")


def _api(method: str, path: str, body=None):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {_token()}", "Accept": "application/vnd.github+json",
                                          "User-Agent": "dads-publish"})
    try:
        with urllib.request.urlopen(req) as r:  # nosec B310
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ensure_repo() -> None:
    status, info = _api("GET", f"/repos/{OWNER}/{REPO}")
    if status == 200:
        if not info.get("private"):
            sys.exit(f"{OWNER}/{REPO} exists but is PUBLIC - refusing to publish. Make it private first.")
        return
    status, info = _api("POST", "/user/repos", {
        "name": REPO, "private": True, "has_wiki": False, "has_projects": False,
        "description": "DADS - release snapshots for local (laptop) installs"})
    if status != 201:
        sys.exit(f"Could not create {OWNER}/{REPO}: HTTP {status} {info.get('message')}")
    print(f"Created private repository {OWNER}/{REPO}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="main_local")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    version = git("show", f"{a.ref}:VERSION").strip()
    commit = git("rev-parse", "--short", a.ref).strip()
    if not a.dry_run:
        ensure_repo()
    url = f"https://github.com/{OWNER}/{REPO}.git"
    work = tempfile.mkdtemp(prefix="dads_user_repo_")
    try:
        empty = subprocess.run(["git", "ls-remote", "--heads", url], capture_output=True, text=True).stdout.strip() == ""
        if a.dry_run and empty:
            git("init", "-q", "-b", "main", cwd=work)
        elif empty:
            git("init", "-q", "-b", "main", cwd=work)
            git("remote", "add", "origin", url, cwd=work)
        else:
            git("clone", "-q", url, work, cwd=ROOT)
        if git("tag", "-l", f"v{version}", cwd=work).strip():
            sys.exit(f"v{version} is already published to {OWNER}/{REPO}.")
        for name in os.listdir(work):  # replace the whole tree with the snapshot
            if name != ".git":
                p = os.path.join(work, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        tar = subprocess.run(["git", "archive", "--format=tar", a.ref], cwd=ROOT, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:  # our own git archive, not outside input
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            t.extractall(work, **kw)  # nosec B202
        for name in EXCLUDE:
            p = os.path.join(work, name)
            if os.path.exists(p):
                os.remove(p)
        git("add", "-A", cwd=work)
        print(git("status", "--short", cwd=work) or "(no file changes)")
        if a.dry_run:
            print(f"Dry run: would publish DADS v{version} ({commit}) to {OWNER}/{REPO}.")
            return 0
        git("-c", "user.name=" + git("config", "user.name").strip(), "-c", "user.email=" + git("config", "user.email").strip(),
            "commit", "-q", "--allow-empty", "-m", f"DADS v{version}\n\nSnapshot of release v{version} (source {commit}).\n\n{TRAILER}",
            cwd=work)
        git("tag", "-a", f"v{version}", "-m", f"DADS v{version}", cwd=work)
        git("push", "-q", "origin", "main", f"v{version}", cwd=work, capture=False)
        print(f"Published DADS v{version} to https://github.com/{OWNER}/{REPO} (tag v{version}).")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

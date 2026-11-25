"""Merge `main` and `main_local` into each other without mixing their run-environment files.

  python scripts/merge_branches.py main          (on main_local: bring main's changes in)
  python scripts/merge_branches.py main_local    (on main: bring main_local's changes in)
  python scripts/merge_branches.py --check       (after fixing conflicts by hand: verify and finish)

What it does: merges the other branch without committing, puts back this branch's own version of every
environment-only file (scripts/branch_env.json) and removes the ones that belong to the other branch,
checks that the Python code still compiles, then commits. See BRANCHES.md.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAILER = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def env_files():
    with open(os.path.join(ROOT, "scripts", "branch_env.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def restore_environment(current: str, groups: dict) -> list:
    """Back to this branch's own environment files; the other branch's files removed."""
    touched = []
    for path in sorted({p for files in groups.values() for p in files}):
        if git("cat-file", "-e", f"HEAD:{path}", check=False).returncode == 0:
            git("checkout", "HEAD", "--", path)  # also resolves a conflict on it
            touched.append(f"kept   {path}")
        elif os.path.exists(os.path.join(ROOT, path)) or git("ls-files", "--", path).stdout.strip():
            git("rm", "-rf", "-q", "--ignore-unmatch", "--", path)
            touched.append(f"removed {path} (belongs to the other branch)")
    return touched


def verify(current: str, groups: dict) -> None:
    for other, files in groups.items():
        if other == current:
            continue
        for path in files:
            if path not in groups[current] and os.path.exists(os.path.join(ROOT, path)):
                sys.exit(f"{path} belongs to '{other}' but is present on '{current}'.")
    for path in groups[current]:
        if not os.path.exists(os.path.join(ROOT, path)):
            sys.exit(f"Environment file {path} of '{current}' is missing.")
    r = subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "main.py"], cwd=ROOT)
    if r.returncode != 0:
        sys.exit("The Python code does not compile after the merge - fix it before committing.")


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit(__doc__)
    groups = env_files()
    current = git("branch", "--show-current").stdout.strip()
    if current not in groups:
        sys.exit(f"Run this from one of {list(groups)} (you are on '{current}').")

    if args[0] == "--check":
        if git("diff", "--name-only", "--diff-filter=U").stdout.strip():
            sys.exit("Conflicts are still unresolved:\n" + git("diff", "--name-only", "--diff-filter=U").stdout)
        for line in restore_environment(current, groups):
            print(line)
        verify(current, groups)
        git("add", "-u")
        git("commit", "-q", "-m", f"Merge into {current} (environment files kept)\n\n{TRAILER}")
        print("Committed. Push when ready:  git push")
        return

    other = args[0]
    if other == current or other not in groups:
        sys.exit(f"Give the OTHER branch: one of {[b for b in groups if b != current]}.")
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        sys.exit("Commit or stash your changes first.")
    git("fetch", "-q", "origin", check=False)
    ref = other if git("rev-parse", "--verify", "-q", other, check=False).returncode == 0 else f"origin/{other}"
    r = git("merge", "--no-ff", "--no-commit", ref, check=False)
    if "Already up to date" in r.stdout:
        print("Nothing to merge.")
        return
    for line in restore_environment(current, groups):
        print(line)
    conflicts = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    if conflicts:
        sys.exit("Conflicts in shared files - resolve these, `git add` them, then run "
                 "`python scripts/merge_branches.py --check`:\n  " + "\n  ".join(conflicts))
    verify(current, groups)
    git("add", "-u")
    git("commit", "-q", "-m", f"Merge {other} into {current} (environment files kept)\n\n{TRAILER}")
    print(f"Merged {other} into {current}. Push when ready:  git push")


if __name__ == "__main__":
    main()

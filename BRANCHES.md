# Branches and run environments

DADS has two long-lived branches. The **application code is the same on both**; only *how it is started
and hosted* differs. Both must keep working when changes are brought across.

| Branch | Environment | Environment-only files |
|---|---|---|
| `main` | Hostinger server (systemd + Nginx + HTTPS, `/root/dads`) | `RUN_ENVIRONMENT.md` (server version), `deploy_server.sh` |
| `main_local` | One computer, nothing exposed to the network | `RUN_ENVIRONMENT.md` (local version), `setup_local.bat/.sh`, `run_local.bat/.sh` |

The authoritative list is `scripts/branch_env.json`.

## The rules
1. **Shared code is changed anywhere and merged both ways.** `src/`, `ui/`, `config/`, `docs/`, `tests/`, `.github/`,
   `README.md`, `CHANGELOG.md`, `VERSION`, `main.py`, `requirements.in` / `requirements.txt` / `constraints.txt` /
   `requirements-dev.txt` and this file are identical on both branches except for your latest un-merged work. Do not put
   server-only or local-only behaviour into them — use environment variables (see `main.py`: `DADS_HOST`, `DADS_PORT`,
   `DADS_NO_BROWSER`, `DADS_RELOAD`) or put it in an environment-only file.
2. **Environment-only files never cross.** Each branch keeps its own version of every file in `scripts/branch_env.json`.
   A file that exists only on one side (for example `setup_local.bat`, `deploy_server.sh`) must not appear on the other.
3. **Merge only with the script**, never with a plain `git merge`:
   ```
   git checkout main_local
   python scripts/merge_branches.py main          # bring server-branch updates into the local branch
   git checkout main
   python scripts/merge_branches.py main_local    # bring local-branch updates into the server branch
   ```
   The script merges, restores this branch's environment files (and removes the other side's), checks the code still
   compiles, and commits. If real conflicts remain in shared files it stops and tells you which; fix them, `git add`
   and `git commit`, then run `python scripts/merge_branches.py --check`.
4. **Adding a new environment-only file** (e.g. a new deploy helper): add it to the right list in
   `scripts/branch_env.json` on *both* branches in the same commit as the file.
5. **Git-ignored runtime data** (`data/`: datasets, models, run history) is never part of either branch and is not
   touched by a merge.
6. `git cherry-pick` of a single shared commit is fine too; for commits that touch environment-only files, use the script
   or re-apply by hand.

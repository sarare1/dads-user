# Changelog

All notable changes to DADS. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions: [Semantic Versioning](https://semver.org/) (see `docs/RELEASING.md`).

## [Unreleased]

## [1.2.0] - 2026-11-25

**Upgrade:** a new dependency (`cryptography`). On the server, migrate once with
`deploy/migrate_to_service_user.sh` (it builds a fresh venv), or run `./deploy_server.sh --deps`.

### Security
- **Model packages carry a signed hash manifest.** `MANIFEST.json` holds the SHA-256 of every file and the
  model's provenance: run, release, dataset name and hash, and seed. The server signs it with an Ed25519 key in
  `data/signing/` (`MANIFEST.json.sig`).
  - `verify_package.py` (in the package) checks a package against a trusted key.
  - `sequence_inference.py` checks the hashes before it loads the model.
  - The public key is at `GET /api/package-key`.
  - Each download is audited with the package's SHA-256.
- **No internal details in the browser.** File paths are cut out of error messages. Unexpected errors show only a
  reference, and the traceback is logged under it in `data/logs/dads.log`.
- **Unescaped fields fixed.** One HTML-escape helper (`ui/escape.js`) replaces five copies, and the Testing
  history table now escapes every field.
- **Server runs as an unprivileged user** (branch `main`): hardened systemd unit and a one-time migration
  script (see that branch's `RUN_ENVIRONMENT.md`).
- New documents: the data handling rule (`docs/DATA_HANDLING.md`) and the security policy and threat model
  (`SECURITY.md`).

### Changed
- The `dot` distance metric is gone. It was always computed as cosine. Runs labelled `dot` still load and work
  as before.

### Added
- `GET /api/health` for monitoring (no sign-in needed).
- Log file `data/logs/dads.log`, rotated. Before, DADS's own INFO messages were dropped.
- Dependency: `cryptography` (package signing). Without it, packages are still hashed, just not signed.

### Removed
- Start-up clean-up of the retired single-pulse pipeline and of the old upload folder.
- Unused code: `assign_pri_pattern`, `sample_pri_marginal`, `DatasetRanges.to_columns`,
  `_prepare_for_analysis`, `purge_non_sequence` and `scripts/setup_pod_env.sh`.
- Unused config keys: `sample_rate_hz` and `sosa_compliance`.

## [1.1.0] - 2026-11-25

### Security: sign-in hardening
- No password or session secret in the source code any more. Accounts live in `data/auth.db` with salted
  PBKDF2-SHA256 password hashes, managed with `scripts/manage_users.py`. More than one named account is possible.
- First start creates `admin` from `DADS_ADMIN_PASSWORD`, or with a random password in
  `data/INITIAL_ADMIN_PASSWORD.txt`. **The old shared password no longer works.**
- Each sign-in gets a new random session that the server can end: Logout, 2 h idle, 12 h maximum, password
  change and disabling an account all end it. Cookie is `SameSite=Strict`.
- Failed sign-ins are throttled per address (HTTP 429).
- Audit trail `data/audit.log`: sign-ins, sign-outs, account changes, every change request, downloads and Live
  Simulation sessions, with user and address, hash-chained (`manage_users.py verify-audit`).
- UI scripts under `/static/` need a session; change requests from other sites are refused.
- `GET /api/me` returns the signed-in user.

## [1.0.0] - 2026-11-25

First versioned release. It includes the evaluation-leakage fixes, so metrics from runs trained before this
release are **not comparable**: retrain to compare like with like.

### Fixed: evaluation leakage
- Class prototypes are built from the training windows only. Validation and open-set windows are only
  scored against them.
- A run's headline accuracy and AUROC are the mean over the cross-validation folds. The kept fold's own
  validation score is still reported, under `kept_fold_validation`.
- The test split's unknown tracks are kept out of every fold's open-set pool, so they no longer influence
  which fold is kept.
- Training is seeded. The seed is recorded with the run as `train_seed`, and fold f uses `train_seed + f`.
- One AUROC implementation is used everywhere, and it counts ties as one half (identical to sklearn's).

### Fixed: server responsiveness and limits
- Slow endpoints run in worker threads instead of on the event loop:
  - CSV uploads and parsing
  - dataset inspection, analytics and download
  - model loading, testing, inference and example tracks
  - the per-pulse model run in Live Simulation

  One large request no longer stalls every user and WebSocket.
- History `limit` must be 1–500, Analytics scatter rows 1–5000 and boxplot classes 1–50. Before, `limit=-1`
  returned everything.
- The cached model is guarded by a lock. Model package export works on a private copy of the model.

### Added: dataset lineage
- Each dataset stores the SHA-256 of its CSV; an upload also stores the hash of the file as uploaded.
- Training and test runs store the hash of their dataset.
- A deleted dataset's name is retired, and a name that any run refers to is never reused.
- Loading a model's dataset checks the hash and refuses a changed file.

### Added: engineering
- CI on every push and pull request: pytest, bandit (fails on medium+ findings) and pip-audit. The Actions are
  pinned to commit SHAs, and Dependabot proposes updates.
- Pinned dependencies: `requirements.in` lists what DADS needs, and `requirements.txt` plus `constraints.txt`
  hold the generated exact pins.
- Upgraded `filelock` to 4.3.1 and `setuptools` to 84.2.1 to fix known vulnerabilities.
- `VERSION`, this changelog, tag-triggered GitHub releases with checksums, and `GET /api/version`. Model
  packages record the release, dataset hash and training seed.
- `DADS_DATA_DIR` moves the data folder (used by the tests).

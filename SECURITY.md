# Security

## Reporting a problem
Report suspected vulnerabilities privately to the project owner, not in a public issue. Include what you
found, how to reproduce it and which version you tested (`GET /api/version`, or the `VERSION` file).

## Supported versions
Only the latest release (see `CHANGELOG.md`) gets fixes. The public server and the tester's installation
should run it.

## Threat model (summary)

| Asset | Threat | Controls |
|---|---|---|
| Datasets and trained models | Disclosure or tampering by an outsider | Sign-in on every page and API; HTTPS through Nginx; data folder readable only by the service account; data handling rule (`docs/DATA_HANDLING.md`) |
| Accounts | Password guessing, stolen session cookie | PBKDF2 password hashes; throttling (HTTP 429); random sessions the server can end; 2 h idle and 12 h total limits; `HttpOnly`, `SameSite=Strict` and `Secure` cookies |
| The server | A bug in DADS or a library used to take it over | Runs as the unprivileged `dads` user from a read-only code folder, under a hardened systemd unit (branch `main`); dependencies pinned and checked by pip-audit in CI; bandit in CI |
| Model packages | A changed model given to a receiver (TorchScript runs code when loaded) | SHA-256 manifest and Ed25519 signature; `sequence_inference.py` checks the hashes before loading; `verify_package.py --trusted-key` |
| Evidence | Denying an action, or hiding one | Hash-chained audit log (`manage_users.py verify-audit`) |
| The browser | Cross-site requests, script injection | Foreign-origin writes refused; Content-Security-Policy; one shared HTML-escape helper (`ui/escape.js`) |
| Internal details | Paths or stack traces shown to users | Error messages cleaned of paths; unexpected errors shown only as a reference that points into `data/logs/dads.log` |

**Known limits:**
- The Content-Security-Policy still allows inline scripts, because the pages are written that way.
- Data is not encrypted at rest beyond what the disk provides.
- There is no MFA.

None of these is acceptable for restricted data. See `docs/DATA_HANDLING.md`.

## Keys and secrets
- `data/auth.db` holds password hashes and session hashes. `data/signing/package_ed25519.pem` is the
  package-signing private key. Both stay on the machine, are never committed (`data/` is in `.gitignore`)
  and are covered by the backups.
- If the signing key may have leaked, delete `data/signing/`. The next download creates a new key. Then
  give receivers the new public key (`GET /api/package-key`) and tell them to stop trusting the old one.

## Changes to the code
- All changes, including AI-assisted ones, go through the same review before they reach `main`:
  - Read the diff.
  - Run `pytest`. CI must be green: tests, bandit and pip-audit.
  - Record what was checked in the commit message or pull request.
- Merge `main` and `main_local` only with `scripts/merge_branches.py`.
- Releases are tags `vX.Y.Z` that match `VERSION`. The release workflow publishes the zip with
  SHA-256 checksums (`docs/RELEASING.md`).
- Turn on branch protection for `main` on GitHub: require pull requests and the CI checks.

# Releasing DADS

A release is a git tag `v<MAJOR>.<MINOR>.<PATCH>` whose number equals the `VERSION` file.

## Version numbers

- **MAJOR**: a change that breaks something. For example, an old model package or run history no longer
  loads, or an API route changes in an incompatible way.
- **MINOR**: new features that keep working with what came before.
- **PATCH**: fixes only.

Every model trained with a release records that release in its run (`app_version`) and model package
(`model_card.json` → `dads_version`). Numbers in the thesis can therefore always be traced to the exact code.

## Steps

1. Make sure CI is green on the branch you release from. Releases are made from `main`, after
   `python scripts/merge_branches.py` has brought `main_local`'s work over.
2. Set the new number in `VERSION`.
3. Move the `## [Unreleased]` notes in `CHANGELOG.md` under a new `## [x.y.z] - YYYY-MM-DD` heading.
4. Commit both files: `git commit -am "Release vx.y.z"`.
5. Tag and push:
   ```bash
   git tag -a vx.y.z -m "DADS vx.y.z"
   git push origin main vx.y.z
   ```
6. The **Release** workflow then runs:
   - It reruns the CI checks.
   - It checks that the tag equals `VERSION`.
   - It publishes a GitHub release with `dads-vx.y.z.zip` (the source at that tag) and `SHA256SUMS`, with the
     changelog section as the release notes.
7. **Server.** Deploy the tagged code with `./deploy_server.sh`. Add `--deps` if `requirements.txt` or
   `constraints.txt` changed; the pinned versions need Python 3.11 or newer.
8. **Tester's machine.** Name the patch zip after the release (`dads-vx.y.z-patch.zip`) and send its SHA-256
   with it, so the tester can check the file arrived intact:
   - Windows: `certutil -hashfile file.zip SHA256`
   - Linux/macOS: `sha256sum file.zip`

## Dependencies

- `requirements.in` is the hand-edited list.
- `requirements.txt` and `constraints.txt` are generated pins. Never edit them by hand.

To upgrade:

```bash
pip install -U <package>          # in the venv
pytest                            # must pass
python scripts/pin_requirements.py
pip-audit -r requirements.txt -r constraints.txt
```

Dependabot opens weekly pull requests for package and GitHub Action updates. After merging a package update,
regenerate the pins as above.

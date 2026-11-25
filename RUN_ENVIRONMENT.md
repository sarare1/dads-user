# Run environment — `main_local` (one computer)

> **This file belongs to the `main_local` branch only.** The `main` branch has its own `RUN_ENVIRONMENT.md`
> (Hostinger server). Never carry one over to the other — see `BRANCHES.md` and use `scripts/merge_branches.py`.

DADS set up to run on a single computer, for one person, with nothing exposed to the network.


**You need:** Python 3.10 or newer (3.13 is what it was developed on) and about 3 GB of free disk space
(PyTorch is large). No GPU is needed; everything runs on the CPU.

### Windows
```
setup_local.bat      (once: creates venv and installs the packages — takes a few minutes)
run_local.bat        (every time: starts DADS and opens http://localhost:8000)
```

### Linux / macOS
```bash
./setup_local.sh     # once   (CPU-only PyTorch, much smaller:  DADS_CPU_TORCH=1 ./setup_local.sh)
./run_local.sh       # every time
```

### By hand
```bash
python -m venv venv
.\venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Stop with Ctrl+C.

**Signing in.** On the first start the console says that user `admin` was created and where its password is:
`data/INITIAL_ADMIN_PASSWORD.txt`. Sign in with it, then choose your own password (at least 12 characters):
```
venv\Scripts\python scripts\manage_users.py passwd admin      (Linux/macOS: venv/bin/python scripts/manage_users.py passwd admin)
```
That also deletes the password file. Forgot the password? Run the same command. More in `README.md` → Sign-in and accounts.

**First run:** go to **Dataset** and generate a Sequence Dataset (or upload your own tracks), then go to
**Training** and click Train Sequence Model — everything else (Testing, Inference, Analytics, Live Simulation)
depends on that.

### Settings (environment variables, all optional)
| Variable | Default | Meaning |
|---|---|---|
| `DADS_PORT` | `8000` | Port to listen on (use another if 8000 is taken) |
| `DADS_HOST` | `127.0.0.1` | `127.0.0.1` = this computer only; `0.0.0.0` = reachable from other computers on your network |
| `DADS_NO_BROWSER` | unset | `1` = do not open the browser on start |
| `DADS_RELOAD` | unset | `1` = restart on code changes (development) |

### Your data
Everything you create lives in the `data/` folder next to the code: datasets, trained models, run history.
It is created on the first start and is not part of git. To move your work to another computer, copy the
whole `data/` folder over (stop DADS first). To back up, copy the same folder.

### Troubleshooting
- *"Port already in use"* — set `DADS_PORT=8001` (Windows: `set DADS_PORT=8001` before `run_local.bat`).
- *`pip install` is slow or fails on torch* — install PyTorch first from <https://pytorch.org/get-started/locally/>
  (choose CPU), then run the setup again.
- *The browser shows an old page after an update* — hard-refresh with Ctrl+F5.
- *Training is slow* — fewer epochs, fewer classes/tracks, or a smaller window k on the Training page; a laptop CPU
  handles the defaults but a long run can take several minutes.

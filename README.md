# SOSA-Aligned Open-Set Radar Emitter Recognition Platform

A research platform for open-set radar emitter classification from **pulse trains**, with a
real-time ESM (Electronic Support Measures) track simulation — built for PhD research into
SOSA-aligned electronic warfare signal processing. Every metric, dataset, and model in this
system is real and computed live (no hardcoded/simulated results): real PyTorch training,
real open-set evaluation.

**Explicit design constraints:** PyTorch + ONNX Runtime only (no TensorRT in the running app —
no NVIDIA GPU on the target dev machine); no database or Docker; synthetic-data-only (no real VITA49
capture or radar hardware).

> **History:** DADS began as a single-pulse classifier (one PDW → one family) with an embedded Qdrant
> vector database. That pipeline — its datasets, backbones, training, testing, inference and live feed —
> was removed once the sequence models superseded it, and Qdrant was removed afterwards because the
> prototype search is a plain exact comparison over a few dozen vectors (see §4). Git history before
> those commits still contains both.

---

## 1. What this system does

This platform trains a neural network to identify which of N known radar "families" a
**de-interleaved track** (the ordered pulses of one emitter) belongs to, while also recognizing
when a track belongs to **none** of them (open-set recognition). The model sees a window of k
consecutive pulses, so it can use what a single pulse cannot show: the PRI modulation pattern
and the frequency agility over time. A track is decided by voting over its windows.

| Page | Route | Purpose |
|---|---|---|
| Training | `/` | Configure and train sequence models; view run history |
| Testing | `/validation` | Evaluate a trained model on fresh / dataset / uploaded tracks |
| Inference | `/inference` | Classify a track (CSV, file or generated example) and see the decision develop |
| Dataset | `/dataset` | Generate or upload the Sequence Dataset |
| Analytics | `/analytics` | EDA of the sequence dataset or an uploaded track file |
| Live Simulation | `/live` | Several simulated emitters at once, each track classified as it grows |
| Help | `/help` | In-app glossary and step-by-step explanation of the whole pipeline |

---

## 2. Architecture overview

```
                    ┌──────────────────────────────────────────────┐
                    │               Dataset page (UI)               │
                    │  generates or uploads the Sequence Dataset:   │
                    │  named datasets: <name>.csv + <name>.meta.json │
                    └────────────────────┬─────────────────────────┘
                                         │
                            ┌────────────┴─────────────┐
                            │  Training page            │
                            │  -> SequenceDualHeadModel │
                            │  -> prototypes + τ_c, τ_d │
                            │  -> SQLite run history    │
                            └────────────┬─────────────┘
                                         │
              ┌──────────────────────────┼───────────────────────────┐
              │                          │                            │
        Testing page               Inference page               Live Simulation
        (fresh tracks /            (a track -> windows ->        (several simulated emitters,
         dataset / upload)          track verdict)                decision after every pulse)
```

---

## 3. Technical stack

- **Backend**: FastAPI + Uvicorn (`--reload` dev server), vanilla Python — no ORM, no task queue.
- **Frontend**: Vanilla HTML/JS (no framework/build step), Chart.js (CDN) for all charts.
- **ML**: PyTorch (model + training loop), scikit-learn (KMeans, silhouette, PCA), ONNX Runtime (portable inference in the downloadable package).
- **Persistence**: SQLite (stdlib `sqlite3`) for training/test run history; flat CSV/JSON files for datasets.
- **No GPU dependency anywhere in the running app** — CPU-only PyTorch.

---

## 4. Model architecture — `SequenceDualHeadModel` (`src/models/sequence_models.py`)

```
track window X (k pulses × F features) -> Temporal encoder -> f (d numbers) ┬─> Classifier Head -> softmax probs (num_classes)
                                                                             └─> Metric Head     -> L2-normalized embedding (d-dim)
```

- **Input**: per pulse `[PRI, ΔPRI, freq, Δfreq, PW, ΔPW, BW, rise, fall, duty cycle, PW×PRI]` (any subset),
  Δ = difference to the previous pulse of the same track, computed by DADS (`src/data/sequence_features.py`).
  No time of arrival — only the per-pulse PRI a de-interleaver reports. Frequency in MHz, bandwidth in Hz.
  Each feature is z-score scaled with the mean/std of the training tracks' pulses; Δ features first get a
  signed-log step (sign·ln(1+|Δ|/median|Δ|)). The feature list, window length and scaling are saved with the
  model, so Inference, Testing and Live Simulation always feed it exactly what it was trained on.
  Any extra numeric column of an uploaded track file (e.g. `aoa_deg`) is also selectable, as its value
  (`col:<name>`) and/or its per-track Δ (`dcol:<name>`); models using one can't run on simulated tracks.
  A blank or zero rise/fall time counts as not measured and is filled from the same track; the first pulse's
  missing PRI takes the 2nd pulse's (no decision before pulse 2).
- **Encoders** (along the pulses, all length-agnostic via pooling), chosen per training run:
  `seq_hybrid` (default — temporal CNN + temporal Transformer fused by a learned layer),
  `seq_cnn` (dilated temporal 1D-CNN), `seq_transformer` (sinusoidal positional encoding), `seq_gru` (bidirectional GRU baseline).
- **Two output heads**:
  - **Classifier head** — a linear layer + softmax over `num_classes` (a property of the generated/uploaded dataset).
  - **Metric head** — a small MLP projecting to a d-dim, L2-normalized embedding, trained via in-batch triplet loss (same-class windows pulled together, different-class pushed apart) using the configured distance metric (`cosine` / `euclidean`; runs from before v1.2.0 may show `dot`, which was always computed as cosine).
- **Embedding size d** (Training page, default 128, options 8–256): the width of the feature vector f and the stored
  embedding. Measured on simulated data: accuracy falls below d = 64 (56% at d = 8), 128 was the smallest size at
  100%, 256 gains nothing. A stored vector is 4·d bytes. See Help → Embedding size.

Total training loss per batch: `cross_entropy(known windows) + triplet_loss + rpl_loss_weight * (RPL auxiliary loss)` (see §5).

**Training / testing** (`src/training/trainer.py`, `src/models/sequence_eval.py`):
- Split by track (never by window) and per family as cross-validation: a held-out test split (default 20 %, kept for the Testing page), the rest in folds (default 4; each fold validates on one bucket and trains on the others, the best fold is kept and the run reports the mean over the folds), 10% of the training pool for calibration; unknown
  tracks (held-out families + noise) are an evaluation-only pool (the test split's unknown tracks stay out of model selection). Training is seeded and the seed recorded. Half the training batches use random 2…k-pulse
  crops so short tracks are decided well.
- Per-window and per-track (vote) accuracy, AUROC against the stored prototypes, and accuracy vs. pulses
  observed — one decision per track after its first n pulses, from those pulses only, as Live Simulation decides.
  Testing can regenerate fresh tracks with harsher missed/spurious rates.
- **Prototype search**: each class keeps up to 3 prototype vectors (K-means centres of its training-window
  embeddings) in the run's `prototypes.json`. A window's nearest prototype is found by an exact comparison with
  every one (cosine / euclidean distance, `trainer.build_distance_fn`) — a few dozen vectors, so no index
  or vector database is needed, and the downloadable package does the same arithmetic. A vector database or
  ANN index would only matter for libraries of many thousands to millions of emitters.
- **Thresholds**: every sequence model has its own τ_c and τ_d. τ_d is auto-calibrated on the held-out calibration
  tracks (known data only) so a chosen share of known windows is accepted, per window length below k (same
  known-track acceptance at every length), or entered manually.
- Uploaded track files keep their own class names; the exported ONNX model accepts any number of pulses.

---

## 5. Open-set recognition — two independently-scored methods

Every training run reports **both** of the following, computed from the same embeddings, so
they're directly comparable:

### 5.1 Embedding-centroid distance threshold (the method the decision rule acts on)
- After training, each known class's validation-window embeddings are clustered into up to **3 sub-centroids**
  via K-means (not a single mean) — `src/training/trainer.py::_class_sub_centroids`.
- **Why sub-centroids, not one mean per class**: families span multiple correlated operating modes with
  different PW/PRI/frequency profiles. A single mean embedding can land in the empty space *between* a class's
  real sub-clusters, making genuine same-class windows look falsely anomalous.
- **Decision logic** (`SequenceClassifier` in `src/models/sequence_eval.py`): combines classifier softmax
  confidence with the distance to the nearest prototype into one of three verdicts —
  `VERIFIED_KNOWN_TARGET` (green), `UNKNOWN_THREAT_ANOMALY` (orange — confident classifier but far from any
  prototype, a possible novel/spoofed emitter), `OOD_NOISE_REJECTED` (red — low confidence regardless of
  distance). A track's verdict is the most frequent window verdict.

### 5.2 Reciprocal Points Learning / ARPL (`src/models/open_set.py`)
A second, independently-trained open-set method, reported as `open_set_auroc_rpl` alongside
the threshold method's `open_set_auroc` — never replacing it.

- **RPL** (Chen et al., ECCV 2020): instead of representing a class by what it IS (a
  centroid), learns a *reciprocal point* per class representing what it ISN'T. Distance to a
  class's reciprocal point IS that class's logit (farther = more confident) — trained via
  cross-entropy on those logits. Open-set score = the confidence gap between the top-1 and
  top-2 farthest reciprocal points.
- **Full ARPL** (Chen et al., TPAMI 2021, opt-in via `rpl_adversarial`): adds a
  `ConfusingSampleGenerator` — a small generator network adversarially trained to synthesize
  embeddings that fool the RPL head into confident (large-gap) acceptance, while the RPL head
  is simultaneously trained to keep such synthesized samples ambiguous (small gap). A genuine
  min-max adversarial game, played directly in embedding space.
- Toggle in Training page's "Reciprocal Points (RPL / ARPL)" panel: **RPL Loss Weight**
  (default 0.1) and **Enable Adversarial Training (ARPL)** (off by default — retrain with it
  on for a genuine RPL-vs-ARPL comparison, both visible side by side in Run History).

---

## 6. Dataset generation — the Sequence Dataset (`src/data/sequence_generator.py`)

Generated from the Dataset page (or upload your own tracks: one row per pulse with the columns
`Frequency, PW, PRI, Modlntype, Modlnval, snr, rise, fall, timeBW, instance, class` plus the optional
`pulse_idx`, `true_class_id`; user-defined extra features and *grow a small dataset into a large one* are on the same page). The *only* dataset the sequence
models train and test on.

- **Real hyperparameters, not hardcoded**: Classes, Holdout Classes, Tracks per Class, Pulses per Track,
  Missed / Spurious pulse rates, Noise Tracks %, Include PGRI and 14 population-range fields
  (frequency / PW / PRI / rise / fall / bandwidth / duty cycle) — all persisted to `sequence_dataset_meta.json`.
- **Open-set protocol**: `Holdout Classes` are entirely excluded from training and evaluated only as
  "unknown", plus whole **noise tracks** with no coherent emitter (`UNKNOWN_NOISE_OOD`).
- **Correlated operating modes** (`src/data/operating_modes.py`): 4 modes (search / acquisition / track /
  illumination) shift frequency / PW / PRI *together*; each emitter dwells in a mode for 15–60 pulses, then a
  biased transition (50% escalate, 30% de-escalate, 20% jump) picks the next one.
- **Pulse width from duty cycle**: each family draws a duty cycle (log-uniform, default 0.1–10%) and
  PW = duty × PRI, kept inside the PW range where possible — no family has pulses longer than its pulse interval.
- **PRI modulation patterns** (`src/data/pri_patterns.py`): constant / jittered / staggered / sliding / wobulated /
  dwell-and-switch, plus optionally **PGRI** (pulse groups; group shape fitted to the family's pulse width).
  A track generates the real time order: stagger cycle, ramp, sine, dwell blocks, pulse groups.
- **Frequency agility** per family: fixed / hopping / stepped / burst.
- **De-interleaving errors**: *missed* pulses (the next PRI becomes the sum of two intervals) and *spurious*
  pulses (another emitter's pulse inserted into the track).
- Ground-truth columns (`pri_pattern`, `freq_agility`, `operating_mode`, `pri_mode_base_us`,
  `pri_deviation_pct`, `missed_before`, `spurious_before`, `is_spurious`) are included in the CSV for inspection
  and analytics — never used as model input. Analytics → PRI Pattern Check verifies each family's deviation
  against its label.

### 6.1 Adding a family later — incremental training (`src/training/incremental.py`)
Training page → *Add New Families*: a kept model (the base) is taught one or more new families from a labelled
dataset, by up to four methods, each producing a child model (`parent_run_id`, `incremental_method` in the run
history; the base is never changed):

| | Method | Data | What happens |
|---|---|---|---|
| A | Frozen encoder + weight imprinting | new only | new classifier row = mean feature vector; prototypes added; no training |
| B | Fine-tune + replay buffer | new + R old tracks per family | a few epochs from the base weights; prototypes and τ_d recomputed |
| C | Fine-tune + distillation (LwF) | new only | the base model as teacher (KL on old-class logits, embedding cosine) |
| D | Full retrain | all old training tracks + new | from scratch — the reference |

All are scored on the same held-out tracks (the base model's validation tracks → forgetting, a share of the new
family's tracks, the new dataset's unknown tracks), next to the base model; each comparison is saved under
`data/incremental_experiments/`. For simulated studies, `POST /api/sequence/dataset/new_families` generates
fresh, labelled tracks of a model's own held-out families (the "new emitter that turned up after deployment").

---

## 7. Live Simulation — tracks

Several simulated emitters transmit at once (up to 12); every pulse arrives already assigned to its
track, as from a de-interleaver. After each pulse from the 2nd, the track's last ≤ k pulses are classified
(with the τ_d calibrated for that length), so you watch each decision firm up as the track grows. Tracks end
and new emitters appear; a configurable share of new tracks is unknown (held-out families or noise).
Ground truth is shown for scoring only — "Correct" means: known track → right class and verified; unknown
track → not verified as known. Hard-capped at 15 minutes server-side. Layout follows AN/SLQ-32 / ISA-101
operator-console research: aggregate tiles and a track table first, diagnostic charts second.

Only a model whose features the track simulator can produce (the six measured columns) can run here; a model
that uses extra dataset columns must be tested on track files.

---

## 8. Dataset page vs. Analytics page

Both are computed by `src/analytics/dataset_analysis.py`, so they always agree. Rise/fall
times of 0 count as "not measured" in both — excluded from charts, counted under data quality.

- **Dataset page (health check)**: composition (known/held-out/noise pulses), class distribution, one small
  histogram per feature with the configured generation range (generated datasets), a data-quality table
  (blank / zero / outlier per column), and the track summary (PRI patterns, frequency agility, example tracks).
- **Analytics page (deep analysis)**: the Sequence Dataset or any uploaded track file, analysed as independent
  pulses (the per-pulse distributions, not the time order). One-shot EDA endpoint (`/api/analytics`):
  - **Overview**: summary statistics, Pearson correlation heatmap, class balance + diversity.
  - **Class Separability**: 2D PCA of the raw per-pulse features with a silhouette score.
  - **Data Quality**, **Feature Usefulness** (mutual information, ANOVA F, known-vs-unknown KS), per-class and known-vs-unknown distributions.
  - **Radar/ESM**: frequency band breakdown, PRI pattern distribution and check, RF spectrum, PRI / PRF histograms, PW vs PRI duty-cycle map, frequency vs PW map, rise vs fall fingerprint, time-bandwidth product, operating mode by class.
  - **Distributions / Box Plots / Scatter Explorer**.

---

## 9. Persistence model

| What | Where | Survives restart? |
|---|---|---|
| Every run's model + prototypes + evaluation/explainability report | `data/models/runs/<run_id>/` (unsaved runs: newest 20 kept) | Yes |
| Which run is in use (Inference / Live Simulation) | `data/models/active_sequence_run.json` | Yes |
| Training run history (full config + metrics) | `data/training_history.db` (SQLite) | Yes |
| Test run history | same SQLite DB, `test_runs` table | Yes |
| Datasets (generated, uploaded for training, uploaded for testing), each `<name>.csv` + `<name>.meta.json`, and the default-dataset pointer | `data/sequence_datasets/` | Yes (until deleted on the Dataset page) |
| Names of deleted datasets (never given out again) | `data/sequence_datasets/_retired.json` | Yes |
| Accounts (hashed passwords) and sign-in sessions | `data/auth.db` (SQLite) | Yes |
| Audit trail (sign-ins, changes, downloads) | `data/audit.log` (hash-chained JSON lines) | Yes |
| Server log (DADS messages and error tracebacks, by reference) | `data/logs/dads.log` (rotated, 5 × 5 MB) | Yes |
| Package-signing key pair (private key: owner-only file) | `data/signing/` (created on the first package download) | Yes |

**Dataset lineage.** A dataset name always means the same data:
- Each dataset's meta file records the SHA-256 of its CSV (`sha256`). An uploaded file also records the hash of the
  file exactly as uploaded (`original_sha256`).
- Every training run stores the hash of the data it trained on (`dataset_sha256`) and the DADS release that
  trained it (`app_version`). Every test run stores the hash of the data it was tested on.
- A deleted dataset's name is retired. A name that any run still refers to is never given to new data either.
- When Testing (test split), example tracks, Analytics or incremental training load a model's dataset, they check
  the hash. If the file has changed since training, they refuse with an error instead of using different data.

At server startup, the settings of the sequence model in use (or the latest run) — embedding size, distance
metric, thresholds, RPL settings — are restored from SQLite so the Training form agrees with
what is persisted. **Apply Configuration never touches a trained model**: it only stores the settings the next
training run starts from.

**Legacy cleanup**: at startup the server deletes what the removed single-pulse pipeline left behind — its runs
and test results (history rows and model files), `classification_datasets/`, `classification_uploads/`,
`auto_test_samples/`, `synthetic_features.json`, `qdrant_storage/` and `current_model*` files. This is idempotent and a no-op on a
fresh install.

### Saved models, switching and explainability

Every training run keeps its trained model under `data/models/runs/<run_id>/` (`src/models/registry.py`):
weights + window + input features/scaling (`model.pt`), class prototypes, and a report
(`src/models/explain.py`, `src/models/sequence_eval.py`) with the confusion matrix, per-class precision/recall/F1,
open-set ROC, verdict breakdown, track accuracy, accuracy-vs-pulses curve, the model's layer-by-layer formulas
with its real dimensions and thresholds, input scaling constants, weight statistics for every layer, and feature
influence (permutation importance + gradient sensitivity). Hybrid runs also get a branch contribution test.
The Help page documents the whole pipeline end to end, with the formula and rationale for every step.

- **Save**: name a run (Run History → double-click → ★ Save Model); saved runs show as ★ name.
  Unsaved runs beyond the newest 20 are pruned; saved and in-use models are never pruned.
- **Delete**: 🗑 Delete in a Run History row (or in a run's details) permanently removes the run, its
  model files and that model's Testing-page results; deleting the model in use clears that choice.
- **Switch**: the model selector on Inference / Live Simulation (or "Use for Inference / Live Simulation" in a
  run's details) makes a model the one in use, server-wide; it survives restarts. Testing can test any kept model.
- **Download**: a deployable .zip that runs without DADS code — `sequence_inference.py` (per-track features →
  scaling → windows → network → prototype distance → verdict → track vote, identical to the server; backends ONNX
  Runtime [TensorRT/CUDA/CPU], TorchScript [CUDA/CPU], or a TensorRT engine), `model.onnx` (dynamic batch × pulses),
  `model.torchscript.pt`, `model.pt`, prototypes, model card, report, and a README with the `trtexec` command for
  this model — or every weight as JSON.

---

## 10. API reference

| Method | Route | Purpose |
|---|---|---|
| GET | `/`, `/validation`, `/inference`, `/dataset`, `/analytics`, `/live`, `/help` | Page routes |
| GET / POST | `/api/config` | Read / update the shared training settings |
| POST | `/api/sequence/train` | Train a sequence model (background) |
| GET | `/api/train/status`, `/api/train/history` | Poll progress / list past runs (`limit` 1–500) |
| GET | `/api/sequence/options`, `/api/sequence/models` | Sequence features/backbones; kept models |
| POST | `/api/sequence/dataset/generate`, `/api/sequence/dataset/upload` | Generate / upload the Sequence Dataset |
| GET | `/api/sequence/dataset`, `/api/sequence/dataset/download` | Dataset summary + example tracks / CSV |
| GET | `/api/dataset/inspect` | Dataset page health check: composition, per-feature mini-histograms, data quality |
| GET | `/api/analytics`, `/api/analytics/sources` | Full EDA payload / datasets available for analysis |
| POST | `/api/sequence/test`, `/api/sequence/test/upload` | Test a model (fresh tracks / dataset / uploaded) |
| GET | `/api/test/history` | Past test runs (`limit` 1–500) |
| GET | `/api/version` | The running DADS release (`VERSION`) |
| GET | `/api/health` | Up/down for monitoring, no sign-in needed: `{"status": "ok"}`, or 503 with the problem |
| GET | `/api/package-key` | The public key that signs model packages (what receivers pin as trusted) |
| POST | `/api/login`, `/api/logout` | Sign in (new session cookie; throttled) / sign out (session ended on the server) |
| GET | `/api/me` | The signed-in user |
| POST | `/api/sequence/models/activate` | Choose the model in use |
| POST | `/api/sequence/infer`, `/api/sequence/example_track` | Classify tracks / generate an example track |
| POST | `/api/sequence/incremental` | Incremental training: methods A–D from a base model on new-family data (background) |
| GET | `/api/sequence/incremental/options`, `/api/sequence/incremental/experiments` | Methods / families still unknown to a model; saved comparisons |
| POST | `/api/sequence/dataset/new_families` | Simulated labelled tracks of a model's held-out families |
| GET / DELETE / POST | `/api/datasets`, `/api/datasets/{name}`, `/api/datasets/{name}/activate` | List / delete / make default |
| WS | `/ws/sequence_live` | Live Simulation |
| GET | `/api/runs/{id}` | Run details: config, metrics, evaluation + explainability report |
| DELETE | `/api/runs/{id}` | Delete a run, its model files and its model's test results |
| POST / DELETE | `/api/runs/{id}/save` | Save under a name / unsave |
| GET | `/api/runs/{id}/download`, `/api/runs/{id}/weights` | Model package (.zip, with a signed hash manifest) / all weights (JSON) |

---

## 11. Project structure

```
src/
  api/server.py            FastAPI app — all routes, config state, sequence training / testing / inference
  data/
    generator.py            Emitter-family archetypes (frequency, duty-cycle-based PW, PRI pattern, …)
    operating_modes.py       Shared search/acquisition/track/illumination mode logic
    pri_patterns.py           Six standard PRI modulation types + PGRI
    sequence_generator.py     Track dataset: PRI patterns + frequency agility over time, missed/spurious pulses
    sequence_features.py      Per-pulse sequence features (incl. Δ), windows, SequenceSpec
  analytics/dataset_analysis.py  Dataset page + Analytics computations (distributions, ranking, quality, radar views)
  models/
    sequence_models.py        Temporal CNN / Transformer / Hybrid / GRU + SequenceDualHeadModel
    sequence_eval.py          Sequence evaluation (track vote, pulses curve), classifier, explainability
    open_set.py               Reciprocal Points Learning + adversarial (ARPL) generator
    explain.py                Shared evaluation report, branch ablation, weights export
    checkpoint.py             Sequence checkpoint bundle (weights + window + features + scaling)
    registry.py               Per-run artifacts, model in use, downloadable package
    package_signing.py        Package manifest, Ed25519 signing, verify_package.py (shipped in every package)
    deploy/sequence_inference.py  Standalone inference script shipped in the model package
  training/trainer.py        Training loop (CE + triplet + RPL/ARPL), prototypes, AUROC, distance functions
  storage/training_store.py  SQLite training/test run history
ui/                         One HTML file per page (vanilla JS, Chart.js via CDN)
config/config.yaml          Startup defaults (overridden by the last real training run)
data/                       All generated/persisted artifacts (see §9)
```

---

## 12. Known limitations (documented deliberately, not hidden)

- **ARPL scope**: implements RPL's core mechanism plus a genuine adversarial generator game in embedding space;
  the original vision-domain ARPL paper's image-feature decoder isn't reproduced.
- **Simulated data**: the track simulator is the only data source besides your own uploads; accuracy on it says
  little about real emitters.
- No TensorRT / GPU support in the running app by design — CPU-only PyTorch (the downloadable package can use
  ONNX Runtime GPU / TensorRT).
- **Engineering debt, known and kept for now:**
  - `src/api/server.py` holds routes, job control and analytics in one large file.
  - The history database stores numbers as text.
  - The Content-Security-Policy allows inline scripts.

  Changing any of these touches most of the code (and every tester patch), so each will be its own release.

---

## 13. Setup & run

The code is the same on every branch; *how it is started* depends on the branch (see `BRANCHES.md`):

| Branch | Runs on | How to start | Details |
|---|---|---|---|
| `main` | The Hostinger server (systemd + Nginx) | `deploy_server.sh` (git pull + restart) | `RUN_ENVIRONMENT.md` |
| `main_local` | One computer | `setup_local` once, then `run_local` | `RUN_ENVIRONMENT.md` |

Plain development run, on any computer:

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt -c constraints.txt    # exact, tested versions (Python 3.11+)
python main.py
```

Opens `http://localhost:8000`. First run: go to **Dataset** and generate a Sequence Dataset
(or upload your own tracks), then go to **Training** and click Train Sequence Model — everything else
(Testing, Inference, Analytics, Live Simulation) depends on that.

### Sign-in and accounts

There is no built-in password. Accounts are stored in `data/auth.db`, with passwords kept only as salted PBKDF2
hashes.

- **First start.** If there are no accounts, the server creates `admin`:
  - with the password in `DADS_ADMIN_PASSWORD` (user name `DADS_ADMIN_USER`), if that is set;
  - otherwise with a random password, written to `data/INITIAL_ADMIN_PASSWORD.txt` (never to the log). The
    console says where the file is.
- **Manage accounts** with `python scripts/manage_users.py` (`list`, `add`, `passwd`, `disable`, `enable`,
  `logout`, `verify-audit`). Changes take effect immediately, also while the server runs. Changing a password
  or disabling an account ends that user's sessions. Passwords need at least 12 characters.
- **Sessions.** Each sign-in gets a new random token; the server keeps only its hash. A session ends after 2 hours
  without use, after 12 hours in total (`DADS_SESSION_IDLE_MINUTES`, `DADS_SESSION_MAX_HOURS`) or on Logout.
  The cookie is `HttpOnly` and `SameSite=Strict`, and `Secure` over HTTPS (or always with `DADS_COOKIE_SECURE=1`).
- **Guessing.** 5 failed sign-ins for one user name from one address, or 20 in total from one address, block
  that address for 15 minutes. The account itself is never locked, so nobody can lock the real user out.
- **Audit trail.** `data/audit.log` gets one JSON line per sign-in, failed sign-in, sign-out, account change,
  change request (every POST/DELETE), download and Live Simulation session, with the user and address. Each line
  holds the SHA-256 of the line before it, so `manage_users.py verify-audit` detects an edited, removed or
  inserted line. (Cutting lines off the end is not detectable from the file alone; copy the log off the machine
  if that matters.)
- Every page, script and API route needs a signed-in session, except the sign-in page, its stylesheet and
  `/api/health`. Requests that change something are refused if they come from another site (`Origin` check).
- **Errors.** Messages shown in the browser never contain server paths. An unexpected error shows only a short
  reference (e.g. `reference 3fa2c19b`); the full traceback is in `data/logs/dads.log` under that reference.

### Model packages: integrity and provenance
Every downloaded package contains `MANIFEST.json`, which holds the SHA-256 of every file plus where the model
came from: run, DADS release, dataset name and hash, and training seed. The server signs the manifest with its
Ed25519 key (`MANIFEST.json.sig`; needs the `cryptography` package, otherwise the package is hashed but unsigned).
- `sequence_inference.py` in the package checks the hashes before it loads any model file and stops on a
  mismatch. This matters because `model.torchscript.pt` runs code when loaded.
- A receiver pins the server's public key once (`GET /api/package-key` while signed in, or `data/signing/package_ed25519.pub` on the server) and checks each package:
  `python verify_package.py package.zip --trusted-key <hex>`. The same option exists on `sequence_inference.py`.
- Each download is recorded in the audit log with the package's SHA-256.

### Data handling
Real (measured) emitter data never goes on the public server. Read `docs/DATA_HANDLING.md` before uploading
anything that was not generated. Security policy and threat model: `SECURITY.md`.

> **Two branches, two environments.** Merge between them only with `python scripts/merge_branches.py <other-branch>`;
> it keeps each branch's own run-environment files. Rules in `BRANCHES.md`.

## 14. Quality gates, dependencies and releases

- **CI** (`.github/workflows/ci.yml`, on every push to `main` / `main_local` and every pull request):
  - `pytest` runs the whole `tests/` suite. Each run uses its own temporary data folder (`DADS_DATA_DIR`), never `data/`.
  - `bandit` scans the code for security problems and fails on medium or higher findings.
  - `pip-audit` checks every pinned package against the known-vulnerability databases.
  - The GitHub Actions themselves are pinned to commit SHAs. Dependabot proposes updates weekly.
- **Pinned dependencies**:
  - `requirements.in` lists what DADS needs (version ranges); edit this file by hand.
  - `requirements.txt` pins those packages to the exact tested versions.
  - `constraints.txt` pins everything they pull in.
  - Regenerate the pins with `python scripts/pin_requirements.py` after an upgrade.
- **Releases**:
  - `VERSION` holds the release number. Releases are git tags `v<VERSION>`.
  - Pushing a tag runs the CI checks and then publishes a GitHub release with the source zip and its `SHA256SUMS`.
  - See `docs/RELEASING.md` and `CHANGELOG.md`.
  - `GET /api/version` reports the running release.

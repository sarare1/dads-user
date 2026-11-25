import os
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.environ.get("DADS_DATA_DIR") or os.path.join(os.path.dirname(__file__), "../../data"),
                       "training_history.db")
IST = timezone(timedelta(hours=5, minutes=30))

_DROPPED_COLUMNS = {"inference_engine", "hnsw_m", "hnsw_ef_construct", "quantization",
                    "rssi_min_dbm", "rssi_max_dbm", "handles_missing", "feature_dropout", "dropped_features"}

_COLUMNS = [
    "backbone_type", "distance_metric",
    "num_classes", "num_holdout_classes", "samples_per_class", "noise_pct", "epochs",
    "rpl_loss_weight", "rpl_adversarial",
    "learning_rate", "triplet_margin", "classifier_confidence_threshold", "ood_distance_threshold",
    "closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss",
    "training_duration_seconds", "loss_curve_json",
    "dataset_seed",
    "freq_min_mhz", "freq_max_mhz", "pw_min_us", "pw_max_us",
    "pri_min_us", "pri_max_us",
    "rise_min_ns", "rise_max_ns",
    "features",  # comma-separated model input feature names; NULL for runs before feature selection
    "bw_min_mhz", "bw_max_mhz", "fall_min_ns", "fall_max_ns",  # NULL for runs before bandwidth/fall time existed
    "saved_name", "saved_at_utc",  # user-given name when the run's model is saved; NULL = not saved
    "model_kind",  # always "sequence"
    "window_k", "window_mode",  # window_mode: "windows" (k-pulse windows) / "track" (one input per track, ≤ window_k pulses); NULL = windows
    "pulses_per_track", "missed_pct", "spurious_pct",
    "include_pgri",  # dataset generated with the PGRI pattern available; NULL = before it existed (= no)
    "duty_min_pct", "duty_max_pct",  # family duty-cycle range; NULL = before PW followed from duty cycle
    "embedding_dim",  # size of the feature vector / embedding; NULL = before it was configurable (128)
    "dataset_name",  # name of the dataset the model was trained on; NULL = before datasets were named
    "balanced_patterns",  # PRI types handed out round-robin to the families; NULL = random assignment (older runs)
    "parent_run_id", "incremental_method",  # incremental training: the base run and method A/B/C/D; NULL = trained normally
    "cv_folds", "test_pct", "include_test",  # cross-validation folds, share of tracks held out as the test split, test split also trained on; NULL = before cross-validation
    "cv_accuracy_mean", "cv_accuracy_std",  # validation accuracy over the folds
    "train_seed",  # seed of the training run (fold f uses train_seed + f); NULL = before training was seeded
    "dataset_sha256",  # SHA-256 of the dataset's CSV the model was trained on; NULL = before hashes were kept
    "app_version",  # DADS release (VERSION file) that trained the model; NULL = before releases were versioned
]

_TEST_COLUMNS = [
    "backbone_type", "distance_metric",
    "dataset_source", "num_known_classes", "num_holdout_classes",
    "closed_set_accuracy", "open_set_auroc",
    "num_known_pulses", "num_openset_pulses", "total_pulses", "dataset_seed",
    "features",
    "run_id", "model_name",  # which trained model (training run id / saved name) was tested
    "model_kind", "track_accuracy",  # sequence-model tests: kind + per-track (majority vote) accuracy
    "dataset_name",  # name of the dataset the model was tested on; NULL = fresh simulated tracks / before names
    "dataset_sha256",  # SHA-256 of that dataset's CSV; NULL = fresh simulated tracks / before hashes were kept
]


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                {", ".join(f"{c} TEXT" for c in _COLUMNS)}
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                {", ".join(f"{c} TEXT" for c in _TEST_COLUMNS)}
            )
        """)
        conn.commit()

        # Safe migration for DB files created before these columns existed — a no-op on a
        # freshly created table (CREATE TABLE above already included them), harmless if run
        # more than once (duplicate-column errors are swallowed).
        for col in _COLUMNS:
            try:
                conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass
        for col in _TEST_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE test_runs ADD COLUMN {col} TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass

        # Drop columns of settings that no longer exist.
        for table in ("training_runs", "test_runs"):
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for col in _DROPPED_COLUMNS & existing:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
                conn.commit()
    finally:
        conn.close()


def _to_ist_string(timestamp_utc: str) -> str:
    dt = datetime.fromisoformat(timestamp_utc).replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def insert_run(record: Dict[str, Any]) -> int:
    """Inserts one completed training run. `record` must contain every key in _COLUMNS;
    numeric/list values are stored as-is (sqlite3 coerces), loss_curve is JSON-encoded."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = dict(record)
        row["loss_curve_json"] = json.dumps(row.pop("loss_curve", row.get("loss_curve_json", [])))
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        columns = ["timestamp_utc"] + _COLUMNS
        values = [timestamp_utc] + [row.get(c) for c in _COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        # Column names come from the fixed column list (never from input); every value is bound with "?".
        cursor = conn.execute(
            f"INSERT INTO training_runs ({', '.join(columns)}) VALUES ({placeholders})", values  # nosec B608
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
    d["saved_at_ist"] = _to_ist_string(d["saved_at_utc"]) if d.get("saved_at_utc") else None
    d["loss_curve"] = json.loads(d.pop("loss_curve_json") or "[]")
    for key in ("closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss", "learning_rate",
                "triplet_margin", "classifier_confidence_threshold", "ood_distance_threshold",
                "training_duration_seconds", "noise_pct", "rpl_loss_weight",
                "freq_min_mhz", "freq_max_mhz", "pw_min_us", "pw_max_us",
                "pri_min_us", "pri_max_us",
                "rise_min_ns", "rise_max_ns", "bw_min_mhz", "bw_max_mhz", "fall_min_ns", "fall_max_ns",
                "missed_pct", "spurious_pct", "duty_min_pct", "duty_max_pct", "test_pct", "cv_accuracy_mean", "cv_accuracy_std"):
        if d.get(key) is not None:
            d[key] = float(d[key])
    for key in ("num_classes", "num_holdout_classes",
                "samples_per_class", "epochs", "dataset_seed", "train_seed", "window_k", "pulses_per_track", "embedding_dim",
                "parent_run_id", "cv_folds"):
        if d.get(key) is not None:
            d[key] = int(float(d[key]))
    d["include_pgri"] = str(d.get("include_pgri")).strip().lower() in ("1", "true", "yes")
    d["include_test"] = str(d.get("include_test")).strip().lower() in ("1", "true", "yes")
    d["balanced_patterns"] = str(d.get("balanced_patterns")).strip().lower() in ("1", "true", "yes")
    if d.get("rpl_adversarial") is not None:
        # sqlite3 stores a Python bool as integer 1/0 (bool is an int subclass), and the TEXT
        # column affinity here can coerce that to the string "1"/"0" on read — so both integer
        # and string truthy forms need handling, not just a literal "true"/"false" string.
        d["rpl_adversarial"] = str(d["rpl_adversarial"]).strip().lower() in ("1", "true", "yes")
    return d


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def set_saved_name(run_id: int, name: Optional[str]) -> None:
    """Saves (names) or unsaves (name=None) a run's model. Names are unique, case-insensitively."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if name is not None:
            clash = conn.execute(
                "SELECT id FROM training_runs WHERE lower(saved_name) = lower(?) AND id != ?", (name, run_id)
            ).fetchone()
            if clash:
                raise ValueError(f"A saved model named '{name}' already exists (run #{clash[0]}).")
        saved_at = datetime.now(timezone.utc).isoformat() if name is not None else None
        conn.execute("UPDATE training_runs SET saved_name = ?, saved_at_utc = ? WHERE id = ?", (name, saved_at, run_id))
        conn.commit()
    finally:
        conn.close()


def delete_run(run_id: int) -> int:
    """Removes a training run's history row and the Testing-page history of its model.
    Returns the number of test runs removed. Test runs recorded before run_id was tracked
    can't be linked to a model and are left alone."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM training_runs WHERE id = ?", (run_id,))
        # run_id is a TEXT column; compare numerically so "7" and "7.0" both match.
        tests = conn.execute("DELETE FROM test_runs WHERE run_id IS NOT NULL AND CAST(run_id AS INTEGER) = ?", (run_id,))
        conn.commit()
        return tests.rowcount
    finally:
        conn.close()


def get_saved_runs() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM training_runs WHERE saved_name IS NOT NULL ORDER BY id DESC").fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_latest_run() -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def referenced_dataset_names() -> List[str]:
    """Every dataset name a training or test run refers to: such a name must never be given to other data."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT dataset_name FROM training_runs WHERE dataset_name IS NOT NULL "
                            "UNION SELECT dataset_name FROM test_runs WHERE dataset_name IS NOT NULL").fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:  # tables not created yet
        return []
    finally:
        conn.close()


def get_all_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _test_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
    for key in ("closed_set_accuracy", "open_set_auroc", "track_accuracy"):
        if d.get(key) is not None:
            d[key] = float(d[key])
    for key in ("num_known_classes", "num_holdout_classes",
                "num_known_pulses", "num_openset_pulses", "total_pulses", "dataset_seed", "run_id"):
        if d.get(key) is not None:
            d[key] = int(float(d[key]))
    return d


def insert_test_run(record: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        columns = ["timestamp_utc"] + _TEST_COLUMNS
        values = [timestamp_utc] + [record.get(c) for c in _TEST_COLUMNS]
        placeholders = ", ".join("?" for _ in columns)
        # Column names come from the fixed column list (never from input); every value is bound with "?".
        cursor = conn.execute(
            f"INSERT INTO test_runs ({', '.join(columns)}) VALUES ({placeholders})", values  # nosec B608
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_test_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM test_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_test_row_to_dict(r) for r in rows]
    finally:
        conn.close()

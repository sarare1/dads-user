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


# Storage type of each column (anything not listed is TEXT). Booleans are stored as INTEGER 0/1.
_REAL = {"closed_set_accuracy", "open_set_auroc", "open_set_auroc_rpl", "final_loss", "learning_rate",
         "triplet_margin", "classifier_confidence_threshold", "ood_distance_threshold",
         "training_duration_seconds", "noise_pct", "rpl_loss_weight",
         "freq_min_mhz", "freq_max_mhz", "pw_min_us", "pw_max_us", "pri_min_us", "pri_max_us",
         "rise_min_ns", "rise_max_ns", "bw_min_mhz", "bw_max_mhz", "fall_min_ns", "fall_max_ns",
         "missed_pct", "spurious_pct", "duty_min_pct", "duty_max_pct", "test_pct",
         "cv_accuracy_mean", "cv_accuracy_std", "track_accuracy"}
_INTEGER = {"num_classes", "num_holdout_classes", "samples_per_class", "epochs", "dataset_seed", "train_seed",
            "window_k", "pulses_per_track", "embedding_dim", "parent_run_id", "cv_folds",
            "num_known_classes", "num_known_pulses", "num_openset_pulses", "total_pulses", "run_id"}
_BOOLEAN = {"rpl_adversarial", "include_pgri", "include_test", "balanced_patterns"}
_TABLES = {"training_runs": _COLUMNS, "test_runs": _TEST_COLUMNS}
TYPED_BACKUP = "training_history.pre-typed.db"  # copy of the history kept by the one-time type migration


def _sql_type(col: str) -> str:
    return "REAL" if col in _REAL else "INTEGER" if col in _INTEGER | _BOOLEAN else "TEXT"


def _create_sql(table: str, name: Optional[str] = None) -> str:
    cols = ", ".join(f"{c} {_sql_type(c)}" for c in _TABLES[table])
    return (f"CREATE TABLE IF NOT EXISTS {name or table} "
            f"(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT NOT NULL, {cols})")


def _convert_expr(col: str) -> str:
    """SQL turning a value stored as text by DADS <= 1.2 ("0.93", "7.0", "1", "True") into its typed form."""
    blank_is_null = f"NULLIF(trim({col}), '')"
    if col in _REAL:
        return f"CAST({blank_is_null} AS REAL)"
    if col in _INTEGER:
        return f"CAST(CAST({blank_is_null} AS REAL) AS INTEGER)"
    if col in _BOOLEAN:
        return (f"CASE WHEN {blank_is_null} IS NULL THEN NULL "
                f"WHEN lower(trim({col})) IN ('1', '1.0', 'true', 'yes') THEN 1 ELSE 0 END")
    return col


def _needs_typing(conn: sqlite3.Connection, table: str) -> bool:
    declared = {row[1]: (row[2] or "").upper() for row in conn.execute(f"PRAGMA table_info({table})")}
    return any(declared.get(c) != _sql_type(c) for c in _TABLES[table] if c in declared)


def _migrate_to_typed(conn: sqlite3.Connection) -> None:
    """Up to DADS 1.2 every column was TEXT, so numbers came back as strings and sorted/compared as text.
    Rebuilds the tables with REAL/INTEGER columns and converts the stored values, once, in one transaction.
    Row ids are kept and the id counter is never lowered, so a deleted run's id is never handed out again
    (model files and test details are named by id). The old file is copied to TYPED_BACKUP first."""
    tables = [t for t in _TABLES if _needs_typing(conn, t)]
    if not tables:
        return
    backup = os.path.join(os.path.dirname(DB_PATH), TYPED_BACKUP)
    if not os.path.exists(backup):
        dst = sqlite3.connect(backup)
        try:
            conn.backup(dst)
        finally:
            dst.close()
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in tables:
            # Table and column names come from the fixed lists above, never from input.
            seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = ?", (table,)).fetchone()
            cols = _TABLES[table]
            conn.execute(f"DROP TABLE IF EXISTS {table}_typed")
            conn.execute(_create_sql(table, f"{table}_typed"))
            conn.execute(f"INSERT INTO {table}_typed (id, timestamp_utc, {', '.join(cols)}) "  # nosec B608
                         f"SELECT id, timestamp_utc, {', '.join(_convert_expr(c) for c in cols)} FROM {table}")
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {table}_typed RENAME TO {table}")
            if seq is not None:
                conn.execute("UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name = ?", (seq[0], table))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.isolation_level = ""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        for table in _TABLES:
            conn.execute(_create_sql(table))
        conn.commit()

        # DB files created before a column existed get it (a no-op on a freshly created table).
        for table, cols in _TABLES.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for col in cols:
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {_sql_type(col)}")
            # Drop columns of settings that no longer exist.
            for col in _DROPPED_COLUMNS & existing:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
            conn.commit()

        _migrate_to_typed(conn)
    finally:
        conn.close()


def _to_ist_string(timestamp_utc: str) -> str:
    dt = datetime.fromisoformat(timestamp_utc).replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def insert_run(record: Dict[str, Any]) -> int:
    """Inserts one completed training run. `record` must contain every key in _COLUMNS; values are stored
    in the column's type (REAL / INTEGER, booleans as 0/1), loss_curve is JSON-encoded."""
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


def _typed(d: Dict[str, Any]) -> Dict[str, Any]:
    """Python types for a row. Columns are typed since DADS 1.3; the conversion still accepts the text
    values of an older file in case its migration was skipped."""
    for key, value in d.items():
        if key in _BOOLEAN:
            d[key] = None if value is None else str(value).strip().lower() in ("1", "1.0", "true", "yes")
        elif value is None:
            continue
        elif key in _REAL or key in _INTEGER:
            if isinstance(value, str) and not value.strip():
                d[key] = None
            else:
                d[key] = float(value) if key in _REAL else int(float(value))
    return d


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = _typed(dict(row))
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
    d["saved_at_ist"] = _to_ist_string(d["saved_at_utc"]) if d.get("saved_at_utc") else None
    d["loss_curve"] = json.loads(d.pop("loss_curve_json") or "[]")
    for key in ("include_pgri", "include_test", "balanced_patterns"):  # NULL = before the option existed = off
        d[key] = bool(d.get(key))
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
        tests = conn.execute("DELETE FROM test_runs WHERE run_id = ?", (run_id,))
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
    d = _typed(dict(row))
    d["timestamp_ist"] = _to_ist_string(d["timestamp_utc"])
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

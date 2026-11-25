"""History database stores numbers as numbers (with a one-time migration of the old all-TEXT files), and the
upload sanitizer cleans markup from text columns without rescanning frames it already checked.
Run: python tests/test_storage_types.py  (or pytest)"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.sequence_features import sanitize_frame
from src.storage import training_store as ts


def _old_style_db(path):
    """A history file as DADS <= 1.2 wrote it: every column TEXT, numbers and booleans as strings."""
    conn = sqlite3.connect(path)
    for table, cols in ts._TABLES.items():
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT NOT NULL, "
                     + ", ".join(f"{c} TEXT" for c in cols) + ")")
    conn.execute("INSERT INTO training_runs (timestamp_utc, closed_set_accuracy, epochs, include_pgri, rpl_adversarial, "
                 "dataset_seed, saved_name, loss_curve_json, learning_rate) "
                 "VALUES ('2026-10-01T10:00:00+00:00', '0.93', '7.0', 'True', '0', '442635', 'my model', '[1.5, 0.5]', '')")
    conn.execute("INSERT INTO training_runs (timestamp_utc, epochs) VALUES ('2026-10-02T10:00:00+00:00', '3')")
    conn.execute("DELETE FROM training_runs WHERE id = 2")  # the id counter stays at 2
    conn.execute("INSERT INTO test_runs (timestamp_utc, run_id, closed_set_accuracy, total_pulses) "
                 "VALUES ('2026-10-01T11:00:00+00:00', '1', '0.9', '1200')")
    conn.commit()
    conn.close()


def test_old_text_history_is_migrated_once_and_keeps_its_values(tmp_path, monkeypatch):
    db = str(tmp_path / "training_history.db")
    monkeypatch.setattr(ts, "DB_PATH", db)
    _old_style_db(db)
    ts.init_db()

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT typeof(closed_set_accuracy), typeof(epochs), typeof(include_pgri), typeof(learning_rate) "
                        "FROM training_runs WHERE id = 1").fetchone() == ("real", "integer", "integer", "null")
    assert conn.execute("SELECT typeof(run_id), typeof(total_pulses) FROM test_runs").fetchone() == ("integer", "integer")
    conn.close()
    assert os.path.exists(tmp_path / ts.TYPED_BACKUP)

    r = ts.get_run(1)
    assert r["closed_set_accuracy"] == 0.93 and r["epochs"] == 7 and r["dataset_seed"] == 442635
    assert r["include_pgri"] is True and r["rpl_adversarial"] is False and r["include_test"] is False
    assert r["learning_rate"] is None and r["saved_name"] == "my model" and r["loss_curve"] == [1.5, 0.5]
    t = ts.get_all_test_runs()[0]
    assert t["run_id"] == 1 and t["total_pulses"] == 1200

    rid = ts.insert_run({c: None for c in ts._COLUMNS} | {"epochs": 4, "loss_curve": []})
    assert rid == 3, "a deleted run's id was handed out again"
    ts.init_db()  # later starts: nothing to migrate
    assert ts.get_run(3)["epochs"] == 4
    assert ts.delete_run(1) == 1  # the test of run 1 goes with it (integer match)


def test_sanitizer_cleans_markup_and_skips_checked_frames():
    df = pd.DataFrame({"cl<a>ss": ["fam<script>", None, "ok", "it's"], "v": [1.0, 2.0, 3.0, 4.0]})
    clean = sanitize_frame(df)
    assert list(clean.columns) == ["cl_a_ss", "v"]
    assert clean["cl_a_ss"].tolist()[0] == "famscript" and clean["cl_a_ss"].tolist()[3] == "its"
    assert pd.isna(clean["cl_a_ss"].tolist()[1]) and clean["v"].tolist() == [1.0, 2.0, 3.0, 4.0]

    same = clean.iloc[:2]  # a slice (one track) of a checked frame is not scanned again
    assert sanitize_frame(same) is same
    grown = clean.copy()
    grown["note"] = ["<b>", "x", "y", "z"]  # a new column: checked again
    assert sanitize_frame(grown)["note"].tolist()[0] == "b"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

"""API hardening: slow endpoints don't run on the event loop, request sizes are capped, and dataset
names + content hashes keep lineage (a name always means the same bytes).
Uses its own temporary data folder. Run: python tests/test_api_hardening.py  (or pytest)"""
import asyncio
import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DADS_DATA_DIR", tempfile.mkdtemp(prefix="dads_test_data_"))  # before importing the server
os.environ.setdefault("DADS_ADMIN_USER", "admin")
os.environ.setdefault("DADS_ADMIN_PASSWORD", "test-only-password-123")

import pandas as pd
from fastapi.testclient import TestClient

from src.api import server
from src.api import common, core
from src.data.sequence_generator import build_sequence_dataset_rows
from src.storage.dataset_store import DatasetStore

assert os.path.abspath(core.DATA_DIR) == os.path.abspath(os.environ["DADS_DATA_DIR"]), "tests must not touch data/"


def _client() -> TestClient:
    c = TestClient(server.app)
    r = c.post("/api/login", json={"username": os.environ["DADS_ADMIN_USER"], "password": os.environ["DADS_ADMIN_PASSWORD"]})
    assert r.status_code == 200
    return c


def _csv_bytes(seed: int = 5) -> bytes:
    rows = build_sequence_dataset_rows(3, [], 4, 24, seed, None, 0.0, 0.0, 0.0, None, False, True)
    return pd.DataFrame(rows).to_csv(index=False).encode()



def _all_routes(routes):
    """Every route, also those of the included route modules (FastAPI keeps them in their own router)."""
    for r in routes:
        inner = getattr(r, "original_router", None)
        yield from (_all_routes(inner.routes) if inner is not None else [r])

def test_heavy_endpoints_do_not_run_on_the_event_loop():
    # A plain `def` endpoint runs in FastAPI's worker threads; `async def` runs on the shared event loop,
    # where one slow CSV parse or model load would stall every user and WebSocket.
    must_be_threaded = {
        "inspect_dataset", "dataset_analytics", "download_sequence_dataset", "sequence_options", "sequence_dataset_info",
        "list_datasets", "delete_dataset", "analytics_sources", "train_history", "test_history", "download_weights",
        "download_model", "generate_sequence_dataset", "expand_sequence_dataset", "test_sequence_model", "sequence_infer",
        "sequence_example_track", "generate_new_family_dataset", "activate_sequence_model", "sequence_models",
    }
    endpoints = {r.endpoint.__name__: r.endpoint for r in _all_routes(server.app.routes) if getattr(r, "endpoint", None)}
    missing = must_be_threaded - endpoints.keys()
    assert not missing, missing
    on_loop = sorted(n for n in must_be_threaded if asyncio.iscoroutinefunction(endpoints[n]))
    assert not on_loop, f"blocking endpoints still on the event loop: {on_loop}"


def test_result_sizes_are_capped():
    with _client() as c:
        assert c.get("/api/train/history?limit=10").status_code == 200
        for url in ("/api/train/history?limit=-1", "/api/train/history?limit=100000", "/api/test/history?limit=-1",
                    "/api/test/history?limit=0", "/api/sequence/incremental/experiments?limit=-1",
                    "/api/analytics?max_scatter_rows=10000000", "/api/analytics?max_boxplot_classes=-1"):
            assert c.get(url).status_code == 422, url


def test_store_hashes_and_never_reuses_a_name():
    d = tempfile.mkdtemp(prefix="dads_store_")
    store = DatasetStore(d, reserved_names=lambda: ["held_by_a_run"])
    df = pd.read_csv(io.BytesIO(_csv_bytes()))
    name = store.save(df, {"source": "uploaded"}, "tracks")
    assert name == "tracks"
    with open(store.csv_path(name), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert store.meta(name)["sha256"] == digest == store.content_hash(name)
    assert store.verify(name, digest) and store.verify(name, None)

    store.delete(name)
    assert store.save(df, {"source": "uploaded"}, "tracks") == "tracks_2", "a deleted dataset's name was reused"
    assert store.save(df, {"source": "uploaded"}, "Tracks").lower() not in ("tracks", "tracks_2"), "names differing only in case collide on Windows"
    assert store.save(df, {"source": "uploaded"}, "held_by_a_run") == "held_by_a_run_2", "a name a run refers to was reused"

    with open(store.csv_path("tracks_2"), "a") as f:  # the stored file is changed behind the app's back
        f.write("\n")
    assert not store.verify("tracks_2", store.meta("tracks_2")["sha256"])


def test_api_upload_records_hash_and_detects_a_changed_dataset():
    data = _csv_bytes(9)
    with _client() as c:
        r = c.post("/api/sequence/dataset/upload", files={"file": ("lineage.csv", data, "text/csv")}, data={"name": "lineage"})
        assert r.status_code == 200, r.text
        name = r.json()["name"]
        listed = {d["name"]: d for d in c.get("/api/datasets").json()["datasets"]}
        assert listed[name]["sha256"] == core.datasets.content_hash(name)
        assert core.datasets.meta(name)["original_sha256"] == hashlib.sha256(data).hexdigest()

        run = {"id": 1, "dataset_name": name, "dataset_sha256": listed[name]["sha256"]}
        common._check_run_dataset(run)  # unchanged: fine
        try:
            common._check_run_dataset(dict(run, dataset_sha256="0" * 64))
            raise AssertionError("a changed dataset was accepted as the run's training data")
        except ValueError:
            pass

        assert c.delete(f"/api/datasets/{name}").status_code == 200
        r = c.post("/api/sequence/dataset/upload", files={"file": ("lineage.csv", data, "text/csv")}, data={"name": name})
        assert r.status_code == 200 and r.json()["name"] != name, "a deleted dataset's name was given to a new upload"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("ALL OK")

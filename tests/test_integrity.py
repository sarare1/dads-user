"""Model-package integrity (hash manifest + Ed25519 signature), error text kept from the browser, the health
endpoint and the retired "dot" metric. Run: python tests/test_integrity.py  (or pytest)"""
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DADS_DATA_DIR", tempfile.mkdtemp(prefix="dads_test_data_"))
os.environ.setdefault("DADS_ADMIN_USER", "admin")
os.environ.setdefault("DADS_ADMIN_PASSWORD", "test-only-password-123")

import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.api import core
from src.api.routers import pages
from src.models import package_signing as ps
from src.models.deploy import sequence_inference
from src.models.registry import ModelRegistry
from src.training.trainer import build_distance_fn

CARD = {"window": 8, "features": ["a"], "input_columns": ["x"], "backbone_type": "gru", "name": "t"}


def _package(tmp_path, signer):
    reg = ModelRegistry(str(tmp_path / "models"))
    d = reg.run_dir(7)
    os.makedirs(d)
    for f, body in (("model.pt", b"weights"), ("prototypes.json", b"{}")):
        open(os.path.join(d, f), "wb").write(body)
    verifier = open(ps.__file__, "rb").read()
    return reg.build_sequence_package(7, CARD, {"model.torchscript.pt": b"ts", "verify_package.py": verifier},
                                      provenance={"dads_version": "test"}, signer=signer)


def _rezip(data: bytes, change: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, change.get(n, zin.read(n)))
        for n, body in change.items():
            if n not in zin.namelist():
                zout.writestr(n, body)
    return out.getvalue()


def _check(tmp_path, data, key=None, name="p.zip"):
    p = tmp_path / name
    p.write_bytes(data)
    return ps.verify_package_path(str(p), key)


@pytest.mark.skipif(not ps.HAVE_CRYPTO, reason="needs cryptography")
def test_signed_package_verifies_and_detects_tampering(tmp_path):
    signer = ps.PackageSigner(str(tmp_path / "data"))
    key = signer.public_key()["public_key_hex"]
    data = _package(tmp_path, signer)
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(data)).read(ps.MANIFEST))
    assert manifest["provenance"] == {"dads_version": "test", "run_id": 7}
    assert set(manifest["files"]) >= {"model.pt", "model.torchscript.pt", "model_card.json", "README.txt"}
    assert _check(tmp_path, data, key)[0]

    assert not _check(tmp_path, _rezip(data, {"model.torchscript.pt": b"evil"}), key)[0]   # changed file
    assert not _check(tmp_path, _rezip(data, {"extra.py": b"import os"}), key)[0]          # added file
    m2 = json.dumps(dict(manifest, provenance={"dads_version": "forged"})).encode()
    assert not _check(tmp_path, _rezip(data, {ps.MANIFEST: m2}), key)[0]                   # manifest re-written
    other = ps.PackageSigner(str(tmp_path / "other")).public_key()["public_key_hex"]
    ok, msg = _check(tmp_path, data, other)
    assert not ok and "different key" in msg                                                # not our server
    assert oct(os.stat(signer.private_path).st_mode & 0o777) == "0o600" or os.name == "nt"


def test_unsigned_package_still_hash_checked(tmp_path):
    data = _package(tmp_path, signer=None)
    assert ps.SIGNATURE not in zipfile.ZipFile(io.BytesIO(data)).namelist()
    ok, msg = _check(tmp_path, data)
    assert ok and "Unsigned" in msg
    assert not _check(tmp_path, _rezip(data, {"model.pt": b"x"}))[0]


def test_standalone_inference_refuses_a_changed_package(tmp_path):
    data = _package(tmp_path, signer=None)
    d = tmp_path / "pkg"
    zipfile.ZipFile(io.BytesIO(data)).extractall(d)
    (d / "results.csv").write_text("user output is fine")   # files the user adds don't fail the check
    assert "match" in sequence_inference.check_package(str(d))
    (d / "model.torchscript.pt").write_bytes(b"evil")
    with pytest.raises(RuntimeError, match="Hash mismatch"):
        sequence_inference.check_package(str(d))
    shutil.rmtree(d)
    os.makedirs(d)
    assert "WARNING" in sequence_inference.check_package(str(d))  # pre-1.2.0 package: warn, don't block


def test_error_text_hides_paths_and_internal_errors():
    msg = core._public_error(ValueError(r"cannot open C:\Users\someone\DADS\data\x.csv or /root/dads/data/y.pt"))
    assert "someone" not in msg and "/root" not in msg and "x.csv" in msg and "y.pt" in msg
    assert core._public_error(ValueError("ratio 3/4")) == "ratio 3/4"
    out = core._explain_error(RuntimeError("secret /root/dads internals"))
    assert "secret" not in out and "reference" in out


def test_health_is_public_and_minimal():
    r = TestClient(server.app).get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_package_key_endpoint_needs_login():
    c = TestClient(server.app)
    assert c.get("/api/package-key").status_code == 401
    c.post("/api/login", json={"username": os.environ["DADS_ADMIN_USER"], "password": os.environ["DADS_ADMIN_PASSWORD"]})
    j = c.get("/api/package-key").json()
    assert j["signing"] is ps.HAVE_CRYPTO and (not j["signing"] or len(j["public_key_hex"]) == 64)


def test_dot_metric_is_retired_but_old_runs_load():
    assert "dot" not in pages.DISTANCE_METRICS
    import torch
    a, b = torch.nn.functional.normalize(torch.randn(3, 4), dim=1), torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    assert torch.allclose(build_distance_fn("dot")(a, b), build_distance_fn("cosine")(a, b))
    with pytest.raises(ValueError):
        build_distance_fn("manhattan")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

"""Per-run model artifacts, the active-model pointer, and the downloadable sequence-model package.

Every successful training run stores its trained model under data/models/runs/<run_id>/:
  model.pt          checkpoint bundle (weights + input features + window + scaling + architecture)
  model.onnx        ONNX export, if one was produced (the downloadable package builds its own)
  prototypes.json   open-set class prototypes (sub-centroids) + their metadata
  report.json       detailed evaluation + explainability (src/models/explain.py, sequence_eval.py)

Saving a run under a name (training_store.set_saved_name) makes it permanent; unsaved runs
beyond the newest MAX_UNSAVED_RUNS are pruned to bound disk use. The active run — the model
serving Inference/Live Simulation — is recorded in active_sequence_run.json and never pruned.
"""
import io
import json
import os
import shutil
import zipfile
from typing import Any, Dict, Iterable, Optional

MAX_UNSAVED_RUNS = 20


class ModelRegistry:
    def __init__(self, models_dir: str):
        self.runs_dir = os.path.join(models_dir, "runs")
        self.models_dir = models_dir
        os.makedirs(self.runs_dir, exist_ok=True)

    def run_dir(self, run_id: int) -> str:
        return os.path.join(self.runs_dir, str(int(run_id)))

    def has_artifacts(self, run_id: int) -> bool:
        return os.path.exists(os.path.join(self.run_dir(run_id), "model.pt"))

    def save_artifacts(self, run_id: int, checkpoint_path: str, onnx_path: Optional[str],
                       centroids: Dict[int, list], metadata: Dict[int, dict], report: Dict[str, Any]) -> None:
        d = self.run_dir(run_id)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(checkpoint_path, os.path.join(d, "model.pt"))
        if onnx_path and os.path.exists(onnx_path):
            shutil.copyfile(onnx_path, os.path.join(d, "model.onnx"))
        with open(os.path.join(d, "prototypes.json"), "w") as f:
            json.dump({"centroids": {str(k): v for k, v in centroids.items()},
                       "metadata": {str(k): v for k, v in metadata.items()}}, f)
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump(report, f)

    def load_report(self, run_id: int) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.run_dir(run_id), "report.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def load_prototypes(self, run_id: int):
        with open(os.path.join(self.run_dir(run_id), "prototypes.json")) as f:
            data = json.load(f)
        return ({int(k): v for k, v in data["centroids"].items()},
                {int(k): v for k, v in data["metadata"].items()})

    def checkpoint_path(self, run_id: int) -> str:
        return os.path.join(self.run_dir(run_id), "model.pt")

    def onnx_path(self, run_id: int) -> Optional[str]:
        path = os.path.join(self.run_dir(run_id), "model.onnx")
        return path if os.path.exists(path) else None

    # The sequence model in use by Inference / Live Simulation.
    def get_active_sequence(self) -> Optional[int]:
        path = os.path.join(self.models_dir, "active_sequence_run.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return int(json.load(f)["run_id"])
        except (ValueError, KeyError, TypeError):
            return None

    def set_active_sequence(self, run_id: Optional[int]) -> None:
        path = os.path.join(self.models_dir, "active_sequence_run.json")
        if run_id is None:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w") as f:
            json.dump({"run_id": int(run_id)}, f)

    def build_sequence_package(self, run_id: int, model_card: Dict[str, Any], extra_files: Dict[str, bytes],
                               provenance: Optional[Dict[str, Any]] = None, signer=None) -> bytes:
        """The deployable zip. MANIFEST.json lists the SHA-256 of every other file (and is signed when `signer`
        can sign), so a receiver can prove the package is unchanged and came from this server."""
        from .package_signing import MANIFEST, SIGNATURE, build_manifest
        d = self.run_dir(run_id)
        files: Dict[str, bytes] = {}
        for fname in ("model.pt", "prototypes.json", "report.json"):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    files[fname] = f.read()
        files["model_card.json"] = json.dumps(model_card, indent=2).encode()
        files.update(extra_files)
        files["README.txt"] = _SEQ_README.format(
            name=model_card.get("name") or f"run_{run_id}", run_id=run_id, k=model_card["window"],
            nf=len(model_card["features"]), features=", ".join(model_card["features"]),
            columns=", ".join(model_card["input_columns"]), backbone=model_card["backbone_type"]).encode()
        manifest, sig = build_manifest(files, dict(provenance or {}, run_id=run_id), signer)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, content in files.items():
                z.writestr(fname, content)
            z.writestr(MANIFEST, manifest)
            if sig:
                z.writestr(SIGNATURE, sig)
        return buf.getvalue()

    def delete_artifacts(self, run_id: int) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)

    def prune(self,run_ids_newest_first: Iterable[int], saved_ids: Iterable[int]) -> list:
        """Deletes artifacts of unsaved runs beyond the newest MAX_UNSAVED_RUNS (never the
        active or a saved run). Returns the pruned run ids."""
        keep = set(saved_ids)
        active = self.get_active_sequence()
        if active is not None:
            keep.add(active)
        pruned, unsaved_seen = [], 0
        for rid in run_ids_newest_first:
            if rid in keep or not self.has_artifacts(rid):
                continue
            unsaved_seen += 1
            if unsaved_seen > MAX_UNSAVED_RUNS:
                shutil.rmtree(self.run_dir(rid), ignore_errors=True)
                pruned.append(rid)
        return pruned


_SEQ_README = """DADS pulse-sequence model package: {name} (training run #{run_id}, {backbone})

Classifies a de-interleaved TRACK (ordered pulses of one emitter) from windows of {k} consecutive
pulses. No DADS code needed.

QUICK START
  pip install numpy pandas onnxruntime        # or onnxruntime-gpu on a CUDA machine
  python sequence_inference.py --csv tracks.csv                    # one verdict per track
  python sequence_inference.py --csv tracks.csv --per-window --out results.csv

  Input CSV, one row per pulse: instance, pulse_idx (order within the track; optional if rows are
  already in order) and the measurement columns: {columns}
  The first pulse of a track may have a blank PRI. No time of arrival is needed.

FILES
  sequence_inference.py  Standalone pipeline: per-track features (Δ values, duty cycle, ...),
                         scaling, windows, network, nearest-prototype distance, decision rule and
                         the per-track majority vote. Same results as the DADS server.
                         Backends: --backend onnxruntime | torchscript | tensorrt
  model.onnx             The network (ONNX opset 17). Input "pulses": float32 [batch, pulses, {nf}],
                         ALREADY SCALED; batch and pulse count are dynamic. Outputs "probs", "embeddings".
  model.torchscript.pt   The same network for PyTorch: torch.jit.load(...)
  model.pt               DADS checkpoint (needs the DADS source)
  prototypes.json        Class prototypes (embedding sub-centroids per class)
  model_card.json        Features and how to compute them, scaling, window, thresholds, metric
  report.json            Evaluation and explainability from training
  MANIFEST.json          SHA-256 of every file above, plus where the model came from (run, release, dataset)
  MANIFEST.json.sig      Ed25519 signature of the manifest by the DADS server (if signing is enabled)
  verify_package.py      Checks the files against the manifest, and the signature against a trusted key

CHECK THE PACKAGE FIRST
  python verify_package.py --trusted-key <the DADS server's public key: GET /api/package-key>
  sequence_inference.py checks the file hashes itself before loading the model and stops if any differ.

IMPORTANT: model.onnx / a TensorRT engine contain only the network. Features must be computed
per track and scaled exactly as sequence_inference.py does (see model_card.json). Feeding raw
values gives wrong results without any error.

CUDA / TENSORRT
  pip install onnxruntime-gpu   then   --backend onnxruntime   (TensorRT / CUDA providers used when present)
  PyTorch on CUDA:                     --backend torchscript
  Native TensorRT engine (dynamic batch and pulse count; TensorRT >= 8.6 for Transformer/Hybrid):
    trtexec --onnx=model.onnx --saveEngine=model.plan \\
            --minShapes=pulses:1x1x{nf} --optShapes=pulses:64x{k}x{nf} --maxShapes=pulses:1024x512x{nf} [--fp16]
    python sequence_inference.py --csv tracks.csv --backend tensorrt --engine model.plan
  GRU models: the ONNX graph uses a GRU op; check that your TensorRT version supports it.

Per-pulse features in order: {features}
"""

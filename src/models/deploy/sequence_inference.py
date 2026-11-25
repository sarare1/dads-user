"""Standalone inference for a DADS pulse-sequence model package — no DADS source code needed.

Input: de-interleaved tracks (ordered pulses of one emitter each), one CSV row per pulse with
instance, optional pulse_idx, and the measurement columns listed in model_card.json.
Output: per track, the class (majority vote over its windows) and the decision rule's verdict;
optionally every window's decision. Results match the DADS server's sequence Inference.

Backends: onnxruntime (CPU / CUDA / TensorRT providers), torchscript (PyTorch, CUDA if present),
tensorrt (an engine built from model.onnx with trtexec).

  python sequence_inference.py --csv tracks.csv
  python sequence_inference.py --csv tracks.csv --per-window --out results.csv --backend torchscript
  python sequence_inference.py --csv tracks.csv --trusted-key <DADS server public key, hex>

Before loading anything, the package's files are checked against MANIFEST.json (verify_package.py); a changed
file stops the run. With --trusted-key the manifest's signature must also be valid for that key.
"""
import argparse
import csv
import json
import os
import sys
import zlib
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

VERDICT_ECM = {
    "VERIFIED_KNOWN_TARGET": ("GREEN", "MONITOR_ONLY"),
    "UNKNOWN_THREAT_ANOMALY": ("ORANGE", "AUTOMATED_JAMMER_ENGAGED"),
    "OOD_NOISE_REJECTED": ("RED", "ELINT_LOGGING"),
}


def check_package(package_dir: str, trusted_key: Optional[str] = None) -> str:
    """Checks the package's files against its manifest (and signature, with trusted_key) BEFORE any model file is
    loaded: model.torchscript.pt runs code when loaded. Raises RuntimeError if the package fails the check."""
    verifier = os.path.join(package_dir, "verify_package.py")
    if not os.path.exists(verifier) or not os.path.exists(os.path.join(package_dir, "MANIFEST.json")):
        if trusted_key:
            raise RuntimeError("This package has no manifest, so its signature can't be checked.")
        return "WARNING: package made before DADS v1.2.0 - no manifest, files not checked."
    import importlib.util
    spec = importlib.util.spec_from_file_location("dads_verify_package", verifier)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok, msg = mod.verify_package_path(package_dir, trusted_key, allow_extra=True)
    if not ok:
        raise RuntimeError(f"Package check failed: {msg}")
    return msg


class DADSSequenceModel:
    def __init__(self, package_dir: str = ".", backend: str = "onnxruntime", engine_path: Optional[str] = None,
                 trusted_key: Optional[str] = None):
        self.dir = package_dir
        self.integrity = check_package(package_dir, trusted_key)
        with open(os.path.join(package_dir, "model_card.json")) as f:
            self.card = json.load(f)
        with open(os.path.join(package_dir, "prototypes.json")) as f:
            protos = json.load(f)
        self.features: List[dict] = self.card["feature_definitions"]
        self.zero_is_missing = set(self.card["zero_is_missing_columns"])
        self.mean = np.asarray(self.card["scaling"]["mean"], dtype=np.float64)
        self.std = np.asarray(self.card["scaling"]["std"], dtype=np.float64)
        dls = self.card["scaling"].get("delta_log_scale")  # Δ features: signed-log step before the z-score
        self.delta_scale = np.asarray(dls, dtype=np.float64) if dls is not None else None
        self.k = int(self.card["window"])
        self.whole = self.card.get("window_mode") == "track"  # track mode: ONE decision on the track's first ≤ k pulses
        self.metric = self.card["distance_metric"]
        self.tau_c = float(self.card["decision_rule"]["classifier_confidence_threshold"])
        self.tau_d = float(self.card["decision_rule"]["ood_distance_threshold"])
        by_len = self.card["decision_rule"].get("ood_distance_threshold_by_length") or {}
        self.tau_d_by_length = {int(L): float(v) for L, v in by_len.items()}
        vecs, labels = [], []
        for cid, subs in protos["centroids"].items():
            for v in (subs if subs and isinstance(subs[0], list) else [subs]):
                vecs.append(v); labels.append(int(cid))
        self.proto = np.asarray(vecs, dtype=np.float64)
        self.class_labels = self.card.get("class_labels") or []
        self.proto_labels = labels
        self.backend = backend
        self._run = {"onnxruntime": self._init_onnxruntime, "torchscript": self._init_torchscript,
                     "tensorrt": self._init_tensorrt}[backend](engine_path)

    # ------------------------------------------------------------------ features (per track)
    def _categorical(self, rows: List[Dict], col: str, kind: str) -> np.ndarray:
        """Text column -> the model's number: `modulation` = position in the card's modulation list (unknown
        text -> after the last), `hash` = crc32(text) % 100003 / 100003. Blank -> the track's most common value."""
        texts = [str(r.get(col) if r.get(col) is not None else "").strip() for r in rows]
        present = [t for t in texts if t]
        fill = Counter(present).most_common(1)[0][0] if present else ""
        mods = self.card.get("modulation_types") or []

        def code(t):
            t = t or fill
            if kind == "modulation":
                return float(mods.index(t)) if t in mods else float(len(mods))
            return (zlib.crc32(t.encode("utf-8")) % 100_003) / 100_003.0
        return np.array([code(t) for t in texts])

    def _column(self, rows: List[Dict], col: str) -> np.ndarray:
        v = np.array([np.nan if (r.get(col) is None or str(r.get(col)).strip() == "") else float(r[col]) for r in rows])
        if col in self.zero_is_missing:
            v[v == 0] = np.nan
        missing = np.isnan(v)
        if missing.all():
            return np.zeros_like(v)
        first = int(np.argmax(~missing))
        v[:first] = v[first]                 # leading gap (the first pulse's PRI) → first measured value
        v[np.isnan(v)] = np.nanmedian(v)     # other gaps → the track's median
        return v

    def track_features(self, rows: List[Dict]) -> np.ndarray:
        """[pulses × features], raw (unscaled) — rows of ONE track in arrival order."""
        cols = {}
        for f in self.features:
            for c in f["columns"]:
                key = (c, f.get("categorical"))
                if key not in cols:
                    if not any(c in r for r in rows):
                        raise ValueError(f"missing column '{c}' needed for feature '{f['name']}'")
                    cols[key] = self._categorical(rows, c, f["categorical"]) if f.get("categorical") else self._column(rows, c)
        out = []
        for f in self.features:
            cat = f.get("categorical")
            a = cols[(f["columns"][0], cat)] * f.get("scale", 1.0)  # the model's unit (e.g. MHz column → Hz)
            if f["op"] == "value":
                out.append(a)
            elif f["op"] == "delta":
                out.append(np.diff(a, prepend=a[0]))
            elif f["op"] == "ratio":
                b = cols[(f["columns"][1], None)]
                out.append(np.where(b > 0, a / np.maximum(b, 1e-12), 0.0))
            elif f["op"] == "product":
                out.append(a * cols[(f["columns"][1], None)])
        return np.stack(out, axis=1)

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Δ features: sign(x)·ln(1 + |x|/c) first (c from model_card.json); then (x − mean) / std."""
        x = np.array(x, dtype=np.float64, copy=True)
        if self.delta_scale is not None:
            for j in np.nonzero(self.delta_scale > 0)[0]:
                x[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]) / self.delta_scale[j])
        return ((x - self.mean) / self.std).astype(np.float32)

    def tau_for(self, n_pulses: int) -> float:
        """τ_d for a window of n_pulses: below the window length, the calibrated τ_d of the largest
        listed length ≤ n (short windows sit farther from the prototypes); τ_d otherwise."""
        if not self.tau_d_by_length or n_pulses >= self.k:
            return self.tau_d
        lengths = [L for L in self.tau_d_by_length if L <= n_pulses] or [min(self.tau_d_by_length)]
        return self.tau_d_by_length[max(lengths)]

    def spans(self, n: int):
        if self.whole:
            return [(0, min(n, self.k))]
        if n <= self.k:
            return [(0, n)]
        s = [(i, i + self.k) for i in range(0, n - self.k + 1, self.k)]
        if s[-1][1] < n:
            s.append((n - self.k, n))
        return s

    # ------------------------------------------------------------------ backends
    def _init_onnxruntime(self, _engine):
        import onnxruntime as ort
        preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(os.path.join(self.dir, "model.onnx"),
                                    providers=[p for p in preferred if p in ort.get_available_providers()])
        self.device = ", ".join(sess.get_providers())
        return lambda x: sess.run(None, {"pulses": x})

    def _init_torchscript(self, _engine):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.jit.load(os.path.join(self.dir, "model.torchscript.pt"), map_location=device).eval()
        self.device = device

        def run(x):
            with torch.no_grad():
                p, e = m(torch.from_numpy(x).to(device))
            return p.cpu().numpy(), e.cpu().numpy()
        return run

    def _init_tensorrt(self, engine_path):
        import tensorrt as trt
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The tensorrt backend needs a CUDA device.")
        engine_path = engine_path or os.path.join(self.dir, "model.plan")
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        outs = [n for n in names if n != inp]
        stream = torch.cuda.Stream()
        self.device = f"TensorRT engine ({os.path.basename(engine_path)})"

        def run(x):
            xb = torch.from_numpy(np.ascontiguousarray(x)).cuda()
            ctx.set_input_shape(inp, tuple(xb.shape))
            ctx.set_tensor_address(inp, xb.data_ptr())
            res = {}
            for n in outs:
                res[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), device="cuda", dtype=torch.float32)
                ctx.set_tensor_address(n, res[n].data_ptr())
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return res["probs"].cpu().numpy(), res["embeddings"].cpu().numpy()
        return run

    # ------------------------------------------------------------------ decisions
    def name(self, class_id: int) -> str:
        return self.class_labels[class_id] if 0 <= class_id < len(self.class_labels) else f"RADAR_FAMILY_{class_id:02d}"

    def _distance(self, e: np.ndarray):
        if self.metric == "euclidean":
            d = np.linalg.norm(e[None, :] - self.proto + 1e-6, axis=1)
        else:  # cosine / dot (embeddings and prototypes are unit length)
            d = 1.0 - (self.proto @ e) / (np.maximum(np.linalg.norm(self.proto, axis=1) * np.linalg.norm(e), 1e-8))
        j = int(np.argmin(d))
        return float(d[j]), self.proto_labels[j]

    def classify_windows(self, windows: List[np.ndarray]) -> List[Dict]:
        results: List[Optional[Dict]] = [None] * len(windows)
        by_len: Dict[int, List[int]] = {}
        for i, w in enumerate(windows):
            by_len.setdefault(len(w), []).append(i)
        for length, idx in by_len.items():
            tau_d = self.tau_for(length)
            x = self.scale(np.stack([windows[i] for i in idx]))
            probs, emb = self._run(x)
            for row, i in enumerate(idx):
                p = np.asarray(probs[row], dtype=np.float64)
                top = int(np.argmax(p)); conf = float(p[top])
                d, near = self._distance(np.asarray(emb[row], dtype=np.float64))
                verdict = ("VERIFIED_KNOWN_TARGET" if conf >= self.tau_c and d <= tau_d
                           else "UNKNOWN_THREAT_ANOMALY" if conf >= self.tau_c else "OOD_NOISE_REJECTED")
                results[i] = {"class_id": top, "class": self.name(top), "confidence": round(conf, 4),
                              "distance": round(d, 5), "nearest_prototype": self.name(near),
                              "verdict": verdict, "ecm_trigger": VERDICT_ECM[verdict][1], "pulses": len(windows[i])}
        return results

    def classify_track(self, rows: List[Dict]) -> Dict:
        uses_pri = any("PRI" in f["columns"] for f in self.features)  # card columns are the names the model was packaged with
        if uses_pri and not any(str(r.get("PRI") or "").strip() not in ("", "nan", "None") for r in rows):
            raise ValueError("a track needs at least 2 pulses: its first pulse has no PRI (no previous pulse)")
        m = self.track_features(rows)
        spans = self.spans(len(m))
        windows = self.classify_windows([m[s:e] for s, e in spans])
        for (s, e), w in zip(spans, windows):
            w["start"], w["end"] = s, e
        # class = most frequent window class, verdict = most frequent window verdict; ties → seen first
        class_id, votes = Counter(w["class_id"] for w in windows).most_common(1)[0]
        verdict = Counter(w["verdict"] for w in windows).most_common(1)[0][0]
        return {"pulses": len(m), "num_windows": len(windows), "class": self.name(class_id),
                "class_votes": f"{votes}/{len(windows)}",
                "confidence": round(float(np.mean([w["confidence"] for w in windows])), 4),
                "distance": round(float(np.median([w["distance"] for w in windows])), 5),
                "verdict": verdict, "ecm_trigger": VERDICT_ECM[verdict][1], "windows": windows}


def read_tracks(path: str) -> Dict[str, List[Dict]]:
    """Rows grouped by instance in file order, each track sorted by pulse_idx when present."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    tracks: Dict[str, List[Dict]] = {}
    for r in rows:
        tracks.setdefault(r.get("instance", r.get("track_id", "track")), []).append(r)
    for t, rs in tracks.items():
        if rs and "pulse_idx" in rs[0] and all(str(r["pulse_idx"]).strip() != "" for r in rs):
            tracks[t] = sorted(rs, key=lambda r: float(r["pulse_idx"]))  # stable: keeps file order on ties
    return tracks


def _main():
    ap = argparse.ArgumentParser(description="DADS pulse-sequence model package inference")
    ap.add_argument("--package", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--csv", required=True, help="tracks: one row per pulse (instance, pulse_idx, measurements)")
    ap.add_argument("--backend", default="onnxruntime", choices=["onnxruntime", "torchscript", "tensorrt"])
    ap.add_argument("--engine", help="TensorRT engine file (tensorrt backend)")
    ap.add_argument("--per-window", action="store_true", help="also output every window's decision")
    ap.add_argument("--out", help="write results as CSV here")
    ap.add_argument("--trusted-key", help="the DADS server's package-signing public key (hex); require a valid signature")
    args = ap.parse_args()
    try:
        model = DADSSequenceModel(args.package, args.backend, args.engine, args.trusted_key)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
    print(model.integrity, file=sys.stderr)
    print(f"Model: {model.card.get('name') or 'run #' + str(model.card['run_id'])} | {model.card['backbone_type']} | "
          f"window {model.k} | backend {args.backend} on {model.device}", file=sys.stderr)
    out_rows = []
    for tid, rows in read_tracks(args.csv).items():
        try:
            r = model.classify_track(rows)
        except ValueError as e:
            print(f"track {tid}: {e}", file=sys.stderr)
            continue
        summary = {k: v for k, v in r.items() if k != "windows"}
        out_rows.append({"instance": tid, "row": "track", **summary})
        if args.per_window:
            out_rows += [{"instance": tid, "row": "window", **w} for w in r["windows"]]
        if not args.out:
            print(json.dumps({"instance": tid, **(r if args.per_window else summary)}))
    if args.out:
        fields = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("instance", "row"), k))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {len(out_rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    _main()

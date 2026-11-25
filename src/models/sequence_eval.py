"""Evaluation and explainability for pulse-sequence models.

Scores are computed against the model's STORED class prototypes (the ones saved at training),
with the shared decision rule, so a test result says exactly what
the deployed model would do. Besides per-window metrics this reports:
  - per-track accuracy: a track's class = majority vote of its windows' predictions;
  - accuracy vs. number of pulses observed: the same model run on windows of 2, 4, 8, … pulses
    (every sequence encoder pools over time, so it accepts any window length).
"""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.sequence_features import (PRI_COLUMN, SEQ_FEATURES, SequenceSpec, causal_window, has_pri, ordered_tracks,
                                         pad_stack, seq_feature_label, track_feature_matrix, windows_of)
from src.models.explain import _rank_auroc

VERDICTS = ("VERIFIED_KNOWN_TARGET", "UNKNOWN_THREAT_ANOMALY", "OOD_NOISE_REJECTED")


def _min_distance(emb: torch.Tensor, protos: List[torch.Tensor], distance_fn) -> np.ndarray:
    with torch.no_grad():
        return torch.stack([distance_fn(emb, p.unsqueeze(0).expand_as(emb)) for p in protos], dim=1).min(dim=1).values.numpy()


def _run(model: nn.Module, x: np.ndarray, batch: int = 512):
    probs, embs = [], []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            p, e = model(torch.from_numpy(x[s:s + batch]))
            probs.append(p); embs.append(e)
    return torch.cat(probs), torch.cat(embs)


def _prep(spec: SequenceSpec, windows: List[np.ndarray]) -> np.ndarray:
    """Raw feature windows -> the model's input array: scaled; windows of different lengths (track
    mode) padded at the end with a mask channel."""
    if len({len(w) for w in windows}) == 1:
        return spec.transform(np.stack(windows))
    return pad_stack([spec.transform(w) for w in windows])


def class_name(class_id: int, class_labels: Optional[List[str]] = None) -> str:
    if class_labels and 0 <= class_id < len(class_labels):
        return str(class_labels[class_id])
    return f"RADAR_FAMILY_{class_id:02d}"


def relabel_by_name(df: pd.DataFrame, class_labels: Optional[List[str]]) -> pd.DataFrame:
    """A labelled track file → the model's class ids, matched by the label NAME (true_label): an
    uploaded training file's class ids were renumbered 0…N−1, so a test file's own ids needn't
    match the model's. Labels the model doesn't know → −1 (unknown). Unchanged without names."""
    if not class_labels or "class" not in df.columns:
        return df
    index = {str(l): i for i, l in enumerate(class_labels)}
    df = df.copy()
    df["true_class_id"] = df["class"].astype(str).map(lambda l: index.get(l, -1)).astype(int)
    return df


def track_vote(classes: List[int], verdicts: List[str]) -> Tuple[int, int, str]:
    """A track's decision from its windows' decisions (in time order): class = most frequent
    window class, verdict = most frequent window verdict; a tie goes to the one seen first.
    Shared by Testing, Inference, Live and the deployment package, so they always agree."""
    class_id, count = Counter(classes).most_common(1)[0]
    verdict = Counter(verdicts).most_common(1)[0][0]
    return int(class_id), int(count), str(verdict)


def _sig(x: float, digits: int = 6) -> float:
    """Round to significant digits — sequence-model distances are tiny (≈ 0.001), so a fixed
    number of decimals could round a threshold to 0."""
    return float(f"{x:.{digits}g}")


def _verdicts(conf: np.ndarray, dist: np.ndarray, tau_c: float, tau_d: float) -> np.ndarray:
    return np.where((conf >= tau_c) & (dist <= tau_d), VERDICTS[0], np.where(conf >= tau_c, VERDICTS[1], VERDICTS[2]))


def calibrate_distance_threshold(model: nn.Module, known_windows: np.ndarray, centroids: Dict[int, List[List[float]]],
                                 distance_fn, coverage: float = 0.95,
                                 short_windows: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, Any]:
    """τ_d such that `coverage` of the KNOWN-class windows given lie within it of their nearest
    prototype — a target acceptance rate for known emitters, computed without looking at any
    unknown data (so it can't inflate the open-set results). Pass windows of held-out
    calibration tracks (not used for training or for building the prototypes) so the rate holds
    on new tracks; validation windows would sit closer to the prototypes built from them."""
    protos = [torch.tensor(v, dtype=torch.float32) for subs in centroids.values() for v in subs]
    _, emb = _run(model, np.asarray(known_windows, dtype=np.float32))
    dist = _min_distance(emb, protos, distance_fn)
    tau = float(np.quantile(dist, coverage))
    out = {"ood_distance_threshold": _sig(tau), "coverage": coverage, "num_windows": int(len(dist)),
           "known_distance_median": _sig(float(np.median(dist))), "known_distance_max": _sig(float(dist.max()))}
    # Shorter windows (the first L pulses of a track) sit farther from the k-pulse prototypes; the
    # same coverage at each length. Never tighter than the k-pulse τ_d.
    by_length = {}
    for L, xs in sorted((short_windows or {}).items()):
        _, e = _run(model, np.asarray(xs, dtype=np.float32))
        by_length[int(L)] = _sig(max(tau, float(np.quantile(_min_distance(e, protos, distance_fn), coverage))))
    if by_length:
        out["by_length"] = by_length
    return out


def tau_for_length(tau_d: float, n_pulses: int, window: int, by_length: Optional[Dict[int, float]], base: Optional[float]) -> float:
    """τ_d for a window of n_pulses: the per-length calibration for windows shorter than k (largest
    calibrated length ≤ n), scaled when τ_d was overridden (tau_d / base); τ_d itself otherwise."""
    if not by_length or n_pulses >= window:
        return tau_d
    lengths = [L for L in by_length if L <= n_pulses]
    if not lengths:
        lengths = [min(by_length)]
    factor = tau_d / base if base else 1.0
    return by_length[max(lengths)] * factor


MAX_DETAIL_TRACKS = 5000  # per-track rows kept in a detailed result


def _ratio(a: float, b: float) -> Optional[float]:
    return round(float(a) / float(b), 4) if b else None


def emitter_breakdown(main: Dict[str, Any], tracks: list, class_labels: Optional[List[str]], tau_c: float, tau_d: float) -> Dict[str, Any]:
    """Per-emitter view of a test: precision / recall / F1 per known emitter, the track-level confusion
    matrix, how each kind of unknown track was handled, the τ_d trade-off and one row per track."""
    y, pred, conf, dist, verdict, tids = main["y"], main["pred"], main["conf"], main["dist"], main["verdict"], main["tracks"]
    VER = VERDICTS[0]
    known = y >= 0
    by_track: Dict[Any, List[int]] = {}
    for i, t in enumerate(tids):
        by_track.setdefault(t, []).append(i)

    per_track = []
    for tid, t, _, label in tracks:
        idx = by_track.get(tid)
        if not idx:
            continue
        idx = np.asarray(idx)
        cls, votes, v = track_vote(pred[idx].tolist(), verdict[idx].tolist())
        raw = str(t["class"].iloc[0]) if "class" in t.columns and pd.notna(t["class"].iloc[0]) else None
        per_track.append({
            "track": str(tid), "true": class_name(label, class_labels) if label >= 0 else (raw or "UNKNOWN"), "true_class_id": int(label),
            "predicted": class_name(cls, class_labels), "predicted_class_id": cls, "verdict": v,
            "correct": bool(cls == label) if label >= 0 else None,
            "flagged_unknown": bool(v != VER) if label < 0 else None,
            "windows": int(len(idx)), "verified_windows": int((verdict[idx] == VER).sum()),
            "confidence": round(float(conf[idx].mean()), 4), "distance": _sig(float(np.median(dist[idx])), 4),
            "pulses": int(len(t))})

    # ---- known emitters: closed-set precision / recall / F1 over windows, track accuracy, rejection
    ids = sorted(set(y[known].tolist()) | set(pred[known].tolist()))
    emitters = []
    for c in ids:
        sel = known & (y == c)
        tp = int((known & (y == c) & (pred == c)).sum())
        fp = int((known & (y != c) & (pred == c)).sum())
        fn = int((sel & (pred != c)).sum())
        prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = round(2 * prec * rec / (prec + rec), 4) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        mine = [pt for pt in per_track if pt["true_class_id"] == c]
        wrong = Counter(pred[sel & (pred != c)].tolist())
        top = wrong.most_common(1)
        emitters.append({
            "class_id": int(c), "emitter": class_name(c, class_labels), "tracks": len(mine), "windows": int(sel.sum()),
            "precision": prec, "recall": rec, "f1": f1,
            "track_accuracy": _ratio(sum(1 for pt in mine if pt["correct"]), len(mine)),
            "verified_rate": _ratio((verdict[sel] == VER).sum(), sel.sum()),
            "rejected_rate": _ratio((verdict[sel] != VER).sum(), sel.sum()),
            "mean_confidence": round(float(conf[sel].mean()), 4) if sel.any() else None,
            "median_distance": _sig(float(np.median(dist[sel])), 4) if sel.any() else None,
            "confused_with": class_name(top[0][0], class_labels) if top else None,
            "confused_share": _ratio(top[0][1], sel.sum()) if top else None})

    # ---- track-level confusion: rows = true emitter (and each unknown label), columns = voted class
    col_ids = sorted({pt["predicted_class_id"] for pt in per_track} | {pt["true_class_id"] for pt in per_track if pt["true_class_id"] >= 0})
    rows = {}
    for pt in per_track:
        key = (0, pt["true_class_id"], pt["true"]) if pt["true_class_id"] >= 0 else (1, -1, pt["true"])
        r = rows.setdefault(key, {"label": pt["true"], "known": pt["true_class_id"] >= 0, "total": 0, "flagged": 0, "counts": [0] * len(col_ids)})
        r["total"] += 1
        r["flagged"] += int(pt["verdict"] != VER)
        r["counts"][col_ids.index(pt["predicted_class_id"])] += 1
    confusion = {"columns": [class_name(c, class_labels) for c in col_ids], "rows": [rows[k] for k in sorted(rows)]}

    # ---- unknown / noise tracks: flagged vs. wrongly accepted
    unknown = []
    for key in sorted(k for k in rows if k[0] == 1):
        label = key[2]
        mine = [pt for pt in per_track if pt["true_class_id"] < 0 and pt["true"] == label]
        wsel = np.isin(tids.astype(str), [m["track"] for m in mine])
        near = Counter(m["predicted"] for m in mine).most_common(1)
        unknown.append({"label": label, "tracks": len(mine), "flagged_rate": _ratio(sum(1 for m in mine if m["flagged_unknown"]), len(mine)),
                        "windows_accepted_as_known": _ratio((verdict[wsel] == VER).sum(), wsel.sum()),
                        "median_distance": _sig(float(np.median(dist[wsel])), 4) if wsel.any() else None,
                        "nearest_known": near[0][0] if near else None})

    # ---- τ_d trade-off on these windows (confidence threshold fixed)
    sweep = []
    if len(dist):
        taus = sorted({float(q) for q in np.quantile(dist, np.linspace(0.02, 0.98, 25))} | {float(tau_d)})
        unk = ~known
        for t in taus:
            ok = (conf >= tau_c) & (dist <= t)
            sweep.append({"tau_d": _sig(t, 4), "known_accepted": _ratio(ok[known].sum(), known.sum()),
                          "unknown_rejected": _ratio((~ok[unk]).sum(), unk.sum()), "current": bool(abs(t - tau_d) < 1e-12)})
    truncated = len(per_track) > MAX_DETAIL_TRACKS
    return {"emitters": emitters, "confusion": confusion, "unknown": unknown, "tau_curve": sweep,
            "per_track": per_track[:MAX_DETAIL_TRACKS], "per_track_total": len(per_track), "per_track_truncated": truncated}


def evaluate_tracks(model: nn.Module, spec: SequenceSpec, df: pd.DataFrame, centroids: Dict[int, List[List[float]]],
                    distance_fn, tau_c: float, tau_d: float, window: Optional[int] = None,
                    curve_lengths: Optional[List[int]] = None, detail: bool = False,
                    class_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Full evaluation of a sequence model on the tracks in `df` (true_class_id −1 = unknown).
    `detail` adds the per-emitter breakdown (Testing page); training reports skip it."""
    model.eval()
    k = int(window or spec.window)
    protos = [torch.tensor(v, dtype=torch.float32) for subs in centroids.values() for v in subs]
    tracks = [(tid, t, track_feature_matrix(t, spec.names), int(t["true_class_id"].iloc[0])) for tid, t in ordered_tracks(df)]

    def score(length: int, stride: int):
        xs, ys, tids = [], [], []
        for tid, _, m, label in tracks:
            for w in windows_of(m, length, stride, spec.whole):
                xs.append(w); ys.append(label); tids.append(tid)
        if not xs:
            return None
        x = _prep(spec, xs)
        probs, emb = _run(model, x)
        ys = np.asarray(ys)
        preds = probs.argmax(dim=-1).numpy()
        conf = probs.max(dim=-1).values.numpy()
        dist = _min_distance(emb, protos, distance_fn) if protos else np.zeros(len(ys))
        return {"y": ys, "pred": preds, "conf": conf, "dist": dist, "tracks": np.asarray(tids, dtype=object),
                "verdict": _verdicts(conf, dist, tau_c, tau_d)}

    main = score(k, k)
    if main is None:
        raise ValueError(f"No window of {k} pulses: every track is shorter than {k} pulses." if not spec.whole
                         else "No track with at least 2 pulses.")
    known = main["y"] >= 0
    out: Dict[str, Any] = {
        "window": k,
        "num_windows_known": int(known.sum()), "num_windows_unknown": int((~known).sum()),
        "num_tracks_known": int(sum(1 for *_, lab in tracks if lab >= 0)),
        "num_tracks_unknown": int(sum(1 for *_, lab in tracks if lab < 0)),
        "closed_set_accuracy": round(float((main["pred"][known] == main["y"][known]).mean()), 4) if known.any() else None,
        "open_set_auroc": None,
        "verdicts": {"known": {v: int((main["verdict"][known] == v).sum()) for v in VERDICTS}},
    }
    if (~known).any():
        auroc = _rank_auroc(main["dist"][known], main["dist"][~known])
        out["open_set_auroc"] = round(auroc, 4) if auroc is not None else None
        out["verdicts"]["unknown"] = {v: int((main["verdict"][~known] == v).sum()) for v in VERDICTS}

    # Per track: the same vote as Inference / Live (track_vote) — known tracks: is the voted class
    # right; unknown tracks: is the track's verdict anything but "verified known".
    correct, flagged = [], []
    for tid, _, _, label in tracks:
        sel = main["tracks"] == tid
        if not sel.any():
            continue
        cls, _, verdict = track_vote(main["pred"][sel].tolist(), main["verdict"][sel].tolist())
        if label >= 0:
            correct.append(int(cls == label))
        else:
            flagged.append(int(verdict != VERDICTS[0]))
    out["track_accuracy"] = round(float(np.mean(correct)), 4) if correct else None
    out["track_unknown_detection"] = round(float(np.mean(flagged)), 4) if flagged else None
    if detail:
        out["detail"] = emitter_breakdown(main, tracks, class_labels, tau_c, tau_d)

    # Accuracy / AUROC vs number of pulses observed: one decision per track after its FIRST n
    # pulses, computed from those pulses only (causal_window) — exactly what Live Simulation
    # decides at that moment. Starts at 2: the first pulse has no PRI yet.
    longest = max(len(m) for _, _, m, _ in tracks)
    lengths = curve_lengths or [n for n in (2, 4, 8, 16, 32, 64, 128) if n <= longest]
    curve = []
    for n in lengths:
        xs, ys = [], []
        for _, t, m, label in tracks:
            if len(t) >= n and (PRI_COLUMN not in spec.columns or has_pri(t.iloc[:n])):
                xs.append(causal_window(t, spec.names, n, k, spec.whole)); ys.append(label)
        if not xs:
            continue
        probs, emb = _run(model, spec.transform(np.stack(xs)))
        ys = np.asarray(ys); kn = ys >= 0
        pred = probs.argmax(dim=-1).numpy()
        dist = _min_distance(emb, protos, distance_fn) if protos else np.zeros(len(ys))
        entry = {"pulses": n, "tracks": int(len(ys)),
                 "accuracy": round(float((pred[kn] == ys[kn]).mean()), 4) if kn.any() else None, "auroc": None}
        if (~kn).any() and kn.any():
            a = _rank_auroc(dist[kn], dist[~kn])
            entry["auroc"] = round(a, 4) if a is not None else None
        curve.append(entry)
    out["pulses_curve"] = curve
    return out


# ---------------------------------------------------------------- classifying one track (Inference / Live / package)

VERDICT_STYLE = {
    "VERIFIED_KNOWN_TARGET": ("GREEN", "MONITOR_ONLY"),
    "UNKNOWN_THREAT_ANOMALY": ("ORANGE", "AUTOMATED_JAMMER_ENGAGED"),
    "OOD_NOISE_REJECTED": ("RED", "ELINT_LOGGING"),
}


class SequenceClassifier:
    """A trained sequence model with everything needed to turn raw pulses into verdicts: its
    feature spec (scaling, window k), stored prototypes, distance metric and thresholds."""

    def __init__(self, model: nn.Module, spec: SequenceSpec, centroids: Dict[int, List[List[float]]],
                 distance_fn, tau_c: float, tau_d: float, tau_d_by_length: Optional[Dict[int, float]] = None,
                 tau_d_base: Optional[float] = None, class_labels: Optional[List[str]] = None):
        self.model, self.spec, self.distance_fn = model.eval(), spec, distance_fn
        self.tau_c, self.tau_d = float(tau_c), float(tau_d)
        self.needs_pri = PRI_COLUMN in spec.columns  # no decision before the 2nd pulse (no PRI yet)
        # Class names: the uploaded dataset's own labels (its class ids were renumbered 0…N−1), else RADAR_FAMILY_xx.
        self.class_labels = list(class_labels) if class_labels else None
        # τ_d per window length below k (auto-calibrated models); base = the calibrated k-pulse τ_d
        self.tau_d_by_length = {int(L): float(v) for L, v in (tau_d_by_length or {}).items()} or None
        self.tau_d_base = float(tau_d_base) if tau_d_base else self.tau_d
        self.protos, self.proto_labels = [], []
        for c, subs in centroids.items():
            for v in subs:
                self.protos.append(torch.tensor(v, dtype=torch.float32))
                self.proto_labels.append(int(c))

    def name(self, class_id: int) -> str:
        return class_name(class_id, self.class_labels)

    def tau_for(self, n_pulses: int) -> float:
        return tau_for_length(self.tau_d, n_pulses, self.spec.window, self.tau_d_by_length, self.tau_d_base)

    def classify(self, windows: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Raw (unscaled) feature windows, possibly of different lengths → one decision each."""
        out: List[Optional[Dict[str, Any]]] = [None] * len(windows)
        by_len: Dict[int, List[int]] = {}
        for i, w in enumerate(windows):
            by_len.setdefault(len(w), []).append(i)
        for length, idx in by_len.items():  # same-length windows run as one batch
            tau_d = self.tau_for(length)
            probs, emb = _run(self.model, _prep(self.spec, [windows[i] for i in idx]))
            with torch.no_grad():
                dists = torch.stack([self.distance_fn(emb, p.unsqueeze(0).expand_as(emb)) for p in self.protos], dim=1)
            nearest = dists.argmin(dim=1).numpy()
            dmin = dists.min(dim=1).values.numpy()
            for row, i in enumerate(idx):
                p = probs[row].numpy()
                top = int(p.argmax())
                conf = float(p[top])
                d = float(dmin[row])
                verdict = _verdicts(np.array([conf]), np.array([d]), self.tau_c, tau_d)[0]
                color, ecm = VERDICT_STYLE[verdict]
                top3 = np.argsort(p)[-3:][::-1]
                out[i] = {"class_id": top, "class": self.name(top), "confidence": round(conf, 4),
                          "top3": [{"class": self.name(int(j)), "prob": round(float(p[j]), 4)} for j in top3],
                          "distance": round(d, 5), "nearest_prototype": self.name(self.proto_labels[int(nearest[row])]),
                          "verdict": str(verdict), "color": color, "ecm_trigger": ecm, "pulses": int(len(windows[i])),
                          "tau_d": _sig(tau_d)}
        return out

    def track_windows(self, matrix: np.ndarray) -> List[Tuple[int, int]]:
        """(start, end) of the windows a track is judged on: consecutive windows of k pulses; a
        leftover tail adds one final window ending at the last pulse; a track shorter than k is
        one window of all its pulses (the model accepts any length)."""
        n, k = len(matrix), self.spec.window
        if self.spec.whole:  # track mode: one decision on the track's first ≤ k pulses
            return [(0, min(n, k))]
        if n <= k:
            return [(0, n)]
        spans = [(s, s + k) for s in range(0, n - k + 1, k)]
        if spans[-1][1] < n:
            spans.append((n - k, n))
        return spans

    def classify_track(self, track: pd.DataFrame, progression_limit: Optional[int] = None) -> Dict[str, Any]:
        """Verdict for one whole track (majority over its windows) + every window's decision +
        the decision as pulses arrive (on the last ≤ k pulses seen so far, as Live Simulation does)."""
        if self.needs_pri and not has_pri(track):
            raise ValueError("A track needs at least 2 pulses: its first pulse has no PRI (no previous pulse).")
        matrix = track_feature_matrix(track, self.spec.names)
        spans = self.track_windows(matrix)
        windows = self.classify([matrix[s:e] for s, e in spans])
        for (s, e), w in zip(spans, windows):
            w["start"], w["end"] = int(s), int(e)
        class_id, class_count, verdict = track_vote([w["class_id"] for w in windows], [w["verdict"] for w in windows])
        color, ecm = VERDICT_STYLE[verdict]
        k = self.spec.window
        # The decision as pulses arrive — from the pulses seen so far only, as Live Simulation does.
        limit = min(len(matrix), progression_limit or 2 * k)
        seen = [n for n in range(1, limit + 1) if not self.needs_pri or has_pri(track.iloc[:n])]
        progression = self.classify([causal_window(track, self.spec.names, n, k, self.spec.whole) for n in seen])
        for n, p in zip(seen, progression):
            p["pulses_seen"] = n
        return {
            "pulses": int(len(matrix)), "window": k, "num_windows": len(windows),
            "class": self.name(class_id), "class_id": class_id,
            "class_votes": f"{class_count}/{len(windows)}",
            "confidence": round(float(np.mean([w["confidence"] for w in windows])), 4),
            "distance": round(float(np.median([w["distance"] for w in windows])), 5),
            "verdict": verdict, "color": color, "ecm_trigger": ecm,
            "verified_windows": int(sum(w["verdict"] == "VERIFIED_KNOWN_TARGET" for w in windows)),
            "windows": windows, "progression": progression,
        }


# ---------------------------------------------------------------- explainability

def _label(name: str) -> str:
    return seq_feature_label(name)


class _Stub:
    def __init__(self, backbone_type: str):
        self.backbone_type = backbone_type


def _backbone_lines(model: nn.Module, n: int, e: int) -> List[Dict[str, Any]]:
    bt = model.backbone_type
    if bt == "seq_hybrid":
        cnn = _backbone_lines(_Stub("seq_cnn"), n, 64)[0]["lines"]
        tf = _backbone_lines(_Stub("seq_transformer"), n, 64)[0]["lines"]
        return [{"title": "Temporal CNN branch (along pulses)", "lines": [l.replace("f = ", "f_cnn = ") for l in cnn]},
                {"title": "Temporal Transformer branch (along pulses)", "lines": [l.replace("f = ", "f_tf = ") for l in tf]},
                {"title": "Hybrid fusion", "lines": [
                    f"f = ReLU( W_h · [f_cnn ; f_tf] + b_h )      # W_h: {e} × 128 — learned weighting of the two branches",
                    "# local patterns (CNN) + relations across the whole window (Transformer); branch contribution below"]}]
    if bt == "seq_cnn":
        lines = [
            "X = [x_1 … x_k]  →  channels = features, axis = pulses   (shape F × k)",
            "h1 = ReLU( Conv1d_{F→64, width 3, dilation 1}(X) )         # each output sees 3 consecutive pulses",
            "h2 = ReLU( Conv1d_{64→128, width 3, dilation 2}(h1) )      # sees 7 consecutive pulses",
            "h3 = ReLU( Conv1d_{128→128, width 3, dilation 4}(h2) )     # sees 15 consecutive pulses",
            "pooled = [ mean over pulses (h3) ; max over pulses (h3) ]  # 256 numbers, any window length",
            f"f = ReLU( W_fc · pooled + b_fc )                           # W_fc: {e} × 256",
        ]
        title = "Temporal 1D-CNN (along pulses)"
    elif bt == "seq_transformer":
        lines = [
            "t_i = W_in · x_i + b_in + PE(i)            # each pulse → a 64-number token; PE = sinusoidal position i",
            "PE(i)_{2m} = sin(i / 10000^{2m/64}),  PE(i)_{2m+1} = cos(i / 10000^{2m/64})",
            "repeat 2 layers:  T ← LN( T + MultiHeadAttention(T) );  T ← LN( T + W_2·ReLU(W_1·T) )   # 4 heads, FFN 128",
            "f = ReLU( W_fc · mean over pulses(T) + b_fc )",
            "# attention compares every pulse with every other; position encoding gives their order",
        ]
        title = "Temporal Transformer (along pulses)"
    else:
        lines = [
            "h_i = GRU(x_i, h_{i−1})  forward and backward over the pulses, 2 layers, 64 units each way",
            "  z = σ(W_z x_i + U_z h_{i−1}),  r = σ(W_r x_i + U_r h_{i−1})",
            "  h̃ = tanh(W x_i + U (r ⊙ h_{i−1})),  h_i = (1 − z) ⊙ h_{i−1} + z ⊙ h̃",
            f"f = ReLU( W_fc · [ mean_i(h_i) ; h_last(→) ; h_last(←) ] + b_fc )      # W_fc: {e} × 256",
        ]
        title = "Bidirectional GRU (along pulses)"
    return [{"title": title, "lines": lines}]


def sequence_architecture_report(model: nn.Module, spec: SequenceSpec, config: Dict[str, Any]) -> Dict[str, Any]:
    hp = config["hyperparameters"]
    n, e = spec.dim, model.classifier_head.in_features
    k = model.classifier_head.out_features
    formulas = [
        {"title": "Input: a window of consecutive pulses of one track", "lines": [
            f"X = [x_1 … x_{spec.window}]   (trained on windows of {spec.window} pulses; any length is accepted)",
            f"x_i = [{', '.join(_label(nm) for nm in spec.names)}]   ({n} features per pulse)",
            "Δv_i = v_i − v_(i−1) within the same track (first pulse: 0; its missing PRI = the 2nd pulse's);  duty cycle = PW / PRI;  PW × PRI = PW · PRI",
            "Δ features: x → sign(x) · ln(1 + |x| / c_j) before scaling, c_j = median |Δ| over the training pulses"
            + ("" if all(n in SEQ_FEATURES for n in spec.names) else ";  dataset columns used as given (value) or as Δ"),
            "x̃_ij = (x_ij − μ_j) / σ_j      μ, σ fitted on the training tracks' pulses (table below)",
        ]},
        *_backbone_lines(model, n, e),
        {"title": "Classifier head (closed-set)", "lines": [
            f"z = W_c · f + b_c   (W_c: {k} × {e});   p = softmax(z);   class = argmax p,  confidence = max p"]},
        {"title": "Metric head (open-set embedding)", "lines": [
            f"e = normalize( W₂ · ReLU(W₁ · f + b₁) + b₂ )   (‖e‖ = 1);  distance = min over stored prototypes"]},
        {"title": "Decision rule", "lines": [
            f"τ_c = {hp['classifier_confidence_threshold']},  τ_d = {hp['ood_distance_threshold']}",
            "Per track: class = majority vote of its windows"]},
        {"title": "Training objective", "lines": [
            "L = CrossEntropy + TripletLoss + λ_RPL · L_RPL",
            "Split by TRACK, per family: 20% validation (prototypes), 10% calibration (τ_d), rest training — never by window",
            "Half of the training batches use a random crop of 2…k pulses, so short tracks are decided well too"]},
    ]
    layers = []
    for name, p in model.named_parameters():
        w = p.detach().cpu().numpy().astype(np.float64)
        layers.append({"name": name, "shape": list(w.shape), "params": int(w.size), "mean": float(w.mean()),
                       "std": float(w.std()), "min": float(w.min()), "max": float(w.max()),
                       "l2_norm": float(np.sqrt((w ** 2).sum()))})
    return {
        "backbone_type": model.backbone_type, "total_params": int(sum(l["params"] for l in layers)),
        "formulas": formulas,
        "scaling": [{"feature": nm, "label": _label(nm), "mean": float(mu), "std": float(sd)}
                    for nm, mu, sd in zip(spec.names, spec.mean, spec.std)],
        "layers": layers, "first_layer": {},
    }


def sequence_feature_importance(model: nn.Module, dataset: Dict[str, Any], spec: SequenceSpec,
                                repeats: int = 5, seed: int = 0) -> List[Dict[str, Any]]:
    """Permutation importance (feature j's whole sequence swapped between windows) and gradient
    sensitivity (mean |∂ log p_y / ∂ x̃|, averaged over pulses), on validation windows."""
    x = np.asarray(dataset["val_x"], dtype=np.float32)
    y = np.asarray(dataset["val_y"])
    if len(y) < 5:
        return []
    model.eval()
    rng = np.random.RandomState(seed)

    def acc(arr):
        p, _ = _run(model, arr)
        return float((p.argmax(dim=-1).numpy() == y).mean())

    base = acc(x)
    xt = torch.from_numpy(x).requires_grad_(True)
    probs, _ = model(xt)
    torch.log(probs.gather(1, torch.from_numpy(y).long().unsqueeze(1)).clamp_min(1e-12)).sum().backward()
    grads = xt.grad.abs().mean(dim=(0, 1)).numpy()
    out = []
    for j, name in enumerate(spec.names):
        drops = []
        for _ in range(repeats):
            sh = x.copy()
            sh[:, :, j] = x[rng.permutation(len(x)), :, j]
            drops.append(base - acc(sh))
        out.append({"feature": name, "label": _label(name), "permutation_drop": round(float(np.mean(drops)), 4),
                    "permutation_std": round(float(np.std(drops)), 4), "gradient_sensitivity": round(float(grads[j]), 6)})
    total = sum(max(0.0, o["permutation_drop"]) for o in out) or 1.0
    for o in out:
        o["permutation_share"] = round(max(0.0, o["permutation_drop"]) / total, 4)
    return sorted(out, key=lambda o: -o["permutation_drop"])

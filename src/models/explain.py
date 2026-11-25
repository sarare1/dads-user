"""Shared evaluation and explainability helpers for trained pulse-sequence models: branch
ablation of the hybrid backbone, the measured evaluation report (confusion matrix, per-class
F1, ROC), the open-set rank AUROC, and the weights export. Sequence-specific reports
(architecture, feature importance, track evaluation) live in sequence_eval.py.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.open_set import rank_auroc as _rank_auroc  # kept under this name for existing imports


# ---------------------------------------------------------------- hybrid branch contribution

def branch_ablation(model: nn.Module, dataset: Dict[str, Any], distance_fn,
                    centroids: Dict[int, List[List[float]]]) -> Optional[Dict[str, Any]]:
    """Sequence Hybrid backbone only: how much the decision depends on each branch. Each branch's 64-d
    output is replaced — by zeros (branch switched off) or by its average over the validation
    rows (branch's pulse-specific information removed, typical level kept) — and the rest of
    the model is run unchanged against the stored prototypes. Also reports each branch's share
    of the fusion layer's input signal: mean ‖W_h[:, branch] · f_branch‖ over validation rows."""
    if getattr(model, "backbone_type", None) != "seq_hybrid":
        return None
    bb = model.backbone
    val_x = torch.from_numpy(np.asarray(dataset["val_x"], dtype=np.float32))
    val_y = np.asarray(dataset["val_y"])
    keep = val_y >= 0
    val_x, val_y = val_x[keep], val_y[keep]
    if len(val_y) < 5:
        return None
    open_x = torch.from_numpy(np.asarray(dataset["openset_x"], dtype=np.float32)) if len(dataset["openset_x"]) else None
    protos = [torch.tensor(v, dtype=torch.float32) for subs in centroids.values() for v in subs]
    model.eval()

    def heads(f_cnn, f_tf):
        # Mirrors TemporalHybridBackbone.forward after the two branches, then the model's heads.
        f = torch.relu(bb.fc(torch.cat([f_cnn, f_tf], dim=-1)))
        probs = torch.softmax(model.classifier_head(f), dim=-1)
        emb = nn.functional.normalize(model.metric_head(f), p=2.0, dim=-1)
        return probs, emb

    def min_dist(e):
        return torch.stack([distance_fn(e, p.unsqueeze(0).expand_as(e)) for p in protos], dim=1).min(dim=1).values.numpy()

    with torch.no_grad():
        k_cnn, k_tf = bb.cnn(val_x), bb.transformer(val_x)
        o_cnn, o_tf = (bb.cnn(open_x), bb.transformer(open_x)) if open_x is not None else (None, None)
        mean_cnn, mean_tf = k_cnn.mean(dim=0, keepdim=True), k_tf.mean(dim=0, keepdim=True)

        def evaluate(transform_cnn, transform_tf):
            probs, emb = heads(transform_cnn(k_cnn), transform_tf(k_tf))
            out = {"accuracy": float((probs.argmax(dim=-1).numpy() == val_y).mean()), "auroc": None}
            if protos and o_cnn is not None:
                _, o_emb = heads(transform_cnn(o_cnn), transform_tf(o_tf))
                out["auroc"] = _rank_auroc(min_dist(emb), min_dist(o_emb))
            return out

        same = lambda t: t
        zero = lambda t: torch.zeros_like(t)
        variants = [
            ("both", "Both branches (the trained model)", same, same),
            ("cnn_zeroed", "1D-CNN branch zeroed — Transformer only", zero, same),
            ("tf_zeroed", "Transformer branch zeroed — 1D-CNN only", same, zero),
            ("cnn_mean", "1D-CNN branch replaced by its average", lambda t: mean_cnn.expand_as(t), same),
            ("tf_mean", "Transformer branch replaced by its average", same, lambda t: mean_tf.expand_as(t)),
        ]
        rows = []
        for key, label, tc, tt in variants:
            rows.append({"variant": key, "label": label, **evaluate(tc, tt)})
        base = rows[0]
        for r in rows:
            r["accuracy_drop"] = round(base["accuracy"] - r["accuracy"], 4)
            r["auroc_drop"] = round(base["auroc"] - r["auroc"], 4) if base["auroc"] is not None and r["auroc"] is not None else None
            r["accuracy"] = round(r["accuracy"], 4)
            r["auroc"] = round(r["auroc"], 4) if r["auroc"] is not None else None

        # Signal share entering the fusion layer: W_h = [W_cnn | W_tf] (embed × 64 each).
        w = bb.fc.weight
        half = k_cnn.shape[1]
        cnn_part = (k_cnn @ w[:, :half].T).norm(dim=1).mean().item()
        tf_part = (k_tf @ w[:, half:].T).norm(dim=1).mean().item()
        total = (cnn_part + tf_part) or 1.0
    return {
        "variants": rows,
        "signal_share": {"cnn": round(cnn_part / total, 4), "transformer": round(tf_part / total, 4)},
        "active_units": {  # ReLU outputs: how many of each branch's 64 numbers are typically non-zero
            "cnn": round(float((k_cnn > 0).float().mean().item()), 4),
            "transformer": round(float((k_tf > 0).float().mean().item()), 4),
        },
    }


# ---------------------------------------------------------------- detailed evaluation

def evaluation_report(model: nn.Module, dataset: Dict[str, Any], distance_fn,
                      centroids: Dict[int, List[List[float]]], ood_threshold: float,
                      conf_threshold: float) -> Dict[str, Any]:
    """Confusion matrix and per-class precision/recall/F1 on the known validation rows, the
    open-set ROC curve (prototype distance as the unknown score), and how the fusion rule's
    verdicts split for known vs. unknown rows at the configured thresholds."""
    model.eval()
    val_x = torch.from_numpy(np.asarray(dataset["val_x"], dtype=np.float32))
    val_y = np.asarray(dataset["val_y"])
    with torch.no_grad():
        probs, emb = model(val_x)
        open_emb = None
        open_probs = None
        if len(dataset["openset_x"]):
            open_probs, open_emb = model(torch.from_numpy(np.asarray(dataset["openset_x"], dtype=np.float32)))
    preds = probs.argmax(dim=-1).numpy()
    classes = sorted(set(int(c) for c in val_y.tolist()) | set(int(c) for c in preds.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(val_y, preds):
        cm[idx[int(t)], idx[int(p)]] += 1
    per_class = []
    for c in classes:
        i = idx[c]
        tp = cm[i, i]
        support = int(cm[i].sum())
        if support == 0:
            continue  # predicted but never a true validation class
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": f"RADAR_FAMILY_{c:02d}", "support": support,
                          "precision": round(float(precision), 4), "recall": round(float(recall), 4),
                          "f1": round(float(f1), 4)})

    report = {
        "classes": [f"RADAR_FAMILY_{c:02d}" for c in classes],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_f1": round(float(np.mean([p["f1"] for p in per_class])), 4) if per_class else None,
        "roc": None,
        "verdicts": None,
    }

    protos = [torch.tensor(v, dtype=torch.float32) for subs in centroids.values() for v in subs]
    if not protos:
        return report

    def min_dist(e: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.stack([distance_fn(e, pr.unsqueeze(0).expand_as(e)) for pr in protos], dim=1).min(dim=1).values.numpy()

    known_d = min_dist(emb)
    known_conf = probs.max(dim=-1).values.numpy()

    def verdicts(conf, d):
        v = np.where((conf >= conf_threshold) & (d <= ood_threshold), "VERIFIED_KNOWN_TARGET",
                     np.where(conf >= conf_threshold, "UNKNOWN_THREAT_ANOMALY", "OOD_NOISE_REJECTED"))
        return {k: int((v == k).sum()) for k in ("VERIFIED_KNOWN_TARGET", "UNKNOWN_THREAT_ANOMALY", "OOD_NOISE_REJECTED")}

    report["verdicts"] = {"known": verdicts(known_conf, known_d)}
    if open_emb is not None:
        unknown_d = min_dist(open_emb)
        report["verdicts"]["unknown"] = verdicts(open_probs.max(dim=-1).values.numpy(), unknown_d)
        scores = np.concatenate([known_d, unknown_d])
        labels = np.concatenate([np.zeros(len(known_d)), np.ones(len(unknown_d))])
        thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
        roc = []
        for t in thresholds[::-1]:
            flagged = scores >= t
            tpr = float((flagged & (labels == 1)).sum() / max(1, (labels == 1).sum()))
            fpr = float((flagged & (labels == 0)).sum() / max(1, (labels == 0).sum()))
            roc.append({"threshold": round(float(t), 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4)})
        report["roc"] = [{"threshold": None, "tpr": 0.0, "fpr": 0.0}] + roc + [{"threshold": None, "tpr": 1.0, "fpr": 1.0}]
    return report


def weights_export(model: nn.Module) -> Dict[str, Any]:
    """Every parameter tensor as nested lists — the complete trained weights, for download."""
    return {name: {"shape": list(p.shape), "values": p.detach().cpu().numpy().tolist()}
            for name, p in model.named_parameters()}

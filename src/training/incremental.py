"""Incremental (class-incremental) training: teach an already trained sequence model new emitter
families without starting over. Four methods, all starting from a trained BASE model:

  A  frozen encoder + weight imprinting   new data only, no gradient step. The new family's
                                          classifier row is set to its mean feature vector and
                                          its prototypes are added. Old rows, prototypes and
                                          thresholds stay exactly as they were — but the new
                                          row now competes with them for every window.
  B  fine-tune with a replay buffer        new data + a few stored tracks of every old family,
                                          a few epochs from the base weights.
  C  fine-tune with distillation (LwF)     new data only; the base model acts as the memory: the
                                          new model must keep the base model's outputs on old
                                          families and its embeddings (Learning without Forgetting).
  D  full retrain                          all old training tracks + the new data, from scratch —
                                          the reference the others are measured against.

Every method is scored on the SAME held-out tracks, never used for training by any method: the base
model's own validation tracks (old families → forgetting), a share of the new family's tracks (how
well it was learned) and the new dataset's unknown tracks (open-set rejection still working).
"""
import copy
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sequence_features import SequenceSpec, load_sequence_dataset_for_training
from src.models.sequence_eval import calibrate_distance_threshold, evaluate_tracks, VERDICTS
from src.models.sequence_models import SequenceDualHeadModel
from src.training import cancel
from src.training.trainer import _in_batch_triplets, build_distance_fn, evaluate_model, seed_training, train_model

METHODS = OrderedDict([
    ("A", "Frozen encoder + weight imprinting (new data only)"),
    ("B", "Fine-tune with replay buffer (new data + a few old tracks per family)"),
    ("C", "Fine-tune with distillation, LwF (new data only)"),
    ("D", "Full retrain on old + new data (reference)"),
])
KD_TEMPERATURE = 2.0


def map_labels(df: pd.DataFrame, label_ids: Dict[str, int]) -> pd.DataFrame:
    """true_class_id from the label NAME (true_label); labels not in `label_ids` → −1 (unknown)."""
    df = df.copy()
    df["true_class_id"] = df["class"].astype(str).map(label_ids).fillna(-1).astype(int)
    return df


def prefix_tracks(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Track ids made unique across the files a training set is assembled from."""
    df = df.copy()
    df["instance"] = prefix + df["instance"].astype(str)
    return df


def grow_model(base: SequenceDualHeadModel, num_classes: int) -> SequenceDualHeadModel:
    """A copy of `base` whose classifier head has `num_classes` rows: the old rows keep their
    weights, added rows start untrained. Backbone and metric head are copied unchanged."""
    model = SequenceDualHeadModel(base.backbone_type, base.n_features, base.classifier_head.in_features, num_classes)
    state = {k: v for k, v in base.state_dict().items() if not k.startswith("classifier_head.")}
    model.load_state_dict(state, strict=False)
    n0 = base.classifier_head.out_features
    with torch.no_grad():
        model.classifier_head.weight[:n0] = base.classifier_head.weight
        model.classifier_head.bias[:n0] = base.classifier_head.bias
    return model


def _features(model: SequenceDualHeadModel, x: np.ndarray, batch: int = 512) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model.backbone(torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)])


def _training_data(df: pd.DataFrame, spec: Optional[SequenceSpec], names: List[str], k: int, seed: int) -> Dict[str, Any]:
    return load_sequence_dataset_for_training(df, names, k, seed, spec=spec)


def _calibrate(model, data: Dict[str, Any], centroids, metric: str, coverage: float) -> Dict[str, Any]:
    use_cal = len(data.get("cal_x", [])) >= 5
    cal = calibrate_distance_threshold(model, data["cal_x"] if use_cal else data["val_x"], centroids,
                                       build_distance_fn(metric), coverage, data.get("cal_short_x"))
    cal["calibrated_on"] = (f"{data['num_cal_tracks']} calibration tracks of this method's training data" if use_cal
                            else "validation windows of this method's training data (too few tracks for a calibration split)")
    cal["mode"] = "auto"
    return cal


def _finetune(model: SequenceDualHeadModel, data: Dict[str, Any], cfg: Dict[str, Any], epochs: int, lr: float,
              teacher: Optional[SequenceDualHeadModel] = None, teacher_ids: Optional[List[int]] = None,
              progress_cb: Optional[Callable] = None, batch_size: int = 64,
              rng: Optional[np.random.RandomState] = None) -> List[float]:
    """Cross-entropy + triplet loss from the current weights (as train_model, without the RPL head:
    RPL is a training-time comparison, not part of the deployed model). With a `teacher`, two
    distillation terms keep the base model's knowledge: its softened class probabilities over the
    old families (KL, temperature T) and its embeddings (1 − cosine) on the same windows."""
    hp = cfg["hyperparameters"]
    distance_fn = build_distance_fn(hp["distance_metric"])
    triplet = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=hp["triplet_margin"])
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])
    crop = data.get("train_crop")
    old = torch.tensor(teacher_ids or [], dtype=torch.long)
    rng = rng if rng is not None else np.random.RandomState()
    curve = []
    model.train()
    for epoch in range(epochs):
        perm, total, n_batches = torch.randperm(len(x)), 0.0, 0
        for s in range(0, len(x), batch_size):
            cancel.check()  # "Stop training"
            idx = perm[s:s + batch_size]
            xb, yb = x[idx], y[idx]
            if crop and xb.shape[1] > crop["min"] and rng.random_sample() < crop["prob"]:
                length = rng.randint(crop["min"], xb.shape[1] + 1)
                start = 0 if crop.get("prefix") else rng.randint(0, xb.shape[1] - length + 1)
                xb = xb[:, start:start + length]
            f = model.backbone(xb)
            logits = model.classifier_head(f)
            emb = F.normalize(model.metric_head(f), dim=-1)
            known = yb >= 0
            loss = ce(logits[known], yb[known]) if known.any() else torch.tensor(0.0)
            a, p, n = _in_batch_triplets(yb, rng)
            if len(a):
                loss = loss + triplet(emb[a], emb[p], emb[n])
            if teacher is not None:
                with torch.no_grad():
                    tf = teacher.backbone(xb)
                    t_logits = teacher.classifier_head(tf)
                    t_emb = F.normalize(teacher.metric_head(tf), dim=-1)
                if len(old):
                    T = KD_TEMPERATURE
                    kd = F.kl_div(F.log_softmax(logits[:, old] / T, dim=-1), F.softmax(t_logits[:, old] / T, dim=-1),
                                  reduction="batchmean") * T * T
                    loss = loss + kd
                loss = loss + (1.0 - (emb * t_emb).sum(dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        curve.append(round(total / max(1, n_batches), 5))
        if progress_cb:
            progress_cb(epoch + 1, epochs, curve[-1])
    model.eval()
    return curve


def run_method(method: str, base: SequenceDualHeadModel, spec: SequenceSpec, base_centroids: Dict[int, list],
               base_thresholds: Dict[str, Any], cfg: Dict[str, Any], num_classes: int, old_ids: List[int],
               new_ids: List[int], new_pool: pd.DataFrame, replay: pd.DataFrame, old_train: pd.DataFrame,
               seed: int, finetune_epochs: int, full_epochs: int, lr_scale: float, coverage: float,
               progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """Trains one method; returns the model, its input spec, prototypes, thresholds and what it used."""
    names, k, metric = spec.names, spec.window, cfg["hyperparameters"]["distance_metric"]
    distance_fn = build_distance_fn(metric)
    rng = seed_training(seed)  # every method starts from the same seed: repeatable and comparable
    lr = cfg["hyperparameters"]["learning_rate"] * lr_scale
    t0 = time.time()
    curve: List[float] = []
    stale: List[int] = []

    if method == "A":
        data = _training_data(new_pool, spec, names, k, seed)
        model = grow_model(base, num_classes)
        model.eval()
        feats = _features(model, data["train_x"])
        w, b = model.classifier_head.weight, model.classifier_head.bias
        with torch.no_grad():
            norm = w[old_ids].norm(dim=1).mean()
            bias = b[old_ids].mean()
            for c in new_ids:
                sel = torch.from_numpy(data["train_y"] == c)
                if sel.any():
                    mean = feats[sel].mean(dim=0)
                    w[c] = mean / mean.norm().clamp_min(1e-8) * norm
                    b[c] = bias
        centroids = {c: v for c, v in base_centroids.items()}
        new_protos = evaluate_model(model, data, distance_fn)["centroids"]
        centroids.update({c: v for c, v in new_protos.items() if c in new_ids})
        thresholds = dict(base_thresholds, mode="inherited",
                          calibrated_on="the base model's thresholds, unchanged (method A uses no old data to recalibrate)")
        used = f"{new_pool['instance'].nunique()} new tracks; no old data; no training"
    elif method in ("B", "C"):
        frames = [new_pool, replay] if method == "B" else [new_pool]
        data = _training_data(pd.concat(frames, ignore_index=True), spec, names, k, seed)
        model = grow_model(base, num_classes)
        curve = _finetune(model, data, cfg, finetune_epochs, lr,
                          teacher=base if method == "C" else None, teacher_ids=old_ids, progress_cb=progress_cb, rng=rng)
        centroids = evaluate_model(model, data, distance_fn)["centroids"]
        # Families with no tracks in this method's data keep the base model's prototypes. Those were
        # computed with the OLD encoder — the price of using no old data (distillation keeps the
        # embeddings close, which is what makes them still roughly valid).
        stale = [c for c in old_ids if c not in centroids]
        for c in stale:
            centroids[c] = base_centroids[c]
        thresholds = _calibrate(model, data, centroids, metric, coverage)
        per_family = replay.groupby("true_class_id")["instance"].nunique().median() if len(replay) else 0
        used = (f"{new_pool['instance'].nunique()} new tracks + {replay['instance'].nunique()} replay tracks "
                f"({per_family:.0f} per old family); {finetune_epochs} epochs from the base weights"
                if method == "B" else f"{new_pool['instance'].nunique()} new tracks; no old data; {finetune_epochs} epochs from the base weights + distillation")
    elif method == "D":
        data = _training_data(pd.concat([old_train, new_pool], ignore_index=True), None, names, k, seed)
        spec = data["feature_spec"]
        model = SequenceDualHeadModel(base.backbone_type, len(names), base.classifier_head.in_features, num_classes)
        run_cfg = copy.deepcopy(cfg)
        run_cfg["model"]["num_classes"] = num_classes
        run_cfg["model"]["embedding_dim"] = base.classifier_head.in_features
        metrics = train_model(model, data, run_cfg, full_epochs, 64, progress_cb, seed)
        curve = metrics["loss_curve"]
        centroids = {int(c): v for c, v in metrics["centroids"].items()}
        thresholds = _calibrate(model, data, centroids, metric, coverage)
        used = (f"{old_train['instance'].nunique()} old + {new_pool['instance'].nunique()} new tracks; "
                f"{full_epochs} epochs from scratch")
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}.")

    return {"method": method, "model": model, "spec": spec, "centroids": centroids, "thresholds": thresholds,
            "loss_curve": curve, "seconds": round(time.time() - t0, 2), "data_used": used, "train_data": data,
            "stale_prototypes": stale}


def score(model, spec: SequenceSpec, centroids: Dict[int, list], metric: str, tau_c: float, tau_d: float,
          eval_old: pd.DataFrame, eval_new: pd.DataFrame, eval_unknown: pd.DataFrame) -> Dict[str, Any]:
    """The comparison numbers on the fixed evaluation tracks. For the base model pass the new
    family's tracks with true_class_id −1: to it they ARE unknown."""
    distance_fn = build_distance_fn(metric)

    def ev(*frames):
        frames = [f for f in frames if f is not None and len(f)]
        if not frames or not any((f["true_class_id"] >= 0).any() for f in frames):
            return None
        return evaluate_tracks(model, spec, pd.concat(frames, ignore_index=True), centroids, distance_fn, tau_c, tau_d)

    every = ev(eval_old, eval_new, eval_unknown)
    old = ev(eval_old)
    learned = eval_new is not None and (eval_new["true_class_id"] >= 0).any()
    new = ev(eval_new) if learned else None
    # Unknown rejection on the same tracks for every model (the base included): old families +
    # the dataset's unknown tracks. For the base model, also: is the new family flagged unknown?
    unk = ev(eval_old, eval_unknown) if eval_unknown is not None and len(eval_unknown) else None
    new_as_unknown = ev(eval_old, eval_new) if not learned and eval_new is not None and len(eval_new) else None

    def verified(res):
        k = (res or {}).get("verdicts", {}).get("known") or {}
        total = sum(k.values())
        return round(k.get(VERDICTS[0], 0) / total, 4) if total else None

    return {
        "old_window_accuracy": old and old["closed_set_accuracy"], "old_track_accuracy": old and old["track_accuracy"],
        "old_verified": verified(old),
        "new_window_accuracy": new and new["closed_set_accuracy"], "new_track_accuracy": new and new["track_accuracy"],
        "new_verified": verified(new),
        "open_set_auroc": unk and unk["open_set_auroc"],
        "unknown_tracks_flagged": unk and unk["track_unknown_detection"],
        "new_flagged_as_unknown": new_as_unknown and new_as_unknown["track_unknown_detection"],
        "overall": every,
    }

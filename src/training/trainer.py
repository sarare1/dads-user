import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

from src.training import cancel
from src.models.open_set import ReciprocalPointsHead, ConfusingSampleGenerator, adversarial_rejection_loss, rpl_auroc, rank_auroc

DEFAULT_RPL_LOSS_WEIGHT = 0.1  # keeps the auxiliary RPL objective from dominating the primary
                               # classifier + triplet training it's being compared against
ADV_MARGIN = 0.1  # how far past its learned radius a confusing sample must be pushed


def _select_holdout_ids(num_classes: int, num_holdout_classes: int, dataset_seed: int) -> List[int]:
    """Deterministically picks which classes are held out from a single seed, so the exact
    same known/unknown split can be reconstructed later (e.g. by /api/test) just by knowing
    dataset_seed + num_holdout_classes — no separate list needs to be stored."""
    if num_holdout_classes <= 0:
        return []
    rng = np.random.RandomState(dataset_seed)
    return sorted(rng.choice(num_classes, size=num_holdout_classes, replace=False).tolist())


def build_distance_fn(metric: str):
    """Builds the embedding-distance function for the configured metric, so the metric head's
    training geometry matches the nearest-prototype distance used at test and inference time.
    Shared by train_model and every standalone evaluation. "dot" is accepted only for runs made before v1.2.0,
    which offered it but always trained and scored with cosine (embeddings have length 1, so the two are equal)."""
    if metric == "euclidean":
        return lambda x, y: nn.functional.pairwise_distance(x, y)
    if metric in ("cosine", "dot"):
        return lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y)
    raise ValueError(f"Unknown distance metric '{metric}'. Use cosine or euclidean.")


def seed_training(seed: Optional[int]) -> np.random.RandomState:
    """Makes a training run repeatable: seeds torch (weight init, batch order, dropout, RPL head) and
    returns the numpy generator for the run's own random choices (crops, triplet mining). Call it
    BEFORE building the model so its initial weights are covered too. seed=None: unseeded."""
    if seed is None:
        return np.random.RandomState()
    torch.manual_seed(int(seed))
    return np.random.RandomState(int(seed) % (2 ** 32))


def _in_batch_triplets(labels: torch.Tensor, rng=np.random) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines (anchor, positive, negative) index triplets from a batch: anchor/positive share
    a label, negative doesn't. Skips OOD-labeled (-1) rows as anchors since they have no
    same-class positive by construction."""
    anchors, positives, negatives = [], [], []
    labels_np = labels.numpy()
    for i, label in enumerate(labels_np):
        if label < 0:
            continue
        same = np.where(labels_np == label)[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        diff = np.where(labels_np != label)[0]
        if len(diff) == 0:
            continue
        anchors.append(i)
        positives.append(rng.choice(same))
        negatives.append(rng.choice(diff))
    idx = (torch.tensor(anchors, dtype=torch.long),
           torch.tensor(positives, dtype=torch.long),
           torch.tensor(negatives, dtype=torch.long))
    return idx


def train_model(model: nn.Module, dataset: Dict[str, Any], config: Dict[str, Any],
                 epochs: int = 30, batch_size: int = 64,
                 progress_cb=None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Trains the dual-head model with real losses: cross-entropy on the classifier head
    (known classes only) and in-batch triplet loss on the metric head, using the configured
    distance metric so the embedding geometry matches the nearest-prototype distance used later.
    Returns evaluation metrics computed on held-out data, not asserted numbers.

    `progress_cb(epoch, epochs, avg_loss)` is called after each epoch if provided, so callers
    (e.g. the API layer) can surface real-time training status instead of a black box.

    `seed` makes the run repeatable (see seed_training; seed the model's construction with the same
    call). Prototypes are built from the TRAINING windows; validation and open-set windows only score.
    """
    rng = seed_training(seed)
    lr = config["hyperparameters"]["learning_rate"]
    margin = config["hyperparameters"]["triplet_margin"]
    metric = config["hyperparameters"]["distance_metric"]
    embed_dim = config["model"]["embedding_dim"]
    num_classes = config["model"]["num_classes"]
    rpl_loss_weight = config["hyperparameters"].get("rpl_loss_weight", DEFAULT_RPL_LOSS_WEIGHT)
    rpl_adversarial = config["hyperparameters"].get("rpl_adversarial", False)
    distance_fn = build_distance_fn(metric)

    # RPL is trained jointly (same optimizer, same batches) as a genuine second open-set
    # method — its loss shapes the shared embedding space alongside the triplet loss, and its
    # own AUROC is reported next to the existing centroid-threshold AUROC for direct comparison.
    rpl_head = ReciprocalPointsHead(num_classes, embed_dim)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(rpl_head.parameters()), lr=lr)

    # Full ARPL: a generator adversarially trained alongside the classifier — it tries to
    # synthesize embeddings that fool the reciprocal-point classifier into accepting them as a
    # known class, while the classifier (via adversarial_rejection_loss, added to the main
    # optimizer step below) is trained to keep rejecting them. Only constructed/trained when
    # explicitly enabled, since the adversarial game is inherently less stable than RPL alone.
    generator, gen_optimizer = None, None
    if rpl_adversarial:
        generator = ConfusingSampleGenerator(num_classes, embed_dim)
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    ce_loss_fn = nn.CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=margin)

    train_x = torch.from_numpy(dataset["train_x"])
    train_y = torch.from_numpy(dataset["train_y"])
    n = train_x.shape[0]
    # Sequence models: some batches use a random shorter crop of every window, so the model learns
    # to decide from few pulses too (Live Simulation decides from the first pulses of a track).
    crop = dataset.get("train_crop") if train_x.dim() == 3 else None

    model.train()
    loss_curve = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            cancel.check()  # "Stop training" — unwinds within one batch
            idx = perm[start:start + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            if crop and xb.shape[1] > crop["min"] and rng.random_sample() < crop["prob"]:
                length = rng.randint(crop["min"], xb.shape[1] + 1)
                start = 0 if crop.get("prefix") else rng.randint(0, xb.shape[1] - length + 1)
                xb = xb[:, start:start + length]
            known_mask = yb >= 0

            # --- main step: model + RPL head, using real data (and, if enabled, rejecting
            # the generator's CURRENT (detached) confusing samples) ---
            optimizer.zero_grad()
            probs, embeddings = model(xb)

            cls_loss = ce_loss_fn(probs[known_mask], yb[known_mask]) if known_mask.any() else torch.tensor(0.0)

            a_idx, p_idx, n_idx = _in_batch_triplets(yb, rng)
            if len(a_idx) > 0:
                trip_loss = triplet_loss_fn(embeddings[a_idx], embeddings[p_idx], embeddings[n_idx])
            else:
                trip_loss = torch.tensor(0.0)

            rpl_dists = rpl_head.distances(embeddings)
            rpl_ce_loss = ce_loss_fn(rpl_head.logits(rpl_dists)[known_mask], yb[known_mask]) \
                if known_mask.any() else torch.tensor(0.0)

            adv_reject_loss = torch.tensor(0.0)
            target_classes = yb[known_mask]
            if rpl_adversarial and len(target_classes) > 0:
                with torch.no_grad():
                    fake_embeddings = generator(target_classes, num_classes)
                adv_reject_loss = adversarial_rejection_loss(rpl_head, fake_embeddings, ADV_MARGIN)

            loss = cls_loss + trip_loss + rpl_loss_weight * (rpl_ce_loss + adv_reject_loss)
            loss.backward()
            optimizer.step()

            # --- generator step: try to fool the CURRENT (just-updated) rpl_head into
            # confidently accepting fresh confusing samples as their target class ---
            if rpl_adversarial and len(target_classes) > 0:
                gen_optimizer.zero_grad()
                fake_embeddings = generator(target_classes, num_classes)
                fake_logits = rpl_head.logits(rpl_head.distances(fake_embeddings))
                gen_loss = ce_loss_fn(fake_logits, target_classes)
                gen_loss.backward()
                gen_optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = round(epoch_loss / max(1, n_batches), 5)
        loss_curve.append(avg_loss)
        if progress_cb is not None:
            progress_cb(epoch + 1, epochs, avg_loss)

    model.eval()
    rpl_head.eval()
    metrics = evaluate_model(model, dataset, distance_fn, rpl_head=rpl_head)
    metrics["loss_curve"] = loss_curve
    metrics["rpl_adversarial"] = rpl_adversarial
    return metrics


def _class_sub_centroids(class_emb: torch.Tensor, max_subclusters: int) -> List[torch.Tensor]:
    """K-means sub-centroids for one class's embeddings, instead of a single mean — a class
    whose true population spans multiple operating modes (search/track/etc., each pulling
    embeddings in a different direction) can have a mean that lands in a low-density gap
    *between* the real sub-clusters, making genuine same-class samples look anomalously far
    from "their own" centroid. Falling back to a single centroid when there's too little data
    to cluster meaningfully (KMeans needs more points than clusters to be well-posed)."""
    n = class_emb.shape[0]
    if n < max_subclusters * 2:
        return [nn.functional.normalize(class_emb.mean(dim=0), dim=0)]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=max_subclusters, n_init=10, random_state=0).fit(class_emb.detach().numpy())
    return [nn.functional.normalize(torch.from_numpy(c.astype(np.float32)), dim=0) for c in km.cluster_centers_]


def _forward_batched(model: nn.Module, x: np.ndarray, batch: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """(probs, embeddings) of `x` in eval mode, in batches (training sets can be large)."""
    if len(x) == 0:
        return torch.zeros(0), torch.zeros(0)
    outs = [model(torch.from_numpy(np.asarray(x[i:i + batch], dtype=np.float32))) for i in range(0, len(x), batch)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])


def evaluate_model(model: nn.Module, dataset: Dict[str, Any], distance_fn,
                    rpl_head: Optional["ReciprocalPointsHead"] = None,
                    max_subclusters: int = 3) -> Dict[str, Any]:
    """Closed-set accuracy on the known validation windows and the open-set AUROC (distance to the
    nearest prototype; validation windows = known, open-set windows = unknown).

    The prototypes are built from the TRAINING windows only, so the windows that are scored never
    helped place the prototypes they are measured against (building them from the validation windows
    made the AUROC optimistic). Each known class contributes up to `max_subclusters` centroids (via
    K-means, see `_class_sub_centroids`). These are also the prototypes the deployed model uses.

    `rpl_head`, when provided (only by train_model's own final call, right after jointly
    training one), additionally reports `open_set_auroc_rpl` — the Reciprocal Points Learning
    method's own AUROC on the exact same val/openset split, for direct comparison against the
    centroid-threshold `open_set_auroc`. RPL is a training-time-only comparison, not a second
    persisted model.
    """
    model.eval()
    with torch.no_grad():
        val_y = np.asarray(dataset["val_y"])
        probs, val_emb = _forward_batched(model, dataset["val_x"])
        known_mask = val_y >= 0
        closed_set_accuracy = float(np.mean(probs.argmax(dim=-1).numpy()[known_mask] == val_y[known_mask]))             if known_mask.any() and len(probs) else 0.0

        train_y = np.asarray(dataset["train_y"])
        _, train_emb = _forward_batched(model, dataset["train_x"])
        centroids: Dict[int, List[torch.Tensor]] = {}
        for c in dataset["known_class_ids"]:
            sel = torch.from_numpy(train_y == c)
            if len(train_emb) and sel.any():
                centroids[c] = _class_sub_centroids(train_emb[sel], max_subclusters)

        all_sub_centroids = [sc for subs in centroids.values() for sc in subs]

        def min_distance_to_centroids(embeddings: torch.Tensor) -> np.ndarray:
            dists = torch.stack([distance_fn(embeddings, sc.unsqueeze(0).expand_as(embeddings))
                                  for sc in all_sub_centroids], dim=1)
            return dists.min(dim=1).values.numpy()

        auroc = None
        auroc_rpl = None
        if len(dataset["openset_x"]) and all_sub_centroids and known_mask.any():
            _, openset_emb = _forward_batched(model, dataset["openset_x"])
            known_emb = val_emb[torch.from_numpy(known_mask)]
            auroc = rank_auroc(min_distance_to_centroids(known_emb), min_distance_to_centroids(openset_emb))
            if rpl_head is not None:
                auroc_rpl = rpl_auroc(rpl_head.ood_scores(known_emb), rpl_head.ood_scores(openset_emb))

    return {
        "closed_set_accuracy": round(closed_set_accuracy, 4),
        "open_set_auroc": round(auroc, 4) if auroc is not None else None,
        "open_set_auroc_rpl": round(auroc_rpl, 4) if auroc_rpl is not None else None,
        "num_known_classes": len(dataset["known_class_ids"]),
        "num_holdout_classes": len(dataset["holdout_class_ids"]),
        "prototypes_from": "training windows",
        "centroids": {int(c): [sc.numpy().tolist() for sc in subs] for c, subs in centroids.items()},
    }

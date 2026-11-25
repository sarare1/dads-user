"""Evaluation must not leak: prototypes come from training windows only, the test split's unknown
tracks stay out of model selection, AUROC counts ties as one half, and a seeded run repeats exactly.
Self-contained (generates its own small dataset). Run: python tests/test_evaluation.py  (or pytest)"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.data.sequence_features import DEFAULT_SEQ_FEATURES, cv_split, load_sequence_dataset_for_training, required_seq_columns, upgrade_columns
from src.data.sequence_generator import build_sequence_dataset_rows
from src.models.open_set import rank_auroc
from src.models.sequence_models import SequenceDualHeadModel
from src.training.trainer import _class_sub_centroids, build_distance_fn, evaluate_model, seed_training, train_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
NUM_CLASSES, HOLDOUT = 4, [3]


def _small_dataset() -> pd.DataFrame:
    rows = build_sequence_dataset_rows(NUM_CLASSES, HOLDOUT, 12, 48, 123, None, 0.02, 0.02, 0.25, None, False, True)
    return upgrade_columns(pd.DataFrame(rows))


def _names(df):
    return [n for n in DEFAULT_SEQ_FEATURES if all(c in df.columns for c in required_seq_columns([n]))]


def _cfg():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    cfg["model"]["num_classes"] = NUM_CLASSES
    cfg["model"]["embedding_dim"] = 32
    cfg["hyperparameters"].setdefault("rpl_loss_weight", 0.1)
    return cfg


def test_rank_auroc_matches_sklearn_with_ties():
    assert rank_auroc([1.0, 1.0], [1.0, 1.0]) == 0.5           # all tied: chance, not 0 or 1
    assert rank_auroc([0.1, 0.2], [0.3, 0.4]) == 1.0
    assert rank_auroc([], [1.0]) is None
    rng = np.random.RandomState(0)
    known, unknown = rng.randint(0, 5, 200).astype(float), rng.randint(2, 8, 150).astype(float)  # many ties
    expected = roc_auc_score(np.r_[np.zeros(200), np.ones(150)], np.r_[known, unknown])
    assert abs(rank_auroc(known, unknown) - expected) < 1e-12


def test_test_split_unknowns_stay_out_of_model_selection():
    df = _small_dataset()
    labels = df.groupby("instance")["true_class_id"].first()
    test_ids, folds = cv_split(df, 5, 25.0, 3, 0.1, include_test=False)
    test_unknown = {t for t in test_ids if labels[t] < 0}
    assert test_unknown, "the test split should contain unknown tracks"
    for train, val, cal, open_ids in folds:
        assert not (open_ids & test_unknown), "test-split unknown tracks leaked into a fold's open-set pool"
        assert not ((train | val | cal) & set(test_ids)), "test-split tracks leaked into training/validation"
    _, folds_incl = cv_split(df, 5, 25.0, 3, 0.1, include_test=True)
    assert test_unknown <= folds_incl[0][3], "with include_test every unknown track is used"


def test_prototypes_come_from_training_windows_only():
    df = _small_dataset()
    names = _names(df)
    _, folds = cv_split(df, 5, 25.0, 3, 0.1, False)
    ds = load_sequence_dataset_for_training(df, names, 16, 5, 0.2, None, 0.1, folds[0])
    seed_training(0)
    model = SequenceDualHeadModel("seq_gru", ds["feature_spec"].dim, 32, NUM_CLASSES).eval()
    dist = build_distance_fn("cosine")
    m = evaluate_model(model, ds, dist)
    with torch.no_grad():
        _, train_emb = model(torch.from_numpy(ds["train_x"]))
    for c in ds["known_class_ids"]:
        expected = _class_sub_centroids(train_emb[torch.from_numpy(ds["train_y"] == c)], 3)
        got = [torch.tensor(v) for v in m["centroids"][c]]
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            assert torch.allclose(g, e, atol=1e-5), f"class {c}: prototype not built from its training windows"
    # Changing ONLY the validation windows must not move any prototype.
    ds2 = dict(ds, val_x=ds["val_x"] + np.float32(3.0))
    m2 = evaluate_model(model, ds2, dist)
    for c in m["centroids"]:
        assert np.allclose(m["centroids"][c], m2["centroids"][c]), "validation windows influenced the prototypes"


def test_seeded_training_repeats_exactly():
    df = _small_dataset()
    names = _names(df)
    _, folds = cv_split(df, 5, 25.0, 3, 0.1, False)
    ds = load_sequence_dataset_for_training(df, names, 16, 5, 0.2, None, 0.1, folds[0])
    results = []
    for _ in range(2):
        seed_training(42)
        model = SequenceDualHeadModel("seq_hybrid", ds["feature_spec"].dim, 32, NUM_CLASSES)
        results.append(train_model(model, copy.deepcopy(ds), _cfg(), 2, 64, None, 42))
    a, b = results
    assert a["loss_curve"] == b["loss_curve"], (a["loss_curve"], b["loss_curve"])
    assert a["closed_set_accuracy"] == b["closed_set_accuracy"] and a["open_set_auroc"] == b["open_set_auroc"]
    for c in a["centroids"]:
        assert np.allclose(a["centroids"][c], b["centroids"][c])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("ALL OK")

"""Every backbone trains on real logits and incremental training accepts track-mode (padded + masked)
inputs. Run: python tests/test_backbone_training.py  (or pytest)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from src.models.sequence_models import SequenceDualHeadModel, SEQ_BACKBONES
from src.training import incremental as inc

CFG = {"hyperparameters": {"learning_rate": 1e-3, "triplet_margin": 0.5, "distance_metric": "cosine"}}


def _track_mode_batch(n_features: int, n: int = 24, L: int = 12, num_classes: int = 3, seed: int = 0):
    """Padded tracks with the mask as one extra last channel - the array form of track mode (pad_stack)."""
    rng = np.random.RandomState(seed)
    x = np.zeros((n, L, n_features + 1), np.float32)
    for i in range(n):
        length = rng.randint(2, L + 1)
        x[i, :length, :n_features] = rng.randn(length, n_features)
        x[i, :length, n_features] = 1.0
    return x, (np.arange(n) % num_classes).astype(np.int64)


def test_logits_match_forward():
    torch.manual_seed(0)
    for bb in SEQ_BACKBONES:
        model = SequenceDualHeadModel(bb, 5, 16, 4).eval()
        x = torch.from_numpy(_track_mode_batch(5)[0])
        with torch.no_grad():
            probs, emb = model(x)
            logits, emb2 = model.logits_and_embeddings(x)
        assert torch.allclose(probs, F.softmax(logits, dim=-1), atol=1e-6), bb
        assert torch.allclose(emb, emb2, atol=1e-6), bb


def test_incremental_accepts_track_mode_inputs():
    torch.manual_seed(0)
    for bb in SEQ_BACKBONES:
        base = SequenceDualHeadModel(bb, 5, 16, 3).eval()
        x, y = _track_mode_batch(5)
        data = {"train_x": x, "train_y": y, "train_crop": {"prob": 1.0, "min": 2, "prefix": True}}
        assert inc._features(inc.grow_model(base, 4), x).shape == (len(x), 16), bb   # method A
        grown = inc.grow_model(base, 4)                                             # methods B / C
        curve = inc._finetune(grown, data, CFG, 1, 1e-3, teacher=base, teacher_ids=[0, 1, 2],
                              rng=np.random.RandomState(0))
        assert np.isfinite(curve).all(), bb


if __name__ == "__main__":
    test_logits_match_forward()
    test_incremental_accepts_track_mode_inputs()
    print("OK")

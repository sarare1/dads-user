"""Padding + mask must not change a track's result: a track padded to a longer length (with its mask)
gives the same output as the same track run alone, unpadded - for every backbone, in training and
eval mode. Run: python tests/test_masking.py  (or pytest)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.models.sequence_models import SequenceDualHeadModel, SEQ_BACKBONES


def test_padding_mask_does_not_change_results(verbose: bool = False):
    torch.manual_seed(0)
    F_, lens, L = 7, [2, 5, 11, 20, 20], 20
    failures = []
    for bb in SEQ_BACKBONES:
        model = SequenceDualHeadModel(bb, F_, 32, 6)
        for mode in ("eval", "train"):
            getattr(model, mode)()
            if mode == "train":  # dropout off: only the masking is under test
                for m in model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.p = 0.0
                    if isinstance(m, torch.nn.MultiheadAttention):
                        m.dropout = 0.0
            tracks = [torch.randn(n, F_) for n in lens]
            x = torch.zeros(len(lens), L, F_ + 1)
            for i, t in enumerate(tracks):
                x[i, :len(t), :F_] = t
                x[i, :len(t), F_] = 1.0
                x[i, len(t):, :F_] = torch.randn(L - len(t), F_) * 5  # garbage in the padding must be ignored
            with torch.no_grad():
                p_pad, e_pad = model(x)
                for i, t in enumerate(tracks):
                    p, e = model(t.unsqueeze(0))
                    d = max(float((p - p_pad[i]).abs().max()), float((e - e_pad[i]).abs().max()))
                    if verbose:
                        print(f"{bb:16s} {mode:5s} len={len(t):2d} max diff {d:.2e} {'OK' if d < 1e-4 else 'FAIL'}")
                    if d >= 1e-4:
                        failures.append((bb, mode, len(t), d))
    assert not failures, f"padding changed the result: {failures}"


if __name__ == "__main__":
    test_padding_mask_does_not_change_results(verbose=True)
    print("ALL OK")

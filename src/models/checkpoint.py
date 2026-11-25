"""Sequence-model checkpoint bundle: weights plus everything needed to rebuild and feed the model.

A bare state_dict doesn't record which input features, window length or scaling a model was
trained on, so loading one after the feature set changed either fails on a shape mismatch or —
worse, when the count happens to match — silently feeds the model the wrong inputs. The
bundle makes the checkpoint the single source of truth for the model's input contract.
"""
from typing import Optional, Tuple

import torch

from src.data.sequence_features import SequenceSpec
from src.models.sequence_models import SequenceDualHeadModel

CHECKPOINT_FORMAT = 2  # 1 = legacy bare state_dict (no feature metadata)


def save_sequence_checkpoint(path: str, model: SequenceDualHeadModel, spec: SequenceSpec) -> None:
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "kind": "sequence",
        "state_dict": model.state_dict(),
        "features": spec.to_dict(),
        "backbone_type": model.backbone_type,
        "embed_dim": model.classifier_head.in_features,
        "num_classes": model.classifier_head.out_features,
    }, path)


def load_sequence_checkpoint(path: str) -> Tuple[Optional[SequenceDualHeadModel], Optional[SequenceSpec], str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors and plain data only — no pickled code
    if not (isinstance(ckpt, dict) and ckpt.get("kind") == "sequence"):
        return None, None, "Not a pulse-sequence model."
    spec = SequenceSpec.from_dict(ckpt["features"])
    model = SequenceDualHeadModel(ckpt["backbone_type"], spec.dim, ckpt["embed_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, spec, f"Loaded sequence model on features {spec.names} (window {spec.window})."

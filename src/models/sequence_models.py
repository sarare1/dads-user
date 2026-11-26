"""Sequence models: classify a window of k consecutive pulses of ONE emitter (a de-interleaved
track), X = [x_1 … x_k] with x_i the selected per-pulse features.

These encoders run along TIME — across pulses — so they can see PRI modulation patterns and frequency agility. Every encoder ends in
pooling over time, so a model trained on windows of k pulses also accepts shorter or longer
windows (used to measure accuracy vs. number of pulses observed).

Two output heads (classifier and metric) read the encoder's feature vector; training is
cross-entropy + triplet + RPL, with class prototypes and a distance/confidence decision rule.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_BACKBONES = {
    "seq_hybrid": "Temporal Hybrid (CNN + Transformer along pulses, fused)",
    "seq_cnn": "Temporal 1D-CNN (convolution along pulses)",
    "seq_transformer": "Temporal Transformer (attention between pulses, positional encoding)",
    "seq_gru": "Bidirectional GRU (recurrent baseline)",
}


class TemporalCNNBackbone(nn.Module):
    """Convolutions along the pulse axis. Dilations 1, 2, 4 give a receptive field of 15
    consecutive pulses — enough to see a stagger cycle, a dwell switch or a ramp."""

    def __init__(self, n_features: int, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):   # x: [B, k, F]; mask: [B, k] (1 = real pulse) or None
        if mask is not None:
            return self._forward_masked(x, mask)
        h = x.transpose(1, 2)                   # [B, F, k] — features are channels, pulses the axis
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))               # [B, 128, k]
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # mean + max over pulses
        return F.relu(self.fc(pooled))

    @torch.jit.unused                           # masked batches are a training-time path; exported models take no mask
    def _forward_masked(self, x, mask):
        # Padded positions are zeroed before and after every layer, so they act exactly like the
        # convolutions' own zero padding: a padded track gives the same result as the unpadded one.
        h = x.transpose(1, 2)
        m = mask.to(h.dtype).unsqueeze(1)       # [B, 1, k]
        h = F.relu(self.conv1(h * m)) * m
        h = F.relu(self.conv2(h)) * m
        h = F.relu(self.conv3(h)) * m           # [B, 128, k]
        mean = h.sum(dim=2) / m.sum(dim=2).clamp(min=1)
        pooled = torch.cat([mean, h.amax(dim=2)], dim=1)  # h >= 0, so padded zeros never win the max
        return F.relu(self.fc(pooled))


class TemporalTransformerBackbone(nn.Module):
    """Each pulse becomes a token; sinusoidal positional encoding gives the pulse order (it
    matters here — the order IS the pattern); attention relates every pulse to every other."""

    def __init__(self, n_features: int, embed_dim: int = 128, d_model: int = 64, max_len: int = 4096):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_encoding", pe, persistent=False)
        self.fc = nn.Linear(d_model, embed_dim)
        self.nhead = 4

    def forward(self, x, mask: Optional[torch.Tensor] = None):   # x: [B, k, F]; mask: [B, k] (1 = real pulse) or None
        t = self.input_proj(x) + self.pos_encoding[: x.shape[1]].unsqueeze(0)
        if mask is not None:
            return self._forward_masked(t, mask)
        # Training: PyTorch's encoder (with its dropout). Inference: the same computation written
        # with length-free reshapes only — nn.MultiheadAttention's ONNX export bakes the window
        # length into its reshapes, so an exported model would only accept exactly k pulses.
        t = self.encoder(t) if self.training else self._encoder_eval(t, None)  # [B, k, d_model]
        return F.relu(self.fc(t.mean(dim=1)))

    @torch.jit.unused
    def _forward_masked(self, t, mask):
        pad = ~mask.bool()                      # True = padding: never attended to
        t = self.encoder(t, src_key_padding_mask=pad) if self.training else self._encoder_eval(t, pad)
        keep = (~pad).to(t.dtype).unsqueeze(-1)
        return F.relu(self.fc((t * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)))

    def _encoder_eval(self, t: torch.Tensor, pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """self.encoder (post-norm, ReLU, batch_first) in eval mode, step by step:
        T = LN(T + MHA(T)), then T = LN(T + W2 ReLU(W1 T)). Same weights, same result."""
        for layer in self.encoder.layers:
            attn = layer.self_attn
            q, k, v = F.linear(t, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            head_dim = q.shape[-1] // self.nhead
            q = q.unflatten(-1, (self.nhead, head_dim)).transpose(1, 2)   # [B, H, L, D]
            k = k.unflatten(-1, (self.nhead, head_dim)).transpose(1, 2)
            v = v.unflatten(-1, (self.nhead, head_dim)).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if pad is not None:
                scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
            w = torch.softmax(scores, dim=-1)
            a = torch.matmul(w, v).transpose(1, 2).flatten(-2)          # [B, L, E]
            t = layer.norm1(t + attn.out_proj(a))
            t = layer.norm2(t + layer.linear2(F.relu(layer.linear1(t))))
        return t



class GRUBackbone(nn.Module):
    """Recurrent baseline: reads the pulses in order (both directions)."""

    def __init__(self, n_features: int, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4 * hidden, embed_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):   # x: [B, k, F]; mask: [B, k] (1 = real pulse) or None
        if mask is not None:
            return self._forward_masked(x, mask)
        out, h = self.gru(x)                    # out: [B, k, 2*hidden]
        last = torch.cat([h[-2], h[-1]], dim=1)  # final state of both directions, top layer
        return F.relu(self.fc(torch.cat([out.mean(dim=1), last], dim=1)))

    @torch.jit.unused
    def _forward_masked(self, x, mask):         # padding is skipped: each track runs over its own pulses only
        lengths = mask.long().sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        mean = out.sum(dim=1) / lengths.to(out.device, out.dtype).unsqueeze(1)
        last = torch.cat([h[-2], h[-1]], dim=1)
        return F.relu(self.fc(torch.cat([mean, last], dim=1)))


class TemporalHybridBackbone(nn.Module):
    """Both temporal encoders in parallel — the CNN for local patterns (a stagger cycle, a step),
    the Transformer for relations across the whole window — fused by a learned layer."""

    def __init__(self, n_features: int, embed_dim: int = 128):
        super().__init__()
        self.cnn = TemporalCNNBackbone(n_features, 64)
        self.transformer = TemporalTransformerBackbone(n_features, 64)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):   # x: [B, k, F]
        return F.relu(self.fc(torch.cat([self.cnn(x, mask), self.transformer(x, mask)], dim=-1)))


class SequenceDualHeadModel(nn.Module):
    """Temporal backbone → shared feature vector f → classifier head + metric head (same heads
    as DualHeadEWModel). Input [batch, pulses, features]; any number of pulses ≥ 1."""

    def __init__(self, backbone_type: str = "seq_cnn", n_features: int = 11, embed_dim: int = 128, num_classes: int = 20):
        super().__init__()
        if backbone_type not in SEQ_BACKBONES:
            raise ValueError(f"Unknown sequence backbone '{backbone_type}'; choose from {list(SEQ_BACKBONES)}.")
        self.backbone_type = backbone_type
        self.n_features = n_features
        self.backbone = {"seq_hybrid": TemporalHybridBackbone, "seq_cnn": TemporalCNNBackbone,
                         "seq_transformer": TemporalTransformerBackbone, "seq_gru": GRUBackbone}[backbone_type](n_features, embed_dim)
        self.classifier_head = nn.Linear(embed_dim, num_classes)
        self.metric_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))

    def logits_and_embeddings(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """(classifier logits, unit-length embeddings). Training losses use this: CrossEntropyLoss
        needs logits, not the softmax probabilities forward() returns. Same inputs as forward()."""
        if x.shape[-1] == self.n_features + 1:
            mask = x[:, :, -1] > 0.5
            x = x[:, :, :-1]
        features = self.backbone(x, mask)
        return self.classifier_head(features), F.normalize(self.metric_head(features), p=2.0, dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """x: [batch, pulses, features]. Tracks of different lengths can share a batch when padded at the
        end: pass `mask` [batch, pulses] (1 = real pulse), or append it to x as one extra last channel
        ([batch, pulses, features + 1]) — the form the dataset arrays use. No mask = every pulse is real."""
        logits, embeddings = self.logits_and_embeddings(x, mask)
        return F.softmax(logits, dim=-1), embeddings

"""Confidence probe:  C_i = max(softmax(W_p h_L)).

A lightweight classification head on the frozen pooled hidden state. Its purpose
is *not* to be a strong classifier but to expose a calibrated internal-confidence
signal ``C_i`` (and normalised entropy) for routing and uncertainty.

Supports the three Experiment-1 pooling types. For ``attention`` pooling the head
owns an :class:`AttentionPool` (trained jointly), consistent with keeping the LLM
backbone frozen. Post-hoc temperature scaling calibrates the confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..llm.pooling import AttentionPool
from .nn_utils import build_mlp, minibatches, to_tensor


class ConfidenceProbe(nn.Module):
    def __init__(self, dim: int, n_classes: int, pooling: str, cfg: Config):
        super().__init__()
        self.pooling = pooling
        self.n_classes = n_classes
        self.attn: Optional[AttentionPool] = None
        if pooling == "attention":
            self.attn = AttentionPool(dim, cfg.pooling.attention_hidden,
                                      cfg.probe.dropout)
        self.head = build_mlp(dim, list(cfg.probe.hidden_dims), n_classes,
                              cfg.probe.dropout)
        # temperature (fit post-hoc); registered so it saves/loads with the module
        self.register_buffer("temperature", torch.ones(1))

    # -- feature -> pooled vector -------------------------------------- #
    def pooled(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling == "attention":
            assert mask is not None, "attention pooling needs a token mask"
            return self.attn(x, mask)
        return x  # last / mean already pooled in the cache

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                apply_temperature: bool = False) -> torch.Tensor:
        logits = self.head(self.pooled(x, mask))
        if apply_temperature:
            logits = logits / self.temperature.clamp(min=1e-2)
        return logits

    # -- inference helpers --------------------------------------------- #
    @torch.no_grad()
    def predict(self, x, mask=None) -> dict:
        self.eval()
        logits = self.forward(to_tensor(x), None if mask is None else to_tensor(mask),
                              apply_temperature=True)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        conf, pred = probs.max(dim=-1)
        ent = -(probs * log_probs).sum(dim=-1)
        norm_ent = ent / float(np.log(self.n_classes)) if self.n_classes > 1 else ent * 0
        return {
            "probs": probs.cpu().numpy().astype(np.float32),
            "confidence": conf.cpu().numpy().astype(np.float32),
            "entropy": norm_ent.cpu().numpy().astype(np.float32),
            "pred": pred.cpu().numpy().astype(np.int64),
        }

    @torch.no_grad()
    def pooled_features(self, x, mask=None) -> np.ndarray:
        self.eval()
        p = self.pooled(to_tensor(x), None if mask is None else to_tensor(mask))
        return p.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
@dataclass
class ProbeData:
    """Pooled features (or token window) + labels for one split."""
    x: np.ndarray                       # [n, dim] or [n, N, dim] (attention)
    y: np.ndarray                       # [n]
    mask: Optional[np.ndarray] = None   # [n, N] for attention


def _fit_temperature(probe: ConfidenceProbe, val: ProbeData) -> None:
    """Optimise a single scalar temperature to minimise val NLL."""
    probe.eval()
    with torch.no_grad():
        logits = probe.forward(to_tensor(val.x),
                               None if val.mask is None else to_tensor(val.mask))
    y = to_tensor(val.y, dtype=torch.long)
    log_temp = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=60)
    nll = nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad()
        loss = nll(logits / log_temp.exp().clamp(min=1e-2), y)
        loss.backward()
        return loss

    opt.step(_closure)
    probe.temperature.copy_(log_temp.exp().detach().clamp(min=1e-2))


def fit_probe(train: ProbeData, val: ProbeData, dim: int, n_classes: int,
              pooling: str, cfg: Config,
              device: str = "cpu") -> Tuple[ConfidenceProbe, dict]:
    """Train the confidence probe; return the module and a small history dict."""
    probe = ConfidenceProbe(dim, n_classes, pooling, cfg).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe.lr,
                            weight_decay=cfg.probe.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    xt = to_tensor(train.x, device)
    yt = to_tensor(train.y, device, dtype=torch.long)
    mt = None if train.mask is None else to_tensor(train.mask, device)

    history = {"train_loss": []}
    for epoch in range(cfg.probe.epochs):
        probe.train()
        epoch_loss = 0.0
        for bidx in minibatches(len(train.y), cfg.probe.batch_size,
                                cfg.seed + epoch):
            b = torch.as_tensor(bidx, device=device)
            logits = probe(xt[b], None if mt is None else mt[b])
            loss = loss_fn(logits, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss.item()) * len(bidx)
        history["train_loss"].append(epoch_loss / len(train.y))

    if cfg.probe.temperature_scale:
        _fit_temperature(probe, val)
    return probe, history

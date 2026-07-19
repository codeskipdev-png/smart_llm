"""Hidden-state pooling strategies compared in Experiment 1.

* ``pool_last`` — last (right-most real) token hidden state. Parameter-free.
* ``pool_mean`` — masked mean over tokens. Parameter-free.
* ``AttentionPool`` — learned additive attention over the cached token window.
  Its (small) parameters are trained *jointly with the confidence probe*, which
  is consistent with the "backbone frozen; train only probe/RBE/calibration"
  rule (the backbone weights never change).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _masked(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Broadcast mask [B, T] -> [B, T, 1] and zero padded positions."""
    return hidden * mask.unsqueeze(-1).to(hidden.dtype)


def pool_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden [B, T, D], mask [B, T] -> [B, D] last real token."""
    lengths = mask.sum(dim=1).long().clamp(min=1)          # [B]
    last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, last_idx).squeeze(1)


def pool_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the token axis. hidden [B, T, D] -> [B, D]."""
    summed = _masked(hidden, mask).sum(dim=1)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
    return summed / counts


class AttentionPool(nn.Module):
    """Additive (Bahdanau) attention pooling with a learned query."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden [B, T, D], mask [B, T]
        e = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)   # [B, T]
        e = e.masked_fill(mask == 0, torch.finfo(e.dtype).min)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)               # [B, T, 1]
        pooled = (alpha * hidden).sum(dim=1)                        # [B, D]
        return self.dropout(pooled)


def pool(kind: str, hidden: torch.Tensor, mask: torch.Tensor,
         attn: "AttentionPool | None" = None) -> torch.Tensor:
    if kind == "last":
        return pool_last(hidden, mask)
    if kind == "mean":
        return pool_mean(hidden, mask)
    if kind == "attention":
        if attn is None:
            raise ValueError("attention pooling requires an AttentionPool module")
        return attn(hidden, mask)
    raise ValueError(f"Unknown pooling kind '{kind}'")

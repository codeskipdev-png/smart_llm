"""Small shared building blocks for the CDKA heads."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


def build_mlp(in_dim: int, hidden_dims: List[int], out_dim: int,
              dropout: float = 0.1) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def standardize_fit(x: np.ndarray):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu) / sd).astype(np.float32)


def to_tensor(x, device="cpu", dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def minibatches(n: int, batch_size: int, seed: int, shuffle: bool = True):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]

"""Uncertainty-Aware Adapter Scaling — the rank schedule (pure logic).

    U(x)  = lam * normalised_entropy + (1 - lam) * (1 - C_i)     (from CDKA)
    r(x)  = r_min + (r_max - r_min) * U(x)                        (continuous)
    r*(x) = nearest available adapter rank to r(x)               (bucketed)

Confident, low-uncertainty inputs get small adapters (little capacity); uncertain
inputs get larger adapters. Compared against static LoRA r in {4, 16, 32}.
"""
from __future__ import annotations

from typing import List

import numpy as np


def continuous_rank(uncertainty: np.ndarray, r_min: int, r_max: int) -> np.ndarray:
    u = np.clip(np.asarray(uncertainty, dtype=np.float64), 0.0, 1.0)
    return r_min + (r_max - r_min) * u


def bucket_rank(cont_rank: np.ndarray, buckets: List[int]) -> np.ndarray:
    """Snap each continuous rank to the nearest available adapter rank."""
    buckets = np.asarray(sorted(buckets), dtype=np.float64)
    cont = np.asarray(cont_rank, dtype=np.float64).reshape(-1, 1)
    nearest = np.argmin(np.abs(cont - buckets.reshape(1, -1)), axis=1)
    return buckets[nearest].astype(np.int64)


def rank_for_uncertainty(uncertainty: np.ndarray, r_min: int, r_max: int,
                         buckets: List[int]) -> np.ndarray:
    return bucket_rank(continuous_rank(uncertainty, r_min, r_max), buckets)

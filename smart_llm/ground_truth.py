"""Ground-truth retrieval-benefit signal (numerically stable).

The naive relative benefit  (Loss_p - Loss_r) / (|Loss_p| + eps)  with a tiny eps
explodes when the parametric model is confidently correct (Loss_p -> 0), producing
targets of order 1e6-1e7 that make the RBE regression meaningless. We therefore
floor the denominator by a fixed constant (a light regularisation of the relative
scale) and clip the result to a bounded range. The *sign* — and hence the oracle
decision 1[Loss_r < Loss_p] — is unchanged, so this only stabilises magnitudes.
"""
from __future__ import annotations

import numpy as np


def stable_benefit(loss_p, loss_r, floor: float = 1.0, clip: float = 5.0):
    """B_true = clip( (Loss_p - Loss_r) / (|Loss_p| + floor), -clip, clip ).

    Works elementwise on arrays or on python floats.
    """
    lp = np.asarray(loss_p, dtype=np.float64)
    lr = np.asarray(loss_r, dtype=np.float64)
    b = (lp - lr) / (np.abs(lp) + floor)
    b = np.clip(b, -clip, clip)
    return b


def oracle_decision(loss_p, loss_r):
    """1 if retrieval lowers the loss, else 0."""
    return (np.asarray(loss_r) < np.asarray(loss_p)).astype(np.int64)

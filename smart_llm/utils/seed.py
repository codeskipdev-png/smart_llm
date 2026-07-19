"""Deterministic seeding across python / numpy / torch."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed all RNGs and (optionally) request deterministic CUDA kernels.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuBLAS determinism for matmul; harmless on CPU.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    return seed

"""Device / dtype helpers. Import ``torch`` lazily so pure-logic modules and
tests do not require it to be present."""
from __future__ import annotations

from typing import Any


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(prefer: str = "cuda") -> Any:
    import torch
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> Any:
    """Map a config dtype string to a torch dtype, falling back safely on CPU."""
    import torch
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = table.get(name.lower(), torch.float32)
    # bf16/fp16 are only meaningful on accelerators; keep fp32 on CPU.
    if not torch.cuda.is_available() and dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    return dtype

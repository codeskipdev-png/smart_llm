"""IO helpers: JSON, compressed npz feature shards, atomic CSV writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path: str, **arrays) -> None:
    """Save named numpy arrays to a compressed ``.npz`` shard."""
    import numpy as np
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def atomic_write_csv(df, path: str, index: bool = False) -> None:
    """Write a pandas DataFrame to CSV atomically (tmp file + rename)."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(Path(path).parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

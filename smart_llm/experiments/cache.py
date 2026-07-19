"""On-disk cache for Stage-1 features (separates heavy LLM work from light CDKA
training). Sharded npz + a global ``meta.csv`` so that:

* last/mean-pooling training never has to load the big token window,
* CDKA can be re-trained/ablated cheaply without re-running the 7B model.

Layout (under ``cache_dir/<dataset>/``)::

    manifest.json                 dims, conditions, shard sizes, config snapshot
    meta.csv                      one row per eval sample (scalars, in order)
    vecs_<s>.npz                  h_last, h_mean, query_emb
    tokens_<s>.npz                h_tokens, token_mask  (attention pooling only)
    cond_<condition>_<s>.npz      centroid, sim, loss_r, pred_r, btrue, oracle, ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.io import load_json, save_json, save_npz, load_npz, atomic_write_csv

META_COLUMNS = ["id", "label", "pred_p", "loss_p", "conf_llm", "entropy_llm",
                "n_prompt_tokens", "t_p"]

COND_KEYS = ("centroid", "sim", "loss_r", "pred_r", "btrue", "oracle",
             "retr_idx", "demo_label_ids", "t_r", "n_tokens_r")


class FeatureWriter:
    def __init__(self, cache_dir: str, dataset: str, conditions: List[str],
                 label_names: List[str], shard_size: int = 1000,
                 config_snapshot: Optional[dict] = None):
        self.dir = Path(cache_dir) / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.conditions = list(conditions)
        self.label_names = list(label_names)
        self.shard_size = shard_size
        self.config_snapshot = config_snapshot or {}
        self._buf_scalars: List[dict] = []
        self._buf_vecs: Dict[str, list] = {"h_last": [], "h_mean": [], "query_emb": []}
        self._buf_tokens: Dict[str, list] = {"h_tokens": [], "token_mask": []}
        self._buf_cond: Dict[str, Dict[str, list]] = {
            c: {k: [] for k in COND_KEYS} for c in conditions}
        self._all_scalars: List[dict] = []
        self._shard = 0
        self._shard_sizes: List[int] = []
        self._timing: dict = {}

    def set_timing(self, timing: dict) -> None:
        """Amortised, non-per-sample costs (embedding/index/search seconds)."""
        self._timing = timing

    # ------------------------------------------------------------------ #
    def add(self, scalars: dict, vecs: dict, tokens: dict, conds: Dict[str, dict]):
        self._buf_scalars.append(scalars)
        for k in self._buf_vecs:
            self._buf_vecs[k].append(vecs[k])
        for k in self._buf_tokens:
            self._buf_tokens[k].append(tokens[k])
        for c in self.conditions:
            for k in self._buf_cond[c]:
                self._buf_cond[c][k].append(conds[c][k])
        if len(self._buf_scalars) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buf_scalars:
            return
        s = self._shard
        save_npz(str(self.dir / f"vecs_{s}.npz"),
                 **{k: np.asarray(v, dtype=np.float32) for k, v in self._buf_vecs.items()})
        save_npz(str(self.dir / f"tokens_{s}.npz"),
                 h_tokens=np.asarray(self._buf_tokens["h_tokens"], dtype=np.float32),
                 token_mask=np.asarray(self._buf_tokens["token_mask"], dtype=np.int64))
        for c in self.conditions:
            b = self._buf_cond[c]
            save_npz(str(self.dir / f"cond_{c}_{s}.npz"),
                     centroid=np.asarray(b["centroid"], dtype=np.float32),
                     sim=np.asarray(b["sim"], dtype=np.float32),
                     loss_r=np.asarray(b["loss_r"], dtype=np.float32),
                     pred_r=np.asarray(b["pred_r"], dtype=np.int64),
                     btrue=np.asarray(b["btrue"], dtype=np.float32),
                     oracle=np.asarray(b["oracle"], dtype=np.int64),
                     retr_idx=np.asarray(b["retr_idx"], dtype=np.int64),
                     demo_label_ids=np.asarray(b["demo_label_ids"], dtype=np.int64),
                     t_r=np.asarray(b["t_r"], dtype=np.float32),
                     n_tokens_r=np.asarray(b["n_tokens_r"], dtype=np.int64))
        self._all_scalars.extend(self._buf_scalars)
        self._shard_sizes.append(len(self._buf_scalars))

        self._buf_scalars = []
        for k in self._buf_vecs:
            self._buf_vecs[k] = []
        for k in self._buf_tokens:
            self._buf_tokens[k] = []
        for c in self.conditions:
            for k in self._buf_cond[c]:
                self._buf_cond[c][k] = []
        self._shard += 1

    def finalize(self):
        self.flush()
        df = pd.DataFrame(self._all_scalars, columns=META_COLUMNS)
        atomic_write_csv(df, str(self.dir / "meta.csv"))
        save_json({
            "dataset": self.dataset,
            "n": int(len(df)),
            "n_shards": self._shard,
            "shard_sizes": self._shard_sizes,
            "conditions": self.conditions,
            "label_names": self.label_names,
            "timing": self._timing,
            "config": self.config_snapshot,
        }, str(self.dir / "manifest.json"))
        return str(self.dir)


# --------------------------------------------------------------------------- #
@dataclass
class Features:
    dataset: str
    label_names: List[str]
    conditions: List[str]
    meta: pd.DataFrame
    h_last: np.ndarray
    h_mean: np.ndarray
    query_emb: np.ndarray
    conds: Dict[str, Dict[str, np.ndarray]]
    h_tokens: Optional[np.ndarray] = None
    token_mask: Optional[np.ndarray] = None
    manifest: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.meta)

    @property
    def labels(self) -> np.ndarray:
        return self.meta["label"].to_numpy(dtype=np.int64)


def load_features(cache_dir: str, dataset: str,
                  conditions: Optional[List[str]] = None,
                  need_tokens: bool = False) -> Features:
    root = Path(cache_dir) / dataset
    manifest = load_json(str(root / "manifest.json"))
    conds = conditions or manifest["conditions"]
    n_shards = manifest["n_shards"]
    meta = pd.read_csv(root / "meta.csv")

    def _cat(files, key):
        return np.concatenate([load_npz(str(f))[key] for f in files], axis=0)

    vec_files = [root / f"vecs_{s}.npz" for s in range(n_shards)]
    h_last = _cat(vec_files, "h_last")
    h_mean = _cat(vec_files, "h_mean")
    query_emb = _cat(vec_files, "query_emb")

    h_tokens = token_mask = None
    if need_tokens:
        tok_files = [root / f"tokens_{s}.npz" for s in range(n_shards)]
        h_tokens = _cat(tok_files, "h_tokens")
        token_mask = _cat(tok_files, "token_mask")

    cond_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for c in conds:
        files = [root / f"cond_{c}_{s}.npz" for s in range(n_shards)]
        cond_arrays[c] = {k: _cat(files, k) for k in COND_KEYS}

    return Features(dataset=dataset, label_names=manifest["label_names"],
                    conditions=list(conds), meta=meta,
                    h_last=h_last, h_mean=h_mean, query_emb=query_emb,
                    conds=cond_arrays, h_tokens=h_tokens, token_mask=token_mask,
                    manifest=manifest)


def cache_exists(cache_dir: str, dataset: str) -> bool:
    return (Path(cache_dir) / dataset / "manifest.json").exists()

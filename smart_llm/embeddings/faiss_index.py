"""FAISS retrieval + the three Experiment-2 retrieval conditions.

Conditions
----------
* ``clean``       : true k nearest neighbours, shown with their real labels.
* ``random``      : k random pool items (noise; a retrieval that ignores content).
* ``adversarial`` : hard negatives — nearest neighbours drawn from *other* classes
                    (``other_class_nn``), i.e. semantically close but wrong-category
                    context. Alternative strategies: ``label_flip`` (clean NN with
                    corrupted displayed labels) and ``shuffle_labels``.

A :class:`RetrievalResult` carries both the retrieved pool ``indices`` and the
``demo_label_ids`` that should be *displayed* in the prompt (these differ from the
pool's true labels only under label-corrupting strategies).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

_log = get_logger("smart_llm.retrieval")


@dataclass
class RetrievalResult:
    condition: str
    indices: np.ndarray         # [n_query, k] int32 -> pool index
    sims: np.ndarray            # [n_query, k] float32 cosine similarity
    demo_label_ids: np.ndarray  # [n_query, k] int32 label to display per demo
    centroid: np.ndarray        # [n_query, dim] float32 mean retrieved embedding

    def mean_sim(self) -> np.ndarray:
        """sim(x, K): mean cosine similarity of the retrieved set. [n_query]"""
        return self.sims.mean(axis=1)


class Retriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index = None
        self.pool_emb: Optional[np.ndarray] = None
        self.pool_labels: Optional[np.ndarray] = None
        self.pool_ids: Optional[List[str]] = None
        self._id_to_row = {}

    # ------------------------------------------------------------------ #
    def fit(self, pool_emb: np.ndarray, pool_labels: np.ndarray,
            pool_ids: List[str]) -> "Retriever":
        import faiss
        pool_emb = np.ascontiguousarray(pool_emb.astype(np.float32))
        dim = pool_emb.shape[1]
        kind = self.cfg.retrieval.index_type
        if kind == "flat_ip":
            index = faiss.IndexFlatIP(dim)
        elif kind == "ivf_flat":
            quant = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quant, dim, self.cfg.retrieval.ivf_nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(pool_emb)
            index.nprobe = min(16, self.cfg.retrieval.ivf_nlist)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.cfg.retrieval.hnsw_m,
                                        faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type '{kind}'")
        index.add(pool_emb)

        self.index = index
        self.pool_emb = pool_emb
        self.pool_labels = np.asarray(pool_labels, dtype=np.int64)
        self.pool_ids = list(pool_ids)
        self._id_to_row = {pid: i for i, pid in enumerate(self.pool_ids)}
        _log.info("FAISS %s index over %d vectors (dim=%d)", kind, len(pool_ids), dim)
        return self

    # ------------------------------------------------------------------ #
    def _centroid(self, indices: np.ndarray) -> np.ndarray:
        # mean of the retrieved pool embeddings per query -> [n, dim]
        return self.pool_emb[indices].mean(axis=1).astype(np.float32)

    def _sims_for(self, query_emb: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # cosine sim (embeddings normalised) between each query and its picks
        picked = self.pool_emb[indices]                 # [n, k, dim]
        return np.einsum("nd,nkd->nk", query_emb, picked).astype(np.float32)

    def retrieve(self, query_emb: np.ndarray, k: Optional[int] = None,
                 condition: str = "clean",
                 query_ids: Optional[List[str]] = None,
                 query_labels: Optional[np.ndarray] = None) -> RetrievalResult:
        assert self.index is not None, "call .fit() first"
        k = k or self.cfg.retrieval.k
        query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
        n = query_emb.shape[0]

        if condition == "clean":
            idx, sims = self._search(query_emb, k, query_ids)
            demo_labels = self.pool_labels[idx]
        elif condition == "random":
            idx = self._random_indices(n, k, query_ids)
            sims = self._sims_for(query_emb, idx)
            demo_labels = self.pool_labels[idx]
        elif condition == "adversarial":
            idx, sims, demo_labels = self._adversarial(query_emb, k, query_ids, query_labels)
        else:
            raise ValueError(f"Unknown retrieval condition '{condition}'")

        return RetrievalResult(condition=condition,
                               indices=idx.astype(np.int32),
                               sims=sims.astype(np.float32),
                               demo_label_ids=demo_labels.astype(np.int32),
                               centroid=self._centroid(idx))

    # ------------------------------------------------------------------ #
    def _search(self, query_emb, k, query_ids):
        """Top-k NN, excluding the query's own pool row when applicable."""
        buffer = k + (2 if self.cfg.retrieval.exclude_self else 0)
        scores, idx = self.index.search(query_emb, buffer)
        if not self.cfg.retrieval.exclude_self or query_ids is None:
            return idx[:, :k], scores[:, :k]
        out_idx, out_sc = [], []
        for row, (qi, srow, irow) in enumerate(zip(query_ids, scores, idx)):
            self_row = self._id_to_row.get(qi, -1)
            keep = [(s, j) for s, j in zip(srow, irow) if j != self_row]
            keep = keep[:k]
            out_sc.append([s for s, _ in keep])
            out_idx.append([j for _, j in keep])
        return np.asarray(out_idx), np.asarray(out_sc)

    def _random_indices(self, n, k, query_ids):
        base = self.cfg.data.seed
        pool_n = len(self.pool_ids)
        out = np.empty((n, k), dtype=np.int64)
        for row in range(n):
            rng = np.random.default_rng(base + row)
            self_row = self._id_to_row.get(query_ids[row], -1) if query_ids else -1
            pick = rng.choice(pool_n, size=min(k + 1, pool_n), replace=False)
            pick = [p for p in pick if p != self_row][:k]
            out[row] = pick
        return out

    def _adversarial(self, query_emb, k, query_ids, query_labels):
        """Hard negatives: nearest neighbours restricted to other classes."""
        strategy = self.cfg.retrieval.adversarial_strategy
        if strategy == "label_flip":
            idx, sims = self._search(query_emb, k, query_ids)
            n_classes = int(self.pool_labels.max()) + 1
            flipped = np.empty_like(idx)
            for row in range(idx.shape[0]):
                rng = np.random.default_rng(self.cfg.data.seed + 7919 + row)
                for c in range(idx.shape[1]):
                    true_lab = int(self.pool_labels[idx[row, c]])
                    choices = [x for x in range(n_classes) if x != true_lab]
                    flipped[row, c] = rng.choice(choices)
            return idx, sims, flipped

        # default: other_class_nn (and shuffle_labels handled downstream)
        if query_labels is None:
            raise ValueError("adversarial 'other_class_nn' needs query_labels")
        buffer = min(max(50, k * 20), self.index.ntotal)
        scores, idx = self.index.search(query_emb, buffer)
        out_idx = np.empty((query_emb.shape[0], k), dtype=np.int64)
        out_sims = np.empty((query_emb.shape[0], k), dtype=np.float32)
        for row in range(query_emb.shape[0]):
            qlab = int(query_labels[row])
            picks = [(s, j) for s, j in zip(scores[row], idx[row])
                     if int(self.pool_labels[j]) != qlab]
            if len(picks) < k:  # fallback: random other-class items
                rng = np.random.default_rng(self.cfg.data.seed + 13 + row)
                others = np.where(self.pool_labels != qlab)[0]
                extra = rng.choice(others, size=k, replace=len(others) < k)
                picks += [(float(np.dot(query_emb[row], self.pool_emb[j])), j)
                          for j in extra]
            picks = picks[:k]
            out_sims[row] = [s for s, _ in picks]
            out_idx[row] = [j for _, j in picks]
        demo_labels = self.pool_labels[out_idx]
        return out_idx, out_sims, demo_labels

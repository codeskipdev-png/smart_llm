"""Sentence-embedding wrapper (BAAI/bge-large-en-v1.5 by default).

bge models expect a retrieval *instruction* prepended to queries but not to
documents; we honour that asymmetry. Embeddings are L2-normalised so that inner
product == cosine similarity, which the FAISS ``flat_ip`` index relies on.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

_log = get_logger("smart_llm.embeddings")


class TextEncoder:
    def __init__(self, cfg: Config, use_alternative: bool = False):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        name = cfg.embedding.alternative_name if use_alternative else cfg.embedding.name
        self.name = name
        device = cfg.embedding.device
        try:
            self.model = SentenceTransformer(name, device=device)
        except Exception as exc:  # pragma: no cover - depends on runtime
            _log.warning("Failed to load %s on %s (%s); retrying on cpu.",
                         name, device, exc)
            self.model = SentenceTransformer(name, device="cpu")
        self.model.max_seq_length = cfg.embedding.max_seq_length
        self.normalize = cfg.embedding.normalize
        self.query_instruction = cfg.embedding.query_instruction
        self._dim = self.model.get_sentence_embedding_dimension()
        cfg.embedding.dim = self._dim
        _log.info("Loaded embedder %s (dim=%d)", name, self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: List[str], instruction: Optional[str]) -> np.ndarray:
        if instruction:
            texts = [instruction + t for t in texts]
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.embedding.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 2000,
        )
        return emb.astype(np.float32)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Encode retrieval-pool documents (no instruction)."""
        return self._encode(texts, instruction=None)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode queries (bge instruction prepended)."""
        return self._encode(texts, instruction=self.query_instruction)

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
        primary = cfg.embedding.alternative_name if use_alternative else cfg.embedding.name
        alt = cfg.embedding.alternative_name
        device = cfg.embedding.device

        # Try, in order: primary@device, primary@cpu, alt@device, alt@cpu. This
        # survives a flaky/rate-limited download of the large model by falling back
        # to a small, reliable one (all-MiniLM). The pipeline is self-consistent at
        # whatever dimension loads (pool + query use the same encoder).
        candidates = [(primary, device), (primary, "cpu")]
        if alt and alt != primary:
            candidates += [(alt, device), (alt, "cpu")]

        self.model = None
        errors = []
        for name, dev in candidates:
            try:
                self.model = SentenceTransformer(name, device=dev)
                self.name = name
                if name != primary:
                    _log.warning("Primary embedder %s unavailable; using fallback %s. "
                                 "Set HF_TOKEN and re-run for the intended embedder.",
                                 primary, name)
                break
            except Exception as exc:  # pragma: no cover - runtime/network dependent
                errors.append(f"{name}@{dev}: {type(exc).__name__}")
        if self.model is None:
            raise RuntimeError("Could not load any sentence embedder. Tried:\n  "
                               + "\n  ".join(errors)
                               + "\nCheck network / set HF_TOKEN, or pre-download the "
                                 "model with `huggingface-cli download`.")

        self.model.max_seq_length = cfg.embedding.max_seq_length
        self.normalize = cfg.embedding.normalize
        # the bge query instruction only applies to the bge model
        self.query_instruction = (cfg.embedding.query_instruction
                                  if self.name == primary else "")
        self._dim = self.model.get_sentence_embedding_dimension()
        cfg.embedding.dim = self._dim
        cfg.embedding.name = self.name          # record what actually loaded
        _log.info("Loaded embedder %s (dim=%d)", self.name, self._dim)

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

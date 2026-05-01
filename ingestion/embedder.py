"""
ingestion/embedder.py — SentenceTransformer implementation of the Embedder port.

SOLID — Dependency Inversion Principle (DIP):
  Implements core/interfaces.Embedder.
  The retrieval layer depends on the Embedder interface, not this class.

SOLID — Single Responsibility Principle (SRP):
  This class does one thing: convert text to vectors.
  Model loading, batching, and normalisation are its only concerns.

KISS Principle:
  No caching layer here — the model is loaded once as a singleton via
  `get_embedder()`. Repeated calls re-use the same object in memory.
"""
from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from config import settings
from core.interfaces import Embedder
from models import Chunk

log = structlog.get_logger()


class SentenceTransformerEmbedder(Embedder):
    """Dense embedder backed by a SentenceTransformer bi-encoder model."""

    def __init__(self, model_name: str, query_prefix: str, batch_size: int) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        log.info("loading_embed_model", model=model_name)
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts (no prefix). Normalised to unit length."""
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with the BGE instruction prefix."""
        return self.embed_texts([self._query_prefix + query])[0]

    def verify_dim(self, expected_dim: int) -> None:
        """Raise ValueError if model output dim differs from config."""
        actual = len(self.embed_texts(["probe"]) [0])
        if actual != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: model={actual}, config={expected_dim}. "
                "Update EMBED_DIM in your .env."
            )


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    """Singleton embedder — loaded once, reused everywhere."""
    return SentenceTransformerEmbedder(
        model_name=settings.embed_model,
        query_prefix=settings.embed_query_prefix,
        batch_size=settings.embed_batch_size,
    )


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Attach embedding vectors to a list of chunks in-place.
    Returns the same list for method chaining.
    """
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    log.info("embedding_chunks", n=len(texts))
    vectors = embedder.embed_texts(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
    log.info("embedding_done", n=len(chunks))
    return chunks
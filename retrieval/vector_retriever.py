"""
retrieval/vector_retriever.py — Thin adapter that retrieves via the VectorStore port.

SOLID — Single Responsibility Principle (SRP):
  This module does ONE thing: embed the query and search the vector store.
  Previously it also contained HyDE expansion (a second responsibility).
  HyDE is now a separate, composable function in retrieval/hyde.py.

SOLID — Dependency Inversion Principle (DIP):
  Depends on `VectorStore` and `Embedder` abstractions, not on Qdrant
  or SentenceTransformer directly.
"""
from __future__ import annotations

import structlog

from config import settings
from core.interfaces import Embedder, VectorStore
from ingestion.embedder import get_embedder
from ingestion.vector_store import get_vector_store
from models import ScoredChunk

log = structlog.get_logger()


def retrieve_vector(
    query: str,
    top_k: int = settings.vector_top_k,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> list[ScoredChunk]:
    """
    Embed the query and return top_k nearest chunks by cosine similarity.

    Both `embedder` and `store` are injectable for testing — default to
    production singletons when not provided.
    """
    _embedder = embedder or get_embedder()
    _store = store or get_vector_store()

    query_vector = _embedder.embed_query(query)
    results = _store.search(settings.qdrant_collection, query_vector, top_k)

    log.debug("vector_retrieved", query=query[:60], n=len(results))
    return results
"""
ingestion/indexer.py — Orchestrates the dual-write ingestion pipeline.

SOLID — Single Responsibility Principle (SRP):
  This module's only job is to coordinate: take embedded chunks, write
  to the vector store, and write to the sparse index.
  It does NOT own client creation, tokenisation, or embedding — those
  live in their respective modules.

KISS Principle:
  `index_chunks()` is a straight three-step sequence:
    1. Ensure vector collection exists.
    2. Upsert to Qdrant.
    3. Build + save BM25.
  No branching, no complex state.
"""
from __future__ import annotations

import structlog

from config import settings
from models import Chunk, IndexStats
from ingestion.vector_store import get_vector_store
from ingestion.sparse_index import BM25SparseIndex

log = structlog.get_logger()


def index_chunks(chunks: list[Chunk]) -> IndexStats:
    """
    Dual-write indexed pipeline:
      chunks (with embeddings) → Qdrant vector store
      chunks (text only)       → BM25 sparse index (pickle on disk)

    Precondition: every chunk must have a non-None embedding.
    """
    if not chunks:
        raise ValueError("index_chunks received an empty list — nothing to index.")

    vector_store = get_vector_store()
    vector_store.ensure_collection(settings.qdrant_collection, settings.embed_dim)
    vector_store.upsert(settings.qdrant_collection, chunks)

    sparse = BM25SparseIndex(
        index_path=settings.bm25_index_path,
        chunks_path=settings.bm25_chunks_path,
    )
    sparse.build(chunks)
    sparse.save()

    unique_sources = len({c.metadata.source for c in chunks})
    stats = IndexStats(
        total_chunks=len(chunks),
        total_documents=unique_sources,
        collection_name=settings.qdrant_collection,
    )
    log.info("indexing_complete", **stats.model_dump(exclude={"indexed_at"}))
    return stats
"""
retrieval/bm25_retriever.py — Thin adapter that retrieves via the SparseIndex port.

SOLID — Dependency Inversion Principle (DIP):
  Depends on `SparseIndex` (abstract), not on `BM25SparseIndex` (concrete).
  Swapping the index backend (e.g. to Elasticsearch BM25) requires
  zero changes here.

SOLID — Single Responsibility Principle (SRP):
  One function. One responsibility: call the sparse index and return results.
"""
from __future__ import annotations

import structlog

from config import settings
from core.interfaces import SparseIndex
from ingestion.sparse_index import get_sparse_index
from models import ScoredChunk

log = structlog.get_logger()


def retrieve_bm25(
    query: str,
    top_k: int = settings.bm25_top_k,
    index: SparseIndex | None = None,
) -> list[ScoredChunk]:
    """
    Return top_k chunks ranked by BM25 score.

    `index` is injectable for testing — defaults to the singleton loaded
    from disk. This makes unit tests possible without a real index file.
    """
    sparse_index = index or get_sparse_index()
    results = sparse_index.search(query, top_k)
    log.debug("bm25_retrieved", query=query[:60], n=len(results))
    return results
"""
retrieval/reranker.py — Cross-encoder reranking stage.

SOLID — Single Responsibility Principle (SRP):
  This module's only job is cross-encoder reranking.
  Previously this lived alongside RRF fusion in one file (fusion_reranker.py),
  giving that file two reasons to change. Now each lives independently.

SOLID — Open/Closed Principle (OCP):
  `rerank()` accepts a `model_name` parameter so callers can switch models
  (e.g. from MiniLM to ms-marco-large) via config without touching this code.

KISS Principle:
  load the model → score pairs → sort → slice. Four steps, no branching.
"""
from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import CrossEncoder

from config import settings
from models import ScoredChunk

log = structlog.get_logger()


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load and cache a cross-encoder model by name."""
    log.info("loading_cross_encoder", model=model_name)
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int = settings.rerank_top_k,
    model_name: str = settings.rerank_model,
) -> list[ScoredChunk]:
    """
    Score (query, chunk) pairs with a cross-encoder and return the top_k.

    Cross-encoder scores are raw logits. We sort on them but do not normalise —
    the generator only needs the ranking, not calibrated probabilities.
    """
    if not candidates:
        return []

    model = _load_cross_encoder(model_name)
    pairs = [(query, item.chunk.text) for item in candidates]
    scores: list[float] = model.predict(pairs).tolist()

    for item, score in zip(candidates, scores):
        item.rerank_score = score

    reranked = sorted(candidates, key=lambda x: x.rerank_score or 0.0, reverse=True)
    log.debug("rerank_done", n_in=len(candidates), n_out=top_k)
    return reranked[:top_k]
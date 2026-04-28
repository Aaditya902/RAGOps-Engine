from __future__ import annotations
 
from models import ScoredChunk
 
 
def reciprocal_rank_fusion(
    *ranked_lists: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """
    Merge an arbitrary number of ranked result lists using RRF.
 
    Formula: score(chunk) = Σ_list  1 / (k + rank_in_list)
 
    Accepts *ranked_lists so callers can fuse two, three, or more lists
    without changing this function — Open/Closed compliance.
 
    Deduplication: chunks appearing in multiple lists have their individual
    bi-encoder scores merged onto one ScoredChunk object and their RRF
    contributions summed.
    """
    rrf_scores: dict[str, float] = {}
    chunk_registry: dict[str, ScoredChunk] = {}
 
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
 
            if cid not in chunk_registry:
                chunk_registry[cid] = item
            else:
                # Merge whichever scores are non-zero from this list
                existing = chunk_registry[cid]
                if item.bm25_score:
                    existing.bm25_score = item.bm25_score
                if item.vector_score:
                    existing.vector_score = item.vector_score
 
    for cid, score in rrf_scores.items():
        chunk_registry[cid].rrf_score = score
 
    return sorted(chunk_registry.values(), key=lambda x: x.rrf_score, reverse=True)
 
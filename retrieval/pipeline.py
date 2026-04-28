from __future__ import annotations
 
from time import perf_counter
 
import structlog
 
from config import settings
from models import RetrievalResult
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.vector_retriever import retrieve_vector
from retrieval.hyde import expand_with_hyde
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
 
log = structlog.get_logger()
 
 
def hybrid_retrieve(
    query: str,
    use_hyde: bool = False,
) -> RetrievalResult:
    """
    Full hybrid retrieval pipeline:
      1. (Optional) HyDE query expansion
      2. BM25 sparse retrieval — top-k by keyword score
      3. Vector dense retrieval — top-k by cosine similarity
      4. RRF fusion — merge and deduplicate both lists
      5. Cross-encoder rerank — re-score top candidates precisely
 
    Returns a RetrievalResult with per-stage latency recorded.
    """
    search_query = expand_with_hyde(query) if use_hyde else query
 
    t_retrieval_start = perf_counter()
    bm25_results = retrieve_bm25(search_query)
    vector_results = retrieve_vector(search_query)
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=settings.rrf_k)
    retrieval_ms = (perf_counter() - t_retrieval_start) * 1000
 
    candidate_pool = fused[: settings.rerank_top_k * settings.rerank_candidate_multiplier]
 
    t_rerank_start = perf_counter()
    final_chunks = rerank(query, candidate_pool)
    rerank_ms = (perf_counter() - t_rerank_start) * 1000
 
    log.info(
        "hybrid_retrieve_done",
        query=query[:60],
        use_hyde=use_hyde,
        n_bm25=len(bm25_results),
        n_vector=len(vector_results),
        n_fused=len(fused),
        n_final=len(final_chunks),
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=round(rerank_ms, 1),
    )
 
    return RetrievalResult(
        query=query,
        chunks=final_chunks,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
    )
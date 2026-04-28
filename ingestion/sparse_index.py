from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from config import settings
from core.interfaces import SparseIndex
from core.tokenizer import tokenize_for_bm25
from models import DocumentMetadata, ScoredChunk, Chunk

log = structlog.get_logger()

class BM25SparseIndex(SparseIndex):

    def __init__(self, index_path:str, chunks_path: str) -> None:
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize_for_bm25(c.text) for c in chunks]
        self._index = BM25Okapi(corpus)
        self._chunks = chunks
        log.info("bm25_index_built", n=len(chunks))

def save(self) -> None:
        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built yet.")
        with open(self._index_path, "wb") as f:
            pickle.dump(self._index, f)
        serialisable = [
            {
                "id": c.id,
                "text": c.text,
                "token_count": c.token_count,
                "metadata": c.metadata.model_dump(mode="json"),
            }
            for c in self._chunks
        ]
        self._chunks_path.write_text(
            json.dumps(serialisable, default=str), encoding="utf-8"
        )
        log.info("bm25_index_saved", path=str(self._index_path))
 
    def load(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self._index_path}. Run ingestion first."
            )
        with open(self._index_path, "rb") as f:
            self._index = pickle.load(f)
        raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(
                id=r["id"],
                text=r["text"],
                token_count=r["token_count"],
                metadata=DocumentMetadata(**r["metadata"]),
            )
            for r in raw
        ]
        log.info("bm25_index_loaded", n=len(self._chunks))
 
    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._index is None:
            raise RuntimeError("Index not loaded — call load() first.")
 
        tokens = tokenize_for_bm25(query)
        raw_scores: np.ndarray = self._index.get_scores(tokens)
 
        scored = sorted(
            zip(raw_scores.tolist(), self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
 
        if not scored:
            return []
 
        max_s, min_s = scored[0][0], scored[-1][0]
        score_range = max_s - min_s if max_s != min_s else 1.0
 
        return [
            ScoredChunk(
                chunk=chunk,
                bm25_score=(raw - min_s) / score_range,
            )
            for raw, chunk in scored
        ]
 
 
@lru_cache(maxsize=1)
def get_sparse_index() -> BM25SparseIndex:
    """
    Singleton sparse index — loaded from disk on first call.
    Subsequent calls return the already-loaded instance.
    """
    idx = BM25SparseIndex(
        index_path=settings.bm25_index_path,
        chunks_path=settings.bm25_chunks_path,
    )
    idx.load()
    return idx
 
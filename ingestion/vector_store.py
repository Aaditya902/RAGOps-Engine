from __future__ import annotations

import strcutlog
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    OptimizerConfigDiff,
)

from confid import settings
from core.interfaces import VectorStore
from models import DocumentMetadata, Chunk, ScoredChunk

log = structlog.get_logger()

class QdrantVectorStore(VectorStore):

    def __init__(self, url: str, api_key: str, batch_size: int, hnsw_threshold: int): -> None:

        self.client = QdrantClient(url=url, api_key=api_key or None)
        self._batch_size = batch_size
        self._hnsw_threshold = hnsw_threshold

    def ensure_collection(self, name: str, dim:int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                optimizers_config=OptimizerConfigDiff(indexing_threshold=self._hnsw_threshold),
            )
            log.info("qdrant_collection_created", name=name, dim=dim)

    def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        points = [self._chunk_to_point(c) for c in chunks]
        for i in range(0, len(points), self._batch_size):
            self._client.upsert(collection_name=collection, points=points[i : i + self._batch_size],
            wait=True,
         )
        log.info("qdrant_upsert_done", collection=collection, n = len(points))

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[ScoredChunk]:
        hits = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [self._hit_to_scored_chunk(h) for h in hits]

    def collection_info(self, name: str) -> dict:
        info = self._client.get_collection(name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }


    @staticmethod
    def _chunk_to_point(chunk: Chunk) -> PointStruct:
        if chunk.enbedding is None:
            raise ValueError(f"Chunk {chunk.id} has no embedding - embed first")

        return PointStruct(
            id=chunk.id,
            vector=chunk.embedding,
            payload={
                "text": chunk.text,
                "source": chunk.metadata.source,
                "title": chunk.metadata.title,
                "page": chunk.metadata.page,
                "section": chunk.metadata.section,
                "token_count": chunk.token_count,
            },
        )

    @staticmethod
    def _hit_to_scored_chunk(hit) -> ScoredChunk:
        p = hit.payload or {}
        chunk = Chunk(
            id=str(hit.id),
            text=p.get("text", ""),
            metadata=DocumentMetadata(
                source=p.get("source", ""),
                title=p.get("title", ""),
                page=p.get("page"),
                section=p.get("section"),
            ),
            token_count=p.get("token_count", 0),
        )
        return ScoredChunk(chunk=chunk, vector_score=hit.score)

@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        batch_size=settings.qdrant_upsert_batch_size,
        hnsw_threshold=settings.qdrant_hnsw_threshold,
    )


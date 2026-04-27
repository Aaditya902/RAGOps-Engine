from __future__ import annotations

from abc import ABC, abstractmethod
from models import chunk, ScoredChunk, IndexStats


class TextExtractor(ABC):

    def supports(self, suffix: str) -> bool:
        pass
    
    
    @abstractmethod
    def extract(self, path: str) -> list[tuple[str, dict]]:
        pass

class Embedder(ABC):

    @abstractmethod
    def embed_texts(self, textx: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, query: str) -> list[float]:
        pass

class VectorStore(ABC):

    @abstractmethod
    def ensure_collection(self, name: str, dim: int) -> None:
        pass

    @abstractmethod
    def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[ScoredChunk]:

class SparseIndex(ABC):

    @abstractmethod
    def build(self, chunks: list[chunks]) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        pass

class LLMCLient(ABC):

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str, int, int]:
        pass


    
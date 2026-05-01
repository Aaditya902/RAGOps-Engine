"""
models.py — Shared domain models.

SOLID — Interface Segregation Principle (ISP):
  Each model carries only the fields its consumers need.
  Scoring fields are split by concern: retrieval scores are on RetrievedChunk,
  generation outputs are on GenerationResponse — no model knows about all layers.

SOLID — Single Responsibility Principle (SRP):
  Models describe data shapes only. No business logic lives here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
import uuid


# ── Ingestion domain ───────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    source: str
    title: str = ""
    author: str = ""
    page: int | None = None
    section: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: DocumentMetadata
    token_count: int = 0
    embedding: list[float] | None = None


class IndexStats(BaseModel):
    total_chunks: int
    total_documents: int
    collection_name: str
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


# ── Retrieval domain ───────────────────────────────────────────────────────────

class ScoredChunk(BaseModel):
    """A chunk annotated with scores from one or more retrieval systems."""
    chunk: Chunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    query: str
    chunks: list[ScoredChunk]
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0


# ── Generation domain ──────────────────────────────────────────────────────────

class CitedSource(BaseModel):
    source_id: str
    source: str
    page: int | None = None
    excerpt: str = ""


class GenerationResult(BaseModel):
    answer: str
    sources: list[CitedSource]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    generation_ms: float = 0.0
    has_uncited_claims: bool = False


# ── API contract (what callers see) ───────────────────────────────────────────

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[CitedSource]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    model: str


# ── Evaluation domain ──────────────────────────────────────────────────────────

class EvalSample(BaseModel):
    question: str
    ground_truth: str = ""
    contexts: list[str] = Field(default_factory=list)
    answer: str = ""


class EvalMetrics(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_samples: int
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    def passed_ci(
        self,
        min_faithfulness: float,
        min_answer_relevancy: float,
        min_context_precision: float,
        min_context_recall: float,
    ) -> tuple[bool, list[str]]:
        """Returns (passed, list_of_failure_messages)."""
        failures: list[str] = []
        checks = [
            (self.faithfulness, min_faithfulness, "faithfulness"),
            (self.answer_relevancy, min_answer_relevancy, "answer_relevancy"),
            (self.context_precision, min_context_precision, "context_precision"),
            (self.context_recall, min_context_recall, "context_recall"),
        ]
        for actual, minimum, name in checks:
            if actual < minimum:
                failures.append(f"{name} {actual:.3f} < threshold {minimum}")
        return len(failures) == 0, failures
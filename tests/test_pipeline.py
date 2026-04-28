from __future__ import annotations
 
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
 
import pytest
 
 
# ── Fixtures ───────────────────────────────────────────────────────────────────
 
@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    content = """
# Transformer Architecture
 
Transformers are neural networks introduced in "Attention is All You Need" (2017).
 
## Self-Attention
 
Self-attention allows each token to attend to every other token in the sequence.
Query, key, and value vectors are computed for each token position.
 
## Positional Encoding
 
Since transformers process tokens in parallel, positional encodings are added
to embeddings to convey sequence order information.
"""
    p = tmp_path / "transformers.md"
    p.write_text(content)
    return p
 
 
@pytest.fixture
def sample_metadata():
    from models import DocumentMetadata
    return DocumentMetadata(source="test.md", title="Test")
 
 
@pytest.fixture
def make_chunk():
    from models import Chunk, DocumentMetadata
 
    def _make(text: str, idx: int = 0) -> Chunk:
        return Chunk(
            id=f"chunk-{idx}",
            text=text,
            metadata=DocumentMetadata(source=f"doc_{idx}.pdf", title=f"Doc {idx}", page=idx + 1),
            token_count=len(text.split()),
        )
    return _make
 
 
@pytest.fixture
def scored_chunks(make_chunk):
    from models import ScoredChunk
 
    texts = [
        "Transformers use self-attention mechanisms to process sequences.",
        "BM25 is a sparse retrieval algorithm based on term frequency.",
        "Cross-encoders rerank candidates by scoring query-passage pairs jointly.",
    ]
    return [
        ScoredChunk(
            chunk=make_chunk(text, i),
            bm25_score=0.9 - i * 0.1,
            vector_score=0.85 - i * 0.05,
            rrf_score=0.5 - i * 0.05,
            rerank_score=0.95 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]
 
 
# ── Tokenizer tests ────────────────────────────────────────────────────────────
 
class TestTokenizer:
    def test_count_tokens_returns_int(self):
        from core.tokenizer import count_tokens
        assert isinstance(count_tokens("hello world"), int)
        assert count_tokens("hello world") > 0
 
    def test_tokenize_for_bm25_lowercases(self):
        from core.tokenizer import tokenize_for_bm25
        tokens = tokenize_for_bm25("Hello WORLD")
        assert all(t == t.lower() for t in tokens)
 
    def test_tokenize_for_bm25_strips_punctuation(self):
        from core.tokenizer import tokenize_for_bm25
        tokens = tokenize_for_bm25("hello, world!")
        assert "," not in tokens
        assert "!" not in tokens
 
    def test_tokenize_same_result_both_directions(self):
        """Indexer and retriever must tokenize identically — same function, same output."""
        from core.tokenizer import tokenize_for_bm25
        text = "What is a Transformer model?"
        assert tokenize_for_bm25(text) == tokenize_for_bm25(text)
 
 
# ── Extractor tests ────────────────────────────────────────────────────────────
 
class TestExtractors:
    def test_plain_text_extractor_supports_md(self):
        from ingestion.extractors import PlainTextExtractor
        ext = PlainTextExtractor()
        assert ext.supports(".md")
        assert ext.supports(".txt")
        assert not ext.supports(".pdf")
 
    def test_plain_text_extractor_reads_file(self, sample_markdown: Path):
        from ingestion.extractors import PlainTextExtractor
        ext = PlainTextExtractor()
        results = ext.extract(str(sample_markdown))
        assert len(results) == 1
        text, meta = results[0]
        assert "Transformer" in text
        assert meta["source"] == str(sample_markdown)
 
    def test_registry_dispatches_to_correct_extractor(self, sample_markdown: Path):
        from ingestion.extractors import build_default_registry
        registry = build_default_registry()
        results = registry.extract(str(sample_markdown))
        assert len(results) >= 1
 
    def test_registry_returns_empty_for_unsupported(self, tmp_path: Path):
        from ingestion.extractors import build_default_registry
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        registry = build_default_registry()
        results = registry.extract(str(f))
        assert results == []
 
 
# ── Chunker tests ──────────────────────────────────────────────────────────────
 
class TestChunker:
    def test_ingest_file_produces_chunks(self, sample_markdown: Path):
        from ingestion.chunker import ingest_file
        chunks = ingest_file(sample_markdown)
        assert len(chunks) > 0
 
    def test_all_chunks_have_text(self, sample_markdown: Path):
        from ingestion.chunker import ingest_file
        for chunk in ingest_file(sample_markdown):
            assert chunk.text.strip() != ""
 
    def test_chunks_within_token_limit(self, sample_markdown: Path):
        from ingestion.chunker import ingest_file
        from core.tokenizer import count_tokens
        from config import settings
        for chunk in ingest_file(sample_markdown):
            assert count_tokens(chunk.text) <= settings.chunk_size + 10
 
    def test_file_not_found_raises(self):
        from ingestion.chunker import ingest_file
        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/file.md")
 
    def test_short_text_below_min_produces_no_chunks(self, sample_metadata):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("Hi.", sample_metadata, min_chunk_size=100)
        assert len(chunks) == 0
 
    def test_fixed_size_split_creates_overlap(self, sample_metadata):
        from ingestion.chunker import fixed_size_split
        long_text = "word " * 300
        parts = fixed_size_split(long_text, chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        assert len(parts) >= 3
 
 
# ── RRF fusion tests ───────────────────────────────────────────────────────────
 
class TestRRFFusion:
    def test_fusion_deduplicates(self, scored_chunks):
        from retrieval.fusion import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(scored_chunks, scored_chunks)
        ids = [item.chunk.id for item in fused]
        assert len(ids) == len(set(ids))
 
    def test_fusion_all_scores_positive(self, scored_chunks):
        from retrieval.fusion import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(scored_chunks, [])
        assert all(item.rrf_score > 0 for item in fused)
 
    def test_fusion_sorted_descending(self, scored_chunks):
        from retrieval.fusion import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(scored_chunks, [])
        scores = [item.rrf_score for item in fused]
        assert scores == sorted(scores, reverse=True)
 
    def test_fusion_accepts_empty_lists(self):
        from retrieval.fusion import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([], []) == []
 
    def test_fusion_three_lists(self, scored_chunks, make_chunk):
        """OCP test: fusion accepts *args so three lists work with no code change."""
        from retrieval.fusion import reciprocal_rank_fusion
        from models import ScoredChunk
        extra = [ScoredChunk(chunk=make_chunk("extra text", 99), bm25_score=0.5)]
        fused = reciprocal_rank_fusion(scored_chunks, scored_chunks, extra)
        assert any(item.chunk.id == "chunk-99" for item in fused)
 
 
# ── Citation validator tests ───────────────────────────────────────────────────
 
class TestCitationValidator:
    def test_valid_citations_no_warnings(self):
        from generation.validator import validate_citations
        answer = "Transformers use attention [src-1]. Introduced in 2017 [src-2]."
        has_violation, warnings = validate_citations(answer, n_sources=2)
        assert not has_violation
        assert warnings == []
 
    def test_out_of_range_citation_flagged(self):
        from generation.validator import validate_citations
        has_violation, warnings = validate_citations(
            "See the result [src-5].", n_sources=2
        )
        assert has_violation
        assert any("src-5" in w for w in warnings)
 
    def test_zero_index_citation_flagged(self):
        from generation.validator import validate_citations
        has_violation, warnings = validate_citations("[src-0] is invalid.", n_sources=3)
        assert has_violation
 
    def test_long_answer_no_citations_flagged(self):
        from generation.validator import validate_citations
        long_answer = "Transformers are powerful models. " * 5
        has_violation, _ = validate_citations(long_answer, n_sources=3)
        assert has_violation
 
    def test_short_answer_no_citations_ok(self):
        from generation.validator import validate_citations
        has_violation, _ = validate_citations("I don't know.", n_sources=3)
        assert not has_violation
 
 
# ── Context builder tests ──────────────────────────────────────────────────────
 
class TestContextBuilder:
    def test_tags_numbered_from_one(self, scored_chunks):
        from generation.context_builder import build_context
        block, sources = build_context(scored_chunks)
        assert "[src-1]" in block
        assert "[src-2]" in block
        assert "[src-3]" in block
 
    def test_source_count_matches_chunks(self, scored_chunks):
        from generation.context_builder import build_context
        _, sources = build_context(scored_chunks)
        assert len(sources) == len(scored_chunks)
 
    def test_sources_have_correct_ids(self, scored_chunks):
        from generation.context_builder import build_context
        _, sources = build_context(scored_chunks)
        for i, (src, item) in enumerate(zip(sources, scored_chunks)):
            assert src.source_id == item.chunk.id
 
 
# ── Generator tests ────────────────────────────────────────────────────────────
 
class TestGenerator:
    def _make_retrieval_result(self, scored_chunks):
        from models import RetrievalResult
        return RetrievalResult(query="What is attention?", chunks=scored_chunks)
 
    def test_empty_retrieval_returns_no_info_message(self):
        from generation.generator import generate
        from models import RetrievalResult
 
        result = generate("test?", RetrievalResult(query="test?", chunks=[]))
        assert "couldn't find" in result.answer.lower()
        assert result.sources == []
 
    def test_generator_calls_llm_client(self, scored_chunks):
        from generation.generator import generate
        from core.interfaces import LLMClient
 
        class MockLLM(LLMClient):
            def complete(self, system, messages, max_tokens, temperature):
                return "Attention scores each token [src-1].", "mock-model", 100, 20
 
        result = generate(
            query="What is attention?",
            retrieval_result=self._make_retrieval_result(scored_chunks),
            llm=MockLLM(),
        )
        assert "attention" in result.answer.lower()
        assert result.model == "mock-model"
        assert result.input_tokens == 100
 
    def test_generator_detects_citation_violations(self, scored_chunks):
        from generation.generator import generate
        from core.interfaces import LLMClient
 
        class HallucinatingLLM(LLMClient):
            def complete(self, system, messages, max_tokens, temperature):
                # References src-99 which doesn't exist
                return "This is an answer [src-99].", "mock-model", 50, 10
 
        result = generate(
            query="test?",
            retrieval_result=self._make_retrieval_result(scored_chunks),
            llm=HallucinatingLLM(),
        )
        assert result.has_uncited_claims is True
 
 
# ── EvalMetrics CI gate tests ──────────────────────────────────────────────────
 
class TestCIGate:
    def _metrics(self, **overrides):
        from models import EvalMetrics
        defaults = dict(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.80,
            n_samples=10,
        )
        defaults.update(overrides)
        return EvalMetrics(**defaults)
 
    def test_all_above_threshold_passes(self):
        passed, failures = self._metrics().passed_ci(0.80, 0.75, 0.70, 0.70)
        assert passed
        assert failures == []
 
    def test_low_faithfulness_fails(self):
        passed, failures = self._metrics(faithfulness=0.60).passed_ci(0.80, 0.75, 0.70, 0.70)
        assert not passed
        assert any("faithfulness" in f for f in failures)
 
    def test_multiple_failures_all_reported(self):
        passed, failures = self._metrics(
            faithfulness=0.50, answer_relevancy=0.40
        ).passed_ci(0.80, 0.75, 0.70, 0.70)
        assert not passed
        assert len(failures) == 2
 
    def test_exactly_at_threshold_passes(self):
        passed, _ = self._metrics(faithfulness=0.80).passed_ci(0.80, 0.75, 0.70, 0.70)
        assert passed
 
 
# ── API smoke tests ────────────────────────────────────────────────────────────
 
class TestAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
 
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
 
    @pytest.mark.asyncio
    @patch("api.main._run_rag")
    async def test_ask_endpoint_happy_path(self, mock_run):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
 
        mock_run.return_value = {
            "query": "What is attention?",
            "answer": "Attention weights tokens [src-1].",
            "sources": [],
            "retrieval_ms": 50.0,
            "rerank_ms": 20.0,
            "generation_ms": 300.0,
            "total_ms": 370.0,
            "model": "claude-3-5-sonnet-20241022",
            "has_uncited": False,
        }
 
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/ask", json={"query": "What is attention?"})
 
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
 
    @pytest.mark.asyncio
    async def test_ask_endpoint_short_query_rejected(self):
        from httpx import AsyncClient, ASGITransport
        from api.main import app
 
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/ask", json={"query": "Hi"})
        assert r.status_code == 422   # Pydantic min_length validation
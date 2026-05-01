"""
api/main.py — FastAPI application.

12-Factor compliance:
  Factor II  (Dependencies): all deps declared in requirements.txt
  Factor III (Config): zero config in this file — all from settings
  Factor VI  (Processes): stateless request handlers; state in Qdrant/BM25 files
  Factor VII (Port binding): host/port from env via settings
  Factor XI  (Logs): JSON logs to stdout via configure_logging()

SOLID — Single Responsibility Principle (SRP):
  Route handlers are thin: validate input → call pipeline → return response.
  Business logic lives in ingestion/, retrieval/, generation/.

KISS Principle:
  No custom middleware beyond CORS.
  `asyncio.to_thread` handles blocking I/O without a separate worker process.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from config import settings
from core.logging import configure_logging
from models import AskResponse, IndexStats

configure_logging()
log = structlog.get_logger()

# ── Prometheus metrics (Factor VIII: observability as a service) ───────────────
REQUEST_COUNT = Counter("rag_requests_total", "Total RAG queries", ["status"])
LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end RAG latency",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10],
)
CITATION_FAILURES = Counter("rag_citation_failures_total", "Responses with uncited claims")


# ── Startup: warm models into memory ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup_warming_models")
    await asyncio.to_thread(_warm_models)
    log.info("startup_complete")
    yield
    log.info("shutdown_complete")


def _warm_models() -> None:
    """
    Pre-load the embedding model and cross-encoder into memory.
    Called once at startup so the first request isn't slow.
    """
    from ingestion.embedder import get_embedder
    from retrieval.reranker import _load_cross_encoder

    get_embedder().embed_texts(["warmup"])
    _load_cross_encoder(settings.rerank_model)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ask My Docs",
    description="Domain-specific RAG — hybrid retrieval, cross-encoder reranking, citation enforcement",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI from /ui directory
_UI_DIR = Path(__file__).parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR)), name="ui")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the UI."""
    index = _UI_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Ask My Docs API", "docs": "/docs", "ui": "/ui/index.html"}


# ── Request models ─────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a file or directory to ingest")
    glob: str = Field("**/*", description="Glob pattern for directory ingestion")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    use_hyde: bool = Field(False, description="Enable HyDE query expansion (slower, higher recall)")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior turns: [{role: user|assistant, content: ...}]",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
async def stats():
    """Qdrant collection statistics."""
    try:
        from ingestion.vector_store import get_vector_store
        info = get_vector_store().collection_info(settings.qdrant_collection)
        return {"collection": settings.qdrant_collection, **info}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/ingest", response_model=IndexStats)
async def ingest(request: IngestRequest):
    """Ingest a file or directory into the dual index."""
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    try:
        return await asyncio.to_thread(_run_ingestion, path, request.glob)
    except Exception as exc:
        log.error("ingest_failed", path=str(path), error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Query the RAG pipeline and return a cited answer."""
    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(_run_rag, request)
        REQUEST_COUNT.labels(status="success").inc()
        LATENCY.observe(time.perf_counter() - t0)
        if result.get("has_uncited"):
            CITATION_FAILURES.inc()
        return result
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        log.error("ask_failed", query=request.query[:60], error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── Pipeline runners (sync, called via to_thread) ──────────────────────────────
def _run_ingestion(path: Path, glob: str) -> IndexStats:
    from ingestion.chunker import ingest_file, ingest_directory
    from ingestion.embedder import embed_chunks
    from ingestion.indexer import index_chunks

    chunks = ingest_file(path) if path.is_file() else ingest_directory(path, glob)
    chunks = embed_chunks(chunks)
    return index_chunks(chunks)


def _run_rag(request: AskRequest) -> dict:
    from retrieval.pipeline import hybrid_retrieve
    from generation.generator import generate

    t0 = time.perf_counter()
    retrieval = hybrid_retrieve(request.query, use_hyde=request.use_hyde)
    result = generate(
        query=request.query,
        retrieval_result=retrieval,
        conversation_history=request.conversation_history or [],
    )
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": request.query,
        "answer": result.answer,
        "sources": [s.model_dump() for s in result.sources],
        "retrieval_ms": retrieval.retrieval_ms,
        "rerank_ms": retrieval.rerank_ms,
        "generation_ms": result.generation_ms,
        "total_ms": total_ms,
        "model": result.model,
        "has_uncited": result.has_uncited_claims,
    }


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
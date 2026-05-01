"""
config.py — Single source of truth for all application settings.

12-Factor App Factor III: Config
  - Every value comes from an environment variable or .env file.
  - No defaults that differ between environments (dev/staging/prod).
  - Secrets (API keys) are required — startup fails loudly if missing.
  - All path-like settings are configurable, not hardcoded.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Primary provider is Google Gemini. GEMINI_API_KEY is required.
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    llm_model: str = Field("gemini-2.0-flash", env="LLM_MODEL")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(0.1, env="LLM_TEMPERATURE")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = Field("BAAI/bge-large-en-v1.5", env="EMBED_MODEL")
    embed_dim: int = Field(1024, env="EMBED_DIM")
    embed_batch_size: int = Field(32, env="EMBED_BATCH_SIZE")
    embed_query_prefix: str = Field(
        "Represent this sentence for searching relevant passages: ",
        env="EMBED_QUERY_PREFIX",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: str = Field("", env="QDRANT_API_KEY")
    qdrant_collection: str = Field("ask_my_docs", env="QDRANT_COLLECTION")
    qdrant_upsert_batch_size: int = Field(256, env="QDRANT_UPSERT_BATCH_SIZE")
    qdrant_hnsw_threshold: int = Field(20_000, env="QDRANT_HNSW_THRESHOLD")

    # ── BM25 persistence (12-Factor: path comes from env, not hardcoded) ──────
    bm25_index_path: str = Field("bm25_index.pkl", env="BM25_INDEX_PATH")
    bm25_chunks_path: str = Field("bm25_chunks.json", env="BM25_CHUNKS_PATH")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    bm25_top_k: int = Field(20, env="BM25_TOP_K")
    vector_top_k: int = Field(20, env="VECTOR_TOP_K")
    rrf_k: int = Field(60, env="RRF_K")
    rerank_top_k: int = Field(5, env="RERANK_TOP_K")
    rerank_candidate_multiplier: int = Field(4, env="RERANK_CANDIDATE_MULTIPLIER")
    rerank_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANK_MODEL"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(100, env="MIN_CHUNK_SIZE")

    # ── Eval CI thresholds ────────────────────────────────────────────────────
    min_faithfulness: float = Field(0.80, env="MIN_FAITHFULNESS")
    min_answer_relevancy: float = Field(0.75, env="MIN_ANSWER_RELEVANCY")
    min_context_precision: float = Field(0.70, env="MIN_CONTEXT_PRECISION")
    min_context_recall: float = Field(0.70, env="MIN_CONTEXT_RECALL")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
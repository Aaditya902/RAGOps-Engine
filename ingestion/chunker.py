"""
ingestion/chunker.py — Converts raw text into overlapping token-bounded chunks.

SOLID — Single Responsibility Principle (SRP):
  This module does one thing: turn (text, metadata) into Chunk objects.
  File I/O is delegated to extractors. Token counting is delegated to
  core/tokenizer. No embedding, no indexing.

KISS Principle:
  Two clear passes — semantic split, then fixed-size if still too large.
  No complex state machines, no inheritance hierarchy.

DRY Principle:
  Token counting and BM25 tokenization both live in core/tokenizer.
  Not duplicated here.
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from core.tokenizer import count_tokens, tokenize_for_bm25
from config import settings
from models import Chunk, DocumentMetadata
from ingestion.extractors import ExtractorRegistry, build_default_registry

log = structlog.get_logger()

# Semantic split: markdown headings, paragraph breaks, sentence boundaries
_SEMANTIC_SPLIT_RE = re.compile(
    r"(\n#{1,6}\s.+\n)"
    r"|(\n\n+)"
    r"|(?<=\.)\s{2,}(?=[A-Z])",
    re.MULTILINE,
)


# ── Splitting logic ────────────────────────────────────────────────────────────

def split_into_semantic_segments(text: str) -> list[str]:
    """Split text on semantic boundaries. Returns non-empty stripped segments."""
    parts = _SEMANTIC_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def fixed_size_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> list[str]:
    """
    Sliding-window token-based split for text that exceeds chunk_size.
    Returns decoded text strings (not token IDs).
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    result: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        segment_tokens = tokens[start:end]
        if len(segment_tokens) >= min_chunk_size:
            result.append(enc.decode(segment_tokens))
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap

    return result


# ── Main chunking function ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    metadata: DocumentMetadata,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    min_chunk_size: int = settings.min_chunk_size,
) -> list[Chunk]:
    """
    Two-pass chunking:
      Pass 1 — merge semantic segments until we approach chunk_size.
      Pass 2 — fixed split for any segment that's still too large.
    """
    segments = split_into_semantic_segments(text)
    chunks: list[Chunk] = []
    buffer = ""

    for segment in segments:
        candidate = f"{buffer} {segment}".strip() if buffer else segment.strip()

        if count_tokens(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer and count_tokens(buffer) >= min_chunk_size:
                chunks.append(_make_chunk(buffer, metadata))

            if count_tokens(segment) > chunk_size:
                for sub in fixed_size_split(segment, chunk_size, chunk_overlap, min_chunk_size):
                    chunks.append(_make_chunk(sub, metadata))
                buffer = ""
            else:
                buffer = segment

    if buffer and count_tokens(buffer) >= min_chunk_size:
        chunks.append(_make_chunk(buffer, metadata))

    log.debug("text_chunked", source=metadata.source, n_chunks=len(chunks))
    return chunks


def _make_chunk(text: str, metadata: DocumentMetadata) -> Chunk:
    return Chunk(text=text, metadata=metadata, token_count=count_tokens(text))


# ── File and directory entry points ───────────────────────────────────────────

def ingest_file(
    path: str | Path,
    registry: ExtractorRegistry | None = None,
) -> list[Chunk]:
    """Extract and chunk a single file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    registry = registry or build_default_registry()
    all_chunks: list[Chunk] = []

    for text, meta_dict in registry.extract(str(path)):
        metadata = DocumentMetadata(**meta_dict)
        all_chunks.extend(chunk_text(text, metadata))

    log.info("file_ingested", path=str(path), n_chunks=len(all_chunks))
    return all_chunks


def ingest_directory(
    directory: str | Path,
    glob: str = "**/*",
    registry: ExtractorRegistry | None = None,
) -> list[Chunk]:
    """Recursively extract and chunk all supported files in a directory."""
    directory = Path(directory)
    registry = registry or build_default_registry()
    supported = {".pdf", ".docx", ".md", ".txt", ".rst"}
    all_chunks: list[Chunk] = []

    for file in sorted(directory.glob(glob)):
        if file.suffix.lower() in supported and file.is_file():
            try:
                all_chunks.extend(ingest_file(file, registry))
            except Exception as exc:
                log.error("file_ingest_failed", path=str(file), error=str(exc))

    log.info("directory_ingested", directory=str(directory), n_chunks=len(all_chunks))
    return all_chunks
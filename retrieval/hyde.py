"""
retrieval/hyde.py — Hypothetical Document Embedding (HyDE) query expansion.

SOLID — Single Responsibility Principle (SRP):
  One public function. One clear contract: str → str.

SOLID — Dependency Inversion Principle (DIP):
  Uses get_llm_client() (which now returns GeminiLLMClient) via the
  LLMClient interface. Previously directly instantiated anthropic.Anthropic —
  that hardcoded dependency is removed.
"""
from __future__ import annotations

import structlog

from config import settings
from generation.llm_client import get_llm_client

log = structlog.get_logger()

_HYDE_SYSTEM = "You are a helpful assistant that writes short factual passages."
_HYDE_PROMPT = (
    "Write a short, factual passage (2-3 sentences) that directly answers "
    "the following question. Output only the passage — no preamble, no labels.\n\n"
    "Question: {query}"
)


def expand_with_hyde(query: str) -> str:
    """
    Generate a hypothetical answer passage and return it as the search query.

    Why this helps: a hypothetical answer lives closer to real answer chunks
    in embedding space than the raw question does — improving recall on
    abstract or multi-hop questions.
    """
    client = get_llm_client()
    hypothetical, _, _, _ = client.complete(
        system=_HYDE_SYSTEM,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        max_tokens=200,
        temperature=0.2,
    )
    log.debug("hyde_expanded", original=query[:60], hypothetical=hypothetical[:80])
    return hypothetical
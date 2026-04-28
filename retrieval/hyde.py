from __future__ import annotations
 
import anthropic
import structlog
 
from config import settings
 
log = structlog.get_logger()
 
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
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=200,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
    )
    hypothetical = response.content[0].text.strip()
    log.debug("hyde_expanded", original=query[:60], hypothetical=hypothetical[:80])
    return hypothetical
 
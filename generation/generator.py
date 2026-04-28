from __future__ import annotations
 
from time import perf_counter
 
import structlog
 
from config import settings
from core.interfaces import LLMClient
from generation.context_builder import build_context
from generation.llm_client import get_llm_client
from generation.validator import validate_citations
from models import GenerationResult, RetrievalResult
 
log = structlog.get_logger()
 
_SYSTEM_PROMPT = """\
You are a precise, factual assistant that answers questions ONLY using the \
provided context passages.
 
Rules you MUST follow:
1. Every factual claim must be followed by a citation: [src-N] where N matches \
the source number in the context block.
2. If the context does not contain enough information, respond with: \
"I don't have enough information in the provided documents to answer this." \
Do NOT use external knowledge or guess.
3. You may synthesise across multiple sources — cite all relevant ones.
4. Keep answers concise. Use bullet points only when listing 3 or more items.
5. Never reproduce large verbatim passages — paraphrase and cite.
"""
 
_CONTEXT_TEMPLATE = """\
--- Context ---
{context_block}
--- End Context ---
 
Question: {question}
"""
 
 
def generate(
    query: str,
    retrieval_result: RetrievalResult,
    conversation_history: list[dict[str, str]] | None = None,
    llm: LLMClient | None = None,
) -> GenerationResult:
    """
    Full generation pipeline:
      1. Build numbered context block from retrieved chunks.
      2. Assemble message history with context-injected user turn.
      3. Call LLM with citation-enforcing system prompt.
      4. Validate citations in the response.
      5. Return structured GenerationResult.
 
    Args:
        query: the user's question.
        retrieval_result: output from the retrieval pipeline.
        conversation_history: prior turns for multi-turn support.
        llm: injectable LLMClient — defaults to the Anthropic singleton.
    """
    chunks = retrieval_result.chunks
 
    if not chunks:
        return GenerationResult(
            answer="I couldn't find relevant information in the documents to answer your question.",
            sources=[],
            model=settings.llm_model,
        )
 
    context_block, sources = build_context(chunks)
    user_message = _CONTEXT_TEMPLATE.format(
        context_block=context_block,
        question=query,
    )
 
    history = list(conversation_history or [])
    history.append({"role": "user", "content": user_message})
 
    client = llm or get_llm_client()
 
    t0 = perf_counter()
    answer, model_name, input_tokens, output_tokens = client.complete(
        system=_SYSTEM_PROMPT,
        messages=history,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    generation_ms = (perf_counter() - t0) * 1000
 
    has_violation, warnings = validate_citations(answer, len(sources))
    if warnings:
        log.warning("citation_violations", warnings=warnings, query=query[:60])
 
    log.info(
        "generation_done",
        query=query[:60],
        n_sources=len(sources),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generation_ms=round(generation_ms, 1),
        has_uncited_claims=has_violation,
    )
 
    return GenerationResult(
        answer=answer,
        sources=sources,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generation_ms=generation_ms,
        has_uncited_claims=has_violation,
    )
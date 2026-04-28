from __future__ import annotations
 
from models import CitedSource, ScoredChunk
 
# The citation tag format — defined once, used everywhere
CITATION_TAG = "[src-{n}]"
 
 
def build_context(
    chunks: list[ScoredChunk],
) -> tuple[str, list[CitedSource]]:
    """
    Convert retrieved chunks into a numbered context block and a source list.
 
    Returns:
        context_text: the block injected between the system prompt and question
        sources: structured CitedSource objects for the API response
    """
    lines: list[str] = []
    sources: list[CitedSource] = []
 
    for i, item in enumerate(chunks, start=1):
        tag = CITATION_TAG.format(n=i)
        lines.append(f"{tag} {item.chunk.text}")
        sources.append(
            CitedSource(
                source_id=item.chunk.id,
                source=item.chunk.metadata.source,
                page=item.chunk.metadata.page,
                excerpt=item.chunk.text[:200],
            )
        )
 
    return "\n\n".join(lines), sources
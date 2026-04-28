from __future__ import annotations
 
import re
 
# Matches [src-N] where N is one or two digits
_CITATION_RE = re.compile(r"\[src-(\d+)\]")
 
 
def validate_citations(
    answer: str,
    n_sources: int,
) -> tuple[bool, list[str]]:
    """
    Check citation validity in a generated answer.
 
    Returns:
        (has_violation, warnings)
 
    A violation is raised when:
      - The answer references [src-N] with N outside [1, n_sources]
      - The answer is longer than 100 chars but contains zero citations
    """
    warnings: list[str] = []
 
    cited_ns = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
 
    for n in cited_ns:
        if n < 1 or n > n_sources:
            warnings.append(
                f"Hallucinated citation [src-{n}]: only {n_sources} sources provided."
            )
 
    if not cited_ns and len(answer) > 100:
        warnings.append("Answer has no citations — claims may be unsupported by context.")
 
    return len(warnings) > 0, warnings
 
from __future__ import annotations

import re
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))

def tokenize_for_bm25(text: str) -> list[str]:
    # Simple whitespace and punctuation tokenizer
    return re.findall(r'\b\w+\b', text.lower())
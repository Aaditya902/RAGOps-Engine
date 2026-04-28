from __future__ import annotations
 
from functools import lru_cache
 
import anthropic
 
from config import settings
from core.interfaces import LLMClient
 
 
class AnthropicLLMClient(LLMClient):
    """LLMClient backed by the Anthropic Messages API."""
 
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
 
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str, int, int]:
        """
        Returns (answer_text, model_name, input_tokens, output_tokens).
        Raises anthropic.APIError on failure — let it propagate to the caller.
        """
        response = self._client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = response.content[0].text.strip()
        return text, response.model, response.usage.input_tokens, response.usage.output_tokens
 
 
@lru_cache(maxsize=1)
def get_llm_client() -> AnthropicLLMClient:
    """Singleton LLM client — one Anthropic connection per process."""
    return AnthropicLLMClient(api_key=settings.anthropic_api_key)
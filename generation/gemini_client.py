"""
generation/gemini_client.py — Google Gemini implementation of the LLMClient port.

SOLID — Dependency Inversion Principle (DIP):
  Implements core/interfaces.LLMClient exactly as AnthropicLLMClient did.
  The generator, HyDE, and evaluator all depend on the interface —
  zero changes were needed in those modules to swap providers.

SOLID — Single Responsibility Principle (SRP):
  This class handles only the Gemini API call.
  Token counting uses Gemini's native count_tokens method for accuracy.

Note on system prompt:
  Gemini's generate_content API accepts a system_instruction parameter
  separate from the message history — equivalent to Anthropic's `system`.
"""
from __future__ import annotations

from functools import lru_cache

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from config import settings
from core.interfaces import LLMClient


class GeminiLLMClient(LLMClient):
    """LLMClient backed by the Google Gemini API."""

    def __init__(self, api_key: str, model_name: str) -> None:
        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, str, int, int]:
        """
        Returns (answer_text, model_name, input_tokens, output_tokens).

        Gemini history format uses 'user'/'model' roles (not 'assistant').
        We convert on the fly so the rest of the codebase stays role-agnostic.
        """
        # Convert OpenAI-style role names → Gemini role names
        gemini_history = [
            {
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [msg["content"]],
            }
            for msg in messages[:-1]   # all turns except the last
        ]

        # The last message is the current user turn — passed as the prompt
        current_prompt = messages[-1]["content"]

        model_with_system = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system,
        )

        chat = model_with_system.start_chat(history=gemini_history)
        response = chat.send_message(
            current_prompt,
            generation_config=GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        answer = response.text.strip()

        # Gemini returns token counts in usage_metadata
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return answer, self._model_name, input_tokens, output_tokens


@lru_cache(maxsize=1)
def get_llm_client() -> GeminiLLMClient:
    """Singleton Gemini client — configured once per process."""
    return GeminiLLMClient(
        api_key=settings.gemini_api_key,
        model_name=settings.llm_model,
    )
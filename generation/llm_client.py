"""
generation/llm_client.py — Provider selector (re-export shim).

OCP / DIP: The generator, hyde.py, and evaluator all import `get_llm_client`
from this module. Switching LLM providers only requires changing this file —
zero changes to any caller.

Current provider: Google Gemini  (GEMINI_API_KEY required in .env).
"""
from generation.gemini_client import GeminiLLMClient as LLMClientImpl, get_llm_client

__all__ = ["LLMClientImpl", "get_llm_client"]
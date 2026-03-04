"""
LLM provider abstraction.

Wraps OllamaLLM with caching, streaming support, and error handling.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from langchain_ollama import OllamaLLM

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_llm(model: str | None = None) -> OllamaLLM:
    """
    Return a cached LLM instance.

    Uses LRU cache to avoid re-creating the connection for the same model.
    """
    model = model or settings.default_model
    logger.info("Initializing LLM: %s", model)
    return OllamaLLM(
        model=model,
        base_url=settings.ollama_base_url,
    )


def stream_response(model: str, prompt: str) -> Iterator[str]:
    """
    Stream tokens from the LLM one at a time.

    Yields individual string tokens as they arrive from Ollama.
    """
    llm = get_llm(model)
    try:
        for chunk in llm.stream(prompt):
            yield chunk
    except Exception as exc:
        logger.error("Streaming error: %s", exc)
        yield f"\n\n⚠️ Error generating response: {exc}"

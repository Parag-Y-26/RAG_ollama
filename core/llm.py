"""
LLM provider abstraction - supports Ollama (local) and cloud providers.

Provider resolution order:
    1. Explicit `provider` argument
    2. settings.llm_provider
    3. Falls back to "ollama"

Supported providers:
    - "ollama"    - local, via OllamaLLM
    - "openai"    - OpenAI GPT models
    - "anthropic" - Anthropic Claude models
    - "groq"      - Groq fast inference
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def get_llm(model: str | None = None, provider: str | None = None) -> Any:
    """
    Return a cached LangChain LLM instance for the given provider and model.

    Args:
        model: Model name. Defaults to settings.default_model.
        provider: One of "ollama", "openai", "anthropic", "groq".
            Defaults to settings.llm_provider.

    Returns:
        A LangChain BaseLLM or BaseChatModel instance.

    Raises:
        ValueError: If the provider is unknown or its API key is missing.
    """
    provider = (provider or settings.llm_provider or "ollama").lower()
    model = model or settings.default_model

    logger.info("Initializing LLM - provider: %s | model: %s", provider, model)

    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=model, base_url=settings.ollama_base_url)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to .streamlit/secrets.toml or your environment."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=settings.openai_api_key)
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .streamlit/secrets.toml or your environment."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key)
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to .streamlit/secrets.toml or your environment."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=settings.groq_api_key)

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        "Choose one of: ollama, openai, anthropic, groq"
    )


def get_available_cloud_models(provider: str) -> list[str]:
    """
    Return a curated list of recommended models for a cloud provider.
    These are static; cloud providers do not expose a list endpoint.
    """
    cloud_models: dict[str, list[str]] = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
    }
    return cloud_models.get(provider, [])

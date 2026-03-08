"""
LLM provider abstraction - supports Ollama (local), Ollama Cloud, and cloud providers.

Provider resolution order:
    1. Explicit `provider` argument
    2. settings.llm_provider
    3. Falls back to "ollama"

Supported providers:
    - "ollama"       - local, via OllamaLLM
    - "ollama_cloud" - Ollama Cloud (OpenAI-compatible), via ChatOpenAI
    - "openai"       - OpenAI GPT models
    - "anthropic"    - Anthropic Claude models
    - "groq"         - Groq fast inference
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
        provider: One of "ollama", "ollama_cloud", "openai", "anthropic",
            "groq", "nvidia". Defaults to settings.llm_provider.

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

    if provider == "ollama_cloud":
        if not settings.ollama_cloud_api_key:
            raise ValueError(
                "Ollama Cloud API key is not set. "
                "Create one at https://ollama.com/settings/keys "
                "then add OLLAMA_API_KEY to .streamlit/secrets.toml"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.ollama_cloud_api_key,
            base_url=settings.ollama_cloud_base_url,
            streaming=True,
            temperature=0.7,
        )

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

    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        # Self-hosted NIM container (local GPU, no real key needed)
        if settings.nvidia_is_self_hosted:
            return ChatNVIDIA(
                model=model,
                base_url=settings.nvidia_nim_base_url,
                nvidia_api_key="no-key",
            )

        # Hosted NVIDIA API Catalog
        if not settings.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Get a free key at https://build.nvidia.com/explore "
                "and add it to .streamlit/secrets.toml.\n"
                "Keys always start with 'nvapi-'."
            )
        return ChatNVIDIA(
            model=model,
            api_key=settings.nvidia_api_key,
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        "Choose one of: ollama, ollama_cloud, openai, anthropic, groq, nvidia"
    )


def get_available_cloud_models(provider: str) -> list[str]:
    """
    Return model names for a cloud provider.
    NVIDIA, OpenAI, and Groq use live discovery; Anthropic uses an expanded
    static list. All live-fetch providers fall back to static lists on error.
    """
    if provider == "nvidia":
        return get_available_nvidia_models()

    if provider == "openai":
        return _fetch_openai_models()

    if provider == "groq":
        return _fetch_groq_models()

    # Providers with static lists (no public model-list API)
    cloud_models: dict[str, list[str]] = {
        "anthropic": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "ollama_cloud": [
            "gpt-oss:120b-cloud",
            "gpt-oss:20b-cloud",
            "deepseek-v3.1:671b-cloud",
            "qwen3-coder:480b-cloud",
            "qwen3.5:397b-cloud",
        ],
    }
    return cloud_models.get(provider, [])


# -- Live model fetching helpers ---------------------------------------------

_OPENAI_FALLBACK_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
]

_GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "whisper-large-v3",
]


def _fetch_openai_models() -> list[str]:
    """Fetch available chat models from OpenAI API. Falls back to static list."""
    try:
        import json
        import urllib.request
        import ssl
        import certifi

        if not settings.openai_api_key:
            return _OPENAI_FALLBACK_MODELS

        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        )
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            # Filter to chat/completion models (skip embeddings, tts, dall-e, etc.)
            if mid and not any(
                x in mid for x in [
                    "embed", "tts", "dall-e", "whisper", "davinci",
                    "babbage", "moderation", "realtime",
                ]
            ):
                models.append(mid)

        if models:
            logger.info("OpenAI: fetched %d live models", len(models))
            return sorted(models)

    except Exception as exc:
        logger.warning("OpenAI live model fetch failed: %s — using fallback", exc)

    return _OPENAI_FALLBACK_MODELS


def _fetch_groq_models() -> list[str]:
    """Fetch available models from Groq API. Falls back to static list."""
    try:
        import json
        import urllib.request
        import ssl
        import certifi

        if not settings.groq_api_key:
            return _GROQ_FALLBACK_MODELS

        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        )
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = [
            m.get("id", "")
            for m in data.get("data", [])
            if m.get("id")
        ]

        if models:
            logger.info("Groq: fetched %d live models", len(models))
            return sorted(models)

    except Exception as exc:
        logger.warning("Groq live model fetch failed: %s — using fallback", exc)

    return _GROQ_FALLBACK_MODELS


# -- NVIDIA NIM live model discovery -----------------------------------------

_NVIDIA_FALLBACK_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    "deepseek-ai/deepseek-r1",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "microsoft/phi-3-medium-128k-instruct",
    "google/gemma-2-27b-it",
]


def get_available_nvidia_models(use_cache: bool = True) -> list[str]:
    """
    Fetch the live list of available models from NVIDIA API Catalog.

    Falls back to a curated static list if the API call fails.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import os

        if settings.nvidia_api_key:
            os.environ["NVIDIA_API_KEY"] = settings.nvidia_api_key

        live_models = ChatNVIDIA.get_available_models()

        chat_models = [
            m.id for m in live_models
            if getattr(m, "model_type", "chat") in ("chat", "completion", None)
            and m.id
        ]

        if chat_models:
            logger.info("NVIDIA NIM: fetched %d live models", len(chat_models))
            return sorted(chat_models)

    except Exception as exc:
        logger.warning(
            "NVIDIA NIM live model fetch failed: %s — using fallback", exc
        )

    return _NVIDIA_FALLBACK_MODELS


def get_ollama_cloud_models_live() -> list[str]:
    """
    Fetch the live model list from Ollama Cloud API.

    Returns the static fallback list if the API key is missing or the
    request fails. Cached by callers (Streamlit cache) for 5 minutes.

    Returns:
        List of model name strings available on Ollama Cloud.
    """
    if not settings.ollama_cloud_api_key:
        return get_available_cloud_models("ollama_cloud")

    try:
        import json
        import ssl
        import urllib.request

        import certifi

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(
            f"{settings.ollama_cloud_base_url}/models",
            headers={
                "Authorization": f"Bearer {settings.ollama_cloud_api_key}",
            },
        )
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            # OpenAI /v1/models returns {"data": [{"id": "...", ...}]}
            models = [m["id"] for m in data.get("data", []) if m.get("id")]
            # Filter to cloud models only (those ending in -cloud)
            cloud_models = [m for m in models if m.endswith("-cloud")]
            if cloud_models:
                return sorted(cloud_models)
    except Exception as exc:
        logger.warning("Could not fetch live Ollama Cloud model list: %s", exc)

    # Fallback to static list
    return get_available_cloud_models("ollama_cloud")

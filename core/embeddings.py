"""
Embedding model abstraction.

Wraps OllamaEmbeddings with caching and error handling.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_embeddings(model: str | None = None) -> OllamaEmbeddings:
    """
    Return a cached embedding model instance.

    Uses LRU cache so the same model string always returns
    the same OllamaEmbeddings object — no re-init overhead.
    """
    model = model or settings.embedding_model
    logger.info("Initializing embedding model: %s", model)
    return OllamaEmbeddings(
        model=model,
        base_url=settings.ollama_base_url,
    )

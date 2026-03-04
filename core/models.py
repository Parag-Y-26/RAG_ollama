"""
Ollama model management, health checking, and embedding configuration.

Provides functions to discover installed models, verify server health,
and resolve the embedding model name.  All configuration values are
read from ``st.secrets`` with graceful fallbacks.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Configuration helpers  (st.secrets → env → default)
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str) -> str:
    """Read a value from ``st.secrets``, falling back to env vars.

    Args:
        key: Secret / environment variable name.
        default: Value to return if neither source contains *key*.

    Returns:
        The resolved string value.
    """
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _ollama_url() -> str:
    """Resolve the Ollama base URL.

    Returns:
        Ollama server base URL (e.g. ``http://localhost:11434``).
    """
    return _get_secret("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# EMBEDDING_MODEL constant
# ---------------------------------------------------------------------------

def _resolve_embedding_model() -> str:
    """Resolve the embedding model name with fallback logic.

    Priority:
        1. ``st.secrets["EMBEDDING_MODEL"]``
        2. ``os.environ["EMBEDDING_MODEL"]``
        3. ``"nomic-embed-text"`` (sensible default)

    Returns:
        Embedding model name string.
    """
    return _get_secret("EMBEDDING_MODEL", "nomic-embed-text")


EMBEDDING_MODEL: str = _resolve_embedding_model()
"""Module-level constant for the active embedding model name."""


# ---------------------------------------------------------------------------
# Health-check dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaHealth:
    """Structured result from ``health_check()``.

    Attributes:
        is_running: Whether the Ollama server responded.
        base_url: The URL that was probed.
        models: List of installed model names (empty if offline).
        error: Error message if the server is unreachable.
    """

    is_running: bool
    base_url: str
    models: list[str]
    error: str


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def get_available_models(timeout: int = 3) -> list[str]:
    """Poll Ollama ``/api/tags`` and return installed model names.

    Cached for 30 seconds via ``st.cache_data`` to avoid
    re-polling Ollama on every Streamlit rerun.

    Excludes embedding-only models (names containing ``embed``).
    Gracefully falls back to a single-element list containing the
    default model if Ollama is unreachable.

    Args:
        timeout: HTTP request timeout in seconds.

    Returns:
        Sorted list of model name strings,
        e.g. ``["deepseek-r1:latest", "gemma3:4b"]``.
    """
    default_model = _get_secret("DEFAULT_LLM_MODEL", "deepseek-r1:latest")

    try:
        url = f"{_ollama_url()}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))

        models: list[str] = []
        for model_info in data.get("models", []):
            name: str = model_info.get("name", "")
            if name and "embed" not in name.lower():
                models.append(name)

        if models:
            return sorted(models)

    except Exception:
        pass  # graceful fallback below

    return [default_model]


def health_check(timeout: int = 3) -> OllamaHealth:
    """Check whether Ollama is running and list installed models.

    Args:
        timeout: HTTP request timeout in seconds.

    Returns:
        An ``OllamaHealth`` dataclass with server status, installed
        models, and any error message.
    """
    base_url = _ollama_url()

    try:
        url = f"{base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))

        all_models: list[str] = sorted(
            m.get("name", "")
            for m in data.get("models", [])
            if m.get("name")
        )

        return OllamaHealth(
            is_running=True,
            base_url=base_url,
            models=all_models,
            error="",
        )

    except urllib.error.URLError as exc:
        return OllamaHealth(
            is_running=False,
            base_url=base_url,
            models=[],
            error=f"Cannot connect to Ollama at {base_url}: {exc.reason}",
        )
    except Exception as exc:
        return OllamaHealth(
            is_running=False,
            base_url=base_url,
            models=[],
            error=f"Unexpected error checking Ollama: {exc}",
        )

"""
Ollama model management and health checks.

Runtime configuration is sourced exclusively from `config.settings`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import streamlit as st

from config.settings import settings


@dataclass(frozen=True)
class OllamaHealth:
    """Structured result from `health_check()`."""

    is_running: bool
    base_url: str
    models: list[str]
    error: str


@st.cache_data(ttl=30, show_spinner=False)
def get_available_models(timeout: int | None = None) -> list[str]:
    """Poll Ollama `/api/tags` and return installed non-embedding models."""
    default_model = settings.default_model
    request_timeout = timeout or settings.ollama_timeout

    try:
        url = f"{settings.ollama_base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))

        models: list[str] = []
        for model_info in data.get("models", []):
            name: str = model_info.get("name", "")
            if name and "embed" not in name.lower():
                models.append(name)

        if models:
            return sorted(models)
    except Exception:
        pass

    return [default_model]


def health_check(timeout: int | None = None) -> OllamaHealth:
    """Check whether Ollama is running and list installed models."""
    base_url = settings.ollama_base_url
    request_timeout = timeout or settings.ollama_timeout

    try:
        url = f"{base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
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

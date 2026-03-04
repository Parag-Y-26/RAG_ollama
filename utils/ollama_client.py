"""
Ollama HTTP API helpers.

Light wrapper around the Ollama REST API — no heavy dependencies.
Handles health checks, model listing, and model pulling.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)


def is_ollama_running() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{settings.ollama_base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=settings.ollama_timeout):
            return True
    except Exception:
        return False


def list_models(exclude_embeddings: bool = True) -> list[str]:
    """
    Fetch installed models from Ollama.

    Returns a list of model name strings, e.g. ["deepseek-r1:latest", "gemma3:4b"].
    Falls back to [settings.default_model] on failure.
    """
    try:
        req = urllib.request.Request(f"{settings.ollama_base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=settings.ollama_timeout) as resp:
            data = json.loads(resp.read().decode())
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                if exclude_embeddings and "embed" in name.lower():
                    continue
                models.append(name)
            if models:
                return sorted(models)
    except Exception as exc:
        logger.warning("Failed to fetch Ollama models: %s", exc)

    return [settings.default_model]


def get_model_info(model_name: str) -> Optional[dict]:
    """Get information about a specific model."""
    try:
        req = urllib.request.Request(
            f"{settings.ollama_base_url}/api/show",
            data=json.dumps({"name": model_name}).encode("utf-8"),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=settings.ollama_timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("Failed to get model info for %s: %s", model_name, exc)
        return None

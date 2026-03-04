"""
Centralized application configuration using Pydantic Settings.

All values can be overridden via environment variables or .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def _env(key: str, default: str = "") -> str:
    """Read from environment with fallback."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Settings dataclass — replaces all hardcoded values in the old codebase
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    # -- Project paths -------------------------------------------------------
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _DATA_DIR
    chroma_db_dir: Path = field(default_factory=lambda: _DATA_DIR / "chroma_db")

    # -- Ollama --------------------------------------------------------------
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_timeout: int = 5  # seconds for health / model-list requests
    default_model: str = field(
        default_factory=lambda: _env("DEFAULT_LLM_MODEL", "deepseek-r1:latest")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text")
    )

    # -- RAG pipeline --------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 5
    max_context_docs: int = 8

    # -- Perplexity ----------------------------------------------------------
    perplexity_api_key: str = field(
        default_factory=lambda: _env("PPLX_API_KEY", "")
    )
    perplexity_model: str = "sonar-pro"
    perplexity_timeout: int = 60

    # -- UI ------------------------------------------------------------------
    app_title: str = "NotebookLM"
    app_icon: str = "🧠"
    page_layout: str = "wide"

    # -- Notebooks -----------------------------------------------------------
    max_notebooks: int = 20
    default_notebook_name: str = "My Notebook"

    @property
    def perplexity_enabled(self) -> bool:
        return bool(self.perplexity_api_key)


# Singleton — import this everywhere
settings = Settings()

"""
Centralized application settings.

`settings` is the single source of truth for runtime configuration.
Values are resolved from `st.secrets`, then environment variables,
then explicit defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def _env(key: str, default: str = "") -> str:
    """Read from st.secrets first, then environment, then default."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


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
    ollama_timeout: int = 5
    default_model: str = field(
        default_factory=lambda: _env("DEFAULT_LLM_MODEL", "deepseek-r1:latest")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text")
    )

    # -- Cloud LLM Providers -------------------------------------------------
    openai_api_key: str = field(
        default_factory=lambda: _env("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: _env("ANTHROPIC_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: _env("GROQ_API_KEY", "")
    )

    # -- Provider selection --------------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "ollama")
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

    # -- Web Search (DuckDuckGo) --------------------------------------------
    web_search_enabled_by_default: bool = False
    web_search_max_results: int = field(
        default_factory=lambda: int(_env("WEB_SEARCH_MAX_RESULTS", "5"))
    )
    web_search_max_fetch: int = field(
        default_factory=lambda: int(_env("WEB_SEARCH_MAX_FETCH", "3"))
    )
    web_search_region: str = field(
        default_factory=lambda: _env("WEB_SEARCH_REGION", "wt-wt")
    )

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

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def anthropic_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()

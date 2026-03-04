"""
Perplexity Sonar API web research integration.

Provides the ``PerplexityResearcher`` class for automated internet
research.  Results are returned as LangChain ``Document`` objects
with full metadata, ready for ``KnowledgeBase.add_source()``.

Security:
    * Uses ``certifi`` for SSL certificate verification — **never**
      disables hostname checking or certificate validation.
    * API key is read from ``st.secrets`` with env-var fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import certifi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERPLEXITY_URL: str = "https://api.perplexity.ai/chat/completions"
_DEFAULT_MODEL: str = "sonar-pro"
_CHUNK_SIZE: int = 1000
_CHUNK_OVERLAP: int = 200
_TIMEOUT: int = 60


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Resolve the Perplexity API key from ``st.secrets`` → env → empty.

    Returns:
        The API key string, or ``""`` if not configured.
    """
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "PPLX_API_KEY" in st.secrets:
            return str(st.secrets["PPLX_API_KEY"])
    except Exception:
        pass
    return os.environ.get("PPLX_API_KEY", "")


# ---------------------------------------------------------------------------
# PerplexityResearcher
# ---------------------------------------------------------------------------

class PerplexityResearcher:
    """Performs web research via the Perplexity Sonar API.

    Usage::

        researcher = PerplexityResearcher()
        docs = researcher.research("Quantum computing 2025 breakthroughs")
        kb.add_source(docs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = _TIMEOUT,
    ) -> None:
        """Initialise the researcher.

        Args:
            api_key: Perplexity API key.  If ``None``, resolved from
                ``st.secrets["PPLX_API_KEY"]`` or the environment.
            model: Perplexity model name (default ``sonar-pro``).
            timeout: HTTP request timeout in seconds.
        """
        self._api_key: str = api_key or _get_api_key()
        self._model: str = model
        self._timeout: int = timeout

        # Build a proper SSL context using certifi's CA bundle.
        # This is the ONLY acceptable way to handle SSL in this project.
        self._ssl_ctx: ssl.SSLContext = ssl.create_default_context(
            cafile=certifi.where()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, topic: str) -> list[Document]:
        """Perform web research on a topic and return chunked Documents.

        The returned Documents carry full metadata (``source``,
        ``source_type``, ``ingested_at``, ``title``, ``chunk_hash``)
        and are ready for ``KnowledgeBase.add_source()``.

        Args:
            topic: The research topic / question.

        Returns:
            List of chunked ``Document`` objects.

        Raises:
            ValueError: If *topic* is blank or the API key is missing.
            ConnectionError: If the API is unreachable.
            PermissionError: If the API key is invalid (HTTP 401/403).
            RuntimeError: For rate limits (HTTP 429) or other API errors.
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("Research topic cannot be empty.")

        if not self._api_key:
            raise ValueError(
                "Perplexity API key is not configured.  "
                "Add PPLX_API_KEY to .streamlit/secrets.toml or your "
                "environment variables."
            )

        raw_content = self._call_api(topic)
        documents = self._build_documents(raw_content, topic)
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, topic: str) -> str:
        """Send the research request to Perplexity and return raw text.

        Args:
            topic: The research topic.

        Returns:
            The assistant's response content as a string.

        Raises:
            ConnectionError: Network-level failure.
            PermissionError: Authentication failure (401/403).
            RuntimeError: Rate limiting (429) or other HTTP errors.
        """
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional research assistant.  "
                        "Provide a comprehensive, factual overview of "
                        "the given topic based on current web search "
                        "results.  Include key facts, dates, names, and "
                        "statistics.  Structure your response with clear "
                        "headings and bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research the following topic and provide a "
                        f"detailed, well-structured summary:\n\n{topic}"
                    ),
                },
            ],
        }

        req = urllib.request.Request(
            _PERPLEXITY_URL,
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with urllib.request.urlopen(
                req,
                context=self._ssl_ctx,
                timeout=self._timeout,
            ) as resp:
                body: dict = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (401, 403):
                raise PermissionError(
                    f"Perplexity API authentication failed (HTTP {status}).  "
                    f"Check that your PPLX_API_KEY is valid."
                ) from exc
            if status == 429:
                raise RuntimeError(
                    "Perplexity API rate limit exceeded (HTTP 429).  "
                    "Please wait a moment and try again."
                ) from exc
            raise RuntimeError(
                f"Perplexity API returned HTTP {status}: {exc.reason}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Perplexity API: {exc.reason}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Perplexity API returned invalid JSON."
            ) from exc

    def _build_documents(
        self,
        content: str,
        topic: str,
    ) -> list[Document]:
        """Chunk the raw research text into annotated Documents.

        Args:
            content: Raw text from the Perplexity API.
            topic: Original research topic (used for metadata).

        Returns:
            List of chunked, fully-annotated ``Document`` objects.
        """
        source_label = f"Perplexity Research: {topic}"
        ingested_at = datetime.now(timezone.utc).isoformat()

        raw_doc = Document(page_content=content, metadata={})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        chunks: list[Document] = splitter.split_documents([raw_doc])

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "source": source_label,
                    "source_type": "perplexity",
                    "ingested_at": ingested_at,
                    "title": topic,
                    "chunk_hash": hashlib.md5(
                        chunk.page_content.encode("utf-8")
                    ).hexdigest(),
                }
            )

        return chunks

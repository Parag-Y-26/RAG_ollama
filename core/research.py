"""
Perplexity Sonar API web research integration.

Provides the `PerplexityResearcher` class for automated internet
research. Results are returned as LangChain `Document` objects
with full metadata, ready for `KnowledgeBase.add_source()`.

Security:
    * Uses `certifi` for SSL certificate verification and never
      disables hostname checking or certificate validation.
    * Runtime configuration is read from `config.settings`.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

import certifi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERPLEXITY_URL: str = "https://api.perplexity.ai/chat/completions"
_CHUNK_SIZE: int = 1000
_CHUNK_OVERLAP: int = 200


class PerplexityResearcher:
    """Perform web research via the Perplexity Sonar API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialise the researcher."""
        self._api_key: str = api_key or settings.perplexity_api_key
        self._model: str = model or settings.perplexity_model
        self._timeout: int = timeout or settings.perplexity_timeout
        self._ssl_ctx: ssl.SSLContext = ssl.create_default_context(
            cafile=certifi.where()
        )

    def research(self, topic: str) -> list[Document]:
        """Perform web research on a topic and return chunked documents."""
        topic = topic.strip()
        if not topic:
            raise ValueError("Research topic cannot be empty.")

        if not self._api_key:
            raise ValueError(
                "Perplexity API key is not configured. Add PPLX_API_KEY "
                "to .streamlit/secrets.toml or your environment variables."
            )

        raw_content = self._call_api(topic)
        return self._build_documents(raw_content, topic)

    def _call_api(self, topic: str) -> str:
        """Send the research request to Perplexity and return raw text."""
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
                        "You are a professional research assistant. "
                        "Provide a comprehensive, factual overview of "
                        "the given topic based on current web search "
                        "results. Include key facts, dates, names, and "
                        "statistics. Structure your response with clear "
                        "headings and bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Research the following topic and provide a "
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
                    f"Perplexity API authentication failed (HTTP {status}). "
                    "Check that your PPLX_API_KEY is valid."
                ) from exc
            if status == 429:
                raise RuntimeError(
                    "Perplexity API rate limit exceeded (HTTP 429). "
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

    def _build_documents(self, content: str, topic: str) -> list[Document]:
        """Chunk the raw research text into annotated documents."""
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

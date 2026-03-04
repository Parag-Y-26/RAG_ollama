"""
Perplexity Sonar API web research.

Performs internet research on a topic and returns LangChain Documents
ready for ingestion into the knowledge base.

SECURITY: Uses proper SSL verification (unlike the original code).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

from langchain_core.documents import Document

from config.settings import settings
from ingestion.base import BaseLoader
from utils.logging import get_logger

logger = get_logger(__name__)


class PerplexityResearcher(BaseLoader):
    """Research a topic via Perplexity Sonar API and ingest the results."""

    def __init__(
        self,
        topic: str,
        notebook_id: str = "default",
        api_key: str | None = None,
    ):
        super().__init__(notebook_id=notebook_id)
        self._topic = topic.strip()
        self._api_key = api_key or settings.perplexity_api_key

    @property
    def source_name(self) -> str:
        return f"Perplexity Research: {self._topic}"

    def load(self) -> list[Document]:
        """Call Perplexity Sonar API and return research as Documents."""
        if not self._api_key:
            raise ValueError(
                "Perplexity API key not configured. "
                "Set PPLX_API_KEY in your environment or .streamlit/secrets.toml"
            )

        if not self._topic:
            raise ValueError("Research topic cannot be empty.")

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.perplexity_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional research assistant. Provide a "
                        "comprehensive, factual overview of the following topic "
                        "based on web search results. Include key facts, dates, "
                        "names, and statistics when available. Structure your "
                        "response with clear headings and bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research the following topic and provide a detailed, "
                        f"well-structured summary:\n\n{self._topic}"
                    ),
                },
            ],
        }

        try:
            req = urllib.request.Request(
                url,
                headers=headers,
                data=json.dumps(payload).encode("utf-8"),
            )
            # NOTE: Using default SSL context (proper verification)
            # The original code disabled SSL — that was a security vulnerability
            with urllib.request.urlopen(req, timeout=settings.perplexity_timeout) as resp:
                result = json.loads(resp.read().decode())
                content = result["choices"][0]["message"]["content"]

            logger.info(
                "Perplexity research complete for '%s' (%d chars)",
                self._topic,
                len(content),
            )

            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": f"Perplexity Research: {self._topic}",
                        "type": "perplexity",
                    },
                )
            ]

        except Exception as exc:
            logger.error("Perplexity research failed for '%s': %s", self._topic, exc)
            raise

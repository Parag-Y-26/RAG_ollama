"""
Web page ingestion with URL validation.
"""

from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from ingestion.base import BaseLoader
from utils.logging import get_logger
from utils.validators import validate_url

logger = get_logger(__name__)


class WebPageLoader(BaseLoader):
    """Load and ingest content from a web page URL."""

    def __init__(self, url: str, notebook_id: str = "default"):
        super().__init__(notebook_id=notebook_id)
        self._url = url.strip()

    @property
    def source_name(self) -> str:
        return f"Web: {self._url}"

    def load(self) -> list[Document]:
        """Fetch and parse the web page."""
        # Validate URL for safety (SSRF prevention)
        is_valid, error = validate_url(self._url)
        if not is_valid:
            raise ValueError(f"Invalid URL: {error}")

        try:
            loader = WebBaseLoader(self._url)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = self._url
                doc.metadata["type"] = "web"

            return docs

        except Exception as exc:
            logger.error("Web extraction failed for '%s': %s", self._url, exc)
            raise

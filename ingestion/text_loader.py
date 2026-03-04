"""
Plain text and markdown ingestion.
"""

from __future__ import annotations

from langchain_core.documents import Document

from ingestion.base import BaseLoader
from utils.logging import get_logger

logger = get_logger(__name__)


class TextLoader(BaseLoader):
    """Load and ingest plain text or markdown content."""

    def __init__(
        self,
        content: str,
        source_name: str = "Pasted Text",
        notebook_id: str = "default",
    ):
        super().__init__(notebook_id=notebook_id)
        self._content = content
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    def load(self) -> list[Document]:
        """Wrap the raw text into a Document."""
        if not self._content.strip():
            logger.warning("Empty text content, nothing to ingest.")
            return []

        return [
            Document(
                page_content=self._content,
                metadata={
                    "source": self._source_name,
                    "type": "text",
                },
            )
        ]

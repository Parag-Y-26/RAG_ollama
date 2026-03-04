"""
YouTube transcript ingestion.
"""

from __future__ import annotations

from langchain_community.document_loaders import YoutubeLoader as LCYoutubeLoader
from langchain_core.documents import Document

from ingestion.base import BaseLoader
from utils.logging import get_logger
from utils.validators import is_youtube_url

logger = get_logger(__name__)


class YouTubeLoader(BaseLoader):
    """Load and ingest a YouTube video transcript."""

    def __init__(self, url: str, notebook_id: str = "default"):
        super().__init__(notebook_id=notebook_id)
        self._url = url.strip()

    @property
    def source_name(self) -> str:
        return f"YouTube: {self._url}"

    def load(self) -> list[Document]:
        """Fetch the transcript from a YouTube video."""
        if not is_youtube_url(self._url):
            raise ValueError(f"Not a valid YouTube URL: {self._url}")

        try:
            loader = LCYoutubeLoader.from_youtube_url(
                self._url, add_video_info=True
            )
            docs = loader.load()

            for doc in docs:
                doc.metadata["type"] = "youtube"
                # Keep any existing metadata (title, etc.) from the loader
                if "source" not in doc.metadata:
                    doc.metadata["source"] = self._url

            return docs

        except Exception as exc:
            logger.error("YouTube extraction failed for '%s': %s", self._url, exc)
            raise

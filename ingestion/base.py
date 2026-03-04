"""
Abstract base for all document ingestion loaders.

Every loader must implement the `load()` method and return
a list of LangChain Document objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from core.vectorstore import add_documents
from utils.logging import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    def __init__(
        self,
        notebook_id: str = "default",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.notebook_id = notebook_id
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    @abstractmethod
    def load(self) -> list[Document]:
        """Load raw documents from the source. Must be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the source for UI display."""
        ...

    def ingest(self) -> int:
        """
        Full pipeline: load → chunk → store.

        Returns the number of chunks added to the vector store.
        """
        logger.info("Starting ingestion from: %s", self.source_name)

        # 1. Load raw documents
        raw_docs = self.load()
        if not raw_docs:
            logger.warning("No documents loaded from: %s", self.source_name)
            return 0

        logger.info("Loaded %d raw documents from: %s", len(raw_docs), self.source_name)

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info("Split into %d chunks", len(chunks))

        # 3. Store in vector database
        stored = add_documents(chunks, notebook_id=self.notebook_id)
        logger.info("Stored %d chunks in notebook '%s'", stored, self.notebook_id)

        return stored

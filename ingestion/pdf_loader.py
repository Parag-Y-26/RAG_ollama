"""
PDF document ingestion.

Handles file upload, temporary file management, and text extraction.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ingestion.base import BaseLoader
from utils.logging import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """Load and ingest a PDF file."""

    def __init__(
        self,
        file_data: bytes,
        filename: str = "uploaded.pdf",
        notebook_id: str = "default",
    ):
        super().__init__(notebook_id=notebook_id)
        self._file_data = file_data
        self._filename = filename

    @property
    def source_name(self) -> str:
        return f"PDF: {self._filename}"

    def load(self) -> list[Document]:
        """Extract text from PDF via a temporary file."""
        tmp_path = None
        try:
            # Write to temp file (PyPDFLoader requires a file path)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", prefix="nb_"
            ) as tmp:
                tmp.write(self._file_data)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Tag each page with the original filename
            for doc in docs:
                doc.metadata["source"] = self._filename
                doc.metadata["type"] = "pdf"

            return docs

        except Exception as exc:
            logger.error("PDF extraction failed for '%s': %s", self._filename, exc)
            raise

        finally:
            # Always clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as cleanup_err:
                    logger.warning("Could not delete temp file %s: %s", tmp_path, cleanup_err)

"""
Document ingestion functions for all supported source types.

Each ``load_*`` function validates its input, extracts raw content,
splits it into chunks via ``RecursiveCharacterTextSplitter``, and
returns fully-annotated ``Document`` objects ready for
``KnowledgeBase.add_source()``.

Every returned ``Document`` carries the following metadata keys:

* ``source`` — the original URL, filename, or identifier.
* ``source_type`` — one of ``youtube``, ``web``, ``pdf``, ``text``.
* ``ingested_at`` — ISO-8601 timestamp of ingestion.
* ``title`` — human-readable title of the source.
* ``chunk_hash`` — MD5 hex digest of the chunk's ``page_content``.
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime, timezone

from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SIZE: int = 1000
_CHUNK_OVERLAP: int = 200

_YT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"^https?://youtu\.be/[\w-]+"),
    re.compile(r"^https?://(?:www\.)?youtube\.com/shorts/[\w-]+"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _chunk_hash(text: str) -> str:
    """Compute the MD5 hex digest of a chunk's content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _split_documents(
    docs: list[Document],
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks and preserve metadata.

    Args:
        docs: Raw ``Document`` objects from a loader.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Chunked ``Document`` list with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _annotate_chunks(
    chunks: list[Document],
    *,
    source: str,
    source_type: str,
    title: str,
) -> list[Document]:
    """Attach required metadata to every chunk.

    Mutates the metadata in-place and returns the same list.

    Args:
        chunks: Pre-split ``Document`` objects.
        source: Original URL / filename / identifier.
        source_type: One of ``youtube``, ``web``, ``pdf``, ``text``.
        title: Human-readable source title.

    Returns:
        The same list with updated metadata on every chunk.
    """
    ingested_at = _now_iso()
    for chunk in chunks:
        chunk.metadata.update(
            {
                "source": source,
                "source_type": source_type,
                "ingested_at": ingested_at,
                "title": title,
                "chunk_hash": _chunk_hash(chunk.page_content),
            }
        )
    return chunks


def _is_youtube_url(url: str) -> bool:
    """Check whether *url* matches a known YouTube pattern.

    Args:
        url: Candidate URL string.

    Returns:
        ``True`` if the URL looks like a YouTube video link.
    """
    return any(pat.match(url) for pat in _YT_PATTERNS)


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def load_youtube(url: str) -> list[Document]:
    """Load and chunk a YouTube video transcript.

    Args:
        url: Full YouTube video URL
            (``https://www.youtube.com/watch?v=…`` or ``https://youtu.be/…``).

    Returns:
        List of chunked ``Document`` objects with standard metadata.

    Raises:
        ValueError: If *url* is empty or not a recognised YouTube URL.
        RuntimeError: If the transcript cannot be fetched.
    """
    url = url.strip()
    if not url:
        raise ValueError("YouTube URL cannot be empty.")

    if not _is_youtube_url(url):
        raise ValueError(
            f"Not a valid YouTube URL: {url}  "
            f"Expected a link like https://www.youtube.com/watch?v=XXXX"
        )

    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        raw_docs = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch YouTube transcript for {url}: {exc}"
        ) from exc

    if not raw_docs:
        raise RuntimeError(
            f"No transcript found for {url}.  "
            f"The video may not have captions enabled."
        )

    # Derive title from loader metadata if available
    title = raw_docs[0].metadata.get("title", url)

    chunks = _split_documents(raw_docs)
    return _annotate_chunks(
        chunks, source=url, source_type="youtube", title=title
    )


def load_webpage(url: str) -> list[Document]:
    """Load and chunk content from a web page.

    Performs basic validation to reject obviously dangerous URLs
    (localhost, private IPs).

    Args:
        url: Fully-qualified ``http`` or ``https`` URL.

    Returns:
        List of chunked ``Document`` objects with standard metadata.

    Raises:
        ValueError: If *url* is blank, uses a disallowed scheme, or
            points to a private address.
        RuntimeError: If the page cannot be fetched or parsed.
    """
    url = url.strip()
    if not url:
        raise ValueError("Web page URL cannot be empty.")

    # Basic scheme check
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"URL must start with http:// or https://  (got: {url})"
        )

    # Block localhost / private IP to prevent SSRF
    _blocked = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.")
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname or ""
    if any(hostname.startswith(b) for b in _blocked):
        raise ValueError(
            f"URLs pointing to localhost / private IPs are blocked: {url}"
        )

    try:
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch web page {url}: {exc}"
        ) from exc

    if not raw_docs:
        raise RuntimeError(f"No content extracted from {url}.")

    # Use the URL as a fallback title
    title = raw_docs[0].metadata.get("title", url)

    chunks = _split_documents(raw_docs)
    return _annotate_chunks(
        chunks, source=url, source_type="web", title=title
    )


def load_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Load and chunk a PDF document from raw bytes.

    Uses ``pypdf`` directly with **per-page** error recovery: if a
    single page fails to parse, it is skipped and a warning is
    recorded — the rest of the document is still ingested.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename (used for metadata and validation).

    Returns:
        List of chunked ``Document`` objects with standard metadata.

    Raises:
        ValueError: If *file_bytes* is empty, *filename* is blank, or
            the file does not look like a valid PDF.
        RuntimeError: If text extraction yields nothing at all.
    """
    if not file_bytes:
        raise ValueError("PDF file is empty (0 bytes received).")

    filename = filename.strip()
    if not filename:
        raise ValueError("Filename cannot be empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError(
            f"Expected a .pdf file, got: {filename}"
        )

    # Quick magic-byte check — PDF files start with %PDF
    if not file_bytes[:5].startswith(b"%PDF"):
        raise ValueError(
            f"File does not appear to be a valid PDF: {filename}"
        )


    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open PDF '{filename}': {exc}"
        ) from exc

    raw_docs: list[Document] = []
    failed_pages: list[int] = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num},
                    )
                )
        except Exception:
            failed_pages.append(page_num)

    if not raw_docs:
        detail = ""
        if failed_pages:
            detail = f"  Pages that failed: {failed_pages}."
        raise RuntimeError(
            f"No text content found in PDF '{filename}'.{detail}  "
            f"The file may contain only images."
        )

    title = filename
    if failed_pages:
        import warnings
        warnings.warn(
            f"PDF '{filename}': pages {failed_pages} failed to parse "
            f"and were skipped.  {len(raw_docs)} pages succeeded.",
            stacklevel=2,
        )

    chunks = _split_documents(raw_docs)
    return _annotate_chunks(
        chunks, source=filename, source_type="pdf", title=title
    )


def load_plaintext(text: str, title: str) -> list[Document]:
    """Wrap a raw text string into chunked ``Document`` objects.

    Args:
        text: The plain text or markdown content to ingest.
        title: A user-supplied title / label for this source.

    Returns:
        List of chunked ``Document`` objects with standard metadata.

    Raises:
        ValueError: If *text* is blank or *title* is blank.
    """
    text = text.strip()
    if not text:
        raise ValueError("Text content cannot be empty.")

    title = title.strip()
    if not title:
        raise ValueError("A title is required for plain-text sources.")

    raw_doc = Document(page_content=text, metadata={})

    chunks = _split_documents([raw_doc])
    return _annotate_chunks(
        chunks, source=title, source_type="text", title=title
    )

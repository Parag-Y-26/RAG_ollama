"""
Input validation utilities.

Prevents SSRF, validates URLs, and sanitises user inputs.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}

# Simple YouTube URL patterns
_YT_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"(?:https?://)?youtu\.be/[\w-]+"),
]


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube video link."""
    return any(p.match(url) for p in _YT_PATTERNS)


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate a URL for safety and correctness.

    Returns (is_valid, error_message).
    """
    url = url.strip()
    if not url:
        return False, "URL cannot be empty."

    parsed = urlparse(url)

    # Scheme check
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"URL scheme '{parsed.scheme}' is not allowed. Use http or https."

    # Host check
    hostname = parsed.hostname or ""
    if not hostname:
        return False, "URL has no hostname."

    if hostname in _BLOCKED_HOSTS:
        return False, "URLs pointing to localhost are not allowed."

    # Check for private IPs (SSRF prevention)
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_reserved:
            return False, "URLs pointing to private/reserved IPs are not allowed."
    except ValueError:
        # Not an IP address — it's a domain name, which is fine
        pass

    return True, ""


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}
_MAX_FILE_SIZE_MB = 50


def validate_uploaded_file(
    filename: str, size_bytes: int
) -> tuple[bool, str]:
    """Validate uploaded file type and size."""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        return False, f"File type '{ext}' not supported. Allowed: {allowed}"

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum: {_MAX_FILE_SIZE_MB} MB."

    return True, ""


# ---------------------------------------------------------------------------
# Text sanitisation
# ---------------------------------------------------------------------------
def sanitise_query(query: str, max_length: int = 2000) -> str:
    """Sanitise and truncate a user query."""
    query = query.strip()
    if len(query) > max_length:
        logger.warning("Query truncated from %d to %d chars", len(query), max_length)
        query = query[:max_length]
    return query

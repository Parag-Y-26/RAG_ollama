"""
Structured logging configuration.

Usage:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"doc_id": "abc"})
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Prevent duplicate handler registration on Streamlit reruns
_INITIALIZED = False


def _setup_root_logger() -> None:
    """Configure the root logger once."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger("notebooklm")
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'notebooklm' namespace."""
    _setup_root_logger()
    return logging.getLogger(f"notebooklm.{name}")

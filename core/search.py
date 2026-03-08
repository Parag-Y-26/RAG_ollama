"""
DuckDuckGo internet search integration.

Provides the ``WebSearcher`` class for free, no-auth internet search.
Results are returned as LangChain ``Document`` objects with full metadata,
ready for ``KnowledgeBase.add_source()``.

Design decisions:
* Uses ``duckduckgo-search`` (DDGS) -- zero API keys, zero cost.
* Fetches full page content for top results using ``WebBaseLoader``
  so the KB gets real content, not just snippets.
* Falls back to snippet-only documents if full-page fetch fails.
* SSRF-safe: inherits URL validation from ``utils.validators``.
* Rate-limit safe: adds a small delay between page fetches.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from datetime import datetime, timezone
from typing import Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logging import get_logger
from utils.validators import validate_url

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SIZE: int = 1000
_CHUNK_OVERLAP: int = 200
_FETCH_DELAY_SECONDS: float = 0.5
_DEFAULT_MAX_RESULTS: int = 5
_DEFAULT_MAX_FETCH: int = 3


class SearchResult:
    """A single DuckDuckGo search result before document conversion."""

    __slots__ = ("title", "url", "snippet", "published")

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        published: str = "",
    ) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet
        self.published = published

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r})"


class WebSearcher:
    """
    Free internet search via DuckDuckGo with optional full-page ingestion.

    Usage - snippet mode (fast, for chat augmentation)::

        searcher = WebSearcher()
        results = searcher.search("Python asyncio tutorial")

    Usage - document mode (for KB ingestion)::

        docs = searcher.search_and_load("Python asyncio tutorial")
        kb.add_source(docs)

    Usage - quick answer mode (for RAG fallback)::

        context = searcher.search_as_context("What is RAG?", max_results=3)
    """

    def __init__(
        self,
        max_results: Optional[int] = None,
        max_fetch: Optional[int] = None,
        region: Optional[str] = None,
        safesearch: str = "moderate",
        fetch_delay: float = _FETCH_DELAY_SECONDS,
    ) -> None:
        """
        Initialise the searcher.

        Args:
            max_results: Number of DDG results to retrieve per query.
            max_fetch: How many results to fetch full page content for.
            region: DDG region string (for example ``"us-en"`` or ``"wt-wt"``).
            safesearch: ``"on"``, ``"moderate"``, or ``"off"``.
            fetch_delay: Seconds to wait between page fetches.
        """
        resolved_max_results = max_results or settings.web_search_max_results or _DEFAULT_MAX_RESULTS
        resolved_max_fetch = (
            settings.web_search_max_fetch
            if max_fetch is None
            else max_fetch
        )

        self._max_results = resolved_max_results
        self._max_fetch = min(resolved_max_fetch, resolved_max_results)
        self._region = region or settings.web_search_region or "wt-wt"
        self._safesearch = safesearch
        self._fetch_delay = fetch_delay

    def search(self, query: str) -> list[SearchResult]:
        """
        Run a DuckDuckGo text search and return structured results.

        Raises:
            ValueError: If query is blank.
            RuntimeError: If DDG is unreachable or returns no results.
        """
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")

        logger.info("DDG search: %r (max_results=%d)", query, self._max_results)

        attempts = [
            (self._region, self._safesearch, "lite"),
            (self._region, "off", "lite"),
            ("us-en", self._safesearch, "lite"),
            ("us-en", "off", "lite"),
            (self._region, self._safesearch, "html"),
            (self._region, "off", "html"),
            ("us-en", self._safesearch, "html"),
            ("us-en", "off", "html"),
            (None, self._safesearch, "html"),
            (None, "on", "html"),
            (None, self._safesearch, "auto"),
        ]
        seen_attempts: set[tuple[str | None, str, str]] = set()
        raw: list[dict[str, str]] = []
        last_error: Exception | None = None

        try:
            from duckduckgo_search import DDGS

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for region, safesearch, backend in attempts:
                    attempt = (region, safesearch, backend)
                    if attempt in seen_attempts:
                        continue
                    seen_attempts.add(attempt)

                    try:
                        with DDGS() as ddgs:
                            search_kwargs = {
                                "safesearch": safesearch,
                                "backend": backend,
                                "max_results": self._max_results,
                            }
                            if region is not None:
                                search_kwargs["region"] = region
                            raw = list(
                                ddgs.text(query, **search_kwargs)
                            )
                    except Exception as exc:
                        last_error = exc
                        continue

                    if raw:
                        if attempt != (self._region, self._safesearch, "lite"):
                            logger.info(
                                "DDG search fallback succeeded for %r using "
                                "region=%s safesearch=%s backend=%s",
                                query,
                                region,
                                safesearch,
                                backend,
                            )
                        break
        except Exception as exc:
            last_error = exc

        if last_error is not None and not raw:
            raise RuntimeError(
                f"DuckDuckGo search failed for query '{query}': {last_error}"
            ) from last_error

        if not raw:
            raise RuntimeError(
                f"DuckDuckGo returned no results for: '{query}'. "
                "Try rephrasing the query."
            )

        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                published=item.get("published", "") or item.get("date", ""),
            )
            for item in raw
            if item.get("href")
        ]

        logger.info("DDG returned %d results", len(results))
        return results

    def search_and_load(
        self,
        query: str,
        absorb_snippets_on_failure: bool = True,
    ) -> list[Document]:
        """
        Search DDG, fetch full page content for top results, return Documents.

        The absorbed result set is treated as a single logical KB source keyed
        by the query label so repeated absorbs of the same query dedupe cleanly
        with the existing vector-store interface.
        """
        results = self.search(query)
        ingested_at = datetime.now(timezone.utc).isoformat()
        source_label = f"DDG Search: {query}"

        all_docs: list[Document] = []

        for idx, result in enumerate(results[: self._max_fetch]):
            is_valid, err = validate_url(result.url)
            if not is_valid:
                logger.warning("Skipping URL (SSRF block): %s - %s", result.url, err)
                continue

            if idx > 0:
                time.sleep(self._fetch_delay)

            page_docs = self._fetch_page(result, source_label, ingested_at)
            if not page_docs and absorb_snippets_on_failure:
                page_docs = self._snippet_to_docs(result, source_label, ingested_at)

            all_docs.extend(page_docs)

        if len(results) > self._max_fetch and absorb_snippets_on_failure:
            for result in results[self._max_fetch :]:
                all_docs.extend(
                    self._snippet_to_docs(result, source_label, ingested_at)
                )

        if not all_docs:
            raise RuntimeError(
                f"Could not extract any content from search results for: '{query}'"
            )

        logger.info(
            "search_and_load: produced %d chunks for query %r",
            len(all_docs),
            query,
        )
        return all_docs

    def search_as_context(
        self,
        query: str,
        max_results: int = 3,
    ) -> str:
        """
        Search DDG and return results formatted as a prompt context string.
        """
        try:
            results = self.search(query)
        except Exception as exc:
            logger.warning("search_as_context failed silently: %s", exc)
            return ""

        parts: list[str] = [
            f"[Web Search Results for: {query}]",
            "",
        ]
        for i, result in enumerate(results[:max_results], 1):
            parts.append(f"[Result {i}]: {result.title}")
            parts.append(f"  URL: {result.url}")
            parts.append(f"  {result.snippet}")
            if result.published:
                parts.append(f"  Published: {result.published}")
            parts.append("")

        return "\n".join(parts)

    def _fetch_page(
        self,
        result: SearchResult,
        source_label: str,
        ingested_at: str,
    ) -> list[Document]:
        """Fetch full page content and chunk it."""
        try:
            loader = WebBaseLoader(result.url)
            raw_docs = loader.load()
            if not raw_docs:
                return []

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=_CHUNK_SIZE,
                chunk_overlap=_CHUNK_OVERLAP,
            )
            chunks = splitter.split_documents(raw_docs)

            for chunk in chunks:
                chunk.page_content = self._prepend_result_header(
                    chunk.page_content,
                    result,
                )
                chunk.metadata.update(
                    self._make_metadata(
                        source=source_label,
                        source_label=source_label,
                        title=source_label,
                        ingested_at=ingested_at,
                        content=chunk.page_content,
                        result=result,
                    )
                )

            logger.debug("Fetched %d chunks from %s", len(chunks), result.url)
            return chunks
        except Exception as exc:
            logger.warning("Page fetch failed for %s: %s", result.url, exc)
            return []

    def _snippet_to_docs(
        self,
        result: SearchResult,
        source_label: str,
        ingested_at: str,
    ) -> list[Document]:
        """Convert a DDG snippet into a single Document (fallback)."""
        if not result.snippet.strip():
            return []

        content = self._prepend_result_header(result.snippet, result)
        doc = Document(
            page_content=content,
            metadata=self._make_metadata(
                source=source_label,
                source_label=source_label,
                title=source_label,
                ingested_at=ingested_at,
                content=content,
                result=result,
            ),
        )
        return [doc]

    @staticmethod
    def _prepend_result_header(content: str, result: SearchResult) -> str:
        """Embed result metadata in stored content for later attribution."""
        published = result.published or "Unknown"
        return (
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Published: {published}\n\n"
            f"{content}"
        )

    @staticmethod
    def _make_metadata(
        source: str,
        source_label: str,
        title: str,
        ingested_at: str,
        content: str,
        result: SearchResult,
    ) -> dict:
        """Build the standard metadata dict for a chunk."""
        return {
            "source": source,
            "source_type": "web_search",
            "title": title,
            "ingested_at": ingested_at,
            "search_label": source_label,
            "result_title": result.title,
            "result_url": result.url,
            "result_published": result.published,
            "chunk_hash": hashlib.md5(content.encode("utf-8")).hexdigest(),
        }

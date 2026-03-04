"""
ChromaDB-backed knowledge base with multi-notebook collection isolation.

Provides the ``KnowledgeBase`` class — the single gateway for all vector
storage operations.  Key design decisions:

* Uses ``chromadb.PersistentClient`` (not the deprecated legacy API).
* Calls ``_persist()`` explicitly after every write for safety.
* Full source-level deduplication via MD5 hash of the source URL.
* Returns typed ``dataclass`` objects — never raw dicts.
* Every public method has type hints and a docstring.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# ---------------------------------------------------------------------------
# Typed return objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceInfo:
    """Metadata about a single ingested source."""

    source_url: str
    source_hash: str
    source_type: str
    title: str
    chunk_count: int
    ingested_at: str


@dataclass(frozen=True)
class SearchResult:
    """A single result from a similarity search."""

    content: str
    source: str
    source_type: str
    title: str
    score: float
    chunk_hash: str


# ---------------------------------------------------------------------------
# Embedding adapter
# ---------------------------------------------------------------------------

class _OllamaEmbeddingAdapter:
    """Adapts ``langchain_ollama.OllamaEmbeddings`` to the chromadb
    ``EmbeddingFunction`` callable interface.

    ChromaDB expects ``__call__(input: list[str]) -> list[list[float]]``.
    """

    def __init__(self, model: str, base_url: str) -> None:
        """Initialise the adapter.

        Args:
            model: Ollama embedding model name (e.g. ``nomic-embed-text``).
            base_url: Ollama server URL (e.g. ``http://localhost:11434``).
        """
        self._model = OllamaEmbeddings(model=model, base_url=base_url)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        return self._model.embed_documents(input)


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """ChromaDB-backed vector store with source-level deduplication.

    Each ``KnowledgeBase`` instance maps to a single ChromaDB collection,
    providing notebook-level isolation.  Sources are tracked by the MD5
    hash of their URL / identifier, enabling fast duplicate checks and
    targeted deletion.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_model: str,
        ollama_base_url: str,
    ) -> None:
        """Initialise the knowledge base.

        Args:
            persist_dir: Filesystem path for ChromaDB on-disk storage.
            collection_name: ChromaDB collection name (notebook isolation).
            embedding_model: Ollama embedding model name.
            ollama_base_url: Ollama server base URL.

        Raises:
            RuntimeError: If ChromaDB data is corrupted.
        """
        self._db_path = Path(persist_dir)
        self._db_path.mkdir(parents=True, exist_ok=True)

        try:
            self._client: chromadb.ClientAPI = chromadb.PersistentClient(
                path=str(self._db_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB database at '{persist_dir}' appears corrupted: "
                f"{exc}. Use the 'Reset Database' button in the knowledge "
                f"panel, or manually delete the '{persist_dir}' folder."
            ) from exc

        self._embedding_fn = _OllamaEmbeddingAdapter(
            model=embedding_model,
            base_url=ollama_base_url,
        )
        self._collection_name = collection_name

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,  # type: ignore[arg-type]
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open collection '{collection_name}': {exc}.  "
                f"The database may be corrupted — try resetting it."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_source(self, documents: list[Document]) -> SourceInfo:
        """Add a set of document chunks from a single source.

        Performs MD5-based deduplication on the source URL.  If the source
        has already been ingested, raises ``ValueError``.

        Args:
            documents: Pre-chunked LangChain ``Document`` objects.  Each
                must carry metadata keys ``source``, ``source_type``,
                ``ingested_at``, ``title``, and ``chunk_hash``.

        Returns:
            A ``SourceInfo`` dataclass describing the ingested source.

        Raises:
            ValueError: If *documents* is empty or the source already exists.
        """
        if not documents:
            raise ValueError("Cannot add an empty document list.")

        # Extract source identifier from the first document
        source_url: str = documents[0].metadata.get("source", "")
        if not source_url:
            raise ValueError("Documents must have a 'source' metadata key.")

        source_hash = self._hash_source(source_url)

        # --- Deduplication check ---
        if self.source_exists(source_url):
            raise ValueError(
                f"Source already ingested: {source_url}  "
                f"Delete it first if you want to re-ingest."
            )

        # --- Prepare chromadb payloads ---
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for idx, doc in enumerate(documents):
            chunk_hash = doc.metadata.get(
                "chunk_hash",
                hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(),
            )
            doc_id = f"{source_hash}_{idx:05d}"

            meta = {
                "source": doc.metadata.get("source", source_url),
                "source_hash": source_hash,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "title": doc.metadata.get("title", ""),
                "ingested_at": doc.metadata.get("ingested_at", ""),
                "chunk_hash": chunk_hash,
                "chunk_index": idx,
            }

            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(meta)

        # --- Write to ChromaDB ---
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        self._persist()

        return SourceInfo(
            source_url=source_url,
            source_hash=source_hash,
            source_type=metadatas[0]["source_type"],
            title=metadatas[0]["title"],
            chunk_count=len(documents),
            ingested_at=metadatas[0]["ingested_at"],
        )

    def delete_source(self, source_url: str) -> bool:
        """Delete all chunks belonging to a source.

        Args:
            source_url: The original source URL / identifier.

        Returns:
            ``True`` if any documents were deleted, ``False`` otherwise.
        """
        source_hash = self._hash_source(source_url)

        # Check existence first
        existing = self._collection.get(
            where={"source_hash": source_hash},
            include=[],
        )
        if not existing["ids"]:
            return False

        self._collection.delete(where={"source_hash": source_hash})
        self._persist()
        return True

    def source_exists(self, source_url: str) -> bool:
        """Check whether a source has already been ingested.

        Args:
            source_url: The original source URL / identifier.

        Returns:
            ``True`` if at least one chunk from this source exists.
        """
        source_hash = self._hash_source(source_url)
        results = self._collection.get(
            where={"source_hash": source_hash},
            limit=1,
            include=[],
        )
        return bool(results["ids"])

    def get_all_sources(self) -> list[SourceInfo]:
        """Return metadata for every unique source in the collection.

        Returns:
            List of ``SourceInfo`` dataclasses, sorted by ingestion time
            (most recent first).
        """
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.get(include=["metadatas"])
        metas: list[dict] = results.get("metadatas", [])

        # Aggregate by source_hash
        source_map: dict[str, dict] = {}
        for meta in metas:
            if not meta:
                continue
            sh = meta.get("source_hash", "")
            if sh not in source_map:
                source_map[sh] = {
                    "source_url": meta.get("source", ""),
                    "source_hash": sh,
                    "source_type": meta.get("source_type", "unknown"),
                    "title": meta.get("title", ""),
                    "chunk_count": 0,
                    "ingested_at": meta.get("ingested_at", ""),
                }
            source_map[sh]["chunk_count"] += 1

        sources = [
            SourceInfo(**info) for info in source_map.values()
        ]
        # Sort by ingestion time, most recent first
        sources.sort(key=lambda s: s.ingested_at, reverse=True)
        return sources

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the knowledge base for chunks similar to a query.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of ``SearchResult`` dataclasses ordered by relevance
            (lowest distance first).
        """
        total = self._collection.count()
        if total == 0:
            return []

        # Clamp top_k to available documents
        effective_k = min(top_k, total)

        results = self._collection.query(
            query_texts=[query],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for content, meta, dist in zip(docs, metas, distances):
            search_results.append(
                SearchResult(
                    content=content or "",
                    source=meta.get("source", "Unknown") if meta else "Unknown",
                    source_type=meta.get("source_type", "unknown") if meta else "unknown",
                    title=meta.get("title", "") if meta else "",
                    score=1.0 - dist if dist is not None else 0.0,
                    chunk_hash=meta.get("chunk_hash", "") if meta else "",
                )
            )

        return search_results

    def clear_all(self) -> None:
        """Delete the entire collection and recreate it empty.

        This permanently removes all ingested sources and their chunks.
        """
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        self._persist()

    def reset_database(self) -> None:
        """Nuclear option: delete and recreate the entire ChromaDB store.

        Use this when the database is corrupted and normal operations
        fail.  Destroys **all** data across **all** collections.
        """
        try:
            del self._client  # release file handles
        except Exception:
            pass
        if self._db_path.exists():
            shutil.rmtree(self._db_path, ignore_errors=True)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        self._persist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Explicitly persist data to disk after a write operation.

        ``chromadb.PersistentClient`` auto-persists on every mutation in
        chromadb ≥ 0.4.  This method is called explicitly after each write
        to satisfy the explicit-persistence contract and to remain
        forward-compatible with any future chromadb changes.
        """
        if hasattr(self._client, "persist"):
            self._client.persist()  # type: ignore[attr-defined]

    @staticmethod
    def _hash_source(source_url: str) -> str:
        """Compute a stable MD5 hex digest for a source identifier.

        Args:
            source_url: The raw source URL or identifier string.

        Returns:
            32-character lowercase hex string.
        """
        return hashlib.md5(source_url.encode("utf-8")).hexdigest()

"""
Streaming RAG chain built with LangChain Expression Language (LCEL).

Provides the ``RAGChain`` class — the intelligence backbone of the app.
Key design decisions:

* **LCEL only** — no legacy ``create_retrieval_chain`` or
  ``create_stuff_documents_chain``.  Chains are composed with the ``|``
  pipe operator.
* **Async streaming** via ``astream()`` — returns a ``StreamingResponse``
  containing an ``AsyncGenerator[str, None]`` of tokens **and** the
  source documents used for attribution.
* **Confidence gating** — if the retriever finds zero relevant chunks
  the RAG path is skipped entirely and the base model is used with an
  explicit disclaimer.
* **Prompt engineering** — the LLM is instructed to answer *only* from
  context, cite sources inline, and refuse to hallucinate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from core.vectorstore import KnowledgeBase, SearchResult


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT: str = """\
You are a precise, citation-aware research assistant.  The user has \
built a personal knowledge base and is asking questions about it.

## Rules — follow these WITHOUT exception

1. Answer **only** from the context provided below.  If the context \
does not contain enough information to answer fully, say exactly: \
"I couldn't find that in your knowledge base." and stop.  \
**Never fabricate or hallucinate information.**
2. For every factual claim, cite the source inline using the format \
[Source N] where N corresponds to the numbered sources below.
3. Be concise but thorough.  Use bullet points or numbered lists \
when appropriate.
4. If multiple sources corroborate a fact, cite all of them, e.g. \
[Source 1][Source 3].
5. If the context partially answers the question, share what you can \
and explicitly note what is missing.

## Context

{context}
"""

_BASE_SYSTEM_PROMPT: str = """\
You are a helpful AI assistant.  The user's knowledge base did not \
contain any information relevant to this question, so you are \
answering from your general training data.

⚠️ **Disclaimer**: This response is NOT sourced from your uploaded \
documents.  It comes from the model's general knowledge and may not \
be perfectly accurate.  For verified answers, add relevant sources to \
your knowledge base first.

Be concise, factual, and honest about uncertainty.\
"""


# ---------------------------------------------------------------------------
# Streaming response container
# ---------------------------------------------------------------------------

@dataclass
class StreamingResponse:
    """Container returned by ``RAGChain.astream()``.

    Attributes:
        source_documents: The ``SearchResult`` objects retrieved from
            the knowledge base (empty in ``base`` mode).
        mode: ``"rag"`` if answering from context, ``"base"`` if the
            knowledge base had no relevant documents.
        token_stream: An async generator that yields string tokens as
            they are produced by the LLM.
    """

    source_documents: list[SearchResult] = field(default_factory=list)
    mode: str = "rag"
    token_stream: AsyncGenerator[str, None] = field(default=None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------

class RAGChain:
    """LCEL-based retrieval-augmented generation chain with streaming.

    Typical usage::

        chain = RAGChain(kb, model="deepseek-r1:latest",
                         ollama_base_url="http://localhost:11434")
        resp = await chain.astream("What is X?")
        # resp.source_documents is available immediately
        async for token in resp.token_stream:
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model: str,
        ollama_base_url: str,
    ) -> None:
        """Initialise the RAG chain.

        Args:
            knowledge_base: The ``KnowledgeBase`` instance to retrieve from.
            model: Ollama model name for generation.
            ollama_base_url: Ollama server URL.
        """
        self._kb = knowledge_base
        self._llm = OllamaLLM(model=model, base_url=ollama_base_url)

        # --- LCEL prompts (never changes after init) ---------------------
        self._rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _BASE_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def astream(
        self,
        query: str,
        top_k: int = 5,
    ) -> StreamingResponse:
        """Run a streaming RAG query.

        1. Retrieves top-k similar chunks from the knowledge base.
        2. If no chunks are found (**confidence gating**), falls back to
           the base model with a disclaimer.
        3. Returns a ``StreamingResponse`` whose ``token_stream`` is an
           async generator of string tokens.

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to retrieve.

        Returns:
            A ``StreamingResponse`` with source documents and a token
            stream ready for consumption.
        """
        # --- Retrieval (synchronous — chromadb is not async) -------------
        results: list[SearchResult] = self._kb.similarity_search(
            query, top_k=top_k
        )

        if not results:
            # Confidence gating — no relevant documents
            return StreamingResponse(
                source_documents=[],
                mode="base",
                token_stream=self._stream_base(query),
            )

        return StreamingResponse(
            source_documents=results,
            mode="rag",
            token_stream=self._stream_rag(query, results),
        )

    # ------------------------------------------------------------------
    # Private streaming generators
    # ------------------------------------------------------------------

    async def _stream_rag(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AsyncGenerator[str, None]:
        """Stream tokens using retrieved context via LCEL.

        Args:
            query: The user's question.
            results: Retrieved ``SearchResult`` objects.

        Yields:
            Individual string tokens as they arrive from the LLM.
        """
        context = self._format_context(results)
        chain = self._rag_prompt | self._llm

        try:
            async for token in chain.astream(
                {"context": context, "input": query}
            ):
                yield token
        except Exception as exc:
            yield (
                f"\n\n⚠️ An error occurred while generating the response: "
                f"{exc}"
            )

    async def _stream_base(
        self,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the base model (no context) via LCEL.

        Args:
            query: The user's question.

        Yields:
            Individual string tokens as they arrive from the LLM.
        """
        chain = self._base_prompt | self._llm

        try:
            async for token in chain.astream({"input": query}):
                yield token
        except Exception as exc:
            yield (
                f"\n\n⚠️ An error occurred while generating the response: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format retrieved results into a numbered context block.

        Args:
            results: The search results to format.

        Returns:
            A string where each source is labelled ``[Source N]`` with
            its title, origin, and content.
        """
        parts: list[str] = []
        for idx, result in enumerate(results, 1):
            header = f"[Source {idx}]: {result.title or result.source}"
            origin = f"  Origin: {result.source} ({result.source_type})"
            parts.append(f"{header}\n{origin}\n{result.content}")
        return "\n\n---\n\n".join(parts)

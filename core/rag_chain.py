"""
Streaming RAG chain built with LangChain Expression Language (LCEL).

Retrieval order:
    1. Knowledge base
    2. Live DuckDuckGo web search
    3. Base model fallback
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm
from core.vectorstore import KnowledgeBase, SearchResult
from utils.logging import get_logger

logger = get_logger(__name__)

_RAG_SYSTEM_PROMPT: str = """\
You are a precise, citation-aware research assistant. The user has \
built a personal knowledge base and is asking questions about it.

## Rules - follow these without exception

1. Answer only from the context provided below. If the context \
does not contain enough information to answer fully, say exactly: \
"I couldn't find that in your knowledge base." and stop. \
Never fabricate or hallucinate information.
2. For every factual claim, cite the source inline using the format \
[Source N] where N corresponds to the numbered sources below.
3. Be concise but thorough. Use bullet points or numbered lists \
when appropriate.
4. If multiple sources corroborate a fact, cite all of them, e.g. \
[Source 1][Source 3].
5. If the context partially answers the question, share what you can \
and explicitly note what is missing.

## Context

{context}
"""

_WEB_SEARCH_SYSTEM_PROMPT: str = """\
You are a precise research assistant. The user's personal knowledge base had no
relevant information, so you have been given live web search results to answer
their question.

## Rules

1. Answer using ONLY the web search results provided below.
2. Cite every factual claim with [Result N] where N matches the numbered results.
3. If the results are insufficient, say so clearly. Do not fabricate.
4. Always note the recency of information where available.
5. Be concise and structured - use bullet points where appropriate.

## Web Search Results

{context}
"""

_BASE_SYSTEM_PROMPT: str = """\
You are a helpful AI assistant. The user's knowledge base did not \
contain any information relevant to this question, so you are \
answering from your general training data.

Disclaimer: This response is not sourced from uploaded documents. \
It comes from the model's general knowledge and may not be perfectly \
accurate. For verified answers, add relevant sources to the knowledge \
base first.

Be concise, factual, and honest about uncertainty.\
"""


@dataclass
class StreamingResponse:
    """Container returned by `RAGChain.astream()`."""

    source_documents: list[SearchResult] = field(default_factory=list)
    mode: str = "rag"
    web_results_used: bool = False
    token_stream: AsyncGenerator[str, None] = field(default=None)  # type: ignore[assignment]


class RAGChain:
    """Retrieval-augmented generation chain with async streaming."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        model: str,
        provider: str = "ollama",
        ollama_base_url: str | None = None,
        enable_web_search: bool = False,
        web_search_max_results: int = 3,
    ) -> None:
        self._kb = knowledge_base
        self._llm = get_llm(model=model, provider=provider)
        self._enable_web_search = enable_web_search

        self._web_searcher = None
        if enable_web_search:
            from core.search import WebSearcher

            self._web_searcher = WebSearcher(
                max_results=web_search_max_results,
                max_fetch=0,
            )

        self._rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._web_search_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _WEB_SEARCH_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._base_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _BASE_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        self._unused_ollama_base_url = ollama_base_url

    async def astream(
        self,
        query: str,
        top_k: int = 5,
    ) -> StreamingResponse:
        """
        Three-tier retrieval: KB -> Web Search -> Base LLM.
        """
        kb_results = self._kb.similarity_search(query, top_k=top_k)

        if kb_results:
            return StreamingResponse(
                source_documents=kb_results,
                mode="rag",
                web_results_used=False,
                token_stream=self._stream_rag(query, kb_results),
            )

        if self._enable_web_search and self._web_searcher is not None:
            try:
                web_context = self._web_searcher.search_as_context(
                    query, max_results=3
                )
                if web_context:
                    return StreamingResponse(
                        source_documents=[],
                        mode="web_search",
                        web_results_used=True,
                        token_stream=self._stream_web_search(query, web_context),
                    )
            except Exception as exc:
                logger.warning("Web search fallback failed: %s", exc)

        return StreamingResponse(
            source_documents=[],
            mode="base",
            web_results_used=False,
            token_stream=self._stream_base(query),
        )

    async def _stream_rag(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AsyncGenerator[str, None]:
        """Stream tokens using retrieved KB context via LCEL."""
        context = self._format_context(results)
        chain = self._rag_prompt | self._llm

        try:
            async for token in chain.astream({"context": context, "input": query}):
                yield self._chunk_to_text(token)
        except Exception as exc:
            yield f"\n\nAn error occurred while generating the response: {exc}"

    async def _stream_web_search(
        self,
        query: str,
        web_context: str,
    ) -> AsyncGenerator[str, None]:
        """Stream an answer from web search context via LCEL."""
        chain = self._web_search_prompt | self._llm
        try:
            async for token in chain.astream(
                {"context": web_context, "input": query}
            ):
                yield self._chunk_to_text(token)
        except Exception as exc:
            yield f"\n\nError generating response from web results: {exc}"

    async def _stream_base(self, query: str) -> AsyncGenerator[str, None]:
        """Stream tokens from the base model with no retrieved context."""
        chain = self._base_prompt | self._llm

        try:
            async for token in chain.astream({"input": query}):
                yield self._chunk_to_text(token)
        except Exception as exc:
            yield f"\n\nAn error occurred while generating the response: {exc}"

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format retrieved KB results into a numbered context block."""
        parts: list[str] = []
        for idx, result in enumerate(results, 1):
            header = f"[Source {idx}]: {result.title or result.source}"
            origin = f"  Origin: {result.source} ({result.source_type})"
            parts.append(f"{header}\n{origin}\n{result.content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _chunk_to_text(chunk: object) -> str:
        """Normalize provider-specific stream chunks to plain text."""
        content = getattr(chunk, "content", chunk)
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    parts.append(str(text))
                else:
                    text = (
                        getattr(part, "text", None)
                        or getattr(part, "content", None)
                        or part
                    )
                    parts.append(str(text))
            return "".join(parts)
        return str(content)

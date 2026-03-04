"""
Streaming chat interface with source citations.

Renders conversation history, handles new messages with token-by-token
streaming via ``st.write_stream()``, and shows collapsible source
attribution below every assistant response.  Includes copy-to-clipboard
and export-as-markdown controls.

The async ``RAGChain.astream()`` is bridged to a sync generator so
that Streamlit's synchronous ``st.write_stream()`` can consume it.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Generator
from pathlib import Path
from typing import Any

import streamlit as st

from core.models import EMBEDDING_MODEL, _get_secret
from core.rag_chain import RAGChain, StreamingResponse
from core.vectorstore import KnowledgeBase, SearchResult
from persistence.chat_store import ChatStore


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _get_kb() -> KnowledgeBase:
    """Return the active ``KnowledgeBase`` from session state."""
    if "kb" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.kb = KnowledgeBase(
            persist_dir=str(base / "data" / "chroma_db"),
            collection_name=st.session_state.get("notebook_id", "default"),
            embedding_model=EMBEDDING_MODEL,
            ollama_base_url=_get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return st.session_state.kb


def _get_chat_store() -> ChatStore:
    """Return the ``ChatStore`` singleton from session state."""
    if "chat_store" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.chat_store = ChatStore(
            str(base / "data" / "chat_history.db")
        )
    return st.session_state.chat_store


def _init_messages() -> None:
    """Load messages from persistent store into session state."""
    if "messages" not in st.session_state:
        store = _get_chat_store()
        notebook_id = st.session_state.get("notebook_id", "default")
        history = store.load_history(notebook_id)
        st.session_state.messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources_cited,
                "mode": "rag" if msg.sources_cited else "base",
            }
            for msg in history
        ]


# ---------------------------------------------------------------------------
# Async → Sync bridge
# ---------------------------------------------------------------------------

def _bridge_async_stream(
    async_gen: Any,
    loop: asyncio.AbstractEventLoop,
) -> Generator[str, None, None]:
    """Convert an async token generator to sync for ``st.write_stream()``.

    Args:
        async_gen: The ``AsyncGenerator[str, None]`` from ``RAGChain``.
        loop: An event loop that stays alive for the streaming duration.

    Yields:
        Individual string tokens.
    """
    while True:
        try:
            token: str = loop.run_until_complete(async_gen.__anext__())
            yield token
        except StopAsyncIteration:
            break


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_chat() -> None:
    """Render the complete chat interface: history + input + controls."""
    _init_messages()

    messages: list[dict] = st.session_state.messages

    # ---- Chat controls (top row) ----------------------------------------
    _render_chat_controls()

    # ---- Empty state ----------------------------------------------------
    if not messages:
        _render_empty_state()

    # ---- History --------------------------------------------------------
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_message_actions(msg, idx)

    # ---- Input ----------------------------------------------------------
    if prompt := st.chat_input("Ask your knowledge base a question…"):
        _handle_user_input(prompt)


# ---------------------------------------------------------------------------
# User input handler
# ---------------------------------------------------------------------------

def _handle_user_input(prompt: str) -> None:
    """Process a new user message: persist, stream, display."""
    prompt = prompt.strip()
    if not prompt:
        return

    notebook_id = st.session_state.get("notebook_id", "default")
    store = _get_chat_store()
    kb = _get_kb()
    model = st.session_state.get("selected_model", "deepseek-r1:latest")

    # 1 — Save & display user message
    store.save_message(notebook_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": [], "mode": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2 — Stream assistant response
    with st.chat_message("assistant"):
        if not st.session_state.get("ollama_running", False):
            err = "⚠️ Ollama is not running. Start Ollama and refresh the page."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": [], "mode": "error"})
            return

        try:
            chain = RAGChain(
                knowledge_base=kb,
                model=model,
                ollama_base_url=_get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

            # Bridge async → sync
            loop = asyncio.new_event_loop()
            try:
                resp: StreamingResponse = loop.run_until_complete(
                    chain.astream(prompt, top_k=5)
                )

                # Stream tokens — st.write_stream returns the full answer
                answer: str = st.write_stream(
                    _bridge_async_stream(resp.token_stream, loop)
                )
            finally:
                loop.close()

            # Sources & mode badge
            source_names: list[str] = [s.source for s in resp.source_documents]
            _render_source_cards(resp.source_documents, resp.mode)

            # 3 — Persist assistant message
            store.save_message(notebook_id, "assistant", answer, sources_cited=source_names)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_names,
                "mode": resp.mode,
            })

        except Exception as exc:
            err = f"⚠️ Something went wrong: {exc}"
            st.error(err)
            st.info("Make sure Ollama is running and your model is downloaded.")
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": [], "mode": "error"})


# ---------------------------------------------------------------------------
# Message actions: sources + copy
# ---------------------------------------------------------------------------

def _render_message_actions(msg: dict, idx: int) -> None:
    """Render source attribution and copy button below an assistant message."""
    col_mode, col_copy = st.columns([5, 1])

    with col_mode:
        if msg.get("mode") == "rag" and msg.get("sources"):
            st.caption("💡 Answered from your knowledge base")
        elif msg.get("mode") == "base":
            st.caption("🌐 Answered from general knowledge — no matching sources")
        elif msg.get("mode") == "error":
            pass  # error messages don't need a badge

    with col_copy:
        _render_copy_button(msg["content"], f"copy_{idx}")

    # Collapsible sources
    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"📎 Sources Used ({len(sources)})", expanded=False):
            for i, s in enumerate(sources, 1):
                # Determine icon
                if "youtube" in s.lower() or "youtu.be" in s.lower():
                    icon = "🎥"
                elif s.lower().endswith(".pdf"):
                    icon = "📄"
                elif "perplexity" in s.lower():
                    icon = "🔍"
                elif s.startswith("http"):
                    icon = "🌐"
                else:
                    icon = "📝"
                st.caption(f"{icon} [{i}] {s}")


def _render_source_cards(results: list[SearchResult], mode: str) -> None:
    """Render inline source cards immediately after streaming finishes."""
    if mode == "rag" and results:
        st.caption("💡 Answered from your knowledge base")
        with st.expander(f"📎 Sources Used ({len(results)})", expanded=False):
            for i, r in enumerate(results, 1):
                if "youtube" in r.source.lower():
                    icon = "🎥"
                elif r.source.lower().endswith(".pdf"):
                    icon = "📄"
                elif "perplexity" in r.source.lower():
                    icon = "🔍"
                elif r.source.startswith("http"):
                    icon = "🌐"
                else:
                    icon = "📝"
                st.markdown(
                    f"**{icon} Source {i}:** {r.title or r.source}",
                )
                st.caption(
                    f"_{r.source_type}_ • Relevance: {r.score:.0%}"
                )
                st.caption(r.content[:200] + "…" if len(r.content) > 200 else r.content)
                if i < len(results):
                    st.markdown("---")
    elif mode == "base":
        st.caption("🌐 No matching sources found — answered from general knowledge")


# ---------------------------------------------------------------------------
# Copy to clipboard
# ---------------------------------------------------------------------------

def _render_copy_button(content: str, key: str) -> None:
    """Inject a copy-to-clipboard button via HTML/JS.

    Uses Base64 encoding to safely pass content through inline JS
    without special-character escaping issues.

    Args:
        content: Text to copy.
        key: Unique Streamlit element key.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    st.markdown(
        f"""<button onclick="
            navigator.clipboard.writeText(atob('{b64}'));
            this.innerText='✅';
            setTimeout(()=>this.innerText='📋',1500);
        " style="background:none;border:1px solid #333;
        border-radius:4px;color:#71717A;cursor:pointer;
        padding:2px 8px;font-size:0.72rem;"
        title="Copy to clipboard">📋</button>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat controls: clear + export
# ---------------------------------------------------------------------------

def _render_chat_controls() -> None:
    """Clear-chat and export buttons."""
    col_clear, col_export, _ = st.columns([1, 1, 4])

    with col_clear:
        if st.button("🗑️ Clear", key="clear_chat_btn", help="Clear chat history"):
            notebook_id = st.session_state.get("notebook_id", "default")
            _get_chat_store().clear_history(notebook_id)
            st.session_state.messages = []
            st.rerun()

    with col_export:
        notebook_id = st.session_state.get("notebook_id", "default")
        md_content = _get_chat_store().export_as_markdown(notebook_id)
        st.download_button(
            "📥 Export",
            data=md_content,
            file_name=f"chat_{notebook_id}.md",
            mime="text/markdown",
            key="export_chat_btn",
            help="Export chat as Markdown",
        )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def _render_empty_state() -> None:
    """Render a welcome screen when no messages exist."""
    st.markdown(
        """<div style="text-align:center;padding:3rem 2rem;animation:fadeIn 0.5s ease;">
            <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
            <div style="font-size:1.1rem;font-weight:600;color:#71717A;margin-bottom:0.5rem;">
                Your AI Knowledge Base
            </div>
            <div style="font-size:0.85rem;color:#52525B;">
                Add sources from the sidebar — PDFs, websites, YouTube videos,
                or run web research — then ask questions about your content.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

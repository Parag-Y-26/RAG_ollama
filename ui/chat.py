"""
Streaming chat interface with source citations and inline web search.

All streaming is fully synchronous — no asyncio bridges needed.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from config.settings import settings
from core.rag_chain import RAGChain, StreamingResponse
from core.vectorstore import KnowledgeBase, SearchResult
from persistence.chat_store import ChatStore


def _get_kb() -> KnowledgeBase:
    """Return the active knowledge base from session state."""
    if "kb" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.kb = KnowledgeBase(
            persist_dir=str(base / "data" / "chroma_db"),
            collection_name=st.session_state.get("notebook_id", "default"),
            embedding_model=settings.embedding_model,
            ollama_base_url=settings.ollama_base_url,
        )
    return st.session_state.kb


def _get_chat_store() -> ChatStore:
    """Return the chat store singleton from session state."""
    if "chat_store" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.chat_store = ChatStore(
            str(base / "data" / "chat_history.db")
        )
    return st.session_state.chat_store


@st.cache_resource
def _get_rag_chain(
    notebook_id: str,
    model: str,
    provider: str,
    enable_web_search: bool,
) -> RAGChain:
    """Return a cached RAG chain per notebook, model, and web-search mode."""
    del notebook_id

    return RAGChain(
        knowledge_base=_get_kb(),
        model=model,
        provider=provider,
        enable_web_search=enable_web_search,
        web_search_max_results=settings.web_search_max_results,
    )


def _init_messages() -> None:
    """Load persisted messages into session state."""
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


def render_chat() -> None:
    """Render the complete chat interface."""
    _init_messages()

    messages: list[dict] = st.session_state.messages
    _render_chat_controls()

    if not messages:
        _render_empty_state()

    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_message_actions(msg, idx)

    if prompt := st.chat_input("Ask your knowledge base a question..."):
        _handle_user_input(prompt)


def _handle_user_input(prompt: str) -> None:
    """Process a new user message, persist it, and stream the reply."""
    prompt = prompt.strip()
    if not prompt:
        return

    # --- Duplicate message guard (prevents double-send on Streamlit rerun) ---
    last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
    if last_msg.get("role") == "user" and last_msg.get("content") == prompt:
        return

    lowered = prompt.lower()
    if lowered == "/search" or lowered.startswith("/search "):
        query = prompt[7:].strip()
        if query:
            _handle_inline_search(query)
        else:
            st.warning("Usage: /search <your query>")
        return

    notebook_id = st.session_state.get("notebook_id", "default")
    store = _get_chat_store()
    provider = st.session_state.get(
        "selected_provider",
        settings.llm_provider or "ollama",
    )
    model = st.session_state.get("selected_model", settings.default_model)
    enable_web_search = st.session_state.get(
        "enable_web_search",
        settings.web_search_enabled_by_default,
    )

    # 1 — Append to session state + persist user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": [], "mode": ""}
    )
    store.save_message(notebook_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2 — Stream assistant response
    with st.chat_message("assistant"):
        # Only block if local Ollama is needed and not running
        if provider == "ollama" and not st.session_state.get("ollama_running", False):
            err = "Ollama is not running. Start Ollama and refresh the page."
            st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                    "sources": [],
                    "mode": "error",
                }
            )
            return

        try:
            chain = _get_rag_chain(
                notebook_id,
                model,
                provider,
                enable_web_search,
            )
            # Fully synchronous — no event loops needed
            resp = chain.stream(prompt, top_k=5)
            answer: str = st.write_stream(resp.token_stream)

            source_names: list[str] = [
                source.source for source in resp.source_documents
            ]
            _render_source_cards(resp.source_documents, resp.mode)

            # 3 — Persist assistant message
            store.save_message(
                notebook_id,
                "assistant",
                answer,
                sources_cited=source_names,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_names,
                    "mode": resp.mode,
                }
            )
        except Exception as exc:
            err = f"Something went wrong: {exc}"
            st.error(err)
            st.info("Make sure Ollama is running and your model is downloaded.")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                    "sources": [],
                    "mode": "error",
                }
            )


def _handle_inline_search(query: str) -> None:
    """
    Handle a /search <query> command typed in the chat.

    Runs a DDG search and displays results inline in the chat thread.
    """
    from core.search import WebSearcher

    notebook_id = st.session_state.get("notebook_id", "default")
    store = _get_chat_store()

    # Dedup guard for /search messages
    last = st.session_state.messages[-1] if st.session_state.messages else {}
    if not (last.get("role") == "user" and last.get("content") == f"/search {query}"):
        store.save_message(notebook_id, "user", f"/search {query}")
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"/search {query}",
                "sources": [],
                "mode": "",
            }
        )

    with st.chat_message("user"):
        st.markdown(f"`/search` {query}")

    with st.chat_message("assistant"):
        with st.spinner(f"Searching…"):
            try:
                searcher = WebSearcher(
                    max_results=settings.web_search_max_results,
                    max_fetch=0,
                )
                results = searcher.search(query)

                lines = [f"**Web search results for:** `{query}`", ""]
                for i, result in enumerate(results, 1):
                    lines.append(f"**{i}. [{result.title}]({result.url})**")
                    lines.append(result.snippet)
                    if result.published:
                        lines.append(f"*{result.published}*")
                    lines.append("")
                answer = "\n".join(lines)

                st.markdown(answer)
                st.caption(
                    f"↳ {len(results)} results from DuckDuckGo · not stored in KB"
                )

                if st.button(
                    f"Absorb all {len(results)} results into KB",
                    key=f"absorb_inline_{abs(hash(query)) % 100000}",
                ):
                    kb = _get_kb()
                    with st.spinner("Absorbing…"):
                        try:
                            docs = searcher.search_and_load(
                                query,
                                absorb_snippets_on_failure=True,
                            )
                            info = kb.add_source(docs)
                            st.success(f"{info.chunk_count} chunks added to your KB.")
                        except ValueError:
                            st.info("These results are already in your KB.")
                        except Exception as exc:
                            st.error(f"Absorb failed: {exc}")

                store.save_message(
                    notebook_id,
                    "assistant",
                    answer,
                    sources_cited=[],
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "mode": "web_search",
                    }
                )
            except RuntimeError as exc:
                # User-friendly rate limit message
                msg = str(exc)
                if "rate limit" in msg.lower() or "attempt" in msg.lower():
                    st.warning(
                        "DuckDuckGo is temporarily rate limiting requests. "
                        "Wait 30 seconds and try again, or use the Sidebar → Search tab."
                    )
                else:
                    st.error(msg)
            except Exception as exc:
                st.error(f"Search failed: {exc}")


def _render_message_actions(msg: dict, idx: int) -> None:
    """Render source attribution and copy button below an assistant message."""
    col_mode, col_copy = st.columns([5, 1])

    with col_mode:
        content = msg.get("content", "")
        mode = msg.get("mode", "")

        if mode == "rag" and msg.get("sources"):
            st.caption("↳ answered from knowledge base")
        elif mode == "web_search" and not content.startswith("⚠️"):
            st.caption("↳ answered from live web search")
        elif mode == "base":
            st.caption("↳ answered from model knowledge")

    with col_copy:
        _render_copy_button(msg["content"], f"copy_{idx}")

    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"Sources Used ({len(sources)})", expanded=False):
            for i, source in enumerate(sources, 1):
                if "youtube" in source.lower() or "youtu.be" in source.lower():
                    icon = "Video"
                elif source.lower().endswith(".pdf"):
                    icon = "PDF"
                elif "perplexity" in source.lower():
                    icon = "Research"
                elif source.startswith("http"):
                    icon = "Web"
                else:
                    icon = "Text"
                st.caption(f"{icon} [{i}] {source}")


def _render_source_cards(results: list[SearchResult], mode: str) -> None:
    """Render inline source cards after streaming finishes."""
    if mode == "rag" and results:
        st.caption("↳ answered from knowledge base")
        with st.expander(f"Sources Used ({len(results)})", expanded=False):
            for i, result in enumerate(results, 1):
                if "youtube" in result.source.lower():
                    label = "Video"
                elif result.source.lower().endswith(".pdf"):
                    label = "PDF"
                elif "perplexity" in result.source.lower():
                    label = "Research"
                elif result.source.startswith("http"):
                    label = "Web"
                else:
                    label = "Text"

                st.markdown(f"**{label} Source {i}:** {result.title or result.source}")
                st.caption(f"{result.source_type} - Relevance: {result.score:.0%}")
                preview = result.content[:200]
                if len(result.content) > 200:
                    preview += "..."
                st.caption(preview)
                if i < len(results):
                    st.markdown("---")
    elif mode == "web_search":
        st.caption("↳ answered from live web search  ·  not stored in KB")
    elif mode == "base":
        st.caption("↳ answered from model knowledge  ·  no sources found")


def _render_copy_button(content: str, key: str) -> None:
    """Inject a copy-to-clipboard button via HTML and inline JS."""
    del key

    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    st.markdown(
        f"""<button onclick="
            navigator.clipboard.writeText(atob('{b64}'));
            this.innerText='OK';
            setTimeout(()=>this.innerText='Copy',1500);
        " style="font-family:'DM Mono','Fira Mono','Courier New',monospace;
        background:#000000;border:1px solid #1C1C1C;
        border-radius:2px;color:#555555;cursor:pointer;
        padding:2px 8px;font-size:0.72rem;"
        title="Copy to clipboard">Copy</button>""",
        unsafe_allow_html=True,
    )


def _render_chat_controls() -> None:
    """Render clear-chat and export buttons."""
    col_clear, col_export, _ = st.columns([1, 1, 4])

    with col_clear:
        if st.button("Clear", key="clear_chat_btn", help="Clear chat history"):
            notebook_id = st.session_state.get("notebook_id", "default")
            _get_chat_store().clear_history(notebook_id)
            st.session_state.messages = []
            st.rerun()

    with col_export:
        notebook_id = st.session_state.get("notebook_id", "default")
        md_content = _get_chat_store().export_as_markdown(notebook_id)
        st.download_button(
            "Export",
            data=md_content,
            file_name=f"chat_{notebook_id}.md",
            mime="text/markdown",
            key="export_chat_btn",
            help="Export chat as Markdown",
        )


def _render_empty_state() -> None:
    """Render a welcome screen when no messages exist."""
    st.markdown(
        """<div class="empty-state">
            <span class="emoji">◈</span>
            <div class="title">Knowledge Base Empty</div>
            <div class="subtitle">
                Add sources from the sidebar.<br>
                PDFs · web pages · YouTube · research.<br><br>
                Or type <code>/search &lt;query&gt;</code> to search the web.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

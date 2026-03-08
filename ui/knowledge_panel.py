"""
Right-side knowledge panel for source inspection and management.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import settings
from core.llm import get_llm
from core.vectorstore import KnowledgeBase, SourceInfo


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


def _icon_for_type(source_type: str, source_url: str) -> str:
    """Return a source icon based on type or URL."""
    mapping = {
        "youtube": "▶",
        "web": "↗",
        "pdf": "▤",
        "text": "≡",
        "perplexity": "◎",
        "web_search": "◌",
    }
    icon = mapping.get(source_type)
    if icon:
        return icon

    url = source_url.lower()
    if "youtube" in url or "youtu.be" in url:
        return "▶"
    if url.endswith(".pdf"):
        return "▤"
    if "perplexity" in url:
        return "◎"
    if url.startswith("http"):
        return "↗"
    return "≡"


def _type_badge(source_type: str) -> str:
    """Return a monochrome type label."""
    return (
        f'<span style="'
        f'border: 1px solid #1C1C1C;'
        f'color: #555555;'
        f'padding: 1px 6px;'
        f'border-radius: 2px;'
        f'font-size: 9px;'
        f'font-weight: 400;'
        f'letter-spacing: 0.1em;'
        f'text-transform: uppercase;'
        f'font-family: DM Mono, monospace;'
        f'">{source_type}</span>'
    )


def _stream_chunk_text(token: object) -> str:
    """Normalize streamed LLM chunks to plain text."""
    content = getattr(token, "content", token)
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


def render_knowledge_panel() -> None:
    """Render the full knowledge panel in the right column."""
    st.markdown("### Sources")

    kb = _get_kb()
    sources = kb.get_all_sources()

    _render_stats(sources)
    if sources:
        _render_kb_summary(sources)

    if not sources:
        st.caption("No sources yet. Add knowledge from the sidebar.")
        return

    search_query = st.text_input(
        "Filter sources",
        key="source_filter",
        placeholder="Type to filter...",
        label_visibility="collapsed",
    )

    filtered = sources
    if search_query:
        query = search_query.lower()
        filtered = [
            source
            for source in sources
            if query in source.title.lower() or query in source.source_url.lower()
        ]

    if not filtered:
        st.caption("No sources match your filter.")
    else:
        st.caption(f"Showing {len(filtered)} of {len(sources)} sources")

    for source in filtered:
        _render_source_row(source, kb)

    st.divider()
    _render_clear_all(kb)
    _render_reset_database(kb)


def _render_stats(sources: list[SourceInfo]) -> None:
    """Display aggregate knowledge base statistics."""
    total_sources = len(sources)
    total_chunks = sum(source.chunk_count for source in sources)
    est_tokens = total_chunks * 250

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sources", total_sources)
    with col2:
        st.metric("Chunks", total_chunks)
    with col3:
        if est_tokens >= 1_000_000:
            label = f"{est_tokens / 1_000_000:.1f}M"
        elif est_tokens >= 1_000:
            label = f"{est_tokens / 1_000:.0f}K"
        else:
            label = str(est_tokens)
        st.metric("Est. Tokens", label)


def _render_source_row(src: SourceInfo, kb: KnowledgeBase) -> None:
    """Render a single source with metadata and delete controls."""
    icon = _icon_for_type(src.source_type, src.source_url)
    badge = _type_badge(src.source_type)
    timestamp = src.ingested_at[:16].replace("T", " ") if src.ingested_at else ""

    st.markdown(
        f"{icon} **{src.title or src.source_url}** {badge}",
        unsafe_allow_html=True,
    )
    st.caption(f"{src.chunk_count} chunks - {timestamp}")

    confirm_key = f"confirm_del_{src.source_hash}"

    if st.session_state.get(confirm_key, False):
        st.warning(f"Delete **{src.title}**? This cannot be undone.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes", key=f"yes_{src.source_hash}", use_container_width=True):
                kb.delete_source(src.source_url)
                st.session_state.pop(confirm_key, None)
                st.success("Deleted.")
                st.rerun()
        with col_no:
            if st.button("Cancel", key=f"no_{src.source_hash}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    else:
        if st.button(
            "Delete",
            key=f"del_{src.source_hash}",
            help=f"Delete {src.title}",
        ):
            st.session_state[confirm_key] = True
            st.rerun()

    st.markdown("---")


def _render_clear_all(kb: KnowledgeBase) -> None:
    """Double-confirmed clear-all action."""
    state_key = "clear_all_step"
    step = st.session_state.get(state_key, 0)

    if step == 0:
        if st.button("Clear All Knowledge", key="clear_all_btn", use_container_width=True):
            st.session_state[state_key] = 1
            st.rerun()
    elif step == 1:
        st.warning("This will permanently delete all sources and chunks.")
        st.warning("Are you absolutely sure?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Yes, delete everything",
                key="clear_all_confirm",
                use_container_width=True,
            ):
                st.session_state[state_key] = 2
                st.rerun()
        with col2:
            if st.button("Cancel", key="clear_all_cancel", use_container_width=True):
                st.session_state[state_key] = 0
                st.rerun()
    elif step == 2:
        kb.clear_all()
        st.session_state[state_key] = 0
        st.session_state.pop("kb_summary", None)
        st.session_state.pop("kb_summary_count", None)
        st.session_state.pop("kb_summary_notebook_id", None)
        st.success("All knowledge cleared.")
        st.rerun()


def _render_kb_summary(sources: list[SourceInfo]) -> None:
    """Generate and display a short summary of the knowledge base."""
    current_count = len(sources)
    notebook_id = st.session_state.get("notebook_id", "default")
    cached_count = st.session_state.get("kb_summary_count", 0)
    cached_summary = st.session_state.get("kb_summary", "")
    cached_notebook_id = st.session_state.get("kb_summary_notebook_id", "")

    if (
        cached_summary
        and cached_count == current_count
        and cached_notebook_id == notebook_id
    ):
        st.info(f"Your knowledge base covers: {cached_summary}")
        return

    manifest_parts: list[str] = []
    for source in sources[:15]:
        manifest_parts.append(
            f"- {source.title} ({source.source_type}, {source.chunk_count} chunks)"
        )
    manifest = "\n".join(manifest_parts)

    provider = st.session_state.get(
        "selected_provider",
        settings.llm_provider or "ollama",
    )
    model = st.session_state.get("selected_model", settings.default_model)
    prompt = (
        "Summarize what this knowledge base covers in EXACTLY 2 sentences. "
        "Be specific about the topics. Do not use bullet points.\n\n"
        f"Sources:\n{manifest}"
    )

    try:
        llm = get_llm(model=model, provider=provider)
        tokens: list[str] = []
        for token in llm.stream(prompt):
            tokens.append(_stream_chunk_text(token))
        summary = "".join(tokens).strip()

        st.session_state.kb_summary = summary
        st.session_state.kb_summary_count = current_count
        st.session_state.kb_summary_notebook_id = notebook_id
        st.info(f"Your knowledge base covers: {summary}")
    except Exception:
        st.session_state.kb_summary = ""
        st.session_state.kb_summary_count = current_count
        st.session_state.kb_summary_notebook_id = notebook_id


def _render_reset_database(kb: KnowledgeBase) -> None:
    """Offer a reset-database button for corruption recovery."""
    with st.expander("Advanced", expanded=False):
        st.caption(
            "If you are experiencing ChromaDB errors, resetting the database "
            "will clear all stored data and rebuild it."
        )
        if st.button("Reset Database", key="reset_db_btn"):
            try:
                kb.reset_database()
                st.session_state.pop("kb_summary", None)
                st.session_state.pop("kb_summary_count", None)
                st.session_state.pop("kb_summary_notebook_id", None)
                st.cache_resource.clear()
                st.success("Database reset. All data has been cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Reset failed: {exc}")

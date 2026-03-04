"""
Right-side knowledge panel — source inspector and management.

Displays every ingested source with type icons, chunk counts,
ingestion timestamps, and inline delete controls.  Provides
search/filter, aggregate stats, auto-generated KB summary,
and a double-confirmed "Clear All Knowledge" action.

Designed to be rendered in the right column of the main layout.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from langchain_ollama import OllamaLLM

from core.models import EMBEDDING_MODEL, _get_secret
from core.vectorstore import KnowledgeBase, SourceInfo


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


def _icon_for_type(source_type: str, source_url: str) -> str:
    """Return an emoji icon based on source type or URL pattern.

    Args:
        source_type: One of ``youtube``, ``web``, ``pdf``, ``text``,
            ``perplexity``, or ``unknown``.
        source_url: The raw source URL / label.

    Returns:
        A single emoji string.
    """
    mapping: dict[str, str] = {
        "youtube": "🎥",
        "web": "🌐",
        "pdf": "📄",
        "text": "📝",
        "perplexity": "🔍",
    }
    icon = mapping.get(source_type, "")
    if icon:
        return icon
    # Fallback heuristics
    url = source_url.lower()
    if "youtube" in url or "youtu.be" in url:
        return "🎥"
    if url.endswith(".pdf"):
        return "📄"
    if "perplexity" in url:
        return "🔍"
    if url.startswith("http"):
        return "🌐"
    return "📝"


def _type_badge(source_type: str) -> str:
    """Return a styled type label.

    Args:
        source_type: Source type string.

    Returns:
        Markdown-formatted badge string.
    """
    colors: dict[str, str] = {
        "youtube": "#FF0000",
        "web": "#4285F4",
        "pdf": "#E94235",
        "text": "#34A853",
        "perplexity": "#20808D",
    }
    color = colors.get(source_type, "#71717A")
    label = source_type.upper()
    return (
        f'<span style="background:{color}20;color:{color};'
        f'padding:1px 8px;border-radius:10px;font-size:0.68rem;'
        f'font-weight:600;letter-spacing:0.04em;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_knowledge_panel() -> None:
    """Render the full knowledge panel in the right column."""
    st.markdown("### 📎 Sources")

    kb = _get_kb()
    sources: list[SourceInfo] = kb.get_all_sources()

    # ---- Aggregate stats ------------------------------------------------
    _render_stats(sources)

    # ---- KB Summary -----------------------------------------------------
    if sources:
        _render_kb_summary(sources)

    if not sources:
        st.caption("No sources yet.  Add knowledge from the sidebar →")
        return

    # ---- Search / filter ------------------------------------------------
    search_query = st.text_input(
        "🔎 Filter sources",
        key="source_filter",
        placeholder="Type to filter…",
        label_visibility="collapsed",
    )

    filtered = sources
    if search_query:
        q = search_query.lower()
        filtered = [s for s in sources if q in s.title.lower() or q in s.source_url.lower()]

    if not filtered:
        st.caption("No sources match your filter.")
    else:
        st.caption(f"Showing {len(filtered)} of {len(sources)} sources")

    # ---- Source list -----------------------------------------------------
    for src in filtered:
        _render_source_row(src, kb)

    # ---- Clear all & Reset -----------------------------------------------
    st.divider()
    _render_clear_all(kb)
    _render_reset_database(kb)


# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------

def _render_stats(sources: list[SourceInfo]) -> None:
    """Display aggregate knowledge base statistics.

    Args:
        sources: All ``SourceInfo`` objects in the collection.
    """
    total_sources = len(sources)
    total_chunks = sum(s.chunk_count for s in sources)
    # Rough estimate: 1 chunk ≈ 1000 chars ≈ 250 tokens
    est_tokens = total_chunks * 250

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Sources", total_sources)
    with col2:
        st.metric("📦 Chunks", total_chunks)
    with col3:
        if est_tokens >= 1_000_000:
            label = f"{est_tokens / 1_000_000:.1f}M"
        elif est_tokens >= 1_000:
            label = f"{est_tokens / 1_000:.0f}K"
        else:
            label = str(est_tokens)
        st.metric("🔤 Est. Tokens", label)


# ---------------------------------------------------------------------------
# Individual source row
# ---------------------------------------------------------------------------

def _render_source_row(src: SourceInfo, kb: KnowledgeBase) -> None:
    """Render a single source with icon, metadata, and delete button.

    Args:
        src: The ``SourceInfo`` to display.
        kb: The ``KnowledgeBase`` for delete operations.
    """
    icon = _icon_for_type(src.source_type, src.source_url)
    badge = _type_badge(src.source_type)
    timestamp = src.ingested_at[:16].replace("T", " ") if src.ingested_at else ""

    # Header row: icon + title + badge
    st.markdown(
        f"{icon} **{src.title or src.source_url}** {badge}",
        unsafe_allow_html=True,
    )
    st.caption(f"{src.chunk_count} chunks • {timestamp}")

    # Delete with confirmation
    confirm_key = f"confirm_del_{src.source_hash}"

    if st.session_state.get(confirm_key, False):
        st.warning(f"Delete **{src.title}**? This cannot be undone.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Yes", key=f"yes_{src.source_hash}", use_container_width=True):
                kb.delete_source(src.source_url)
                st.session_state.pop(confirm_key, None)
                st.success("Deleted.")
                st.rerun()
        with col_no:
            if st.button("❌ Cancel", key=f"no_{src.source_hash}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    else:
        if st.button(
            "🗑️ Delete",
            key=f"del_{src.source_hash}",
            help=f"Delete {src.title}",
        ):
            st.session_state[confirm_key] = True
            st.rerun()

    st.markdown("---")


# ---------------------------------------------------------------------------
# Clear all knowledge — double confirmation
# ---------------------------------------------------------------------------

def _render_clear_all(kb: KnowledgeBase) -> None:
    """Double-confirmed 'Clear All Knowledge' action.

    Args:
        kb: The ``KnowledgeBase`` to clear.
    """
    state_key = "clear_all_step"
    step = st.session_state.get(state_key, 0)

    if step == 0:
        if st.button(
            "⚠️ Clear All Knowledge",
            key="clear_all_btn",
            use_container_width=True,
        ):
            st.session_state[state_key] = 1
            st.rerun()

    elif step == 1:
        st.warning("This will **permanently delete** all sources and chunks.")
        st.warning("Are you absolutely sure?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔴 Yes, delete everything",
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
        st.success("✅ All knowledge cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Auto-generated KB summary
# ---------------------------------------------------------------------------

def _render_kb_summary(sources: list[SourceInfo]) -> None:
    """Generate and display a 2-sentence summary of the knowledge base.

    The summary is cached in ``st.session_state["kb_summary"]`` and
    regenerated when the source count changes (indicating new sources).

    Args:
        sources: Current list of ``SourceInfo`` objects.
    """
    current_count = len(sources)
    cached_count = st.session_state.get("kb_summary_count", 0)
    cached_summary = st.session_state.get("kb_summary", "")

    # Only regenerate if source count changed
    if cached_summary and cached_count == current_count:
        st.info(f"📖 **Your knowledge base covers:** {cached_summary}")
        return

    # Build a source manifest for the LLM
    manifest_parts: list[str] = []
    for s in sources[:15]:  # cap at 15 to avoid token overflow
        manifest_parts.append(f"- {s.title} ({s.source_type}, {s.chunk_count} chunks)")
    manifest = "\n".join(manifest_parts)

    model = st.session_state.get("selected_model", "deepseek-r1:latest")
    ollama_url = _get_secret("OLLAMA_BASE_URL", "http://localhost:11434")

    prompt = (
        "Summarize what this knowledge base covers in EXACTLY 2 sentences. "
        "Be specific about the topics.  Do not use bullet points.\n\n"
        f"Sources:\n{manifest}"
    )

    try:
        llm = OllamaLLM(model=model, base_url=ollama_url)
        # Use streaming (project rule) but collect tokens into a string
        tokens: list[str] = []
        for token in llm.stream(prompt):
            tokens.append(token)
        summary = "".join(tokens).strip()

        # Cache it
        st.session_state.kb_summary = summary
        st.session_state.kb_summary_count = current_count
        st.info(f"📖 **Your knowledge base covers:** {summary}")

    except Exception:
        # Non-critical — just skip the summary
        st.session_state.kb_summary = ""
        st.session_state.kb_summary_count = current_count


# ---------------------------------------------------------------------------
# Reset database (corruption recovery)
# ---------------------------------------------------------------------------

def _render_reset_database(kb: KnowledgeBase) -> None:
    """Offer a nuclear 'Reset Database' button for corruption recovery.

    Args:
        kb: The ``KnowledgeBase`` to reset.
    """
    with st.expander("🛠️ Advanced", expanded=False):
        st.caption(
            "If you're experiencing errors related to ChromaDB, "
            "resetting the database will fix corruption issues."
        )
        if st.button("🔴 Reset Database", key="reset_db_btn"):
            try:
                kb.reset_database()
                st.session_state.pop("kb_summary", None)
                st.cache_resource.clear()
                st.success("✅ Database reset. All data has been cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Reset failed: {exc}")


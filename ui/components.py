"""
Reusable Streamlit UI components.

Provides source citation cards, empty states, status indicators,
and other shared UI elements.
"""

from __future__ import annotations

import streamlit as st


def render_source_cards(sources: list[dict]) -> None:
    """Render source citation cards below an answer."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            source_name = src.get("source", "Unknown")
            preview = src.get("preview", "")
            st.markdown(
                f"""<div class="source-card">
                    <div class="source-title">📄 [{i}] {source_name}</div>
                    <div class="source-preview">{preview}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_empty_chat_state() -> None:
    """Render the empty chat welcome screen."""
    st.markdown(
        """<div class="empty-state">
            <span class="emoji">🧠</span>
            <div class="title">Your AI Knowledge Base</div>
            <div class="subtitle">
                Add sources from the sidebar — PDFs, websites, YouTube videos,
                or run web research — then ask questions about your content.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_ollama_status(is_running: bool) -> None:
    """Show Ollama connection status in the sidebar."""
    if is_running:
        st.markdown("🟢 Ollama Connected")
    else:
        st.markdown("🔴 Ollama Offline")
        st.caption(
            "Start Ollama to use AI features. "
            "[Download Ollama →](https://ollama.com)"
        )


def render_notebook_stats(chunk_count: int, source_count: int) -> None:
    """Show knowledge base statistics."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📦 Chunks", chunk_count)
    with col2:
        st.metric("📄 Sources", source_count)


def render_notebook_chip(name: str) -> None:
    """Render a notebook name as a styled chip."""
    st.markdown(
        f'<span class="notebook-chip">📓 {name}</span>',
        unsafe_allow_html=True,
    )


def render_mode_badge(mode: str) -> None:
    """Show whether the response used RAG or direct LLM mode."""
    if mode == "rag":
        st.caption("💡 Answered from your knowledge base")
    else:
        st.caption("🌐 Answered from general knowledge (no matching sources found)")

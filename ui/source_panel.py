"""
Source panel — view and manage documents in the knowledge base.
"""

from __future__ import annotations

import streamlit as st

from core.vectorstore import get_source_documents, get_collection_count
from utils.logging import get_logger

logger = get_logger(__name__)


def render_source_panel() -> None:
    """Render the source management panel in the main area."""
    notebook_id = st.session_state.get("active_notebook_id", "default")

    sources = get_source_documents(notebook_id)
    total_chunks = get_collection_count(notebook_id)

    if not sources:
        st.caption("No sources added yet. Use the sidebar to add knowledge.")
        return

    st.caption(f"**{len(sources)}** sources • **{total_chunks}** total chunks")

    for src in sources:
        source_name = src["source"]
        chunk_count = src["chunk_count"]

        # Determine icon based on source type
        if "youtube" in source_name.lower() or "youtu.be" in source_name.lower():
            icon = "🎥"
        elif source_name.endswith(".pdf"):
            icon = "📄"
        elif "perplexity" in source_name.lower():
            icon = "🔍"
        elif source_name.startswith("http"):
            icon = "🌐"
        else:
            icon = "📝"

        st.markdown(
            f"""<div class="source-card">
                <div class="source-title">{icon} {source_name}</div>
                <div class="source-preview">{chunk_count} chunks</div>
            </div>""",
            unsafe_allow_html=True,
        )

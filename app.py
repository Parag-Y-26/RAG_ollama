"""
notebooklm application entrypoint.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import settings
from core.models import health_check
from core.vectorstore import KnowledgeBase
from notebooks.manager import NotebookManager
from persistence.chat_store import ChatStore
from ui.styles import inject_styles

st.set_page_config(
    page_title="notebooklm",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

_health = health_check()
if not _health.is_running:
    st.markdown("# Ollama is not running")
    st.error(_health.error)
    st.markdown("### Start the server")
    st.code(
        "# macOS / Linux\ncurl -fsSL https://ollama.com/install.sh | sh\nollama serve",
        language="bash",
    )
    st.code(
        "# Windows - download from https://ollama.com/download\n"
        "# Then open a terminal and run:\nollama serve",
        language="bash",
    )
    st.markdown("### Pull required models")
    st.code(
        "ollama pull deepseek-r1:latest\nollama pull nomic-embed-text",
        language="bash",
    )
    st.info("After starting Ollama, refresh this page.")
    st.stop()


_BASE = Path(__file__).resolve().parent


@st.cache_resource
def _init_kb(notebook_id: str) -> KnowledgeBase:
    """Create or retrieve a cached knowledge base for a notebook."""
    return KnowledgeBase(
        persist_dir=str(_BASE / "data" / "chroma_db"),
        collection_name=notebook_id,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )


@st.cache_resource
def _init_store() -> ChatStore:
    """Create or retrieve the shared chat store."""
    return ChatStore(str(_BASE / "data" / "chat_history.db"))


@st.cache_resource
def _get_notebook_manager() -> NotebookManager:
    """Create or retrieve the notebook manager."""
    return NotebookManager()


nb_manager = _get_notebook_manager()
nb_manager.ensure_default_exists()
all_notebooks = nb_manager.list_notebooks()
notebook_names = [nb.name for nb in all_notebooks]
notebook_ids = [nb.id for nb in all_notebooks]

current_notebook_id = st.session_state.get("notebook_id", "")
current_index = (
    notebook_ids.index(current_notebook_id)
    if current_notebook_id in notebook_ids
    else 0
)

nb_col, new_col = st.columns([5, 1])
with nb_col:
    selected_name = st.selectbox(
        "Notebook",
        notebook_names,
        index=current_index,
        label_visibility="collapsed",
    )
    notebook_id = notebook_ids[notebook_names.index(selected_name)]

with new_col:
    with st.popover("＋", help="New notebook"):
        new_name = st.text_input("Name", key="new_nb_name")
        if st.button("Create", key="create_nb"):
            clean_name = new_name.strip()
            if not clean_name:
                st.warning("Please enter a name.")
            else:
                try:
                    created = nb_manager.create_notebook(clean_name)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state.notebook_id = created.id
                    st.session_state.new_nb_name = ""
                    st.session_state.pop("messages", None)
                    st.session_state.pop("kb", None)
                    st.session_state.pop("kb_summary", None)
                    st.session_state.pop("kb_summary_count", None)
                    st.session_state.pop("kb_summary_notebook_id", None)
                    st.rerun()

previous_notebook_id = st.session_state.get("notebook_id", "")
if previous_notebook_id != notebook_id:
    st.session_state.pop("messages", None)
    st.session_state.pop("kb", None)
    st.session_state.pop("kb_summary", None)
    st.session_state.pop("kb_summary_count", None)
    st.session_state.pop("kb_summary_notebook_id", None)

try:
    st.session_state.kb = _init_kb(notebook_id)
except RuntimeError as exc:
    st.error(f"Database error: {exc}")
    if st.button("Reset Database and Retry"):
        import shutil

        db_path = _BASE / "data" / "chroma_db"
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)
        _init_kb.clear()
        st.rerun()
    st.stop()

st.session_state.chat_store = _init_store()
st.session_state.notebook_id = notebook_id

from ui.chat import render_chat  # noqa: E402
from ui.knowledge_panel import render_knowledge_panel  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

render_sidebar()

show_panel = st.toggle("Sources", value=True, help="Toggle knowledge panel")

if show_panel:
    chat_col, panel_col = st.columns([7, 3])
    with chat_col:
        render_chat()
    with panel_col:
        render_knowledge_panel()
else:
    render_chat()

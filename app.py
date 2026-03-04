"""
NotebookLM — AI Knowledge Base powered by Ollama.

Entrypoint that wires together every layer: sidebar, chat, knowledge
panel.  Handles Ollama health gating, multi-notebook switching, and
theme injection.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.models import EMBEDDING_MODEL, _get_secret, health_check
from core.vectorstore import KnowledgeBase
from persistence.chat_store import ChatStore

# ──────────────────── page config ────────────────────
st.set_page_config(
    page_title="NotebookLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────── obsidian / cyan theme ──────────
st.markdown(
    """<style>
    :root{--bg:#0A0A0A;--surface:#111;--border:#1F1F1F;--accent:#00FFFF;--text:#E4E4E7;--muted:#71717A}
    .stApp{background:var(--bg);color:var(--text)}
    [data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
    h1,h2,h3{color:var(--accent)!important}
    .stButton>button{border-color:color-mix(in srgb,var(--accent) 30%,transparent);color:var(--accent);transition:.2s}
    .stButton>button:hover{background:color-mix(in srgb,var(--accent) 10%,transparent);border-color:var(--accent)}
    .stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background:var(--surface)!important;border-color:var(--border)!important;color:var(--text)!important}
    .stMetric label{color:var(--muted)!important}.stMetric [data-testid="stMetricValue"]{color:var(--accent)!important}
    .stChatMessage{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important}
    [data-testid="stExpander"]{border-color:var(--border)!important}
    ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
    .stDivider{border-color:var(--border)!important}
    div[data-testid="stTabs"] button{color:var(--muted)!important}
    div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important}
    .stDownloadButton>button{border-color:var(--border);color:var(--muted)}
    </style>""",
    unsafe_allow_html=True,
)

# ──────────────────── ollama health gate ─────────────
_health = health_check()
if not _health.is_running:
    st.markdown("# 🔴 Ollama is not running")
    st.error(_health.error)
    st.markdown("### Start the server")
    st.code("# macOS / Linux\ncurl -fsSL https://ollama.com/install.sh | sh\nollama serve", language="bash")
    st.code("# Windows — download from https://ollama.com/download\n# Then open a terminal and run:\nollama serve", language="bash")
    st.markdown("### Pull required models")
    st.code("ollama pull deepseek-r1:latest\nollama pull nomic-embed-text", language="bash")
    st.info("After starting Ollama, **refresh this page**.")
    st.stop()


# ──────────────────── cached resources ───────────────
_BASE = Path(__file__).resolve().parent


@st.cache_resource
def _init_kb(notebook_id: str) -> KnowledgeBase:
    """Create (or retrieve cached) KnowledgeBase for a notebook."""
    return KnowledgeBase(
        persist_dir=str(_BASE / "data" / "chroma_db"),
        collection_name=notebook_id,
        embedding_model=EMBEDDING_MODEL,
        ollama_base_url=_get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


@st.cache_resource
def _init_store() -> ChatStore:
    """Create (or retrieve cached) ChatStore singleton."""
    return ChatStore(str(_BASE / "data" / "chat_history.db"))


# ──────────────────── notebook selector ──────────────
if "notebooks" not in st.session_state:
    st.session_state.notebooks = ["Default"]

nb_col, new_col = st.columns([5, 1])
with nb_col:
    notebook_id = st.selectbox(
        "📓 Notebook",
        st.session_state.notebooks,
        label_visibility="collapsed",
    )
with new_col:
    with st.popover("➕", help="Create a new notebook"):
        new_name = st.text_input("Notebook name", key="new_nb_name")
        if st.button("Create", key="create_nb") and new_name.strip():
            clean = new_name.strip().replace(" ", "_").lower()
            if clean not in st.session_state.notebooks:
                st.session_state.notebooks.append(clean)
                st.rerun()
            else:
                st.warning("Already exists.")

# Detect notebook switch → reset messages
prev = st.session_state.get("notebook_id", "")
if prev != notebook_id:
    st.session_state.pop("messages", None)

try:
    st.session_state.kb = _init_kb(notebook_id)
except RuntimeError as exc:
    st.error(f"🔴 Database error: {exc}")
    if st.button("🔄 Reset Database & Retry"):
        import shutil
        db_path = _BASE / "data" / "chroma_db"
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)
        _init_kb.clear()  # clear st.cache_resource
        st.rerun()
    st.stop()

st.session_state.chat_store = _init_store()
st.session_state.notebook_id = notebook_id

# ──────────────────── layout ─────────────────────────
from ui.sidebar import render_sidebar  # noqa: E402
from ui.chat import render_chat  # noqa: E402
from ui.knowledge_panel import render_knowledge_panel  # noqa: E402

render_sidebar()

show_panel = st.toggle("📎 Sources", value=True, help="Toggle knowledge panel")

if show_panel:
    chat_col, panel_col = st.columns([7, 3])
else:
    chat_col, panel_col = st.container(), None

with chat_col:
    render_chat()

if panel_col is not None:
    with panel_col:
        render_knowledge_panel()

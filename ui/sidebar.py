"""
Left sidebar UI — Ollama health, model selection, Perplexity research,
and tabbed source ingestion (YouTube, Website, PDF, Plain text).

Every section provides visual feedback via spinners, success toasts,
error toasts, and progress indicators.  All heavy work is delegated
to ``core.ingestion``, ``core.research``, and ``core.vectorstore``.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import streamlit as st
from pypdf import PdfReader

from core.ingestion import load_youtube, load_webpage, load_pdf, load_plaintext
from core.models import health_check, get_available_models, EMBEDDING_MODEL, _get_secret
from core.research import PerplexityResearcher
from core.vectorstore import KnowledgeBase


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _get_kb() -> KnowledgeBase:
    """Return the active ``KnowledgeBase``, creating one if needed."""
    if "kb" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.kb = KnowledgeBase(
            persist_dir=str(base / "data" / "chroma_db"),
            collection_name=st.session_state.get("notebook_id", "default"),
            embedding_model=EMBEDDING_MODEL,
            ollama_base_url=_get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return st.session_state.kb


def _record_ingestion(source: str, chunks: int) -> None:
    """Push a recent-ingestion entry into session state for display."""
    if "recent_ingestions" not in st.session_state:
        st.session_state.recent_ingestions = []
    st.session_state.recent_ingestions.append({
        "source": source,
        "chunks": chunks,
        "time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
    })
    # Keep only last 5
    st.session_state.recent_ingestions = st.session_state.recent_ingestions[-5:]


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

_YT_ID_RE = [
    re.compile(r"youtube\.com/watch\?v=([\w-]+)"),
    re.compile(r"youtu\.be/([\w-]+)"),
    re.compile(r"youtube\.com/shorts/([\w-]+)"),
]


def _extract_yt_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL."""
    for pat in _YT_ID_RE:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the full left sidebar."""
    with st.sidebar:
        st.markdown("## 🧠 NotebookLM")
        _render_health_indicator()
        st.divider()
        _render_model_selector()
        st.divider()
        _render_research_section()
        st.divider()
        _render_ingestion_tabs()
        _render_recent_ingestions()


# ---------------------------------------------------------------------------
# Section 1 — Ollama health
# ---------------------------------------------------------------------------

def _render_health_indicator() -> None:
    """Show Ollama server status as a green or red dot."""
    status = health_check()

    if status.is_running:
        model_count = len(status.models)
        st.markdown(
            f"🟢 **Ollama Online** — {model_count} model{'s' if model_count != 1 else ''} installed"
        )
    else:
        st.markdown("🔴 **Ollama Offline**")
        st.caption(status.error or "Cannot reach Ollama server.")
        st.caption("[Download Ollama →](https://ollama.com)")

    # Cache the status for other sections
    st.session_state.ollama_running = status.is_running


# ---------------------------------------------------------------------------
# Section 2 — Model selector
# ---------------------------------------------------------------------------

def _render_model_selector() -> None:
    """Model picker — only shows models confirmed running in Ollama."""
    st.markdown("### ⚙️ Model")

    if not st.session_state.get("ollama_running", False):
        st.caption("Start Ollama to select a model.")
        return

    models: list[str] = get_available_models()

    col_sel, col_ref = st.columns([4, 1])
    with col_ref:
        if st.button("🔄", key="refresh_models", help="Refresh model list"):
            get_available_models.clear()  # clear st.cache_data
            st.rerun()

    with col_sel:
        current = st.session_state.get("selected_model", models[0] if models else "")
        try:
            idx = models.index(current)
        except ValueError:
            idx = 0
        selected = st.selectbox(
            "Active model",
            models,
            index=idx,
            key="model_selectbox",
            label_visibility="collapsed",
        )
        st.session_state.selected_model = selected

    # Hint: if we only have the fallback model, show pull command
    if len(models) == 1:
        st.caption(
            f"Only the default model is available.  "
            f"Pull more with:"
        )
        st.code(f"ollama pull {models[0]}", language="bash")
        st.caption("Try: `ollama pull gemma3:4b` or `ollama pull llama3.2`")


# ---------------------------------------------------------------------------
# Section 3 — Perplexity research
# ---------------------------------------------------------------------------

def _render_research_section() -> None:
    """Perplexity Sonar API web-research input."""
    st.markdown("### 🔍 Web Research")

    api_key = _get_secret("PPLX_API_KEY", "")
    if not api_key:
        st.caption("Add `PPLX_API_KEY` to `.streamlit/secrets.toml` to enable.")
        return

    topic = st.text_input(
        "Research topic",
        key="research_topic",
        placeholder="e.g. Transformer architecture advances 2025",
    )

    if st.button("🌐 Research & Absorb", key="research_btn", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic first.")
            return

        kb = _get_kb()
        with st.spinner("🌐 Researching the web…"):
            try:
                researcher = PerplexityResearcher(api_key=api_key)
                docs = researcher.research(topic)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"✅ Absorbed {info.chunk_count} chunks of research!")
            except PermissionError as exc:
                st.error(f"🔑 API key issue: {exc}")
            except ConnectionError as exc:
                st.error(f"🌐 Network error: {exc}")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Research failed: {exc}")


# ---------------------------------------------------------------------------
# Section 4 — Source ingestion tabs
# ---------------------------------------------------------------------------

def _render_ingestion_tabs() -> None:
    """Tabbed ingestion: YouTube, Website, PDF, Plain text."""
    st.markdown("### 📚 Add Knowledge")

    tab_yt, tab_web, tab_pdf, tab_txt = st.tabs(
        ["🎥 YouTube", "🌐 Website", "📄 PDF", "📝 Text"]
    )

    with tab_yt:
        _tab_youtube()

    with tab_web:
        _tab_website()

    with tab_pdf:
        _tab_pdf()

    with tab_txt:
        _tab_plaintext()


def _tab_youtube() -> None:
    """YouTube URL ingestion with thumbnail preview."""
    url = st.text_input(
        "YouTube URL",
        key="yt_url_input",
        placeholder="https://www.youtube.com/watch?v=…",
    )

    # Thumbnail preview
    if url:
        vid_id = _extract_yt_id(url)
        if vid_id:
            st.image(
                f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
                width=260,
                caption="Video thumbnail",
            )
        else:
            st.caption("⚠️ Not a recognised YouTube URL.")

    if st.button("Absorb YouTube", key="absorb_yt", use_container_width=True):
        if not url:
            st.warning("Paste a YouTube URL first.")
            return
        kb = _get_kb()
        with st.spinner("Fetching transcript…"):
            try:
                docs = load_youtube(url)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"✅ {info.title} — {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


def _tab_website() -> None:
    """Web page ingestion with favicon preview."""
    url = st.text_input(
        "Website URL",
        key="web_url_input",
        placeholder="https://example.com/article",
    )

    # Favicon preview
    if url and url.startswith("http"):
        domain = urlparse(url).hostname
        if domain:
            col_icon, col_domain = st.columns([1, 5])
            with col_icon:
                st.image(
                    f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                    width=24,
                )
            with col_domain:
                st.caption(domain)

    if st.button("Absorb Webpage", key="absorb_web", use_container_width=True):
        if not url:
            st.warning("Paste a URL first.")
            return
        kb = _get_kb()
        with st.spinner("Extracting web content…"):
            try:
                docs = load_webpage(url)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"✅ {info.title} — {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


def _tab_pdf() -> None:
    """PDF upload with page count display."""
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

    if uploaded is not None:
        try:
            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            page_count = len(reader.pages)
            st.caption(f"📄 **{uploaded.name}** — {page_count} page{'s' if page_count != 1 else ''}")
        except Exception:
            st.caption(f"📄 **{uploaded.name}**")

    if st.button("Absorb PDF", key="absorb_pdf", use_container_width=True):
        if uploaded is None:
            st.warning("Upload a PDF first.")
            return
        kb = _get_kb()
        with st.spinner("Processing PDF…"):
            try:
                docs = load_pdf(uploaded.getvalue(), uploaded.name)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"✅ {info.title} — {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


def _tab_plaintext() -> None:
    """Plain text / markdown paste ingestion."""
    text = st.text_area(
        "Paste text or markdown",
        key="text_paste_input",
        height=120,
        placeholder="Paste notes, articles, or any raw text…",
    )
    title = st.text_input(
        "Source title",
        key="text_title_input",
        placeholder="e.g. Lecture Notes — Week 5",
    )

    if st.button("Absorb Text", key="absorb_text", use_container_width=True):
        if not text or not text.strip():
            st.warning("Paste some text first.")
            return
        if not title or not title.strip():
            st.warning("Give it a title so you can find it later.")
            return
        kb = _get_kb()
        with st.spinner("Processing text…"):
            try:
                docs = load_plaintext(text, title)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"✅ {info.title} — {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


# ---------------------------------------------------------------------------
# Recent ingestions feed
# ---------------------------------------------------------------------------

def _render_recent_ingestions() -> None:
    """Show the last few successful ingestions at the bottom of sidebar."""
    ingestions = st.session_state.get("recent_ingestions", [])
    if not ingestions:
        return

    st.divider()
    st.markdown("### 📋 Recent")
    for item in reversed(ingestions[-5:]):
        st.caption(f"✅ {item['source'][:35]} — {item['chunks']} chunks — {item['time']}")

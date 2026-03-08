"""
Sidebar UI for health, model selection, search, research, and ingestion.
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

from config.settings import settings
from core.ingestion import load_pdf, load_plaintext, load_webpage, load_youtube
from core.models import health_check
from core.research import PerplexityResearcher
from core.vectorstore import KnowledgeBase


def _get_kb() -> KnowledgeBase:
    """Return the active knowledge base, creating one if needed."""
    if "kb" not in st.session_state:
        base = Path(__file__).resolve().parent.parent
        st.session_state.kb = KnowledgeBase(
            persist_dir=str(base / "data" / "chroma_db"),
            collection_name=st.session_state.get("notebook_id", "default"),
            embedding_model=settings.embedding_model,
            ollama_base_url=settings.ollama_base_url,
        )
    return st.session_state.kb


def _record_ingestion(source: str, chunks: int) -> None:
    """Push a recent-ingestion entry into session state for display."""
    if "recent_ingestions" not in st.session_state:
        st.session_state.recent_ingestions = []
    st.session_state.recent_ingestions.append(
        {
            "source": source,
            "chunks": chunks,
            "time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        }
    )
    st.session_state.recent_ingestions = st.session_state.recent_ingestions[-5:]


_YT_ID_RE = [
    re.compile(r"youtube\.com/watch\?v=([\w-]+)"),
    re.compile(r"youtu\.be/([\w-]+)"),
    re.compile(r"youtube\.com/shorts/([\w-]+)"),
]


def _extract_yt_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL."""
    for pattern in _YT_ID_RE:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def render_sidebar() -> None:
    """Render the full left sidebar."""
    with st.sidebar:
        st.markdown(
            '<p class="sidebar-title">◈ notebooklm</p>',
            unsafe_allow_html=True,
        )
        _render_health_indicator()
        _sidebar_rule()
        _render_model_selector()
        _sidebar_rule()
        _render_research_section()
        _sidebar_rule()
        _render_ingestion_tabs()
        _render_recent_ingestions()


def _sidebar_rule() -> None:
    """Render a monochrome sidebar section separator."""
    st.markdown(
        '<div style="border-top:1px solid #1C1C1C;margin:1.25rem 0;"></div>',
        unsafe_allow_html=True,
    )


def _render_health_indicator() -> None:
    """Show Ollama server status."""
    status = health_check()

    if status.is_running:
        st.markdown(
            f'<p class="status-online">● ollama · {len(status.models)} models</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="status-offline">○ ollama offline</p>',
            unsafe_allow_html=True,
        )

    st.session_state.ollama_running = status.is_running


def _render_model_selector() -> None:
    """Provider picker, model picker, and auto web-search toggle."""
    from core.llm import get_available_cloud_models
    from core.models import get_available_models as get_ollama_models

    st.markdown("### Model")

    available_providers: list[str] = ["ollama"]
    if settings.openai_enabled:
        available_providers.append("openai")
    if settings.anthropic_enabled:
        available_providers.append("anthropic")
    if settings.groq_enabled:
        available_providers.append("groq")

    current_provider = st.session_state.get(
        "selected_provider",
        settings.llm_provider or "ollama",
    )
    provider = st.selectbox(
        "Provider",
        available_providers,
        index=(
            available_providers.index(current_provider)
            if current_provider in available_providers
            else 0
        ),
        key="provider_selectbox",
        label_visibility="visible",
    )
    st.session_state.selected_provider = provider

    if provider == "ollama":
        if not st.session_state.get("ollama_running", False):
            st.caption("Start Ollama to select a model.")
        else:
            models = get_ollama_models()
            col_sel, col_ref = st.columns([4, 1])
            with col_ref:
                if st.button("↺", key="refresh_models", help="Refresh"):
                    get_ollama_models.clear()
                    st.rerun()
            with col_sel:
                _pick_model(models, provider)
    else:
        models = get_available_cloud_models(provider)
        if not models:
            st.caption(f"No models configured for {provider}.")
        else:
            _pick_model(models, provider)

        key_map = {
            "openai": settings.openai_enabled,
            "anthropic": settings.anthropic_enabled,
            "groq": settings.groq_enabled,
        }
        if key_map.get(provider):
            st.caption("✓ API key configured")
        else:
            st.caption(f"✗ Add {provider.upper()}_API_KEY to secrets.toml")

    st.markdown("---")
    web_search_enabled = st.toggle(
        "Auto web search",
        value=st.session_state.get(
            "enable_web_search",
            settings.web_search_enabled_by_default,
        ),
        key="web_search_toggle",
        help=(
            "When ON: if your KB has no answer, the assistant "
            "automatically searches DuckDuckGo and answers from web results. "
            "Results are NOT stored in your KB."
        ),
    )
    st.session_state.enable_web_search = web_search_enabled

    if web_search_enabled:
        st.caption("● web search active")
    else:
        st.caption("○ web search off")


def _pick_model(models: list[str], provider: str) -> None:
    """Shared model selectbox logic."""
    current = st.session_state.get("selected_model", models[0] if models else "")
    idx = models.index(current) if current in models else 0
    selected = st.selectbox(
        "Model",
        models,
        index=idx,
        key=f"model_selectbox_{provider}",
        label_visibility="visible",
    )
    st.session_state.selected_model = selected


def _render_research_section() -> None:
    """Perplexity research input."""
    st.markdown("### Web Research")

    api_key = settings.perplexity_api_key
    if not api_key:
        st.caption("Add `PPLX_API_KEY` to `.streamlit/secrets.toml` to enable.")
        return

    topic = st.text_input(
        "Research topic",
        key="research_topic",
        placeholder="e.g. Transformer architecture advances 2025",
    )

    if st.button("Research and Absorb", key="research_btn", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic first.")
            return

        kb = _get_kb()
        with st.spinner("Researching the web..."):
            try:
                researcher = PerplexityResearcher(api_key=api_key)
                docs = researcher.research(topic)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"Absorbed {info.chunk_count} chunks of research.")
            except PermissionError as exc:
                st.error(f"API key issue: {exc}")
            except ConnectionError as exc:
                st.error(f"Network error: {exc}")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Research failed: {exc}")


def _render_ingestion_tabs() -> None:
    """Tabbed ingestion for YouTube, Website, PDF, Text, and Search."""
    st.markdown("### Knowledge")

    tab_yt, tab_web, tab_pdf, tab_txt, tab_search = st.tabs(
        ["YouTube", "Website", "PDF", "Text", "Search"]
    )

    with tab_yt:
        _tab_youtube()
    with tab_web:
        _tab_website()
    with tab_pdf:
        _tab_pdf()
    with tab_txt:
        _tab_plaintext()
    with tab_search:
        _tab_search()


def _tab_youtube() -> None:
    """YouTube URL ingestion with thumbnail preview."""
    url = st.text_input(
        "YouTube URL",
        key="yt_url_input",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if url:
        video_id = _extract_yt_id(url)
        if video_id:
            st.image(
                f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                width=260,
                caption="Video thumbnail",
            )
        else:
            st.caption("Not a recognized YouTube URL.")

    if st.button("Absorb YouTube", key="absorb_yt", use_container_width=True):
        if not url:
            st.warning("Paste a YouTube URL first.")
            return

        kb = _get_kb()
        with st.spinner("Fetching transcript..."):
            try:
                docs = load_youtube(url)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"{info.title} - {info.chunk_count} chunks")
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
        with st.spinner("Extracting web content..."):
            try:
                docs = load_webpage(url)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"{info.title} - {info.chunk_count} chunks")
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
            suffix = "s" if page_count != 1 else ""
            st.caption(f"{uploaded.name} - {page_count} page{suffix}")
        except Exception:
            st.caption(uploaded.name)

    if st.button("Absorb PDF", key="absorb_pdf", use_container_width=True):
        if uploaded is None:
            st.warning("Upload a PDF first.")
            return

        kb = _get_kb()
        with st.spinner("Processing PDF..."):
            try:
                docs = load_pdf(uploaded.getvalue(), uploaded.name)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"{info.title} - {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


def _tab_plaintext() -> None:
    """Plain text or markdown ingestion."""
    text = st.text_area(
        "Paste text or markdown",
        key="text_paste_input",
        height=120,
        placeholder="Paste notes, articles, or any raw text...",
    )
    title = st.text_input(
        "Source title",
        key="text_title_input",
        placeholder="e.g. Lecture Notes - Week 5",
    )

    if st.button("Absorb Text", key="absorb_text", use_container_width=True):
        if not text or not text.strip():
            st.warning("Paste some text first.")
            return
        if not title or not title.strip():
            st.warning("Give it a title so you can find it later.")
            return

        kb = _get_kb()
        with st.spinner("Processing text..."):
            try:
                docs = load_plaintext(text, title)
                info = kb.add_source(docs)
                _record_ingestion(info.title, info.chunk_count)
                st.success(f"{info.title} - {info.chunk_count} chunks")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed: {exc}")


def _tab_search() -> None:
    """
    DuckDuckGo search tab - search the web and absorb results into the KB.
    """
    from core.search import WebSearcher

    query = st.text_input(
        "Search query",
        key="ddg_search_query",
        placeholder="e.g. LangChain LCEL streaming 2025",
    )

    col_n, col_mode = st.columns([2, 3])
    with col_n:
        result_options = [3, 5, 8, 10]
        default_results = (
            settings.web_search_max_results
            if settings.web_search_max_results in result_options
            else 5
        )
        max_results = st.selectbox(
            "Results",
            options=result_options,
            index=result_options.index(default_results),
            key="ddg_max_results",
            label_visibility="visible",
        )
    with col_mode:
        fetch_full = st.checkbox(
            "Fetch full pages",
            value=settings.web_search_max_fetch > 0,
            key="ddg_fetch_full",
            help=(
                "ON: fetch full page content for top results (slower, more thorough). "
                "OFF: use snippets only (faster, less content)."
            ),
        )

    col_preview, col_absorb = st.columns(2)

    with col_preview:
        if st.button("Preview", key="ddg_preview_btn", use_container_width=True):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                searcher = WebSearcher(
                    max_results=max_results,
                    max_fetch=0,
                    region=settings.web_search_region,
                )
                with st.spinner("Searching..."):
                    try:
                        results = searcher.search(query.strip())
                        st.session_state["ddg_preview_results"] = results
                        st.session_state["ddg_preview_query"] = query.strip()
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")
                        st.session_state.pop("ddg_preview_results", None)
                        st.session_state.pop("ddg_preview_query", None)

    with col_absorb:
        if st.button("Absorb", key="ddg_absorb_btn", use_container_width=True):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                kb = _get_kb()
                max_fetch = max_results if fetch_full else 0
                searcher = WebSearcher(
                    max_results=max_results,
                    max_fetch=max_fetch,
                    region=settings.web_search_region,
                )
                with st.spinner("Searching and absorbing..."):
                    try:
                        docs = searcher.search_and_load(
                            query.strip(),
                            absorb_snippets_on_failure=True,
                        )
                        info = kb.add_source(docs)
                        _record_ingestion(info.title, info.chunk_count)
                        st.success(
                            f"{info.chunk_count} chunks absorbed from "
                            f"web search: '{query.strip()[:40]}'"
                        )
                        st.session_state.pop("ddg_preview_results", None)
                        st.session_state.pop("ddg_preview_query", None)
                    except ValueError as exc:
                        st.warning(str(exc))
                    except RuntimeError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Absorb failed: {exc}")

    preview_results = st.session_state.get("ddg_preview_results", [])
    preview_query = st.session_state.get("ddg_preview_query", query)
    if preview_results:
        st.markdown("---")
        st.caption(f"{len(preview_results)} results for '{preview_query}'")
        for i, result in enumerate(preview_results, 1):
            title = result.title[:60] if result.title else result.url
            with st.expander(f"{i}. {title}", expanded=(i == 1)):
                st.caption(result.url)
                st.markdown(result.snippet)
                if result.published:
                    st.caption(f"Published: {result.published}")


def _render_recent_ingestions() -> None:
    """Show the last few successful ingestions."""
    ingestions = st.session_state.get("recent_ingestions", [])
    if not ingestions:
        return

    _sidebar_rule()
    st.markdown("### Recent")
    for item in reversed(ingestions[-5:]):
        st.caption(
            f"{item['source'][:35]} - {item['chunks']} chunks - {item['time']}"
        )

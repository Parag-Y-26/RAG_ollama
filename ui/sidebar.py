"""
Sidebar UI for health, model selection, API keys, search, research, and ingestion.
"""

from __future__ import annotations

import io
import os
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


# ---------------------------------------------------------------------------
# Provider display labels
# ---------------------------------------------------------------------------

_PROVIDER_LABELS: dict[str, str] = {
    "ollama": "Ollama (local)",
    "ollama_cloud": "Ollama Cloud",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "groq": "Groq",
    "nvidia": "NVIDIA NIM",
}


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


# ===================================================================
# Sidebar entry point
# ===================================================================


def render_sidebar() -> None:
    """Render the full left sidebar."""
    with st.sidebar:
        st.markdown(
            '<p class="sidebar-title">◈ notebooklm</p>',
            unsafe_allow_html=True,
        )
        _render_health_indicator()
        _sidebar_rule()
        _render_api_keys_section()
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


# ===================================================================
# Health indicator
# ===================================================================


def _render_health_indicator() -> None:
    """Show Ollama local server status and cloud key status."""
    # Re-read settings in case runtime key was patched
    from config.settings import settings as _settings

    status = health_check()
    provider = st.session_state.get("selected_provider", "ollama")

    if status.is_running:
        st.markdown(
            f'<p class="status-online">● ollama local · {len(status.models)} models</p>',
            unsafe_allow_html=True,
        )
    else:
        if provider == "ollama":
            st.markdown(
                '<p class="status-offline">○ ollama offline</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-online" style="color:#555555;">○ ollama local offline</p>',
                unsafe_allow_html=True,
            )

    # Ollama Cloud status
    if _settings.ollama_cloud_enabled:
        st.markdown(
            '<p class="status-online">● ollama cloud · key configured</p>',
            unsafe_allow_html=True,
        )

    # NVIDIA NIM status
    if _settings.nvidia_enabled:
        if _settings.nvidia_is_self_hosted:
            st.markdown(
                f'<p class="status-online">● nvidia nim · self-hosted</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-online">● nvidia nim · api catalog</p>',
                unsafe_allow_html=True,
            )

    st.session_state.ollama_running = status.is_running


# ===================================================================
# API Keys section
# ===================================================================


def _render_api_keys_section() -> None:
    """
    Collapsible API keys manager.

    Keys entered here are stored in os.environ for the current process.
    For persistence across restarts, add them to .streamlit/secrets.toml.
    """
    from config.settings import settings as _settings

    with st.expander("API Keys", expanded=False):
        st.caption("Keys last for this session only.")
        st.caption("For persistence: add to `.streamlit/secrets.toml`")

        # ── Ollama Cloud ───────────────────────────────────────────
        st.markdown("**Ollama Cloud**")
        st.caption("Get your key: [ollama.com/settings/keys](https://ollama.com/settings/keys)")
        _render_key_input(
            label="OLLAMA_API_KEY",
            session_key="_runtime_ollama_cloud_key",
            env_key="OLLAMA_API_KEY",
            current_value=_settings.ollama_cloud_api_key,
            placeholder="ollama_...",
        )

        st.divider()

        # ── NVIDIA NIM ─────────────────────────────────────────────────
        st.markdown("**NVIDIA NIM**")
        st.caption("Free key: [build.nvidia.com](https://build.nvidia.com/explore)")
        _render_key_input(
            label="NVIDIA_API_KEY",
            session_key="_runtime_nvidia_key",
            env_key="NVIDIA_API_KEY",
            current_value=_settings.nvidia_api_key,
            placeholder="nvapi-...",
        )

        st.divider()

        # ── OpenAI ─────────────────────────────────────────────────
        st.markdown("**OpenAI**")
        _render_key_input(
            label="OPENAI_API_KEY",
            session_key="_runtime_openai_key",
            env_key="OPENAI_API_KEY",
            current_value=_settings.openai_api_key,
            placeholder="sk-...",
        )

        # ── Anthropic ──────────────────────────────────────────────
        st.markdown("**Anthropic**")
        _render_key_input(
            label="ANTHROPIC_API_KEY",
            session_key="_runtime_anthropic_key",
            env_key="ANTHROPIC_API_KEY",
            current_value=_settings.anthropic_api_key,
            placeholder="sk-ant-...",
        )

        # ── Groq ───────────────────────────────────────────────────
        st.markdown("**Groq**")
        _render_key_input(
            label="GROQ_API_KEY",
            session_key="_runtime_groq_key",
            env_key="GROQ_API_KEY",
            current_value=_settings.groq_api_key,
            placeholder="gsk_...",
        )

        # ── Perplexity ─────────────────────────────────────────────
        st.markdown("**Perplexity** (web research)")
        _render_key_input(
            label="PPLX_API_KEY",
            session_key="_runtime_pplx_key",
            env_key="PPLX_API_KEY",
            current_value=_settings.perplexity_api_key,
            placeholder="pplx-...",
        )


def _render_key_input(
    label: str,
    session_key: str,
    env_key: str,
    current_value: str,
    placeholder: str,
) -> None:
    """Reusable key input widget for a single provider."""
    display_value = st.session_state.get(session_key, "") or current_value
    new_val = st.text_input(
        label,
        value=display_value,
        type="password",
        key=f"key_input_{env_key}",
        placeholder=placeholder,
        label_visibility="visible",
    )
    if new_val and new_val != display_value:
        _patch_runtime_key(env_key, new_val)
        st.session_state[session_key] = new_val
        # Clear cached LLM so it picks up the new key
        from core.llm import get_llm

        get_llm.cache_clear()
        st.rerun()


def _patch_runtime_key(env_key: str, value: str) -> None:
    """
    Inject a key into the live environment so config.settings can read it.

    os.environ is process-wide so the change persists for the duration
    of the Streamlit process. We also rebuild the Settings singleton
    so cached properties pick up the new values.
    """
    os.environ[env_key] = value

    # Reconstruct the module-level settings singleton
    from config import settings as settings_module

    settings_module.settings = settings_module.Settings()


# ===================================================================
# Model selector
# ===================================================================


def _render_model_selector() -> None:
    """Provider picker, model picker, and auto web-search toggle."""
    from core.llm import get_available_cloud_models, get_ollama_cloud_models_live
    from core.models import get_available_models as get_ollama_models

    # Re-read settings in case runtime key was patched
    from config.settings import settings as _settings

    st.markdown("### Model")

    # All providers are always visible — users enter keys via the
    # API Keys expander above. This avoids chicken-and-egg problem.
    available_providers: list[str] = [
        "ollama",
        "ollama_cloud",
        "openai",
        "anthropic",
        "groq",
        "nvidia",
    ]

    # Provider display labels
    provider_labels = [_PROVIDER_LABELS.get(p, p) for p in available_providers]

    current_provider = st.session_state.get(
        "selected_provider",
        _settings.llm_provider or "ollama",
    )
    try:
        current_idx = available_providers.index(current_provider)
    except ValueError:
        current_idx = 0

    selected_label = st.selectbox(
        "Provider",
        provider_labels,
        index=current_idx,
        key="provider_selectbox",
        label_visibility="visible",
    )
    # Map label back to internal provider ID
    provider = available_providers[provider_labels.index(selected_label)]
    st.session_state.selected_provider = provider

    # ── Model selection per provider ──────────────────────────────
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

    elif provider == "ollama_cloud":
        models = get_ollama_cloud_models_live()
        if not models:
            st.caption("No cloud models found.")
        else:
            col_sel, col_ref = st.columns([4, 1])
            with col_ref:
                if st.button("↺", key="refresh_cloud_models", help="Refresh cloud models"):
                    st.rerun()
            with col_sel:
                _pick_model(models, provider)
        st.caption("✓ Ollama Cloud key configured")
        st.caption("Models run on ollama.com infrastructure")

    elif provider == "nvidia":
        _render_nvidia_model_selector()

    else:
        # OpenAI, Anthropic, Groq — live fetch when key is available,
        # static fallback otherwise (models always shown)
        models = get_available_cloud_models(provider)
        if not models:
            st.caption(f"No models configured for {provider}.")
        else:
            # Refresh button for providers with live fetch
            if provider in ("openai", "groq"):
                col_sel, col_ref = st.columns([4, 1])
                with col_ref:
                    if st.button("↺", key=f"refresh_{provider}_models", help="Refresh models"):
                        st.rerun()
                with col_sel:
                    _pick_model(models, provider)
            else:
                _pick_model(models, provider)

        key_map = {
            "openai": _settings.openai_enabled,
            "anthropic": _settings.anthropic_enabled,
            "groq": _settings.groq_enabled,
        }
        if key_map.get(provider):
            st.caption("✓ API key configured · live models")
        else:
            st.caption(f"✗ Add {provider.upper()}_API_KEY in ▸ API Keys above")

    st.markdown("---")
    web_search_enabled = st.toggle(
        "AUTO WEB SEARCH",
        value=st.session_state.get(
            "enable_web_search",
            _settings.web_search_enabled_by_default,
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
        st.markdown(
            '<p class="status-online">○ web search active</p>',
            unsafe_allow_html=True,
        )
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
    # If model changed, invalidate the cached RAG chain
    if selected != st.session_state.get("selected_model"):
        from ui.chat import _get_rag_chain

        _get_rag_chain.clear()
    st.session_state.selected_model = selected


def _render_nvidia_model_selector() -> None:
    """
    NVIDIA NIM model selector with live discovery and mode indicator.
    Shows whether using hosted API Catalog or self-hosted local NIM.
    """
    from core.llm import get_available_nvidia_models
    from config.settings import settings as _settings

    # Mode indicator
    if _settings.nvidia_is_self_hosted:
        st.caption(f"○ self-hosted NIM · {_settings.nvidia_nim_base_url}")
    elif _settings.nvidia_api_key:
        key_hint = _settings.nvidia_api_key[:10] + "…"
        st.caption(f"● API Catalog · {key_hint}")
    else:
        st.caption("✗ Add NVIDIA_API_KEY or NVIDIA_NIM_BASE_URL to secrets.toml")
        return

    # Model selector with live refresh
    col_sel, col_ref = st.columns([4, 1])

    with col_ref:
        if st.button("↺", key="refresh_nvidia_models", help="Fetch live model list"):
            st.session_state.pop("nvidia_models_cache", None)
            st.rerun()

    # Fetch models — use session state as lightweight cache
    if "nvidia_models_cache" not in st.session_state:
        with st.spinner("Fetching NVIDIA models…"):
            st.session_state.nvidia_models_cache = get_available_nvidia_models()

    models = st.session_state.nvidia_models_cache

    with col_sel:
        current = st.session_state.get("selected_model", models[0] if models else "")
        idx = models.index(current) if current in models else 0
        selected = st.selectbox(
            "Model",
            models,
            index=idx,
            key="model_selectbox_nvidia",
            label_visibility="visible",
        )
        if selected != st.session_state.get("selected_model"):
            from ui.chat import _get_rag_chain

            _get_rag_chain.clear()
        st.session_state.selected_model = selected

    # Context window hint for selected model
    _CONTEXT_HINTS = {
        "meta/llama-3.3-70b-instruct": "128K ctx · flagship",
        "meta/llama-3.1-8b-instruct": "128K ctx · fast",
        "nvidia/llama-3.1-nemotron-70b-instruct": "32K ctx · NVIDIA optimized",
        "nvidia/nemotron-mini-4b-instruct": "4K ctx · lightweight",
        "deepseek-ai/deepseek-r1": "64K ctx · reasoning",
        "mistralai/mixtral-8x22b-instruct-v0.1": "65K ctx",
        "microsoft/phi-3-medium-128k-instruct": "128K ctx · efficient",
        "google/gemma-2-27b-it": "8K ctx",
    }
    hint = _CONTEXT_HINTS.get(selected, "")
    if hint:
        st.caption(hint)


# ===================================================================
# Research section
# ===================================================================


def _render_research_section() -> None:
    """Perplexity research input."""
    from config.settings import settings as _settings

    st.markdown("### Web Research")

    api_key = _settings.perplexity_api_key
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


# ===================================================================
# Ingestion tabs
# ===================================================================


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

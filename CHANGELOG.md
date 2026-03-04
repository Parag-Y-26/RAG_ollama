# Changelog

All notable changes from the original monolithic codebase to this
production rebuild.

---

## [2.0.0] — 2026-03-04

### Architecture

- **Monolith → Modular**: Replaced the single 318-line `app.py` with
  a clean, layered architecture: `core/`, `persistence/`, `ui/`, with
  a ~60-line entrypoint.
- **Separation of concerns**: Business logic (`core/`) has zero
  Streamlit imports. UI code (`ui/`) has zero ChromaDB imports.
- **Typed dataclasses** throughout — `SourceInfo`, `SearchResult`,
  `ChatMessage`, `OllamaHealth`, `StreamingResponse` replace raw dicts.

### Bugs Fixed

- **`langchain_classic` import** — removed the non-existent package;
  all imports now use `langchain` / `langchain-core` / `langchain-ollama`.
- **SSL verification bypass** (`verify=False`) — replaced `CERT_NONE`
  with `certifi`-backed SSL contexts. Certificate verification is
  never disabled.
- **Perplexity API key fallback** — key is now resolved from
  `st.secrets` → env var, never hardcoded.
- **Temp file cleanup on Windows** — PDF loader now uses `pypdf`
  directly (in-memory `BytesIO`), eliminating temp-file permission
  errors entirely.
- **Redundant LLM instantiation** — model is cached via
  `st.cache_resource`; no longer re-created on every rerun.
- **ChromaDB deprecation** — migrated from legacy `Chroma()` to
  `chromadb.PersistentClient` with explicit `_persist()` calls.

### Security

- **Certifi SSL** for all external HTTPS calls (Perplexity API).
- **SSRF protection** — web URL ingestion blocks `localhost`,
  `127.0.0.1`, `0.0.0.0`, `::1`, and `169.254.*` addresses.
- **Secret management** — all API keys and configuration read from
  `.streamlit/secrets.toml` via `st.secrets`, never hardcoded.
- **PDF magic-byte validation** — rejects non-PDF uploads before
  processing.

### New Features

- **Multi-notebook support** — create and switch between isolated
  knowledge bases, each backed by its own ChromaDB collection.
- **Streaming RAG responses** — token-by-token display via
  `st.write_stream()` with async-to-sync bridge.
- **LCEL RAG chain** — built with LangChain Expression Language
  (`prompt | llm`), no legacy chain classes.
- **Confidence gating** — when the retriever finds 0 relevant chunks,
  falls back to base model with an explicit disclaimer.
- **Inline source citations** — LLM is prompted to cite `[Source N]`
  for every factual claim.
- **Collapsible source cards** — each assistant message has an
  expandable "Sources Used" section showing chunk text and relevance %.
- **Chat persistence** — SQLite-backed `ChatStore` with WAL journal
  mode, survives app restarts.
- **Markdown chat export** — download full conversation as `.md`.
- **Copy-to-clipboard** — per-message copy button via Base64 JS.
- **Perplexity web research** — topic-based internet research
  absorbed directly into the knowledge base.
- **YouTube thumbnail preview** — shown before ingestion via
  YouTube's image API.
- **Website favicon preview** — shown via Google's S2 favicon service.
- **PDF page count** — displayed before ingestion via `pypdf`.
- **Knowledge panel** — right-side inspector with search/filter,
  per-source delete (with confirmation), double-confirmed "Clear All",
  and aggregate stats (sources, chunks, estimated tokens).
- **Auto-generated KB summary** — LLM produces a 2-sentence overview
  of the knowledge base, regenerated when sources change.
- **Ollama health gate** — full-page error with exact install and
  model pull commands if Ollama is offline.
- **Model selector** — dynamic dropdown showing only installed
  models, with refresh button and pull hints.
- **Reset Database** — nuclear option for ChromaDB corruption
  recovery, accessible from the knowledge panel.

### Performance

- `get_available_models()` cached via `@st.cache_data(ttl=30)`.
- `KnowledgeBase` and `ChatStore` cached via `@st.cache_resource`.
- No expensive operations inside the Streamlit render loop.

### Error Handling

- Per-page PDF parsing — failed pages are skipped with a warning,
  remaining pages are still ingested.
- ChromaDB corruption — caught at init with descriptive error and
  "Reset Database" recovery button.
- Missing Ollama model — shows exact `ollama pull <model>` command.
- Perplexity API — granular error handling for 401 (auth), 429
  (rate limit), and network failures.
- All exceptions surface as Streamlit toasts — never raw tracebacks.

### Developer Experience

- Module-level docstrings on every file.
- Type hints and docstrings on every function.
- Professional `README.md` with ASCII architecture diagram.
- `.streamlit/secrets.toml.example` template.
- Updated `.gitignore` for `data/`, `__pycache__/`, `.env`.

### Removed

- `ask_pplx.py` — superseded by `core/research.py`.
- `pull_deepseek.py` — superseded by README instructions.
- Embedded CSS in `app.py` — replaced with structured theme injection.
- `langchain_classic` dependency — does not exist.

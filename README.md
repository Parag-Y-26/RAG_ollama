# 🧠 NotebookLM

A production-grade AI knowledge base powered by **local LLMs** (Ollama)
with retrieval-augmented generation, multi-notebook isolation, web
research, and a premium dark UI.

---

## ✨ Features

| Feature                    | Details                                                   |
| -------------------------- | --------------------------------------------------------- |
| **Multi-source ingestion** | YouTube transcripts, web pages, PDFs, plain text          |
| **Streaming RAG**          | Token-by-token responses with inline source citations     |
| **Multi-notebook**         | Isolated ChromaDB collections per notebook                |
| **Perplexity research**    | Internet research via Sonar API absorbed into your KB     |
| **Confidence gating**      | Falls back to base model with disclaimer when KB is empty |
| **Chat persistence**       | SQLite-backed history with markdown export                |
| **Premium UI**             | Obsidian dark theme, cyan accents, collapsible panels     |
| **Source inspector**       | Search, filter, delete, stats — all inline                |

---

## 📋 Prerequisites

### 1. Python 3.11+

```bash
python --version
```

### 2. Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer
# https://ollama.com/download
```

### 3. Pull required models

```bash
# LLM (generation)
ollama pull deepseek-r1:latest

# Embedding model (vector search)
ollama pull nomic-embed-text
```

Verify Ollama is running:

```bash
ollama list
```

---

## 🚀 Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd NOTEBOOKLM

# Create virtual environment
python -m venv .venv

# Activate it
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your Perplexity API key
(optional — only needed for web research):

```toml
PPLX_API_KEY = "pplx-xxxxxxxxxxxx"
```

---

## ▶️ Run

```bash
# Make sure Ollama is running first
ollama serve

# Start the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🏗️ Architecture

```
NOTEBOOKLM/
│
├── app.py                    ← Entrypoint: theme, health gate, layout
│
├── core/                     ← Business logic (no Streamlit imports)
│   ├── vectorstore.py        │  KnowledgeBase (ChromaDB + dedup)
│   ├── ingestion.py          │  load_youtube / web / pdf / text
│   ├── rag_chain.py          │  LCEL streaming RAG + confidence gate
│   ├── research.py           │  Perplexity Sonar API (certifi SSL)
│   └── models.py             │  Ollama health + model discovery
│
├── persistence/              ← Durable storage
│   └── chat_store.py         │  SQLite chat history
│
├── ui/                       ← Streamlit rendering (no business logic)
│   ├── sidebar.py            │  Ingestion tabs, model picker, health
│   ├── chat.py               │  Streaming chat + source cards
│   └── knowledge_panel.py    │  Source inspector + stats
│
├── data/                     ← Runtime data (gitignored)
│   ├── chroma_db/            │  Vector embeddings
│   └── chat_history.db       │  SQLite messages
│
├── .streamlit/
│   ├── config.toml           │  Streamlit settings
│   └── secrets.toml          │  API keys (never committed)
│
├── requirements.txt
└── README.md
```

### Data flow

```
User Question
     │
     ▼
┌──────────┐    similarity_search()   ┌────────────────┐
│ RAGChain │ ◄─────────────────────── │ KnowledgeBase  │
│  (LCEL)  │    top-k chunks          │  (ChromaDB)    │
└────┬─────┘                          └───────┬────────┘
     │                                        ▲
     │  astream() tokens                      │ add_source()
     ▼                                        │
┌──────────┐                          ┌───────┴────────┐
│   Chat   │                          │   Ingestion    │
│   UI     │                          │  (YouTube/Web/ │
└──────────┘                          │   PDF/Text)    │
                                      └────────────────┘
```

---

## 📖 Usage Guide

1. **Add sources** — use the sidebar tabs to ingest YouTube videos,
   web pages, PDFs, or paste raw text.
2. **Ask questions** — the AI answers strictly from your sources with
   inline `[Source N]` citations.
3. **Web research** — enter a topic and click "Research & Absorb" to
   pull internet research into your KB.
4. **Switch notebooks** — use the dropdown at the top to create and
   switch between isolated knowledge bases.
5. **Manage sources** — the right panel lets you search, inspect,
   and delete individual sources.
6. **Export chat** — click the download button to save your
   conversation as Markdown.

---

## 🔒 Security

- **SSL**: All external HTTPS calls use `certifi` — certificate
  verification is never disabled.
- **SSRF protection**: Web URLs are validated against localhost and
  private IP ranges before fetching.
- **Secrets**: API keys are loaded from `st.secrets` — never
  hardcoded or logged.
- **Local-first**: LLM inference runs entirely on your machine via
  Ollama — no data leaves your network (unless you use Perplexity).

---

## 📝 License

MIT

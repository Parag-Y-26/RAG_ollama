"""
Utility package for NotebookLM.

Shared helpers that are used across ``core/``, ``ingestion/``, and ``ui/``.

Modules
-------
logging
    Structured logging with Streamlit-rerun-safe initialisation.
validators
    URL validation (SSRF-safe), file type/size checks, query sanitisation.
ollama_client
    Lightweight Ollama REST API helpers for health checks and model listing.
"""

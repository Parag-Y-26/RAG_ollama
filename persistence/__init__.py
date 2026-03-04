"""
Persistence package for NotebookLM.

Provides durable storage for chat history using SQLite — a built-in
Python module with zero external dependencies.

Modules
-------
chat_store
    ``ChatStore`` — SQLite-backed chat history with per-notebook
    isolation, timestamp tracking, source citation storage, and
    markdown export.
"""

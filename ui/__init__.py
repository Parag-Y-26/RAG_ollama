"""
UI package for NotebookLM.

All Streamlit-specific rendering lives here, cleanly separated from
business logic in ``core/`` and ``ingestion/``.

Modules
-------
styles
    Premium Obsidian-dark CSS theme injection.
components
    Reusable widgets: source cards, empty states, status indicators.
sidebar
    Sidebar layout: notebook selector, model picker, ingestion tabs.
chat
    Streaming chat interface with token-by-token display.
source_panel
    Source document viewer and management panel.
"""

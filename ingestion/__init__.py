"""
Document ingestion package for NotebookLM.

Provides a pluggable loader architecture.  Every loader extends
``BaseLoader`` and implements the ``load → chunk → store`` pipeline.

Modules
-------
base
    Abstract ``BaseLoader`` with the shared ingest pipeline.
pdf_loader
    PDF file extraction via ``PyPDFLoader``.
web_loader
    Web page scraping with SSRF-safe URL validation.
youtube_loader
    YouTube transcript fetching.
text_loader
    Plain-text / Markdown ingestion.
perplexity_researcher
    Internet research via Perplexity Sonar API (``certifi`` SSL).
"""

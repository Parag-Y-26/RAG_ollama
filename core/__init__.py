"""
Core RAG pipeline package for NotebookLM.

Contains the fundamental building blocks of the retrieval-augmented
generation system:

Modules
-------
embeddings
    Cached Ollama embedding model factory.
llm
    LLM provider abstraction with streaming-first design.
vectorstore
    ChromaDB ``PersistentClient`` management with multi-notebook
    collection isolation and explicit ``.persist()`` calls.
rag_chain
    Streaming RAG pipeline with source citation and graceful
    fallback to direct LLM mode when no documents are available.
"""

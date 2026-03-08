"""
Persistent multi-notebook management.

Each notebook maps to an isolated knowledge base collection and chat
history. Notebook metadata is stored in `data/notebooks.json`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

_NOTEBOOKS_FILE = settings.data_dir / "notebooks.json"


@dataclass
class Notebook:
    """A single notebook / knowledge base."""

    id: str
    name: str
    created_at: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Notebook":
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )


class NotebookManager:
    """CRUD operations for notebooks with file-based persistence."""

    def __init__(self) -> None:
        self._ensure_data_dir()
        self._notebooks: dict[str, Notebook] = self._load()

    def list_notebooks(self) -> list[Notebook]:
        """Return all notebooks sorted by creation date, newest first."""
        return sorted(
            self._notebooks.values(),
            key=lambda nb: nb.created_at,
            reverse=True,
        )

    def get_notebook(self, notebook_id: str) -> Optional[Notebook]:
        """Get a notebook by ID."""
        return self._notebooks.get(notebook_id)

    def create_notebook(self, name: str, description: str = "") -> Notebook:
        """Create a new notebook with a unique display name."""
        clean_name = name.strip() or settings.default_notebook_name
        normalized = clean_name.casefold()

        if len(self._notebooks) >= settings.max_notebooks:
            raise ValueError(
                f"Maximum of {settings.max_notebooks} notebooks reached."
            )
        if any(nb.name.casefold() == normalized for nb in self._notebooks.values()):
            raise ValueError(f"A notebook named '{clean_name}' already exists.")

        notebook = Notebook(
            id=uuid.uuid4().hex[:12],
            name=clean_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description.strip(),
        )
        self._notebooks[notebook.id] = notebook
        self._save()
        logger.info("Created notebook: %s (%s)", notebook.name, notebook.id)
        return notebook

    def rename_notebook(self, notebook_id: str, new_name: str) -> bool:
        """Rename a notebook if the destination name is available."""
        notebook = self._notebooks.get(notebook_id)
        clean_name = new_name.strip()
        if not notebook or not clean_name:
            return False
        if any(
            nb.id != notebook_id and nb.name.casefold() == clean_name.casefold()
            for nb in self._notebooks.values()
        ):
            return False

        notebook.name = clean_name
        self._save()
        return True

    def delete_notebook(self, notebook_id: str) -> bool:
        """Delete notebook metadata and attempt to clear its persisted data."""
        if notebook_id not in self._notebooks:
            return False

        try:
            kb = self._build_knowledge_base(notebook_id)
            kb.clear_all()
        except Exception as exc:
            logger.warning(
                "Failed to clear knowledge base for notebook %s: %s",
                notebook_id,
                exc,
            )

        try:
            from persistence.chat_store import ChatStore

            store = ChatStore(str(settings.data_dir / "chat_history.db"))
            store.clear_history(notebook_id)
            store.close()
        except Exception as exc:
            logger.warning(
                "Failed to clear chat history for notebook %s: %s",
                notebook_id,
                exc,
            )

        del self._notebooks[notebook_id]
        self._save()
        logger.info("Deleted notebook: %s", notebook_id)
        return True

    def get_notebook_stats(self, notebook_id: str) -> dict:
        """Get source and chunk counts for a notebook."""
        kb = self._build_knowledge_base(notebook_id)
        sources = kb.get_all_sources()
        chunk_count = sum(source.chunk_count for source in sources)
        return {
            "chunk_count": chunk_count,
            "source_count": len(sources),
            "sources": sources,
        }

    def ensure_default_exists(self) -> Notebook:
        """Ensure at least one notebook exists."""
        if not self._notebooks:
            return self.create_notebook(settings.default_notebook_name)
        return self.list_notebooks()[-1]

    def _ensure_data_dir(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Notebook]:
        """Load notebooks from the JSON metadata file."""
        if not _NOTEBOOKS_FILE.exists():
            return {}

        try:
            data = json.loads(_NOTEBOOKS_FILE.read_text(encoding="utf-8"))
            return {
                notebook["id"]: Notebook.from_dict(notebook)
                for notebook in data.get("notebooks", [])
            }
        except Exception as exc:
            logger.error("Failed to load notebooks: %s", exc)
            return {}

    def _save(self) -> None:
        """Persist notebooks to the JSON metadata file."""
        try:
            data = {
                "notebooks": [nb.to_dict() for nb in self._notebooks.values()]
            }
            _NOTEBOOKS_FILE.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save notebooks: %s", exc)

    @staticmethod
    def _build_knowledge_base(notebook_id: str):
        """Construct a KnowledgeBase instance for a notebook on demand."""
        from core.vectorstore import KnowledgeBase

        return KnowledgeBase(
            persist_dir=str(settings.chroma_db_dir),
            collection_name=notebook_id,
            embedding_model=settings.embedding_model,
            ollama_base_url=settings.ollama_base_url,
        )

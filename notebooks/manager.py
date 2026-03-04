"""
Multi-notebook management.

Each notebook is an isolated knowledge base with its own:
- ChromaDB collection
- Chat history
- Source list
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings
from core.vectorstore import delete_collection, get_collection_count, get_source_documents
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
    def from_dict(cls, data: dict) -> Notebook:
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )


class NotebookManager:
    """CRUD operations for notebooks with file-based persistence."""

    def __init__(self):
        self._ensure_data_dir()
        self._notebooks: dict[str, Notebook] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_notebooks(self) -> list[Notebook]:
        """Return all notebooks sorted by creation date (newest first)."""
        return sorted(
            self._notebooks.values(),
            key=lambda nb: nb.created_at,
            reverse=True,
        )

    def get_notebook(self, notebook_id: str) -> Optional[Notebook]:
        """Get a notebook by ID."""
        return self._notebooks.get(notebook_id)

    def create_notebook(self, name: str, description: str = "") -> Notebook:
        """Create a new notebook."""
        if len(self._notebooks) >= settings.max_notebooks:
            raise ValueError(
                f"Maximum of {settings.max_notebooks} notebooks reached."
            )

        notebook = Notebook(
            id=uuid.uuid4().hex[:12],
            name=name.strip() or settings.default_notebook_name,
            created_at=datetime.now().isoformat(),
            description=description.strip(),
        )
        self._notebooks[notebook.id] = notebook
        self._save()
        logger.info("Created notebook: %s (%s)", notebook.name, notebook.id)
        return notebook

    def rename_notebook(self, notebook_id: str, new_name: str) -> bool:
        """Rename a notebook."""
        nb = self._notebooks.get(notebook_id)
        if not nb:
            return False
        # Dataclass is not frozen, so we can mutate
        object.__setattr__(nb, "name", new_name.strip())
        self._save()
        return True

    def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook and its ChromaDB collection."""
        if notebook_id not in self._notebooks:
            return False

        # Delete the vector collection
        delete_collection(notebook_id)

        del self._notebooks[notebook_id]
        self._save()
        logger.info("Deleted notebook: %s", notebook_id)
        return True

    def get_notebook_stats(self, notebook_id: str) -> dict:
        """Get stats for a notebook (chunk count, source count)."""
        sources = get_source_documents(notebook_id)
        chunk_count = get_collection_count(notebook_id)
        return {
            "chunk_count": chunk_count,
            "source_count": len(sources),
            "sources": sources,
        }

    def ensure_default_exists(self) -> Notebook:
        """Ensure at least one notebook exists; create default if needed."""
        if not self._notebooks:
            return self.create_notebook(settings.default_notebook_name)
        return self.list_notebooks()[-1]  # oldest notebook

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_data_dir(self) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Notebook]:
        """Load notebooks from JSON file."""
        if not _NOTEBOOKS_FILE.exists():
            return {}
        try:
            data = json.loads(_NOTEBOOKS_FILE.read_text(encoding="utf-8"))
            return {
                nb["id"]: Notebook.from_dict(nb)
                for nb in data.get("notebooks", [])
            }
        except Exception as exc:
            logger.error("Failed to load notebooks: %s", exc)
            return {}

    def _save(self) -> None:
        """Persist notebooks to JSON file."""
        try:
            data = {
                "notebooks": [nb.to_dict() for nb in self._notebooks.values()]
            }
            _NOTEBOOKS_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to save notebooks: %s", exc)

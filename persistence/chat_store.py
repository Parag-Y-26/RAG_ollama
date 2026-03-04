"""
SQLite-backed chat history store.

Provides the ``ChatStore`` class for persisting conversation messages
across Streamlit reruns and application restarts.  Uses Python's
built-in ``sqlite3`` module — no extra dependencies required.

Schema
------
``messages`` table:

    ============== ======= =============================================
    Column         Type    Description
    ============== ======= =============================================
    id             INTEGER Auto-incrementing primary key
    notebook_id    TEXT    Notebook this message belongs to
    role           TEXT    ``"user"`` or ``"assistant"``
    content        TEXT    The message body (markdown supported)
    timestamp      TEXT    ISO-8601 UTC timestamp
    sources_cited  TEXT    JSON-encoded list of source strings (nullable)
    ============== ======= =============================================
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatMessage:
    """Immutable representation of a single chat message.

    Attributes:
        id: Database row ID.
        notebook_id: Owning notebook identifier.
        role: ``"user"`` or ``"assistant"``.
        content: Message text (may contain markdown).
        timestamp: ISO-8601 UTC creation time.
        sources_cited: List of source labels cited in this message.
    """

    id: int
    notebook_id: str
    role: str
    content: str
    timestamp: str
    sources_cited: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ChatStore
# ---------------------------------------------------------------------------

class ChatStore:
    """SQLite-backed persistent chat history.

    Each ``ChatStore`` instance manages a single SQLite database file.
    Messages are isolated by ``notebook_id``, so multiple notebooks
    can share one database without interference.

    Usage::

        store = ChatStore("data/chat_history.db")
        store.save_message("nb_001", "user", "What is RAG?")
        store.save_message("nb_001", "assistant", "RAG is ...",
                           sources_cited=["paper.pdf"])
        history = store.load_history("nb_001")
    """

    _CREATE_TABLE_SQL: str = """
        CREATE TABLE IF NOT EXISTS messages (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id    TEXT    NOT NULL,
            role           TEXT    NOT NULL,
            content        TEXT    NOT NULL,
            timestamp      TEXT    NOT NULL,
            sources_cited  TEXT
        );
    """

    _CREATE_INDEX_SQL: str = """
        CREATE INDEX IF NOT EXISTS idx_messages_notebook
        ON messages (notebook_id, timestamp);
    """

    def __init__(self, db_path: str) -> None:
        """Open (or create) the SQLite database.

        Args:
            db_path: Filesystem path for the ``.db`` file.  Parent
                directories are created automatically.
        """
        parent = Path(db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

        self._db_path: str = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path,
            check_same_thread=False,  # safe for Streamlit's threading
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._conn.execute(self._CREATE_INDEX_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_message(
        self,
        notebook_id: str,
        role: str,
        content: str,
        sources_cited: Optional[list[str]] = None,
    ) -> ChatMessage:
        """Persist a single chat message.

        Args:
            notebook_id: The notebook this message belongs to.
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
            sources_cited: Optional list of cited source names /
                URLs.  Stored as a JSON array.

        Returns:
            The ``ChatMessage`` that was written (including its
            database-assigned ``id``).
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        sources_json: Optional[str] = (
            json.dumps(sources_cited) if sources_cited else None
        )

        cursor = self._conn.execute(
            """
            INSERT INTO messages (notebook_id, role, content, timestamp, sources_cited)
            VALUES (?, ?, ?, ?, ?)
            """,
            (notebook_id, role, content, timestamp, sources_json),
        )
        self._conn.commit()

        return ChatMessage(
            id=cursor.lastrowid or 0,
            notebook_id=notebook_id,
            role=role,
            content=content,
            timestamp=timestamp,
            sources_cited=sources_cited or [],
        )

    def load_history(
        self,
        notebook_id: str,
        limit: int = 200,
    ) -> list[ChatMessage]:
        """Load the chat history for a notebook.

        Args:
            notebook_id: Target notebook.
            limit: Maximum number of messages to return (most recent
                last, matching natural conversation order).

        Returns:
            Chronologically ordered list of ``ChatMessage`` objects.
        """
        cursor = self._conn.execute(
            """
            SELECT id, notebook_id, role, content, timestamp, sources_cited
            FROM messages
            WHERE notebook_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (notebook_id, limit),
        )

        messages: list[ChatMessage] = []
        for row in cursor.fetchall():
            sources = (
                json.loads(row["sources_cited"])
                if row["sources_cited"]
                else []
            )
            messages.append(
                ChatMessage(
                    id=row["id"],
                    notebook_id=row["notebook_id"],
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    sources_cited=sources,
                )
            )

        return messages

    def clear_history(self, notebook_id: str) -> int:
        """Delete all messages for a notebook.

        Args:
            notebook_id: Target notebook.

        Returns:
            Number of messages deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM messages WHERE notebook_id = ?",
            (notebook_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    def export_as_markdown(self, notebook_id: str) -> str:
        """Export a notebook's full chat history as Markdown.

        Produces a human-readable Markdown document with timestamps,
        role labels, message content, and cited sources.

        Args:
            notebook_id: Target notebook.

        Returns:
            A Markdown-formatted string.  Returns a brief placeholder
            message if the history is empty.
        """
        messages = self.load_history(notebook_id, limit=10_000)

        if not messages:
            return f"# Chat Export — {notebook_id}\n\n_No messages._\n"

        export_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: list[str] = [
            f"# Chat Export — {notebook_id}",
            f"_Exported on {export_time}_",
            "",
            "---",
            "",
        ]

        for msg in messages:
            ts = msg.timestamp[:19].replace("T", " ")  # trim microseconds
            role_label = "🧑 **User**" if msg.role == "user" else "🤖 **Assistant**"
            lines.append(f"### {role_label}  _{ts}_")
            lines.append("")
            lines.append(msg.content)

            if msg.sources_cited:
                cited = ", ".join(f"`{s}`" for s in msg.sources_cited)
                lines.append("")
                lines.append(f"> 📎 Sources: {cited}")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call multiple times.
        """
        try:
            self._conn.close()
        except Exception:
            pass

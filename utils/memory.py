"""
SQLite-based memory system for the Discord bot.
Stores recent messages per channel for context.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("MEMORY_DB_PATH", "/tmp/bot_memory.db")


class MemoryStore:
    """Simple SQLite-backed message memory per channel."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    guild_id TEXT,
                    author_name TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_created 
                ON messages(channel_id, created_at)
            """)
            conn.commit()

    def add_message(self, channel_id: str, guild_id: Optional[str],
                    author_name: str, author_id: str, content: str):
        """Store a message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (channel_id, guild_id, author_name, author_id, content) VALUES (?, ?, ?, ?, ?)",
                (channel_id, guild_id, author_name, author_id, content)
            )
            conn.commit()

    def get_recent(self, channel_id: str, limit: int = 20,
                   exclude_author_id: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Get recent messages for a channel, returns (author_name, content, created_at).
           Optionally exclude messages from a specific author (e.g. the bot itself)."""
        with sqlite3.connect(self.db_path) as conn:
            if exclude_author_id:
                rows = conn.execute(
                    """SELECT author_name, content, created_at FROM messages 
                       WHERE channel_id = ? AND author_id != ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (channel_id, exclude_author_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT author_name, content, created_at FROM messages 
                       WHERE channel_id = ? 
                       ORDER BY created_at DESC LIMIT ?""",
                    (channel_id, limit)
                ).fetchall()
        return list(reversed(rows))

    def get_user_history(self, author_id: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Get recent messages from a specific user across all channels."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT content, created_at FROM messages 
                   WHERE author_id = ? 
                   ORDER BY created_at DESC LIMIT ?""",
                (author_id, limit)
            ).fetchall()
        return list(reversed(rows))

    def cleanup_old(self, days: int = 7):
        """Remove messages older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
            conn.commit()

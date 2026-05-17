"""
SQLite audit log for autonomous bot actions.
"""

import os
import sqlite3
from contextlib import closing
from typing import Dict, List, Optional

DB_PATH = os.getenv("ACTION_AUDIT_DB_PATH", "/tmp/fellasbot_actions.db")


class ActionAuditStore:
    """Stores why the bot acted, for admin tuning/debugging."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id TEXT,
                    channel_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    topic_id INTEGER,
                    probability REAL,
                    roll REAL,
                    message_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_controls (
                    channel_id TEXT PRIMARY KEY,
                    learning_enabled INTEGER NOT NULL DEFAULT 1,
                    style_enabled INTEGER NOT NULL DEFAULT 1,
                    topics_enabled INTEGER NOT NULL DEFAULT 1,
                    starters_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_enabled INTEGER NOT NULL DEFAULT 1,
                    quiet_enabled INTEGER NOT NULL DEFAULT 0,
                    tracking_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_rate REAL NOT NULL DEFAULT 1.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(channel_controls)").fetchall()
            }
            if "tracking_enabled" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN tracking_enabled INTEGER NOT NULL DEFAULT 1"
                )
            if "spontaneous_rate" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN spontaneous_rate REAL NOT NULL DEFAULT 1.0"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS control_aliases (
                    alias TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    guild_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_action_audit_channel_created
                ON action_audit(channel_id, created_at DESC)
                """
            )
            conn.commit()

    def record(
        self,
        *,
        channel_id: str,
        guild_id: Optional[str] = None,
        action_type: str,
        reason: str = "",
        topic_id: Optional[int] = None,
        probability: Optional[float] = None,
        roll: Optional[float] = None,
        message_id: Optional[str] = None,
    ) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO action_audit
                    (guild_id, channel_id, action_type, reason, topic_id, probability, roll, message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (guild_id, channel_id, action_type, reason, topic_id, probability, roll, message_id),
            )
            conn.commit()

    def get_recent(self, channel_id: str, limit: int = 10) -> List[Dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM action_audit
                WHERE channel_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (channel_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_last(self, channel_id: str) -> Optional[Dict]:
        rows = self.get_recent(channel_id, limit=1)
        return rows[0] if rows else None

    def get_channel_controls(self, channel_id: str) -> Dict[str, bool]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM channel_controls WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        if not row:
            return {
                "learning_enabled": True,
                "style_enabled": True,
                "topics_enabled": True,
                "starters_enabled": True,
                "spontaneous_enabled": True,
                "quiet_enabled": False,
                "tracking_enabled": True,
                "spontaneous_rate": 1.0,
            }
        data = dict(row)
        return {
            "learning_enabled": bool(data["learning_enabled"]),
            "style_enabled": bool(data["style_enabled"]),
            "topics_enabled": bool(data["topics_enabled"]),
            "starters_enabled": bool(data["starters_enabled"]),
            "spontaneous_enabled": bool(data["spontaneous_enabled"]),
            "quiet_enabled": bool(data["quiet_enabled"]),
            "tracking_enabled": bool(data["tracking_enabled"]),
            "spontaneous_rate": float(data["spontaneous_rate"]),
        }

    def set_channel_control(self, channel_id: str, control: str, enabled: bool) -> None:
        allowed = {
            "learning": "learning_enabled",
            "style": "style_enabled",
            "topics": "topics_enabled",
            "starters": "starters_enabled",
            "spontaneous": "spontaneous_enabled",
            "quiet": "quiet_enabled",
            "tracking": "tracking_enabled",
        }
        column = allowed.get(control)
        if not column:
            raise ValueError(f"Unknown channel control: {control}")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO channel_controls (channel_id)
                VALUES (?)
                ON CONFLICT(channel_id) DO NOTHING
                """,
                (channel_id,),
            )
            conn.execute(
                f"UPDATE channel_controls SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE channel_id = ?",
                (1 if enabled else 0, channel_id),
            )
            conn.commit()

    def set_spontaneous_rate(self, channel_id: str, rate: float) -> None:
        if rate < 0 or rate > 2:
            raise ValueError("spontaneous rate must be between 0 and 2")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO channel_controls (channel_id)
                VALUES (?)
                ON CONFLICT(channel_id) DO NOTHING
                """,
                (channel_id,),
            )
            conn.execute(
                """
                UPDATE channel_controls
                SET spontaneous_rate = ?, updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ?
                """,
                (rate, channel_id),
            )
            conn.commit()

    def bind_alias(self, alias: str, channel_id: str, guild_id: Optional[str] = None) -> None:
        clean_alias = alias.strip().lower()
        if not clean_alias:
            raise ValueError("Alias cannot be empty")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO control_aliases (alias, channel_id, guild_id)
                VALUES (?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    guild_id = excluded.guild_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (clean_alias, channel_id, guild_id),
            )
            conn.commit()

    def unbind_alias(self, alias: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM control_aliases WHERE alias = ?", (alias.strip().lower(),))
            conn.commit()

    def resolve_alias(self, alias_or_channel_id: str) -> Optional[str]:
        value = alias_or_channel_id.strip()
        if value.isdigit():
            return value
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT channel_id FROM control_aliases WHERE alias = ?",
                (value.lower(),),
            ).fetchone()
        return row[0] if row else None

    def list_aliases(self) -> List[Dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT alias, channel_id, guild_id FROM control_aliases ORDER BY alias"
            ).fetchall()
        return [dict(row) for row in rows]

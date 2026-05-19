"""
SQLite audit log for autonomous bot actions.
"""

import os
import sqlite3
from contextlib import closing
from typing import Dict, List, Optional

DB_PATH = os.getenv("ACTION_AUDIT_DB_PATH", "/tmp/discord_llm_bot_actions.db")


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
                    trigger_type TEXT,
                    user_response_mode TEXT,
                    channel_mode TEXT,
                    final_action TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            audit_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(action_audit)").fetchall()
            }
            for column in ("trigger_type", "user_response_mode", "channel_mode", "final_action"):
                if column not in audit_columns:
                    conn.execute(f"ALTER TABLE action_audit ADD COLUMN {column} TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_controls (
                    channel_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL DEFAULT 'normal',
                    learning_enabled INTEGER NOT NULL DEFAULT 1,
                    style_enabled INTEGER NOT NULL DEFAULT 1,
                    topics_enabled INTEGER NOT NULL DEFAULT 1,
                    starters_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_react_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_reply_enabled INTEGER NOT NULL DEFAULT 0,
                    quiet_enabled INTEGER NOT NULL DEFAULT 0,
                    tracking_enabled INTEGER NOT NULL DEFAULT 1,
                    spontaneous_rate REAL NOT NULL DEFAULT 1.0,
                    spontaneous_react_rate REAL NOT NULL DEFAULT 1.0,
                    spontaneous_reply_rate REAL NOT NULL DEFAULT 0.0,
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
            if "spontaneous_react_enabled" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN spontaneous_react_enabled INTEGER NOT NULL DEFAULT 1"
                )
            if "spontaneous_reply_enabled" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN spontaneous_reply_enabled INTEGER NOT NULL DEFAULT 0"
                )
            if "spontaneous_react_rate" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN spontaneous_react_rate REAL NOT NULL DEFAULT 1.0"
                )
            if "spontaneous_reply_rate" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN spontaneous_reply_rate REAL NOT NULL DEFAULT 0.0"
                )
            if "mode" not in columns:
                conn.execute(
                    "ALTER TABLE channel_controls ADD COLUMN mode TEXT NOT NULL DEFAULT 'normal'"
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
                CREATE TABLE IF NOT EXISTS user_response_controls (
                    user_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL DEFAULT 'normal',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_privacy_controls (
                    user_id TEXT PRIMARY KEY,
                    remember_enabled INTEGER NOT NULL DEFAULT 1,
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
        trigger_type: Optional[str] = None,
        user_response_mode: Optional[str] = None,
        channel_mode: Optional[str] = None,
        final_action: Optional[str] = None,
    ) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO action_audit
                    (
                        guild_id, channel_id, action_type, reason, topic_id, probability, roll, message_id,
                        trigger_type, user_response_mode, channel_mode, final_action
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    channel_id,
                    action_type,
                    reason,
                    topic_id,
                    probability,
                    roll,
                    message_id,
                    trigger_type,
                    user_response_mode,
                    channel_mode,
                    final_action,
                ),
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
                "spontaneous_react_enabled": True,
                "spontaneous_reply_enabled": False,
                "quiet_enabled": False,
                "tracking_enabled": True,
                "spontaneous_rate": 1.0,
                "spontaneous_react_rate": 1.0,
                "spontaneous_reply_rate": 0.0,
                "mode": "normal",
            }
        data = dict(row)
        return {
            "learning_enabled": bool(data["learning_enabled"]),
            "style_enabled": bool(data["style_enabled"]),
            "topics_enabled": bool(data["topics_enabled"]),
            "starters_enabled": bool(data["starters_enabled"]),
            "spontaneous_enabled": bool(data["spontaneous_enabled"]),
            "spontaneous_react_enabled": bool(data.get("spontaneous_react_enabled", data["spontaneous_enabled"])),
            "spontaneous_reply_enabled": bool(data.get("spontaneous_reply_enabled", False)),
            "quiet_enabled": bool(data["quiet_enabled"]),
            "tracking_enabled": bool(data["tracking_enabled"]),
            "spontaneous_rate": float(data["spontaneous_rate"]),
            "spontaneous_react_rate": float(data.get("spontaneous_react_rate", data["spontaneous_rate"])),
            "spontaneous_reply_rate": float(data.get("spontaneous_reply_rate", 0.0)),
            "mode": data.get("mode") or "normal",
        }

    def set_channel_control(self, channel_id: str, control: str, enabled: bool) -> None:
        allowed = {
            "learning": "learning_enabled",
            "style": "style_enabled",
            "topics": "topics_enabled",
            "starters": "starters_enabled",
            "spontaneous": "spontaneous_enabled",
            "spontaneous_react": "spontaneous_react_enabled",
            "spontaneous_reply": "spontaneous_reply_enabled",
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
        self.set_spontaneous_react_rate(channel_id, rate)

    def set_spontaneous_react_rate(self, channel_id: str, rate: float) -> None:
        if rate < 0 or rate > 2:
            raise ValueError("spontaneous react rate must be between 0 and 2")
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
                SET spontaneous_rate = ?, spontaneous_react_rate = ?, updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ?
                """,
                (rate, rate, channel_id),
            )
            conn.commit()

    def set_spontaneous_reply_rate(self, channel_id: str, rate: float) -> None:
        if rate < 0 or rate > 2:
            raise ValueError("spontaneous reply rate must be between 0 and 2")
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
                SET spontaneous_reply_rate = ?, updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ?
                """,
                (rate, channel_id),
            )
            conn.commit()

    def set_channel_mode(self, channel_id: str, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in {"normal", "quiet", "observe-only", "ignore", "no-learning"}:
            raise ValueError("channel mode must be normal, quiet, observe-only, ignore, or no-learning")
        presets = {
            "normal": {
                "learning_enabled": 1,
                "style_enabled": 1,
                "topics_enabled": 1,
                "starters_enabled": 1,
                "spontaneous_enabled": 1,
                "spontaneous_react_enabled": 1,
                "spontaneous_reply_enabled": 0,
                "quiet_enabled": 0,
                "tracking_enabled": 1,
            },
            "quiet": {
                "learning_enabled": 1,
                "style_enabled": 1,
                "topics_enabled": 1,
                "starters_enabled": 0,
                "spontaneous_enabled": 0,
                "spontaneous_react_enabled": 0,
                "spontaneous_reply_enabled": 0,
                "quiet_enabled": 1,
                "tracking_enabled": 1,
            },
            "observe-only": {
                "learning_enabled": 1,
                "style_enabled": 1,
                "topics_enabled": 1,
                "starters_enabled": 0,
                "spontaneous_enabled": 0,
                "spontaneous_react_enabled": 0,
                "spontaneous_reply_enabled": 0,
                "quiet_enabled": 1,
                "tracking_enabled": 1,
            },
            "ignore": {
                "learning_enabled": 0,
                "style_enabled": 0,
                "topics_enabled": 0,
                "starters_enabled": 0,
                "spontaneous_enabled": 0,
                "spontaneous_react_enabled": 0,
                "spontaneous_reply_enabled": 0,
                "quiet_enabled": 1,
                "tracking_enabled": 0,
            },
            "no-learning": {
                "learning_enabled": 0,
                "style_enabled": 0,
                "topics_enabled": 0,
                "starters_enabled": 1,
                "spontaneous_enabled": 1,
                "spontaneous_react_enabled": 1,
                "spontaneous_reply_enabled": 0,
                "quiet_enabled": 0,
                "tracking_enabled": 1,
            },
        }[mode]
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
                SET mode = ?,
                    learning_enabled = ?,
                    style_enabled = ?,
                    topics_enabled = ?,
                    starters_enabled = ?,
                    spontaneous_enabled = ?,
                    spontaneous_react_enabled = ?,
                    spontaneous_reply_enabled = ?,
                    quiet_enabled = ?,
                    tracking_enabled = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ?
                """,
                (
                    mode,
                    presets["learning_enabled"],
                    presets["style_enabled"],
                    presets["topics_enabled"],
                    presets["starters_enabled"],
                    presets["spontaneous_enabled"],
                    presets["spontaneous_react_enabled"],
                    presets["spontaneous_reply_enabled"],
                    presets["quiet_enabled"],
                    presets["tracking_enabled"],
                    channel_id,
                ),
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

    def get_user_response_mode(self, user_id: str) -> str:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT mode FROM user_response_controls WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return row[0] if row else "normal"

    def set_user_response_mode(self, user_id: str, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in {"normal", "prompted", "strict"}:
            raise ValueError("user response mode must be normal, prompted, or strict")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO user_response_controls (user_id, mode)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    mode = excluded.mode,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (str(user_id), mode),
            )
            conn.commit()

    def list_user_response_modes(self) -> List[Dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT user_id, mode, updated_at FROM user_response_controls ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_user_privacy(self, user_id: str) -> Dict[str, bool]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT remember_enabled FROM user_privacy_controls WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return {"remember_enabled": bool(row[0]) if row else True}

    def set_user_remember_enabled(self, user_id: str, enabled: bool) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO user_privacy_controls (user_id, remember_enabled)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    remember_enabled = excluded.remember_enabled,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (str(user_id), 1 if enabled else 0),
            )
            conn.commit()

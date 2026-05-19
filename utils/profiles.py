"""
User profile learning system.
Maintains a running entry per user — traits, facts, preferences, and observations
gathered from chat interactions. Used to personalize responses.
"""

import os
import sqlite3
import json
import logging
import csv
from contextlib import closing
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

PROFILES_DB = os.getenv("PROFILES_DB", "/tmp/bot_profiles.db")


def _normalize_trigger_text(text: str) -> str:
    return " ".join((text or "").casefold().split())


class UserProfileStore:
    """Per-user profile storage. One row per user, appended/updated over time."""

    def __init__(self, db_path: str = PROFILES_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    known_facts TEXT NOT NULL DEFAULT '[]',
                    personality_notes TEXT NOT NULL DEFAULT '',
                    likes TEXT NOT NULL DEFAULT '[]',
                    dislikes TEXT NOT NULL DEFAULT '[]',
                    relationships TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id TEXT NOT NULL DEFAULT '',
                    trigger_text TEXT NOT NULL,
                    trigger_key TEXT NOT NULL,
                    reaction TEXT NOT NULL,
                    set_by TEXT NOT NULL DEFAULT '',
                    set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(guild_id, trigger_key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_custom_triggers_guild
                ON custom_triggers(guild_id, trigger_key)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_trigger_imports (
                    source TEXT NOT NULL,
                    guild_id TEXT NOT NULL DEFAULT '',
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source, guild_id)
                )
            """)
            conn.commit()

    def set_trigger(self, trigger_text: str, reaction: str, guild_id: str = "", set_by: str = "") -> None:
        trigger = (trigger_text or "").strip()
        emoji = (reaction or "").strip()
        key = _normalize_trigger_text(trigger)
        if not key:
            raise ValueError("Trigger text cannot be empty")
        if not emoji:
            raise ValueError("Reaction cannot be empty")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO custom_triggers (guild_id, trigger_text, trigger_key, reaction, set_by, set_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id, trigger_key) DO UPDATE SET
                    trigger_text = excluded.trigger_text,
                    reaction = excluded.reaction,
                    set_by = excluded.set_by,
                    set_at = CURRENT_TIMESTAMP
            """, (str(guild_id or ""), trigger, key, emoji, str(set_by or "")))
            conn.commit()

    def forget_trigger(self, trigger_text: str, guild_id: str = "") -> bool:
        key = _normalize_trigger_text(trigger_text)
        if not key:
            return False
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM custom_triggers WHERE guild_id = ? AND trigger_key = ?",
                (str(guild_id or ""), key),
            )
            deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def list_triggers(self, guild_id: str = "", limit: int = 25) -> List[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM custom_triggers
                WHERE guild_id = ?
                ORDER BY set_at DESC, id DESC
                LIMIT ?
            """, (str(guild_id or ""), int(limit))).fetchall()
        return [dict(row) for row in rows]

    def find_trigger_match(self, content: str, guild_id: str = "") -> Optional[Dict[str, Any]]:
        text = _normalize_trigger_text(content)
        if not text:
            return None
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM custom_triggers
                WHERE guild_id IN (?, '')
                ORDER BY LENGTH(trigger_key) DESC, id ASC
            """, (str(guild_id or ""),)).fetchall()
        for row in rows:
            trigger_key = row["trigger_key"]
            if trigger_key and trigger_key == text:
                return dict(row)
        return None

    def import_triggers_csv(self, path: str, guild_id: str = "") -> int:
        if not path or not os.path.exists(path):
            return 0
        count = 0
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f):
                trigger = (row.get("trigger") or "").strip()
                reaction = (row.get("response") or "").strip()
                if not trigger or not reaction:
                    continue
                self.set_trigger(
                    trigger,
                    reaction,
                    guild_id=guild_id,
                    set_by=(row.get("set_by") or row.get("command_word") or "csv"),
                )
                count += 1
        return count

    def import_triggers_csv_once(self, path: str, guild_id: str = "") -> int:
        if not path or not os.path.exists(path):
            return 0
        source = os.path.abspath(path)
        guild = str(guild_id or "")
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT row_count FROM custom_trigger_imports WHERE source = ? AND guild_id = ?",
                (source, guild),
            ).fetchone()
        if row:
            return 0
        count = self.import_triggers_csv(source, guild_id=guild)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO custom_trigger_imports (source, guild_id, row_count, imported_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (source, guild, count))
            conn.commit()
        return count

    def upsert_user(self, user_id: str, display_name: str):
        """Ensure a user exists in the store. Creates or updates name/timestamps."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO user_profiles (user_id, display_name, last_seen, message_count)
                VALUES (?, ?, CURRENT_TIMESTAMP, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    display_name = CASE
                        WHEN excluded.display_name != '' THEN excluded.display_name
                        ELSE user_profiles.display_name
                    END,
                    last_seen = CURRENT_TIMESTAMP,
                    message_count = message_count + 1
            """, (user_id, display_name))
            conn.commit()

    def add_fact(self, user_id: str, fact: str, display_name: str = ""):
        """Append a fact about a user. Deduplicates."""
        self.upsert_user(user_id, display_name)
        facts = self.get_facts(user_id)
        if fact not in facts:
            facts.append(fact)
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "UPDATE user_profiles SET known_facts = ? WHERE user_id = ?",
                    (json.dumps(facts), user_id)
                )
                conn.commit()

    def add_note(self, user_id: str, note: str, display_name: str = ""):
        """Append a personality observation."""
        self.upsert_user(user_id, display_name)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                UPDATE user_profiles
                SET personality_notes = personality_notes || '\n' || ?,
                    last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (note, user_id))
            conn.commit()

    def add_preference(self, user_id: str, kind: str, item: str, display_name: str = ""):
        """Add a like or dislike. kind='like' or 'dislike'."""
        self.upsert_user(user_id, display_name)
        col = "likes" if kind == "like" else "dislikes"
        items = self._get_json_list(user_id, col)
        if item not in items:
            items.append(item)
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    f"UPDATE user_profiles SET {col} = ? WHERE user_id = ?",
                    (json.dumps(items), user_id)
                )
                conn.commit()

    def _get_json_list(self, user_id: str, col: str) -> List[str]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                f"SELECT {col} FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            try:
                return json.loads(row[0] or "[]")
            except json.JSONDecodeError:
                return []
        return []

    def get_facts(self, user_id: str) -> List[str]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT known_facts FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            try:
                return json.loads(row[0] or "[]")
            except json.JSONDecodeError:
                return []
        return []

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the full profile dict for a user."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_profile_summary(self, user_id: str) -> str:
        """Get a compact, LLM-friendly summary of a user's profile."""
        profile = self.get_profile(user_id)
        if not profile:
            return ""

        parts = []
        name = profile["display_name"]
        parts.append(f"**{name}** (seen {profile['message_count']} messages)")

        facts = json.loads(profile["known_facts"] or "[]")
        if facts:
            parts.append("  Facts:")
            for f in facts:
                parts.append(f"    - {f}")

        notes = profile["personality_notes"].strip()
        if notes:
            parts.append("  Personality:")
            for line in notes.split("\n"):
                line = line.strip()
                if line:
                    parts.append(f"    - {line}")

        likes = json.loads(profile["likes"] or "[]")
        if likes:
            parts.append(f"  Likes: {', '.join(likes)}")

        dislikes = json.loads(profile["dislikes"] or "[]")
        if dislikes:
            parts.append(f"  Dislikes: {', '.join(dislikes)}")

        return "\n".join(parts)

    def get_all_summaries(self, limit: int = 10) -> str:
        """Get summaries of all known users, limited to the most active."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT user_id FROM user_profiles ORDER BY message_count DESC LIMIT ?",
                (limit,)
            ).fetchall()

        parts = []
        for (user_id,) in rows:
            summary = self.get_profile_summary(user_id)
            if summary:
                parts.append(summary)

        return "\n\n".join(parts) if parts else "No user profiles yet."

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        """Get all profile dicts, sorted by activity."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM user_profiles ORDER BY message_count DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def reset_profile(self, user_id: str):
        """Wipe a user's profile data but keep the record."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                UPDATE user_profiles SET
                    known_facts = '[]',
                    personality_notes = '',
                    likes = '[]',
                    dislikes = '[]',
                    relationships = '{}'
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()

    def delete_profile(self, user_id: str):
        """Fully delete a user's profile."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()

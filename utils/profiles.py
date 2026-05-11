"""
User profile learning system.
Maintains a running entry per user — traits, facts, preferences, and observations
gathered from chat interactions. Used to personalize responses.
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

PROFILES_DB = os.getenv("PROFILES_DB", "/tmp/bot_profiles.db")


class UserProfileStore:
    """Per-user profile storage. One row per user, appended/updated over time."""

    def __init__(self, db_path: str = PROFILES_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
            conn.commit()

    def upsert_user(self, user_id: str, display_name: str):
        """Ensure a user exists in the store. Creates or updates name/timestamps."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_profiles (user_id, display_name, last_seen, message_count)
                VALUES (?, ?, CURRENT_TIMESTAMP, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    display_name = excluded.display_name,
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE user_profiles SET known_facts = ? WHERE user_id = ?",
                    (json.dumps(facts), user_id)
                )
                conn.commit()

    def add_note(self, user_id: str, note: str, display_name: str = ""):
        """Append a personality observation."""
        self.upsert_user(user_id, display_name)
        with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE user_profiles SET {col} = ? WHERE user_id = ?",
                    (json.dumps(items), user_id)
                )
                conn.commit()

    def _get_json_list(self, user_id: str, col: str) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM user_profiles ORDER BY message_count DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def reset_profile(self, user_id: str):
        """Wipe a user's profile data but keep the record."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()

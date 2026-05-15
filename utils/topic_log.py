"""
Recurring topic learning and starter selection for the Discord orchestrator.
"""

import json
import logging
import os
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from utils.learning_safety import sanitize_recent_messages

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("TOPIC_LOG_DB_PATH", "/tmp/fellasbot_topics.db")

MAX_LABEL = 80
MAX_SUMMARY = 500
MAX_SEED_PROMPT = 300
LLMGenerate = Callable[..., Awaitable[str]]


def _truncate(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    return None


class TopicLogStore:
    """SQLite-backed recurring topic store."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id TEXT,
                    channel_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    seed_prompt TEXT NOT NULL DEFAULT '',
                    score REAL NOT NULL DEFAULT 1.0,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    started_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    muted INTEGER NOT NULL DEFAULT 0,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_started_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topics_channel_score
                ON topics(channel_id, score DESC, last_seen_at DESC)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_channel_label
                ON topics(channel_id, label)
                """
            )
            columns = [row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()]
            if "muted" not in columns:
                conn.execute("ALTER TABLE topics ADD COLUMN muted INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    def upsert_topics(
        self, channel_id: str, guild_id: Optional[str], topics: List[Dict[str, Any]]
    ) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            for topic in topics:
                label = _truncate(topic.get("label"), MAX_LABEL).lower()
                if not label:
                    continue
                summary = _truncate(topic.get("summary"), MAX_SUMMARY)
                seed_prompt = _truncate(topic.get("seed_prompt"), MAX_SEED_PROMPT)
                score = max(0.0, min(5.0, float(topic.get("score") or 1.0)))

                existing = conn.execute(
                    "SELECT summary, seed_prompt FROM topics WHERE channel_id = ? AND label = ?",
                    (channel_id, label),
                ).fetchone()
                if existing:
                    chosen_summary = summary if len(summary) > len(existing[0] or "") else existing[0]
                    chosen_seed = seed_prompt if len(seed_prompt) > len(existing[1] or "") else existing[1]
                    conn.execute(
                        """
                        UPDATE topics SET
                            guild_id = ?,
                            summary = ?,
                            seed_prompt = ?,
                            score = score + 0.5,
                            seen_count = seen_count + 1,
                            last_seen_at = CURRENT_TIMESTAMP
                        WHERE channel_id = ? AND label = ?
                        """,
                        (guild_id, chosen_summary, chosen_seed, channel_id, label),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO topics (guild_id, channel_id, label, summary, seed_prompt, score)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (guild_id, channel_id, label, summary, seed_prompt, score),
                    )
            conn.commit()

    def get_candidate_topics(self, channel_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM topics
                WHERE channel_id = ? AND score > 0 AND muted = 0
                ORDER BY score DESC, last_seen_at DESC
                LIMIT ?
                """,
                (channel_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_topic(self, topic_id: int) -> Optional[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return dict(row) if row else None

    def mark_started(self, topic_id: int) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                UPDATE topics SET
                    started_count = started_count + 1,
                    score = MAX(0, score - 0.75),
                    last_started_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (topic_id,),
            )
            conn.commit()

    def mark_success(self, topic_id: int) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                UPDATE topics SET
                    success_count = success_count + 1,
                    score = score + 1.0
                WHERE id = ?
                """,
                (topic_id,),
            )
            conn.commit()

    def mark_ignored(self, topic_id: int) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE topics SET score = MAX(0, score - 0.5) WHERE id = ?",
                (topic_id,),
            )
            conn.commit()

    def update_topic(
        self,
        topic_id: int,
        *,
        label: Optional[str] = None,
        summary: Optional[str] = None,
        seed_prompt: Optional[str] = None,
    ) -> None:
        updates = []
        values = []
        if label is not None:
            updates.append("label = ?")
            values.append(_truncate(label, MAX_LABEL).lower())
        if summary is not None:
            updates.append("summary = ?")
            values.append(_truncate(summary, MAX_SUMMARY))
        if seed_prompt is not None:
            updates.append("seed_prompt = ?")
            values.append(_truncate(seed_prompt, MAX_SEED_PROMPT))
        if not updates:
            return
        values.append(topic_id)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(f"UPDATE topics SET {', '.join(updates)} WHERE id = ?", tuple(values))
            conn.commit()

    def boost_topic(self, topic_id: int, amount: float = 1.0) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE topics SET score = score + ? WHERE id = ?",
                (amount, topic_id),
            )
            conn.commit()

    def set_topic_muted(self, topic_id: int, muted: bool = True) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE topics SET muted = ? WHERE id = ?",
                (1 if muted else 0, topic_id),
            )
            conn.commit()

    def delete_topic(self, topic_id: int) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
            conn.commit()

    def clear_channel_topics(self, channel_id: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM topics WHERE channel_id = ?", (channel_id,))
            conn.commit()

    def decay_scores(self, days: int = 14) -> None:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        delete_cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE topics SET score = score * 0.8 WHERE last_seen_at < ?",
                (cutoff,),
            )
            conn.execute(
                "DELETE FROM topics WHERE score < 0.2 AND last_seen_at < ?",
                (delete_cutoff,),
            )
            conn.commit()


class TopicLearner:
    """Extracts recurring topics and chooses starter candidates."""

    def __init__(
        self,
        store: TopicLogStore,
        interval: int = 40,
        context_limit: int = 60,
        starter_cooldown_seconds: int = 7200,
    ):
        self.store = store
        self.interval = max(1, interval)
        self.context_limit = max(1, context_limit)
        self.starter_cooldown_seconds = starter_cooldown_seconds

    def should_learn(self, message_count: int) -> bool:
        return message_count > 0 and message_count % self.interval == 0

    async def learn_from_recent(
        self,
        channel_id: str,
        guild_id: Optional[str],
        recent_messages: List[Tuple[str, str, str]],
        llm_generate: LLMGenerate,
    ) -> None:
        messages = sanitize_recent_messages(recent_messages)
        if len(messages) < 20:
            return
        messages = messages[-self.context_limit :]
        convo = "\n".join(f"{name}: {content}" for name, content, _ in messages)
        prompt = (
            "Extract recurring public conversation topics from this Discord channel.\n"
            "Return strict JSON only: {\"topics\": [{\"label\": str, \"summary\": str, "
            "\"seed_prompt\": str, \"score\": number}]}.\n"
            "Use stable short labels. Store no private details. Include at most five topics.\n\n"
            f"Conversation:\n{convo}"
        )
        try:
            raw = await llm_generate(
                prompt,
                system_prompt="You summarize recurring Discord topics into safe strict JSON.",
            )
            parsed = json.loads(raw.strip())
            topics = parsed.get("topics") if isinstance(parsed, dict) else None
            if isinstance(topics, list):
                self.store.upsert_topics(channel_id, guild_id, topics[:5])
        except Exception as exc:
            logger.warning("Topic learning failed channel_id=%s: %s", channel_id, exc)

    def choose_starter_topic(
        self, channel_id: str, now: Optional[datetime] = None, rng=random
    ) -> Optional[Dict[str, Any]]:
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        candidates = []
        for topic in self.store.get_candidate_topics(channel_id, limit=8):
            started = _parse_timestamp(topic.get("last_started_at"))
            if started and (now - started).total_seconds() < self.starter_cooldown_seconds:
                continue
            candidates.append(topic)
        if not candidates:
            return None

        total = sum(max(0.1, float(t.get("score") or 0.0)) for t in candidates)
        roll = rng.random() * total
        upto = 0.0
        for topic in candidates:
            upto += max(0.1, float(topic.get("score") or 0.0))
            if upto >= roll:
                return topic
        return candidates[0]

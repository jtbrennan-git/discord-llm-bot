"""
Feedback-based learning system for the Discord bot.
Tracks reactions to bot messages and stores lessons for self-improvement.
"""

import sqlite3
import os
import logging
from contextlib import closing
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "/tmp/bot_feedback.db")

# Reactions that signal positive feedback (bot did good)
POSITIVE_REACTIONS = {"👍", "❤️", "🔥", "😂", "💀", "🤣", "😭", "💯", "🫡", "👏", "true", "based", "real", "yes", "this"}

# Reactions that signal negative feedback (bot messed up)
NEGATIVE_REACTIONS = {"👎", "cringe", "mid", "bad", "wtf", "💩", "🤮", "no", "stop", "shut up", "ew"}

# Reactions that signal "okay but could be better"
NEUTRAL_REACTIONS = {"👀", "hmm", "interesting", "ok", "sure", "whatever", "🤔", "meh"}


@dataclass
class FeedbackEntry:
    """A single feedback entry from a reaction."""
    message_id: str
    channel_id: str
    author_id: str
    author_name: str
    bot_message: str
    reaction: str
    sentiment: str  # positive, negative, neutral
    timestamp: str


class FeedbackStore:
    """SQLite-backed feedback storage and learning system."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    bot_message TEXT NOT NULL,
                    reaction TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    reviewer_id TEXT NOT NULL DEFAULT '',
                    bot_message TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_labels_message
                ON quality_labels(message_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_message
                ON feedback(message_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_sentiment
                ON feedback(sentiment)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique_reaction
                ON feedback(message_id, author_id, reaction)
            """)
            conn.commit()

    def classify_reaction(self, reaction: str) -> str:
        """Classify a reaction as positive, negative, or neutral."""
        reaction_lower = reaction.lower().strip()
        if reaction_lower.startswith("<:lolll:") or reaction_lower == ":lolll:":
            return "positive"
        if reaction_lower in POSITIVE_REACTIONS:
            return "positive"
        elif reaction_lower in NEGATIVE_REACTIONS:
            return "negative"
        elif reaction_lower in NEUTRAL_REACTIONS:
            return "neutral"
        # Default: assume neutral if unknown
        return "neutral"

    def add_feedback(self, message_id: str, channel_id: str,
                     author_id: str, author_name: str,
                     bot_message: str, reaction: str):
        """Store a reaction as feedback."""
        sentiment = self.classify_reaction(reaction)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO feedback
                   (message_id, channel_id, author_id, author_name, bot_message, reaction, sentiment)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message_id, channel_id, author_id, author_name, bot_message, reaction, sentiment)
            )
            conn.commit()
        logger.debug(f"Feedback stored: {sentiment} reaction '{reaction}' on message {message_id}")

    def add_feedback_from_payload(self, payload: dict):
        """Add feedback from a discord.Reaction payload dict."""
        self.add_feedback(
            message_id=payload["message_id"],
            channel_id=payload["channel_id"],
            author_id=payload["user_id"],
            author_name=payload["user_name"],
            bot_message=payload["bot_message"],
            reaction=payload["reaction_emoji"],
        )

    def get_feedback_summary(self, limit: int = 50) -> List[dict]:
        """Get recent feedback grouped by sentiment."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """SELECT sentiment, reaction, COUNT(*) as count,
                          GROUP_CONCAT(DISTINCT author_name) as users
                   FROM feedback
                   GROUP BY sentiment, reaction
                   ORDER BY count DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return [{"sentiment": r[0], "reaction": r[1], "count": r[2], "users": r[3]} for r in rows]

    def get_negative_examples(self, limit: int = 10) -> List[Tuple[str, str]]:
        """Get bot messages that received negative feedback."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """SELECT bot_message, reaction
                   FROM feedback
                   WHERE sentiment = 'negative'
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        return list(rows)

    def get_recent_feedback_context(self, limit: int = 20) -> str:
        """Get recent feedback as a summary string for the LLM to learn from."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            # Count by sentiment
            sentiment_counts = conn.execute(
                """SELECT sentiment, COUNT(*) FROM feedback
                   GROUP BY sentiment"""
            ).fetchall()

            # Get recent negative examples
            negative = conn.execute(
                """SELECT bot_message, reaction
                   FROM feedback
                   WHERE sentiment = 'negative'
                   ORDER BY timestamp DESC
                   LIMIT 5"""
            ).fetchall()

            # Get most common positive reactions
            top_reactions = conn.execute(
                """SELECT reaction, COUNT(*)
                   FROM feedback
                   WHERE sentiment = 'positive'
                   GROUP BY reaction
                   ORDER BY COUNT(*) DESC
                   LIMIT 5"""
            ).fetchall()

        summary_parts = []
        sentiment_str = ", ".join([f"{s}: {c}" for s, c in sentiment_counts])
        summary_parts.append(f"Total feedback breakdown: {sentiment_str}")

        if negative:
            summary_parts.append("\nMessages users didn't like (with their reactions):")
            for msg, reaction in negative:
                short_msg = msg[:100] + "..." if len(msg) > 100 else msg
                summary_parts.append(f'  - "{short_msg}" → {reaction}')

        if top_reactions:
            summary_parts.append("\nReactions users liked most:")
            for reaction, count in top_reactions:
                summary_parts.append(f"  - {reaction} (used {count} times)")

        return "\n".join(summary_parts)

    def add_lesson(self, category: str, lesson: str, confidence: float = 1.0):
        """Store a lesson the bot has learned."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            # Check if similar lesson exists
            existing = conn.execute(
                "SELECT id FROM lessons WHERE lesson = ? AND category = ?",
                (lesson, category)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE lessons SET confidence = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (confidence, existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO lessons (category, lesson, confidence) VALUES (?, ?, ?)",
                    (category, lesson, confidence)
                )
            conn.commit()

    def add_quality_label(
        self,
        message_id: str,
        label: str,
        note: str = "",
        reviewer_id: str = "",
        bot_message: str = "",
    ) -> None:
        label = label.strip().lower()
        if label not in {"good", "bad", "too-much", "too-friendly", "missed-opportunity", "wrong-tone"}:
            raise ValueError("unknown quality label")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO quality_labels (message_id, label, note, reviewer_id, bot_message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(message_id), label, note.strip(), str(reviewer_id), bot_message),
            )
            conn.commit()

    def get_quality_labels(self, limit: int = 20) -> List[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT message_id, label, note, reviewer_id, bot_message, created_at
                FROM quality_labels
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_lessons(self, category: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Get stored lessons, optionally filtered by category."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            if category:
                rows = conn.execute(
                    "SELECT category, lesson, confidence FROM lessons WHERE category = ? ORDER BY confidence DESC",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT category, lesson, confidence FROM lessons ORDER BY confidence DESC"
                ).fetchall()
        return rows

    def cleanup_old(self, days: int = 30):
        """Remove feedback older than N days."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff,))
            conn.commit()


class FeedbackTracker:
    """Higher-level feedback tracker that wraps FeedbackStore and manages
    tracked bot messages, reaction counting, and lesson generation."""

    def __init__(self, db_path: str = DB_PATH):
        self.store = FeedbackStore(db_path)
        self._tracked_messages: dict = {}  # message_id -> content
        self.lessons: list = []
        self.load_lessons()

    def register_message(self, message_id: str, content: str):
        """Track a bot message so we can later record reactions against it."""
        self._tracked_messages[message_id] = content

    def add_reaction(self, message_id: str, emoji: str, user_id: str):
        """Record a reaction on a tracked bot message."""
        if message_id not in self._tracked_messages:
            return
        content = self._tracked_messages[message_id]
        self.store.add_feedback(
            message_id=message_id,
            channel_id="unknown",
            author_id=user_id,
            author_name="unknown",
            bot_message=content,
            reaction=emoji,
        )

    def add_quality_label(self, message_id: str, label: str, reviewer_id: str, note: str = "") -> bool:
        """Apply a manual quality label to a tracked bot message."""
        content = self._tracked_messages.get(message_id, "")
        self.store.add_quality_label(message_id, label, note=note, reviewer_id=reviewer_id, bot_message=content)
        return True

    @property
    def total_positive(self) -> int:
        return self._count_sentiment("positive")

    @property
    def total_negative(self) -> int:
        return self._count_sentiment("negative")

    def compute_sentiment(self, message_id: str) -> float:
        """Return positive-minus-negative score normalized by known reactions."""
        with closing(sqlite3.connect(self.store.db_path)) as conn:
            rows = conn.execute(
                "SELECT sentiment, COUNT(*) FROM feedback WHERE message_id = ? GROUP BY sentiment",
                (message_id,),
            ).fetchall()
        counts = {sentiment: count for sentiment, count in rows}
        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)
        total = positive + negative + counts.get("neutral", 0)
        if total == 0:
            return 0.0
        return (positive - negative) / total

    def _count_sentiment(self, sentiment: str) -> int:
        with closing(sqlite3.connect(self.store.db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE sentiment = ?",
                (sentiment,),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_feedback_context(self) -> Optional[str]:
        """Get recent feedback as context for the LLM."""
        return self.store.get_recent_feedback_context()

    def get_stats(self) -> dict:
        """Return feedback statistics."""
        summary = self.store.get_feedback_summary()
        total_positive = sum(s["count"] for s in summary if s["sentiment"] == "positive")
        total_negative = sum(s["count"] for s in summary if s["sentiment"] == "negative")
        total_tracked = len(self._tracked_messages)
        return {
            "total_positive": total_positive,
            "total_negative": total_negative,
            "total_tracked_messages": total_tracked,
            "quality_labels": len(self.store.get_quality_labels(limit=1000)),
        }

    def load_lessons(self):
        """Load existing lessons from the store."""
        raw = self.store.get_lessons()
        self.lessons = [f"{cat}: {lesson}" for cat, lesson, conf in raw]

    def update_lessons(self):
        """Analyze negative feedback and generate/upsert lessons."""
        negatives = self.store.get_negative_examples(limit=10)
        if not negatives:
            self.lessons = []
            return
        for msg, reaction in negatives:
            lesson = f"Users reacted {reaction} to messages like: {msg[:80]}..."
            self.store.add_lesson("negative_feedback", lesson)
        self.load_lessons()



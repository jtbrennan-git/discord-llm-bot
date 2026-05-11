"""
Feedback-based learning system for the Discord bot.
Tracks reactions to bot messages and stores lessons for self-improvement.
"""

import sqlite3
import os
import logging
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
        with sqlite3.connect(self.db_path) as conn:
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
                CREATE INDEX IF NOT EXISTS idx_feedback_message
                ON feedback(message_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_sentiment
                ON feedback(sentiment)
            """)
            conn.commit()

    def classify_reaction(self, reaction: str) -> str:
        """Classify a reaction as positive, negative, or neutral."""
        reaction_lower = reaction.lower().strip()
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO feedback 
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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

    def get_lessons(self, category: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Get stored lessons, optionally filtered by category."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff,))
            conn.commit()



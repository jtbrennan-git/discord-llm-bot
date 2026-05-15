"""
Channel-level style guide learning and storage.

The orchestrator owns this data. Learned style is compact prompt context, not a
runtime change to the underlying model or Hermes profile.
"""

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from utils.learning_safety import sanitize_recent_messages

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("STYLE_GUIDE_DB_PATH", "/tmp/fellasbot_style.db")

MAX_STYLE_SUMMARY = 600
MAX_PATTERN = 120
MAX_HUMOR_NOTES = 300
MAX_ITEMS = 5
MIN_PROMPT_CONFIDENCE = 0.35

LLMGenerate = Callable[..., Awaitable[str]]


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _clean_list(value: Any, limit: int = MAX_PATTERN) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = _truncate(item, limit)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= MAX_ITEMS:
            break
    return cleaned


def _json_list(value: str) -> List[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


class StyleGuideStore:
    """SQLite-backed channel style guides."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_style_guides (
                    channel_id TEXT PRIMARY KEY,
                    guild_id TEXT,
                    style_summary TEXT NOT NULL DEFAULT '',
                    do_patterns TEXT NOT NULL DEFAULT '[]',
                    avoid_patterns TEXT NOT NULL DEFAULT '[]',
                    common_phrases TEXT NOT NULL DEFAULT '[]',
                    humor_notes TEXT NOT NULL DEFAULT '',
                    energy_level TEXT NOT NULL DEFAULT 'neutral',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert_channel_style(
        self, channel_id: str, guild_id: Optional[str], style: Dict[str, Any]
    ) -> None:
        """Create or replace the learned style guide for a channel."""
        confidence = float(style.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        energy = str(style.get("energy_level") or "neutral").strip().lower()
        if energy not in {"low", "neutral", "high"}:
            energy = "neutral"

        payload = {
            "style_summary": _truncate(style.get("style_summary"), MAX_STYLE_SUMMARY),
            "do_patterns": json.dumps(_clean_list(style.get("do_patterns"))),
            "avoid_patterns": json.dumps(_clean_list(style.get("avoid_patterns"))),
            "common_phrases": json.dumps(_clean_list(style.get("common_phrases"))),
            "humor_notes": _truncate(style.get("humor_notes"), MAX_HUMOR_NOTES),
            "energy_level": energy,
            "confidence": confidence,
            "sample_count": int(style.get("sample_count") or 0),
        }

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO channel_style_guides
                    (channel_id, guild_id, style_summary, do_patterns, avoid_patterns,
                     common_phrases, humor_notes, energy_level, confidence, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    guild_id = excluded.guild_id,
                    style_summary = excluded.style_summary,
                    do_patterns = excluded.do_patterns,
                    avoid_patterns = excluded.avoid_patterns,
                    common_phrases = excluded.common_phrases,
                    humor_notes = excluded.humor_notes,
                    energy_level = excluded.energy_level,
                    confidence = excluded.confidence,
                    sample_count = excluded.sample_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    channel_id,
                    guild_id,
                    payload["style_summary"],
                    payload["do_patterns"],
                    payload["avoid_patterns"],
                    payload["common_phrases"],
                    payload["humor_notes"],
                    payload["energy_level"],
                    payload["confidence"],
                    payload["sample_count"],
                ),
            )
            conn.commit()

    def get_channel_style(self, channel_id: str) -> Optional[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM channel_style_guides WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["do_patterns"] = _json_list(result["do_patterns"])
        result["avoid_patterns"] = _json_list(result["avoid_patterns"])
        result["common_phrases"] = _json_list(result["common_phrases"])
        return result

    def get_prompt_context(self, channel_id: str) -> str:
        style = self.get_channel_style(channel_id)
        if not style or float(style.get("confidence") or 0.0) < MIN_PROMPT_CONFIDENCE:
            return ""

        parts = []
        if style["style_summary"]:
            parts.append(f"Summary: {style['style_summary']}")
        parts.append(f"Energy: {style['energy_level']}")
        if style["do_patterns"]:
            parts.append("Do: " + "; ".join(style["do_patterns"]))
        if style["avoid_patterns"]:
            parts.append("Avoid: " + "; ".join(style["avoid_patterns"]))
        if style["common_phrases"]:
            parts.append("Common phrases, use sparingly: " + "; ".join(style["common_phrases"]))
        if style["humor_notes"]:
            parts.append(f"Humor: {style['humor_notes']}")
        return "\n".join(parts)

    def clear_channel_style(self, channel_id: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM channel_style_guides WHERE channel_id = ?", (channel_id,))
            conn.commit()

    def cleanup_stale(self, days: int = 60) -> None:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM channel_style_guides WHERE updated_at < ?", (cutoff,))
            conn.commit()


class StyleGuideLearner:
    """Extracts channel-level style summaries from recent messages."""

    def __init__(self, store: StyleGuideStore, interval: int = 50, context_limit: int = 80):
        self.store = store
        self.interval = max(1, interval)
        self.context_limit = max(1, context_limit)

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
        if len(messages) < 25:
            return
        messages = messages[-self.context_limit :]
        convo = "\n".join(f"{name}: {content}" for name, content, _ in messages)
        prompt = (
            "Extract a compact group-level Discord channel style guide from this conversation.\n"
            "Return strict JSON only with keys: style_summary, do_patterns, avoid_patterns, "
            "common_phrases, humor_notes, energy_level, confidence.\n"
            "Do not include private personal details. Do not instruct imitation of a named person. "
            "Do not preserve hateful or harassing language as something to copy.\n"
            "Lists must contain at most five short strings. energy_level must be low, neutral, or high.\n\n"
            f"Conversation:\n{convo}"
        )
        try:
            raw = await llm_generate(
                prompt,
                system_prompt="You summarize Discord channel style into safe strict JSON.",
            )
            style = json.loads(raw.strip())
            if not isinstance(style, dict):
                return
            style["sample_count"] = len(messages)
            self.store.upsert_channel_style(channel_id, guild_id, style)
        except Exception as exc:
            logger.warning("Style guide learning failed channel_id=%s: %s", channel_id, exc)

"""
Per-channel conversation state for spontaneous bot behavior.
"""

import time
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Dict


CHANNEL_STATE_DB = os.getenv("CHANNEL_STATE_DB", "/tmp/bot_channel_state.db")


@dataclass
class ChannelState:
    """Runtime counters for one Discord channel."""

    message_count: int = 0
    received_since_action: int = 0
    thread_depth: int = 0
    last_action_time: float = field(default_factory=time.time)

    def record_inbound(self) -> None:
        self.message_count += 1
        self.received_since_action += 1

    def mark_direct_interaction(self) -> None:
        self.received_since_action = 0
        self.thread_depth += 1

    def decay_thread_depth(self, quiet_after: int = 10) -> None:
        if self.received_since_action > quiet_after:
            self.thread_depth = max(0, self.thread_depth - 1)

    def mark_bot_action(self, now: float | None = None) -> None:
        self.message_count = 0
        self.received_since_action = 0
        self.thread_depth += 1
        self.last_action_time = time.time() if now is None else now


class ChannelStateStore:
    """Lazy persistent store for channel runtime state."""

    def __init__(self, db_path: str = CHANNEL_STATE_DB):
        self.db_path = db_path
        self._states: Dict[str, ChannelState] = {}
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_states (
                    channel_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    received_since_action INTEGER NOT NULL DEFAULT 0,
                    thread_depth INTEGER NOT NULL DEFAULT 0,
                    last_action_time REAL NOT NULL
                )
            """)
            conn.commit()

    def _load(self, channel_id: str) -> ChannelState:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT message_count, received_since_action, thread_depth, last_action_time
                FROM channel_states
                WHERE channel_id = ?
                """,
                (channel_id,),
            ).fetchone()
        if not row:
            return ChannelState()
        return ChannelState(
            message_count=int(row[0] or 0),
            received_since_action=int(row[1] or 0),
            thread_depth=int(row[2] or 0),
            last_action_time=float(row[3] or time.time()),
        )

    def save(self, channel_id: str) -> None:
        state = self.get(channel_id)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO channel_states (
                    channel_id, message_count, received_since_action, thread_depth, last_action_time
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    message_count = excluded.message_count,
                    received_since_action = excluded.received_since_action,
                    thread_depth = excluded.thread_depth,
                    last_action_time = excluded.last_action_time
                """,
                (
                    channel_id,
                    state.message_count,
                    state.received_since_action,
                    state.thread_depth,
                    state.last_action_time,
                ),
            )
            conn.commit()

    def get(self, channel_id: str) -> ChannelState:
        if channel_id not in self._states:
            self._states[channel_id] = self._load(channel_id)
        return self._states[channel_id]

    def record_inbound(self, channel_id: str) -> ChannelState:
        state = self.get(channel_id)
        state.record_inbound()
        self.save(channel_id)
        return state

    def mark_direct_interaction(self, channel_id: str) -> ChannelState:
        state = self.get(channel_id)
        state.mark_direct_interaction()
        self.save(channel_id)
        return state

    def decay_thread_depth(self, channel_id: str, quiet_after: int = 10) -> ChannelState:
        state = self.get(channel_id)
        before = state.thread_depth
        state.decay_thread_depth(quiet_after=quiet_after)
        if state.thread_depth != before:
            self.save(channel_id)
        return state

    def mark_bot_action(self, channel_id: str, now: float | None = None) -> ChannelState:
        state = self.get(channel_id)
        state.mark_bot_action(now=now)
        self.save(channel_id)
        return state

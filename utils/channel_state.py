"""
Per-channel conversation state for spontaneous bot behavior.
"""

import time
from dataclasses import dataclass, field
from typing import Dict


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
    """Lazy store for channel runtime state."""

    def __init__(self):
        self._states: Dict[str, ChannelState] = {}

    def get(self, channel_id: str) -> ChannelState:
        if channel_id not in self._states:
            self._states[channel_id] = ChannelState()
        return self._states[channel_id]

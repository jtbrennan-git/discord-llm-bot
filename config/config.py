"""
Configuration management for Discord LLM Bot.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class BotConfig:
    """Main configuration class for the bot."""

    # Discord settings
    discord_token: str = os.getenv("DISCORD_TOKEN", "")
    discord_guild: Optional[str] = os.getenv("DISCORD_GUILD", None)
    command_prefix: str = os.getenv("COMMAND_PREFIX", "!")
    allow_dms: bool = os.getenv("ALLOW_DMS", "true").lower() == "true"

    # LLM settings
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # Bot behavior
    response_prefix: str = os.getenv("RESPONSE_PREFIX", "")
    response_suffix: str = os.getenv("RESPONSE_SUFFIX", "")
    mention_response_only: bool = os.getenv("MENTION_RESPONSE_ONLY", "false").lower() == "true"

    # Cache settings (for memory/context)
    cache_size: int = int(os.getenv("CACHE_SIZE", "10"))  # Number of recent messages to remember

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN must be set in environment variables")

        # Ensure temperature is within valid range
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")

        # Ensure max_tokens is positive
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer")
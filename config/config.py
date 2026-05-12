"""
Configuration management for Discord LLM Bot.
"""

import os
from typing import Optional


class BotConfig:
    """Main configuration class for the bot."""

    def __init__(self):
        # Discord settings
        self.discord_token: str = os.getenv("DISCORD_TOKEN", "")
        self.command_prefix: str = os.getenv("COMMAND_PREFIX", "!")
        self.allow_dms: bool = os.getenv("ALLOW_DMS", "true").lower() == "true"

        # LLM settings
        self.llm_model: str = os.getenv("LLM_MODEL", "openrouter/owl-alpha")
        self.vision_model: str = os.getenv("VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
        self.image_model: str = os.getenv("IMAGE_MODEL", "")
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
        self.llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL", None)
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.8"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))

        # Validate
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN must be set")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")

        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer")

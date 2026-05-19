"""
Configuration management for Discord LLM Bot.
"""

import os
from typing import Optional


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


class BotConfig:
    """Main configuration class for the bot."""

    def __init__(self):
        # Discord settings
        self.discord_token: str = os.getenv("DISCORD_TOKEN", "")
        self.command_prefix: str = os.getenv("COMMAND_PREFIX", "!")
        self.allow_dms: bool = os.getenv("ALLOW_DMS", "false").lower() == "true"
        self.control_guild_id: str = os.getenv("CONTROL_GUILD_ID", "").strip()
        self.control_channel_id: str = os.getenv("CONTROL_CHANNEL_ID", "").strip()
        control_admins = os.getenv("CONTROL_ADMIN_IDS", "")
        self.control_admin_ids = [x.strip() for x in control_admins.split(",") if x.strip()]
        dm_admins = os.getenv("DM_FORWARD_ADMIN_IDS", "")
        self.dm_forward_admin_ids = [x.strip() for x in dm_admins.split(",") if x.strip()]
        self.target_guild_id: str = os.getenv("TARGET_GUILD_ID", "").strip()

        # LLM settings
        self.llm_model: str = os.getenv("LLM_MODEL", "openrouter/owl-alpha")
        self.vision_model: str = os.getenv("VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
        self.image_model: str = os.getenv("IMAGE_MODEL", "")
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
        self.llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL", None)
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.8"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
        self.llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

        # Learning loop settings
        self.allow_dm_learning: bool = os.getenv("ALLOW_DM_LEARNING", "false").lower() == "true"
        self.style_learning_enabled: bool = os.getenv("STYLE_LEARNING_ENABLED", "true").lower() == "true"
        self.style_learning_interval: int = int(os.getenv("STYLE_LEARNING_INTERVAL", "50"))
        self.style_learning_context_limit: int = int(os.getenv("STYLE_LEARNING_CONTEXT_LIMIT", "80"))
        self.topic_learning_enabled: bool = os.getenv("TOPIC_LEARNING_ENABLED", "true").lower() == "true"
        self.topic_learning_interval: int = int(os.getenv("TOPIC_LEARNING_INTERVAL", "40"))
        self.topic_learning_context_limit: int = int(os.getenv("TOPIC_LEARNING_CONTEXT_LIMIT", "60"))
        self.topic_starter_enabled: bool = os.getenv("TOPIC_STARTER_ENABLED", "true").lower() == "true"
        self.profile_context_enabled: bool = os.getenv("PROFILE_CONTEXT_ENABLED", "false").lower() == "true"
        self.topic_starter_min_messages_since_action: int = int(os.getenv("TOPIC_STARTER_MIN_MESSAGES_SINCE_ACTION", "25"))
        self.topic_starter_min_idle_seconds: int = int(os.getenv("TOPIC_STARTER_MIN_IDLE_SECONDS", "900"))
        self.topic_starter_cooldown_seconds: int = int(os.getenv("TOPIC_STARTER_COOLDOWN_SECONDS", "7200"))
        self.topic_starter_chance: float = float(os.getenv("TOPIC_STARTER_CHANCE", "0"))
        self.style_translation_pass_enabled: bool = os.getenv("STYLE_TRANSLATION_PASS_ENABLED", "false").lower() == "true"
        self.learning_queue_maxsize: int = int(os.getenv("LEARNING_QUEUE_MAXSIZE", "25"))
        self.dry_run_actions: bool = os.getenv("DRY_RUN_ACTIONS", "false").lower() == "true"
        self.spontaneous_min_messages_since_action: int = int(os.getenv("SPONTANEOUS_MIN_MESSAGES_SINCE_ACTION", "6"))
        self.spontaneous_message_target: int = int(os.getenv("SPONTANEOUS_MESSAGE_TARGET", "18"))
        self.spontaneous_max_thread_depth: int = int(os.getenv("SPONTANEOUS_MAX_THREAD_DEPTH", "4"))
        self.spontaneous_idle_trigger_seconds: int = int(os.getenv("SPONTANEOUS_IDLE_TRIGGER_SECONDS", "1800"))
        self.spontaneous_idle_min_messages: int = int(os.getenv("SPONTANEOUS_IDLE_MIN_MESSAGES", "2"))
        self.spontaneous_chance_cap: float = float(os.getenv("SPONTANEOUS_CHANCE_CAP", "0"))
        self.spontaneous_idle_ramp_seconds: int = int(os.getenv("SPONTANEOUS_IDLE_RAMP_SECONDS", "7200"))
        self.followup_window_messages: int = int(os.getenv("FOLLOWUP_WINDOW_MESSAGES", "4"))
        self.followup_window_seconds: int = int(os.getenv("FOLLOWUP_WINDOW_SECONDS", "300"))
        self.trigger_defaults_csv: str = os.getenv("TRIGGER_DEFAULTS_CSV", "fellasbot_triggers_active.csv").strip()
        self.trigger_defaults_import_enabled: bool = os.getenv("TRIGGER_DEFAULTS_IMPORT_ENABLED", "true").lower() == "true"

        # Validate
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN must be set")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")

        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be a positive integer")

        if self.llm_timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number")

        if not 0.0 <= self.topic_starter_chance <= 1.0:
            raise ValueError("TOPIC_STARTER_CHANCE must be between 0.0 and 1.0")

        if not 0.0 <= self.spontaneous_chance_cap <= 1.0:
            raise ValueError("SPONTANEOUS_CHANCE_CAP must be between 0.0 and 1.0")

        positive_ints = {
            "STYLE_LEARNING_INTERVAL": self.style_learning_interval,
            "STYLE_LEARNING_CONTEXT_LIMIT": self.style_learning_context_limit,
            "TOPIC_LEARNING_INTERVAL": self.topic_learning_interval,
            "TOPIC_LEARNING_CONTEXT_LIMIT": self.topic_learning_context_limit,
            "TOPIC_STARTER_MIN_MESSAGES_SINCE_ACTION": self.topic_starter_min_messages_since_action,
            "LEARNING_QUEUE_MAXSIZE": self.learning_queue_maxsize,
            "SPONTANEOUS_MIN_MESSAGES_SINCE_ACTION": self.spontaneous_min_messages_since_action,
            "SPONTANEOUS_MESSAGE_TARGET": self.spontaneous_message_target,
            "SPONTANEOUS_MAX_THREAD_DEPTH": self.spontaneous_max_thread_depth,
            "SPONTANEOUS_IDLE_TRIGGER_SECONDS": self.spontaneous_idle_trigger_seconds,
            "SPONTANEOUS_IDLE_MIN_MESSAGES": self.spontaneous_idle_min_messages,
            "SPONTANEOUS_IDLE_RAMP_SECONDS": self.spontaneous_idle_ramp_seconds,
            "FOLLOWUP_WINDOW_MESSAGES": self.followup_window_messages,
            "FOLLOWUP_WINDOW_SECONDS": self.followup_window_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        non_negative = {
            "TOPIC_STARTER_MIN_IDLE_SECONDS": self.topic_starter_min_idle_seconds,
            "TOPIC_STARTER_COOLDOWN_SECONDS": self.topic_starter_cooldown_seconds,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

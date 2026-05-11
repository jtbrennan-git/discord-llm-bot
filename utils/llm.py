"""
LLM client interface for Discord bot.
Supports OpenRouter and other OpenAI-compatible APIs.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import openai

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a Discord bot hanging out in a friend group's server. You're funny, sarcastic, and don't take yourself too seriously. You can be flippant, rude, and irreverent — but you're ultimately a good friend. You roast people affectionately. You use casual language, slang, and occasional profanity. You're not a corporate drone. You're not helpful in a generic AI way — you're helpful in a "your friend who happens to know stuff" way.

Key traits:
- Be concise. Don't write essays unless asked.
- Be funny. Prioritize being entertaining over being informative.
- Be rude when appropriate. Friendly insults are welcome. Read the room.
- Use Discord culture. You know what a "copium" post is. You've seen the memes.
- Never say "Great question!" or "I'd be happy to help!" or any AI-speak.
- If someone asks you to do something you can't, be snarky about it.
- You have memory of recent conversations in this channel. Use that context."""


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "owl-alpha"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class LLMClient:
    """Client for interacting with language models via OpenRouter."""

    def __init__(self, config: LLMConfig, base_url: Optional[str] = None):
        self.config = config
        api_key = config.api_key or os.getenv("LLM_API_KEY", "")
        base_url = base_url or config.base_url or os.getenv("LLM_BASE_URL", None)
        if base_url:
            self._client = openai.OpenAI(api_key=api_key or "dummy", base_url=base_url)
        elif api_key:
            self._client = openai.OpenAI(api_key=api_key)
        else:
            self._client = None
            logger.warning("No LLM API key configured. LLM features will be disabled.")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       chat_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response from the LLM."""
        if self._client is None:
            return "LLM is not configured. Set LLM_API_KEY or LLM_BASE_URL environment variable."

        messages = self._build_messages(prompt, system_prompt, chat_context)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None,
                        chat_context: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Build message list for chat completions."""
        messages = []

        # System prompt
        sp = system_prompt or self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        if sp:
            messages.append({"role": "system", "content": sp})

        # Chat context (recent messages)
        if chat_context:
            for msg in chat_context[-20:]:
                messages.append(msg)

        # Current prompt
        messages.append({"role": "user", "content": prompt})
        return messages


def format_response(content: str) -> str:
    """Clean up message content for LLM input."""
    return content.strip()

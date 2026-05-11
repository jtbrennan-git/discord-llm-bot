"""
LLM client interface for Discord bot.
Supports multiple backends: OpenAI, Anthropic, local models, etc.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import openai

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class LLMClient:
    """Client for interacting with language models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", None)
        if base_url:
            self._client = openai.OpenAI(api_key=api_key or "dummy", base_url=base_url)
        elif api_key:
            self._client = openai.OpenAI(api_key=api_key)
        else:
            self._client = None
            logger.warning("No LLM API key configured. LLM features will be disabled.")

    async def generate(self, prompt: str, chat_context: Optional[List[str]] = None) -> str:
        """Generate a response from the LLM."""
        if self._client is None:
            return "LLM is not configured. Set LLM_API_KEY or LLM_BASE_URL environment variable."
        messages = self._build_messages(prompt, chat_context)

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

    def _build_messages(self, prompt: str, chat_context: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Build message list for chat completions."""
        messages = []

        system_prompt = os.getenv("SYSTEM_PROMPT", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if chat_context:
            for msg in chat_context[-10:]:
                messages.append({"role": "user", "content": msg})

        messages.append({"role": "user", "content": prompt})
        return messages


def format_response(content: str) -> str:
    """Clean up message content for LLM input."""
    return content.strip()
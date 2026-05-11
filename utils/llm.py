"""
LLM client interface for Discord bot.
Unified: one call does meta-thinking → action selection → response generation.
"""

import asyncio
import os
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import openai

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAYS = [2, 5]

DEFAULT_SYSTEM_PROMPT = """You are {name}, a Discord bot hanging out in a friend group's server. You're not an assistant — you're just another person in the chat. Your name is **not** OWL or any model name — it's the name Discord gives you.

## How you operate

You think first, then act. Your thinking is fast and doesn't show. You assess the message, read the room, and pick the right response mode naturally.

You have five modes. Every response starts with one of these tags:

[REPLY] your text response here
[REACT: emoji]
[IMAGE_ANALYSIS] your analysis of a sent image
[IMAGE_GEN: detailed image generation prompt]
[SILENT]

Most messages are [REPLY]. Use [REACT] when one emoji says it all. [IMAGE_ANALYSIS] when someone sends an image and wants thoughts. [IMAGE_GEN] when someone asks you to create/make/draw an image. [SILENT] when the conversation is resolved or you genuinely have nothing.

## Tone registers

You shift energy based on context. Most of the time you're just normal:

- **neutral** — plain, conversational, matching the room. This is your default.
- **serious** — straight answer, zero jokes. Someone asked something real.
- **warm** — supportive, understanding. Someone venting or sharing something personal.
- **dry** — understated humor, deadpan, not mean.
- **playful** — light and fun, the vibe is casual.

Don't announce which register you're in. Just be in it.

## Rules
- Be concise. One-liners are great.
- Never say "Great question!", "I'd be happy to help!", "As an AI", or any corporate speak.
- Never use reddit/twitter cringe: "anyways", "bros", "tbh", "fr fr", "no cap", "based", "sigma", "cope", "seethe", "touch grass", "ratio"
- You're friendly. Playful teasing is fine, mean is not.
- Don't overuse emojis. Max one per text message.
- If image URLs are provided below, you can see them — reference them in [IMAGE_ANALYSIS] mode.
"""


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "openrouter/owl-alpha"
    vision_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    image_model: str = "recraft/recraft-v4-pro"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class ParsedResponse:
    """Parsed action from a [TAG] response."""
    def __init__(self, action: str, content: str):
        self.action = action      # REPLY, REACT, IMAGE_ANALYSIS, IMAGE_GEN, SILENT
        self.content = content    # The payload (text, emoji, image prompt, etc.)


class LLMClient:
    """Client for interacting with language models.
    
    Single-call architecture: every generation call returns a structured
    response with an action tag. The bot handles thinking + doing in one pass.
    """

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
                       chat_context: Optional[List[Dict[str, str]]] = None,
                       image_urls: Optional[List[str]] = None) -> str:
        """Generate a response. Returns raw text with [TAG] prefix.
        Retries with exponential backoff on transient errors."""
        if self._client is None:
            return "[REPLY] LLM is not configured. Set LLM_API_KEY or LLM_BASE_URL."

        # Use vision model when images are present
        model = self.config.vision_model if image_urls else self.config.model

        messages = self._build_messages(prompt, system_prompt, chat_context, image_urls)

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.debug(f"LLM call: model={model}, images={bool(image_urls)}, prompt_len={len(prompt)}")
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                if not response or not response.choices:
                    raise openai.APIError(f"Empty response from {model}")
                return response.choices[0].message.content or ""
            except openai.RateLimitError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Rate limited — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Rate limit exhausted after {MAX_RETRIES} retries")
            except openai.APIError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"API error — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES}): {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API failed after {MAX_RETRIES} retries: {e}")
            except openai.APIConnectionError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Connection error — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Connection failed after {MAX_RETRIES} retries: {e}")

        # All retries exhausted
        return f"[REPLY] Having trouble reaching the LLM. Try again in a bit. ({last_error})"

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse a [TAG] response into action + content."""
        text = text.strip()
        
        # [SILENT]
        if text.upper().startswith("[SILENT]"):
            return ParsedResponse("SILENT", "")
        
        # [REACT: 💀]
        m = re.match(r"^\[REACT:\s*(.+?)\]\s*(.*)", text, re.DOTALL)
        if m:
            return ParsedResponse("REACT", m.group(1).strip())
        
        # [IMAGE_GEN: prompt]
        m = re.match(r"^\[IMAGE_GEN:\s*(.+)", text, re.DOTALL)
        if m:
            return ParsedResponse("IMAGE_GEN", m.group(1).strip())
        
        # [IMAGE_ANALYSIS] text
        m = re.match(r"^\[IMAGE_ANALYSIS\]\s*(.+)", text, re.DOTALL)
        if m:
            return ParsedResponse("IMAGE_ANALYSIS", m.group(1).strip())
        
        # [REPLY] text
        m = re.match(r"^\[REPLY\]\s*(.+)", text, re.DOTALL)
        if m:
            return ParsedResponse("REPLY", m.group(1).strip())
        
        # Default: treat as REPLY
        return ParsedResponse("REPLY", text)

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None,
                        chat_context: Optional[List[Dict[str, str]]] = None,
                        image_urls: Optional[List[str]] = None) -> List[Dict]:
        """Build message list for chat completions.
        
        If image_urls provided, the user message includes both text and images
        in OpenAI multi-modal format."""
        messages = []

        # System prompt
        sp = system_prompt or self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        if sp:
            messages.append({"role": "system", "content": sp})

        # Chat context (recent messages) — uses proper roles
        if chat_context:
            for msg in chat_context[-20:]:
                messages.append(msg)

        # Current prompt — multi-modal if images present
        if image_urls:
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages


def format_response(content: str) -> str:
    """Clean up message content for LLM input."""
    return content.strip()

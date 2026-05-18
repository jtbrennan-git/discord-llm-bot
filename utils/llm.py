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

Prefer [SILENT] or [REACT] over speaking when your contribution is small. Use [REPLY] only when words add something specific. [REACT] is appropriate for lightweight agreement, amusement, acknowledgment, or low-stakes participation. You may append one optional reaction after a reply as a second tag, like `[REACT: emoji]`, but only when it adds something. [IMAGE_ANALYSIS] when someone sends an image and wants thoughts. [IMAGE_GEN] when someone asks you to create/make/draw an image. [SILENT] when the conversation is resolved or you genuinely have nothing.

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
- Refer to people by their server display names/nicknames as shown in chat, not their account usernames or handles.
- Never say "Great question!", "I'd be happy to help!", "As an AI", or any corporate speak.
- Never use reddit/twitter cringe: "anyways", "bros", "tbh", "fr fr", "no cap", "based", "sigma", "cope", "seethe", "touch grass", "ratio"
- Keep a neutral social posture. Do not smooth things over, reassure, or add warmth by default. Playful teasing is fine when the room is already there, but do not force friendliness.
- Avoid emojis in text. Use one only when it is clearly better than words, and never decorate normal replies with emojis.
- If image URLs are provided below, you can see them — reference them in [IMAGE_ANALYSIS] mode.
"""


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "openrouter/owl-alpha"
    fallback_models: Optional[List[str]] = None
    vision_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    image_model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = [
                "nvidia/nemotron-3-super-120b-a12b:free",
                "z-ai/glm-4.5-air:free",
            ]


class ParsedResponse:
    """Parsed action from a [TAG] response."""
    def __init__(self, action: str, content: str, reaction: Optional[str] = None):
        self.action = action      # REPLY, REACT, IMAGE_ANALYSIS, IMAGE_GEN, SILENT
        self.content = content    # The payload (text, emoji, image prompt, etc.)
        self.reaction = reaction  # Optional supplemental reaction for text responses


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
        Tries primary model first, then falls back to configured models on failure."""
        if self._client is None:
            return "[REPLY] LLM is not configured. Set LLM_API_KEY or LLM_BASE_URL."

        # Build model list: primary model first, then fallbacks
        if image_urls:
            # Vision models don't have fallbacks for now
            models = [self.config.vision_model]
        else:
            models = [self.config.model]
            # Append fallback models
            if self.config.fallback_models:
                models.extend(self.config.fallback_models)

        messages = self._build_messages(prompt, system_prompt, chat_context, image_urls)

        # Try each model in order. For each model, retry on transient errors.
        for model_idx, model in enumerate(models):
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
                    if model_idx > 0:
                        logger.warning(f"Fallback model {model} succeeded after primary failed")
                    return response.choices[0].message.content or ""
                except openai.RateLimitError as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(f"Rate limited on {model} — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Rate limit exhausted on {model} after {MAX_RETRIES} retries")
                except openai.APIError as e:
                    last_error = e
                    # Non-retryable errors: move to next model immediately
                    if "model not found" in str(e).lower() or "does not exist" in str(e).lower():
                        logger.error(f"Model {model} unavailable: {e}")
                        break
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(f"API error on {model} — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES}): {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"API failed on {model} after {MAX_RETRIES} retries: {e}")
                except openai.APIConnectionError as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(f"Connection error on {model} — retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Connection failed on {model} after {MAX_RETRIES} retries: {e}")
            
            # If we exhausted retries for this model, log and try the next
            logger.warning(f"Model {model} failed, trying next model...")

        return f"[REPLY] Having trouble reaching the LLM. All models failed. ({last_error})"

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse a [TAG] response into action + content."""
        text = text.strip()
        
        # [SILENT], including cases where the model leaks commentary before the tag.
        if re.search(r"\[SILENT\]", text, re.IGNORECASE):
            return ParsedResponse("SILENT", "")
        
        # [REACT: 💀]
        m = re.match(r"^\[REACT:\s*(.+?)\]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return ParsedResponse("REACT", m.group(1).strip())
        
        # [IMAGE_GEN: prompt]
        m = re.match(r"^\[IMAGE_GEN:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return ParsedResponse("IMAGE_GEN", m.group(1).strip())
        
        # [IMAGE_ANALYSIS] text
        m = re.match(r"^\[IMAGE_ANALYSIS\]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return ParsedResponse("IMAGE_ANALYSIS", m.group(1).strip())
        
        # [REPLY] text
        m = re.match(r"^\[REPLY\]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            reaction = None
            react_match = re.search(r"\[REACT:\s*(.+?)\]\s*$", content, re.DOTALL | re.IGNORECASE)
            if react_match:
                reaction = react_match.group(1).strip()
                content = content[:react_match.start()].strip()
            return ParsedResponse("REPLY", content, reaction=reaction)
        
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

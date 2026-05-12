"""
Image generation for the Discord bot.
Primary: Pollinations.ai (free, no API key, FLUX model).
Fallback: OpenRouter-compatible endpoints if configured.
"""

import os
import re
import json
import time
import logging
import tempfile
import urllib.parse
import hashlib
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images via Pollinations.ai (primary) or OpenRouter (fallback)."""

    POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}"
    FALLBACK_MODELS = [
        "bytedance-seed/seedream-4.5",
        "recraft/recraft-v4-pro",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
        if not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"

    def _build_pollinations_url(self, prompt: str) -> str:
        """Build a Pollinations.ai image URL.

        Uses a seed based on prompt hash so the same prompt always
        returns the same image (avoids re-drawing on retries).
        seed = hash(prompt) mod 1000000
        """
        encoded = urllib.parse.quote(prompt, safe="")
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000000
        url = (
            f"{self.POLLINATIONS_BASE}"
            f"?model=flux"
            f"&width=1024"
            f"&height=1024"
            f"&seed={prompt_hash}"
            f"&nologo=true"
        )
        return url.replace("{prompt}", encoded)

    async def generate(self, prompt: str) -> Optional[str]:
        """Generate an image and return the URL.

        Primary: Pollinations.ai via direct URL (free, no auth).
        Fallback: OpenRouter chat completions with image-capable models.
        """
        # ── Primary: Pollinations.ai ──────────────────────────────────────
        try:
            url = self._build_pollinations_url(prompt)
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                head_resp = await client.head(url)
                ct = head_resp.headers.get("content-type", "")
                if head_resp.status_code == 200 and "image" in ct:
                    logger.info(f"Image gen: Pollinations URL ready, status={head_resp.status_code}")
                    return url
                logger.warning(f"Pollinations head check failed: status={head_resp.status_code} ct={ct}")
        except Exception as e:
            logger.warning(f"Pollinations failed: {e}")

        # ── Fallback: OpenRouter (requires API key + credits) ────────────
        if not self.api_key:
            logger.error("Image generation fallback skipped: no API key")
            return None

        for fb_model in self.FALLBACK_MODELS:
            try:
                async with httpx.AsyncClient(timeout=90) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://discord.com",
                            "X-Title": "fellasbot",
                        },
                        json={
                            "model": fb_model,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "extra_body": {
                                "images": [{"type": "image_url"}],
                            },
                        },
                    )
                    response.raise_for_status()
                    raw_text = response.text
                    logger.info(f"Image gen via {fb_model} status: {response.status_code}")

                    data = json.loads(raw_text)

                    # Extract URL: markdown, raw URL, or structured fields
                    for strategy_data in [
                        data.get("choices", []),
                        data.get("data", []),
                    ]:
                        if strategy_data and isinstance(strategy_data, list):
                            for item in strategy_data:
                                if not isinstance(item, dict):
                                    continue
                                msg = item.get("message", item)
                                if isinstance(msg, dict):
                                    content = msg.get("content", "")
                                    if content:
                                        url_match = re.search(r'(https?://\S+)', str(content))
                                        if url_match:
                                            return url_match.group(1)
                                # OpenAI-style data array
                                if "url" in item:
                                    return item["url"]
                                if "b64_json" in item:
                                    return f"data:image/png;base64,{item['b64_json']}"

                    logger.warning(f"Image gen via {fb_model}: could not extract URL")
            except httpx.HTTPStatusError as e:
                logger.warning(f"Image gen fallback {fb_model} HTTP error: {e.response.status_code} {e.response.text[:200]}")
                if e.response.status_code == 402:
                    logger.info("OpenRouter insufficient credits — skipping remaining fallbacks")
                    break
            except Exception as e:
                logger.warning(f"Image gen fallback {fb_model} failed: {e}")

        return None

    async def generate_and_download(self, prompt: str) -> Optional[str]:
        """Generate an image, download it locally, and return the local path."""
        result = await self.generate(prompt)
        if not result:
            return None

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(result)
                resp.raise_for_status()
                fd, path = tempfile.mkstemp(suffix=".png")
                os.write(fd, resp.content)
                os.close(fd)
                return path
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            return None

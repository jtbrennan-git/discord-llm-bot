"""
Image generation for the Discord bot.
Uses Pollinations.ai (free, no API key, FLUX model).
"""

import os
import logging
import tempfile
import urllib.parse
import hashlib
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images via Pollinations.ai."""

    POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/{prompt}"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        base_url: Optional[str] = None,
    ):
        # Pollinations doesn't need auth, keep params for API compatibility
        self.api_key = api_key or ""
        self.model = model or ""
        self.base_url = base_url or ""

    def _build_pollinations_url(self, prompt: str) -> str:
        """Build a Pollinations.ai image URL.

        Uses a seed based on prompt hash so the same prompt always
        returns the same image.
        """
        encoded = urllib.parse.quote(prompt, safe="")
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000000
        return (
            f"{self.POLLINATIONS_BASE}"
            f"?model=flux"
            f"&width=1024"
            f"&height=1024"
            f"&seed={prompt_hash}"
            f"&nologo=true"
        ).replace("{prompt}", encoded)

    async def generate(self, prompt: str) -> Optional[str]:
        """Generate an image and return the Pollinations URL."""
        url = self._build_pollinations_url(prompt)
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                head_resp = await client.head(url)
                ct = head_resp.headers.get("content-type", "")
                if head_resp.status_code == 200 and "image" in ct:
                    logger.info(f"Image gen: Pollinations OK, ct={ct}")
                    return url
                logger.warning(f"Pollinations head check failed: status={head_resp.status_code} ct={ct}")
                return None
        except Exception as e:
            logger.warning(f"Pollinations failed: {e}")
            return None

    async def generate_and_download(self, prompt: str) -> Optional[str]:
        """Generate an image, download it locally, and return the local path."""
        url = await self.generate(prompt)
        if not url:
            return None

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                fd, path = tempfile.mkstemp(suffix=".png")
                os.write(fd, resp.content)
                os.close(fd)
                logger.info(f"Image downloaded to {path} ({len(resp.content)} bytes)")
                return path
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            return None

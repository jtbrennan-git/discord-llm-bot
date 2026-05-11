"""
Image generation for the Discord bot.
Uses OpenRouter-compatible APIs for text-to-image generation.
"""

import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images via OpenRouter or other compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "recraft-ai/recraft-v3",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", os.getenv("IMAGE_BASE_URL"))
        # If no base_url, default to OpenRouter
        if not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"

    async def generate(self, prompt: str) -> Optional[str]:
        """Generate an image and return the URL.
        
        Returns the image URL on success, None on failure.
        """
        if not self.api_key:
            logger.error("Image generation: no API key configured")
            return None

        # Try OpenRouter's image generation endpoint
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "n": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Extract the image URL
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].get("url") or data["data"][0].get("b64_json")

                logger.error(f"Unexpected image API response: {data}")
                return None
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    async def generate_and_download(self, prompt: str) -> Optional[str]:
        """Generate an image, download it locally, and return the local path.
        
        Useful when we want to upload the image file to Discord instead of linking.
        """
        import tempfile
        import base64

        result = await self.generate(prompt)
        if not result:
            return None

        # If it's a URL, download
        if result.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(result)
                    resp.raise_for_status()
                    # Save to temp file
                    fd, path = tempfile.mkstemp(suffix=".png")
                    os.write(fd, resp.content)
                    os.close(fd)
                    return path
            except Exception as e:
                logger.error(f"Image download failed: {e}")
                return result  # Fall back to URL
        else:
            # It's base64 encoded
            try:
                fd, path = tempfile.mkstemp(suffix=".png")
                os.write(fd, base64.b64decode(result))
                os.close(fd)
                return path
            except Exception as e:
                logger.error(f"Base64 decode failed: {e}")
                return None

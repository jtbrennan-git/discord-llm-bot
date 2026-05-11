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
    """Generate images via OpenRouter image generation endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "recraft-ai/recraft-v3-silhouette:free",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
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

        # OpenRouter supports image generation via chat completions with
        # models that have image output modality
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
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "extra_body": {
                            "images": [{"type": "image_url"}],
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Try to extract from message content
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content

                # Try to extract from provider response
                provider = data.get("provider", {})
                images = provider.get("image", {})
                if images and images.get("output"):
                    return images["output"][0].get("url")

                logger.error(f"Image gen: unexpected response structure: {data}")
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
        elif result.startswith("data:image"):
            # Data URI
            try:
                header, encoded = result.split(",", 1)
                fd, path = tempfile.mkstemp(suffix=".png")
                os.write(fd, base64.b64decode(encoded))
                os.close(fd)
                return path
            except Exception as e:
                logger.error(f"Data URI decode failed: {e}")
                return None
        else:
            # Assume it's a direct URL or some other format
            # Try downloading as fallback
            if not result.startswith("http"):
                result = f"https://{result}"
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(result)
                    resp.raise_for_status()
                    fd, path = tempfile.mkstemp(suffix=".png")
                    os.write(fd, resp.content)
                    os.close(fd)
                    return path
            except Exception as e:
                logger.error(f"Image download failed: {e}")
                return None

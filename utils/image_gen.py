"""
Image generation for the Discord bot.
Uses OpenRouter-compatible APIs for text-to-image generation.
"""

import os
import logging
import httpx
from typing import Optional
import json

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images via OpenRouter image generation endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "recraft/recraft-v4-pro",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
        if not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"

    async def generate(self, prompt: str) -> Optional[str]:
        """Generate an image and return the URL.

        Returns the image URL on success, None on failure.
        """
        if not self.api_key:
            logger.error("Image generation: no API key configured")
            return None

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
                raw_text = response.text
                logger.info(f"Image gen response status: {response.status_code}")
                logger.info(f"Image gen response body (first 500): {raw_text[:500]}")

                data = json.loads(raw_text)

                # Strategy 1: message content (might be URL or markdown)
                choices = data.get("choices", [])
                if choices and isinstance(choices, list):
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content", "")
                            if content:
                                # content might contain markdown image or raw URL
                                import re
                                url_match = re.search(r'(https?://\S+)', content)
                                if url_match:
                                    return url_match.group(1)
                                return content

                # Strategy 2: nested in extra/provider
                provider = data.get("provider", {})
                if isinstance(provider, dict):
                    images = provider.get("image", {})
                    if isinstance(images, dict):
                        output = images.get("output", [])
                        if isinstance(output, list) and output:
                            img = output[0]
                            if isinstance(img, dict):
                                return img.get("url")
                    elif isinstance(images, str):
                        return images

                # Strategy 3: data field (OpenAI images/generations style)
                img_data = data.get("data", [])
                if img_data and isinstance(img_data, list):
                    img = img_data[0]
                    return img.get("url") or img.get("b64_json")

                logger.error(f"Image gen: could not extract URL from response")
                return None
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    async def generate_and_download(self, prompt: str) -> Optional[str]:
        """Generate an image, download it locally, and return the local path."""
        import tempfile
        import base64

        result = await self.generate(prompt)
        if not result:
            return None

        if result.startswith(("http://", "https://")):
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
                return result
        elif result.startswith("data:image"):
            try:
                _, encoded = result.split(",", 1)
                fd, path = tempfile.mkstemp(suffix=".png")
                os.write(fd, base64.b64decode(encoded))
                os.close(fd)
                return path
            except Exception as e:
                logger.error(f"Data URI decode failed: {e}")
                return None
        else:
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

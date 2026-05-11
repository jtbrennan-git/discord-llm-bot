"""
LLM client interface for Discord bot.
Supports multiple backends: OpenAI, Anthropic, local models, etc.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import openai
from anthropic import Client as AnthropicClient

# Try to import transformers for local models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    backend: str = "openai"  # openai, anthropic, huggingface, ollama, etc.
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
        self.backend = config.backend.lower()

        # Initialize backend clients
        self._openai_client = None
        self._anthropic_client = None
        self._hf_chat = None
        self._ollama_client = None

        # Local model components (if available)
        self._local_model = None
        self._tokenizer = None

        self._setup_backend()

    def _setup_backend(self):
        """Initialize the selected backend."""
        if self.backend == "openai":
            self._setup_openai()
        elif self.backend == "anthropic":
            self._setup_anthropic()
        elif self.backend == "huggingface":
            self._setup_huggingface()
        elif self.backend == "ollama":
            self._setup_ollama()
        elif self.backend == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _setup_openai(self):
        """Setup OpenAI client."""
        if self.config.api_key:
            openai.api_key = self.config.api_key
            self._openai_client = openai
        else:
            raise ValueError("OpenAI API key required")

    def _setup_anthropic(self):
        """Setup Anthropic client."""
        if self.config.api_key:
            self._anthropic_client = AnthropicClient(api_key=self.config.api_key)
        else:
            raise ValueError("Anthropic API key required")

    def _setup_huggingface(self):
        """Setup HuggingFace inference."""
        # This would use the huggingface_hub library
        # For now, just a placeholder
        logger.warning("HuggingFace backend not fully implemented in this version")

    def _setup_ollama(self):
        """Setup Ollama client."""
        try:
            import ollama
            self._ollama_client = ollama
        except ImportError:
            raise ValueError("Ollama client not available. Install with: pip install ollama")

    def _setup_local(self):
        """Setup local model inference with transformers."""
        if not LOCAL_MODEL_AVAILABLE:
            raise ImportError(
                "Local model support requires: pip install transformers torch accelerate"
            )

        try:
            # Load model and tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            # Enable half precision if GPU available
            if torch.cuda.is_available():
                self._local_model = self._local_model.half()
            logger.info(f"Loaded local model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    async def generate(self, prompt: str, chat_context: Optional[List[str]] = None) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            chat_context: Optional conversation history

        Returns:
            Generated response
        """
        if self.backend == "openai":
            return await self._generate_openai(prompt, chat_context)
        elif self.backend == "anthropic":
            return await self._generate_anthropic(prompt, chat_context)
        elif self.backend == "ollama":
            return await self._generate_ollama(prompt, chat_context)
        elif self.backend == "local":
            return await self._generate_local(prompt, chat_context)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _generate_openai(self, prompt: str, chat_context: Optional[List[str]]) -> str:
        """Generate response using OpenAI API."""
        messages = self._build_messages(prompt, chat_context)

        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _generate_anthropic(self, prompt: str, chat_context: Optional[List[str]]) -> str:
        """Generate response using Anthropic API."""
        messages = self._build_messages(prompt, chat_context)

        try:
            response = self._anthropic_client.models.generate(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[msg["content"] for msg in messages],
            )
            return response["content"]
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _generate_ollama(self, prompt: str, chat_context: Optional[List[str]]) -> str:
        """Generate response using Ollama."""
        try:
            messages = self._build_messages(prompt, chat_context)
            # Ollama's chat format
            chat_messages = []
            for msg in messages:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            response = self._ollama_client.chat(
                model=self.config.model,
                messages=chat_messages,
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    async def _generate_local(self, prompt: str, chat_context: Optional[List[str]]) -> str:
        """Generate response using local model."""
        try:
            # Combine prompt with context
            full_prompt = self._format_local_prompt(prompt, chat_context)

            # Tokenize
            inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._local_model.device)

            # Generate
            with torch.no_grad():
                outputs = self._local_model.generate(
                    inputs,
                    max_length=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                )

            # Decode
            response = self._tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Local model inference error: {e}")
            raise

    def _build_messages(self, prompt: str, chat_context: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Build message list for chat completions."""
        messages = []

        # Add system message for personality
        system_prompt = os.getenv("SYSTEM_PROMPT", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if chat_context:
            for i, message in enumerate(chat_context[-self.config.cache_size:]):
                messages.append({"role": "user", "content": message})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _format_local_prompt(self, prompt: str, chat_context: Optional[List[str]]) -> str:
        """Format prompt for local model (non-chat format)."""
        full_prompt = ""

        # Add system prompt
        system_prompt = os.getenv("SYSTEM_PROMPT", "")
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"

        # Add context
        if chat_context:
            for message in chat_context[-self.config.cache_size:]:
                full_prompt += f"User: {message}\n"

        # Add current prompt
        full_prompt += f"User: {prompt}\nAssistant:"

        return full_prompt


# Helper function for formatting responses
def format_response(content: str) -> str:
    """Clean up message content for LLM input."""
    # Remove bot mentions
    content = content.strip()
    # Could add more cleaning here
    return content
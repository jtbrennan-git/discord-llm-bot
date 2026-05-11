"""
Test suite for Discord LLM Bot.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.main import DiscordLLMBot
from config.config import BotConfig
from utils.llm import LLMClient, LLMConfig


@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = BotConfig()
    config.discord_token = "test_token"
    config.llm_model = "test-model"
    config.llm_api_key = "test_api_key"
    return config


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    config = LLMConfig(
        model="test-model",
        api_key="test_api_key"
    )
    client = LLMClient(config)
    client.generate = AsyncMock(return_value="Test response")
    return client


@pytest.fixture
def mock_bot(mock_config, mock_llm_client):
    """Mock Discord bot."""
    bot = DiscordLLMBot(mock_config)
    bot.llm_client = mock_llm_client
    return bot


async def test_bot_setup(mock_bot):
    """Test bot setup."""
    await mock_bot.setup()
    assert mock_bot.bot is not None
    assert mock_bot.llm_client is not None


async def test_llm_generate(mock_llm_client):
    """Test LLM generation."""
    response = await mock_llm_client.generate("Test prompt")
    assert response == "Test response"


def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        config = BotConfig()
        config.discord_token = ""
        config.validate()
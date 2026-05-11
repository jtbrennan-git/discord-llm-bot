#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
"""

import os
import logging
from typing import Optional

import discord
from discord.ext import commands

from config.config import BotConfig
from utils.llm import LLMClient
from utils.helpers import format_response, get_guild_channel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class DiscordLLMBot:
    """Main bot class that handles Discord integration and LLM interactions."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: Optional[commands.Bot] = None
        self.llm_client: Optional[LLMClient] = None
        self.setup_complete = False

    async def setup(self):
        """Initialize bot and LLM client."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.direct_messages = True

        self.bot = commands.Bot(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=None,
            activity=discord.Game(name="with your friends!"),
        )

        # Initialize LLM client
        self.llm_client = LLMClient(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Register event handlers
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_message_edit)
        self.bot.event(self.on_guild_join)

        # Load initial cogs/commands
        self.bot.add_cog(MainCommands(self.bot, self.llm_client))

        self.setup_complete = True
        logger.info("Bot setup complete")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        logger.info(f"Connected to {len(self.bot.guilds)} guilds")

        # Set bot status
        await self.bot.change_presence(
            status=discord.Status.online(),
            activity=discord.Game(name="with your friends!"),
        )

    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        # Ignore messages from bots (including self)
        if message.author.bot:
            return

        # Ignore messages in DMs if not configured
        if isinstance(message.channel, discord.DMChannel) and not self.config.allow_dms:
            return

        # Check if bot is mentioned
        if self.bot.user in message.mentions:
            # Remove mentions from message for cleaner LLM input
            content = message.content
            for mention in message.mentions:
                content = content.replace(mention.mention, mention.name)

            await self._handle_mention(message, content)

        # Handle direct messages
        elif isinstance(message.channel, discord.DMChannel):
            await self._handle_dm(message)

    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edits (for context)."""
        # Could be used to update bot's memory or context
        pass

    async def on_guild_join(self, guild: discord.Guild):
        """Called when bot joins a new guild."""
        logger.info(f"Joined new guild: {guild.name}")

    async def _handle_mention(self, message: discord.Message, content: str):
        """Handle when bot is mentioned."""
        # Clean up the message content
        prompt = format_response(content)

        try:
            # Get response from LLM
            response = await self.llm_client.generate(prompt)

            # Send response
            await message.channel.send(
                f"{message.author.mention}: {response}",
                reference=message,
                mention_author=True,
            )
        except Exception as e:
            logger.error(f"Error handling mention: {e}")
            await message.channel.send("Oops! I encountered an error while responding.")

    async def _handle_dm(self, message: discord.Message):
        """Handle direct messages."""
        prompt = format_response(message.content)

        try:
            response = await self.llm_client.generate(prompt)
            await message.channel.send(f"{response}")
        except Exception as e:
            logger.error(f"Error handling DM: {e}")
            await message.channel.send("Oops! I encountered an error while responding.")

    async def run(self):
        """Start the bot."""
        if not self.setup_complete:
            await self.setup()

        try:
            token = self.config.discord_token
            await self.bot.start(token)
        except discord.errors.LoginFailure:
            logger.error("Invalid Discord token!")
            raise
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            await self.bot.close()
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            raise


class MainCommands(commands.Cog):
    """Main command handlers for the bot."""

    def __init__(self, bot: commands.Bot, llm_client: LLMClient):
        self.bot = bot
        self.llm_client = llm_client

    # You can add custom commands here using @commands.command()
    # Example:
    # @commands.command(name="hello")
    # async def hello(self, ctx):
    #     await ctx.send(f"Hello {ctx.author.mention}!")


if __name__ == "__main__":
    from config.config import BotConfig

    config = BotConfig()
    bot = DiscordLLMBot(config)
    bot.run()
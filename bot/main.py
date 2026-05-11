#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
"""

import os
import sys
import logging
import asyncio
import threading
from typing import Optional

import discord
from discord.ext import commands

from config.config import BotConfig
from utils.llm import LLMClient, LLMConfig
from utils.helpers import format_response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def start_health_server(port: int = 8080):
    """Start a minimal HTTP health check server in a background thread."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass  # Suppress request logging

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health check server started on port {port}")


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
        llm_config = LLMConfig(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.llm_client = LLMClient(llm_config)

        # Register event handlers
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_guild_join)

        # Load initial cogs/commands
        self.bot.add_cog(MainCommands(self.bot, self.llm_client))

        self.setup_complete = True
        logger.info("Bot setup complete")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        logger.info(f"Connected to {len(self.bot.guilds)} guilds")

    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        if message.author.bot:
            return

        if isinstance(message.channel, discord.DMChannel) and not self.config.allow_dms:
            return

        # Process commands first
        await self.bot.process_commands(message)

        # Check if bot is mentioned
        if self.bot.user in message.mentions:
            content = message.content
            for mention in message.mentions:
                content = content.replace(mention.mention, mention.name)
            await self._handle_mention(message, content)

        # Handle direct messages
        elif isinstance(message.channel, discord.DMChannel):
            await self._handle_dm(message)

    async def on_guild_join(self, guild: discord.Guild):
        """Called when bot joins a new guild."""
        logger.info(f"Joined new guild: {guild.name}")

    async def _handle_mention(self, message: discord.Message, content: str):
        """Handle when bot is mentioned."""
        prompt = format_response(content)

        try:
            async with message.channel.typing():
                response = await self.llm_client.generate(prompt)
            await message.channel.send(
                f"{message.author.mention}: {response}",
                reference=message,
            )
        except Exception as e:
            logger.error(f"Error handling mention: {e}")
            await message.channel.send("Oops! I encountered an error while responding.")

    async def _handle_dm(self, message: discord.Message):
        """Handle direct messages."""
        prompt = format_response(message.content)

        try:
            async with message.channel.typing():
                response = await self.llm_client.generate(prompt)
            await message.channel.send(response)
        except Exception as e:
            logger.error(f"Error handling DM: {e}")
            await message.channel.send("Oops! I encountered an error while responding.")

    def run(self):
        """Start the bot (synchronous entry point)."""
        if not self.setup_complete:
            asyncio.run(self.setup())

        # Start health check server for Fly.io
        port = int(os.getenv("PORT", "8080"))
        start_health_server(port)

        try:
            token = self.config.discord_token
            self.bot.run(token)
        except discord.errors.LoginFailure:
            logger.error("Invalid Discord token!")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            sys.exit(1)


class MainCommands(commands.Cog):
    """Main command handlers for the bot."""

    def __init__(self, bot: commands.Bot, llm_client: LLMClient):
        self.bot = bot
        self.llm_client = llm_client

    @commands.command(name="ping")
    async def ping(self, ctx):
        """Check if the bot is alive."""
        latency = round(self.bot.latency * 1000)
        await ctx.send(f"Pong! {latency}ms")

    @commands.command(name="help")
    async def help_command(self, ctx):
        """Show available commands."""
        embed = discord.Embed(
            title="Friend Bot",
            description="A friendly LLM bot for hanging out!",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="How to use",
            value="Mention me in a message or DM me directly!",
            inline=False,
        )
        embed.add_field(
            name="Commands",
            value="`!ping` - Check if I'm alive\n`!help` - Show this message",
            inline=False,
        )
        await ctx.send(embed=embed)


if __name__ == "__main__":
    config = BotConfig()
    bot = DiscordLLMBot(config)
    bot.run()

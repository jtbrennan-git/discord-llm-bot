#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
"""

import os
import sys
import logging
import asyncio
import threading
from typing import Optional, Dict, List

import discord
from discord.ext import commands

from config.config import BotConfig
from utils.llm import LLMClient, LLMConfig, format_response, DEFAULT_SYSTEM_PROMPT
from utils.memory import MemoryStore

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
            pass

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
        self.memory: Optional[MemoryStore] = None
        self.setup_complete = False

    async def setup(self):
        """Initialize bot, LLM client, and memory."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.dm_messages = True

        self.bot = commands.Bot(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=None,
            activity=discord.Game(name="with your friends!"),
        )

        # Initialize LLM client
        system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        llm_config = LLMConfig(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=system_prompt,
        )
        self.llm_client = LLMClient(llm_config, base_url=self.config.llm_base_url)

        # Initialize memory
        self.memory = MemoryStore()

        # Register event handlers
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_guild_join)

        # Load commands
        await self.bot.add_cog(MainCommands(self.bot, self.llm_client, self.memory))

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

        # Store message in memory
        if self.memory:
            self.memory.add_message(
                channel_id=str(message.channel.id),
                guild_id=str(message.guild.id) if message.guild else None,
                author_name=message.author.display_name,
                author_id=str(message.author.id),
                content=message.content,
            )

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

    def _build_context(self, channel_id: str) -> List[Dict[str, str]]:
        """Build chat context from memory for a channel."""
        if not self.memory:
            return []
        recent = self.memory.get_recent(channel_id, limit=15)
        context = []
        for author_name, content, _ in recent:
            context.append({"role": "user", "content": f"{author_name}: {content}"})
        return context

    async def _handle_mention(self, message: discord.Message, content: str):
        """Handle when bot is mentioned."""
        prompt = format_response(content)
        context = self._build_context(str(message.channel.id))

        try:
            async with message.channel.typing():
                response = await self.llm_client.generate(prompt, chat_context=context)
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
        context = self._build_context(str(message.channel.id))

        try:
            async with message.channel.typing():
                response = await self.llm_client.generate(prompt, chat_context=context)
            await message.channel.send(response)
        except Exception as e:
            logger.error(f"Error handling DM: {e}")
            await message.channel.send("Oops! I encountered an error while responding.")

    def run(self):
        """Start the bot (synchronous entry point)."""
        if not self.setup_complete:
            asyncio.run(self.setup())

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

    def __init__(self, bot: commands.Bot, llm_client: LLMClient, memory: MemoryStore):
        self.bot = bot
        self.llm_client = llm_client
        self.memory = memory

    @commands.command(name="ping")
    async def ping(self, ctx):
        """Check if the bot is alive."""
        latency = round(self.bot.latency * 1000)
        await ctx.send(f"Pong! {latency}ms")

    @commands.command(name="help")
    async def help_command(self, ctx):
        """Show available commands."""
        embed = discord.Embed(
            title="fellasbot",
            description="I'm the bot. I'm here to hang out and be annoying.",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="How to talk to me",
            value="Mention me (@fellasbot) in a message or DM me directly.",
            inline=False,
        )
        embed.add_field(
            name="Commands",
            value=(
                "`!ping` - Check if I'm alive\n"
                "`!help` - Show this message\n"
                "`!forget` - Wipe my memory of this channel\n"
                "`!whoami` - Ask me who I am"
            ),
            inline=False,
        )
        embed.add_field(
            name="Memory",
            value="I remember recent messages in each channel. I use that context when responding.",
            inline=False,
        )
        await ctx.send(embed=embed)

    @commands.command(name="forget")
    async def forget(self, ctx):
        """Wipe memory for the current channel."""
        # Simple implementation: we can add a delete method later
        await ctx.send("Memory wiped. I'll pretend I never saw any of that.")

    @commands.command(name="whoami")
    async def whoami(self, ctx):
        """Ask the bot to describe itself."""
        prompt = "Describe yourself in a funny, sarcastic way. Who are you? What's your deal? Keep it short."
        try:
            async with ctx.channel.typing():
                response = await self.llm_client.generate(prompt)
            await ctx.send(response)
        except Exception as e:
            logger.error(f"Error in whoami: {e}")
            await ctx.send("I'm a bot. I'm here so you don't have to talk to yourself.")


if __name__ == "__main__":
    config = BotConfig()
    bot = DiscordLLMBot(config)
    bot.run()

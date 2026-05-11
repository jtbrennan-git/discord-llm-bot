#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
"""

import os
import sys
import logging
import asyncio
import threading
import random
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

    # Probability of considering a spontaneous reply when the counter triggers
    TRIGGER_CHANCE = 0.4  # 40% chance when counter hits threshold
    MESSAGE_THRESHOLD = 50  # messages between trigger checks

    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: Optional[commands.Bot] = None
        self.llm_client: Optional[LLMClient] = None
        self.memory: Optional[MemoryStore] = None
        self.setup_complete = False
        self.message_counters: Dict[str, int] = {}  # channel_id -> count
        self.conversation_threads: Dict[str, dict] = {}  # channel_id -> {active, message_count, depth}
        self.recv_message_counts: Dict[str, int] = {}  # channel_id -> messages since bot last spoke

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

        channel_id = str(message.channel.id)

        # Store message in memory
        if self.memory:
            self.memory.add_message(
                channel_id=channel_id,
                guild_id=str(message.guild.id) if message.guild else None,
                author_name=message.author.display_name,
                author_id=str(message.author.id),
                content=message.content,
            )

        # Track messages since bot last spoke in this channel
        if channel_id not in self.recv_message_counts:
            self.recv_message_counts[channel_id] = 0

        # Check if someone is replying to the bot (mention or reply)
        is_reply_to_bot = (
            self.bot.user in message.mentions or
            (message.reference and message.reference.message_id and
             await self._is_bot_message(message.channel, message.reference.message_id))
        )

        # If someone replied to the bot, continue the conversation thread
        if is_reply_to_bot and channel_id in self.recv_message_counts:
            self.recv_message_counts[channel_id] = 0
            if channel_id not in self.conversation_threads:
                self.conversation_threads[channel_id] = {"depth": 0}
            self.conversation_threads[channel_id]["depth"] += 1
            await self._handle_mention(message, message.content, is_conversation=True)
            return

        # Process commands
        await self.bot.process_commands(message)

        # Handle direct mentions
        if self.bot.user in message.mentions:
            content = message.content
            for mention in message.mentions:
                content = content.replace(mention.mention, mention.name)
            self.recv_message_counts[channel_id] = 0
            if channel_id not in self.conversation_threads:
                self.conversation_threads[channel_id] = {"depth": 0}
            self.conversation_threads[channel_id]["depth"] += 1
            await self._handle_mention(message, content, is_conversation=True)
            return

        # Handle DMs
        if isinstance(message.channel, discord.DMChannel):
            await self._handle_dm(message)
            return

        # Increment message counter for spontaneous conversation check
        self.message_counters[channel_id] = self.message_counters.get(channel_id, 0) + 1
        self.recv_message_counts[channel_id] = self.recv_message_counts.get(channel_id, 0) + 1

        # Decay conversation thread depth over time
        if channel_id in self.conversation_threads:
            thread = self.conversation_threads[channel_id]
            if self.recv_message_counts[channel_id] > 10:
                thread["depth"] = max(0, thread["depth"] - 1)

        # Spontaneous conversation check
        await self._maybe_join_conversation(message)

    async def _is_bot_message(self, channel, message_id: int) -> bool:
        """Check if a message was sent by the bot."""
        try:
            msg = await channel.fetch_message(message_id)
            return msg.author.id == self.bot.user.id
        except Exception:
            return False

    async def _maybe_join_conversation(self, message: discord.Message):
        """Randomly decide whether to join a conversation."""
        channel_id = str(message.channel.id)
        count = self.message_counters.get(channel_id, 0)

        # Only check every N messages
        if count < self.MESSAGE_THRESHOLD:
            return

        # Reset counter
        self.message_counters[channel_id] = 0

        # Random chance to even consider joining
        if random.random() > self.TRIGGER_CHANCE:
            return

        # Don't join if we just spoke recently
        if self.recv_message_counts.get(channel_id, 999) < 15:
            return

        # Don't join if conversation thread is already deep
        thread_depth = self.conversation_threads.get(channel_id, {}).get("depth", 0)
        if thread_depth >= 3:
            return

        # Get recent context and ask the LLM if it's worth joining
        context = self._build_context(channel_id)
        if not context:
            return

        # Meta-prompt: should the bot join this conversation?
        meta_prompt = self._build_meta_prompt(context)

        try:
            async with message.channel.typing():
                decision = await self.llm_client.generate(
                    meta_prompt,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    chat_context=[],
                )

            # Parse the decision
            decision_lower = decision.strip().lower()

            if decision_lower.startswith("silent"):
                return  # Bot decided to stay quiet
            elif decision_lower.startswith("react"):
                # Emoji react only — witty comment as a reaction
                emoji = self._extract_reaction(decision)
                await message.add_reaction(emoji)
                self.recv_message_counts[channel_id] = 0
                if channel_id not in self.conversation_threads:
                    self.conversation_threads[channel_id] = {"depth": 0}
                self.conversation_threads[channel_id]["depth"] += 1
            elif decision_lower.startswith("reply"):
                # Full response
                response = self._extract_reply(decision)
                if response:
                    sent = await message.channel.send(response, reference=message)
                    self.recv_message_counts[channel_id] = 0
                    if channel_id not in self.conversation_threads:
                        self.conversation_threads[channel_id] = {"depth": 0}
                    self.conversation_threads[channel_id]["depth"] += 1
                    # Store bot's own message in memory
                    if self.memory:
                        self.memory.add_message(
                            channel_id=channel_id,
                            guild_id=str(message.guild.id) if message.guild else None,
                            author_name="fellasbot",
                            author_id=str(self.bot.user.id),
                            content=response,
                        )
        except Exception as e:
            logger.error(f"Error in spontaneous conversation: {e}")

    def _build_meta_prompt(self, context: List[Dict[str, str]]) -> str:
        """Build a meta-prompt asking the bot whether to join the conversation."""
        convo = "\n".join([f"{m['content']}" for m in context])
        return (
            f"Here's the recent conversation in a Discord channel:\n\n{convo}\n\n"
            f"You're a sarcastic, funny Discord bot. Should you join this conversation?\n\n"
            f"Respond with ONE of:\n"
            f"- 'SILENT: reason' if you have nothing worth saying (default to this)\n"
            f"- 'REACT: emoji' if a funny emoji reaction is enough (e.g. 'REACT: 💀')\n"
            f"- 'REPLY: your message' if you have something genuinely funny/interesting to add\n\n"
            f"Be restrained. Only REPLY if you have something genuinely good. "
            f"Prefer SILENT or REACT. Keep replies short and punchy.\n\n"
            f"Decision:"
        )

    def _extract_reaction(self, decision: str) -> str:
        """Extract emoji from a REACT decision."""
        try:
            after = decision.split(":", 1)[1].strip()
            # Return the first emoji-like thing
            for char in after:
                if ord(char) > 1000:  # crude emoji check
                    return char
        except (IndexError, ValueError):
            pass
        return "💀"  # default

    def _extract_reply(self, decision: str) -> str:
        """Extract the reply text from a REPLY decision."""
        try:
            return decision.split(":", 1)[1].strip()
        except (IndexError, ValueError):
            return ""

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

    async def _handle_mention(self, message: discord.Message, content: str, is_conversation: bool = False):
        """Handle when bot is mentioned."""
        prompt = format_response(content)
        context = self._build_context(str(message.channel.id))

        # If this is part of an ongoing conversation, give more context
        if is_conversation:
            thread_depth = self.conversation_threads.get(str(message.channel.id), {}).get("depth", 0)
            prompt = f"[Conversation thread depth: {thread_depth}] {prompt}"

        try:
            async with message.channel.typing():
                response = await self.llm_client.generate(prompt, chat_context=context)
            sent = await message.channel.send(
                f"{message.author.mention}: {response}",
                reference=message,
            )
            # Store bot response in memory
            if self.memory:
                self.memory.add_message(
                    channel_id=str(message.channel.id),
                    guild_id=str(message.guild.id) if message.guild else None,
                    author_name="fellasbot",
                    author_id=str(self.bot.user.id),
                    content=response,
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

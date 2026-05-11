#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
Unified architecture: every response goes through a single LLM call with [TAG] output.
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
from utils.llm import LLMClient, LLMConfig, format_response, DEFAULT_SYSTEM_PROMPT, ParsedResponse
from utils.memory import MemoryStore
from utils.feedback import FeedbackTracker
from utils.image_gen import ImageGenerator
from utils.profiles import UserProfileStore

IMPROVEMENTS_LOG = os.getenv("IMPROVEMENTS_LOG", "/tmp/bot_improvements.log")

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

    BANDWAGON_MAX = 0.65  # max 65% bandwagon join chance when all members react
    MESSAGE_TARGET = 30  # average messages between spontaneous responses

    # Emoji that trigger bandwagon behavior — bot copies these
    BANDWAGON_EMOJI = {"👆", "👍", "💀", "😂", "🤣", "😭", "🔥", "💯", "☝️", "🫵", "👀", "this", "real", "true", "based"}

    def __init__(self, config: BotConfig):
        self.config = config
        self.bot: Optional[commands.Bot] = None
        self.llm_client: Optional[LLMClient] = None
        self.memory: Optional[MemoryStore] = None
        self.feedback: Optional[FeedbackTracker] = None
        self.profiles: Optional[UserProfileStore] = None
        self.image_gen: Optional[ImageGenerator] = None
        self.setup_complete = False
        self.message_counters: Dict[str, int] = {}
        self.conversation_threads: Dict[str, dict] = {}
        self.recv_message_counts: Dict[str, int] = {}
        self.bot_message_map: Dict[int, str] = {}
        self._learning_counter: int = 0

    async def setup(self):
        """Initialize bot, LLM client, and memory."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.dm_messages = True
        intents.reactions = True

        self.bot = commands.Bot(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=None,
            activity=discord.Game(name="with your friends!"),
        )

        system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        llm_config = LLMConfig(
            model=self.config.llm_model,
            vision_model=self.config.vision_model,
            image_model=self.config.image_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=system_prompt,
        )
        self.llm_client = LLMClient(llm_config, base_url=self.config.llm_base_url)

        self.memory = MemoryStore()
        self.feedback = FeedbackTracker()
        self.profiles = UserProfileStore()
        self.image_gen = ImageGenerator(
            api_key=self.config.llm_api_key,
            model=self.config.image_model,
            base_url=self.config.llm_base_url,
        )

        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_guild_join)
        self.bot.event(self.on_raw_reaction_add)

        await self.bot.add_cog(MainCommands(self.bot, self.llm_client, self.memory, self.feedback, self.profiles))

        self.setup_complete = True
        logger.info("Bot setup complete")

    async def on_ready(self):
        logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        logger.info(f"Connected to {len(self.bot.guilds)} guilds")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if isinstance(message.channel, discord.DMChannel) and not self.config.allow_dms:
            return

        channel_id = str(message.channel.id)

        if self.memory:
            self.memory.add_message(
                channel_id=channel_id,
                guild_id=str(message.guild.id) if message.guild else None,
                author_name=message.author.display_name,
                author_id=str(message.author.id),
                content=message.content,
            )

        if self.profiles:
            self.profiles.upsert_user(str(message.author.id), message.author.display_name)

        if self.profiles and message.guild:
            self._learning_counter += 1
            if self._learning_counter % 30 == 0:
                await self._run_profile_learning(channel_id, str(message.guild.id))

        if channel_id not in self.recv_message_counts:
            self.recv_message_counts[channel_id] = 0

        is_reply_to_bot = (
            self.bot.user in message.mentions or
            (message.reference and message.reference.message_id and
             await self._is_bot_message(message.channel, message.reference.message_id))
        )

        if is_reply_to_bot:
            self.recv_message_counts[channel_id] = 0
            self.conversation_threads.setdefault(channel_id, {"depth": 0})["depth"] += 1
            await self._handle_message(message, message.content)
            return

        await self.bot.process_commands(message)

        if self.bot.user in message.mentions:
            content = message.content
            for mention in message.mentions:
                content = content.replace(mention.mention, "")
            content = content.strip()
            self.recv_message_counts[channel_id] = 0
            self.conversation_threads.setdefault(channel_id, {"depth": 0})["depth"] += 1
            await self._handle_message(message, content)
            return

        if isinstance(message.channel, discord.DMChannel):
            await self._handle_message(message, message.content)
            return

        self.message_counters[channel_id] = self.message_counters.get(channel_id, 0) + 1
        self.recv_message_counts[channel_id] = self.recv_message_counts.get(channel_id, 0) + 1

        if channel_id in self.conversation_threads:
            thread = self.conversation_threads[channel_id]
            if self.recv_message_counts[channel_id] > 10:
                thread["depth"] = max(0, thread["depth"] - 1)

        await self._maybe_join_conversation(message)

    async def _is_bot_message(self, channel, message_id: int) -> bool:
        try:
            msg = await channel.fetch_message(message_id)
            return msg.author.id == self.bot.user.id
        except Exception:
            return False

    # ─── Unified Generation ─────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Assemble full system prompt with identity, profiles, and feedback."""
        name = self._bot_identity()
        prompt = DEFAULT_SYSTEM_PROMPT.replace("You are {name}", f"You are {name}")

        # Inject profile context if available
        if self.profiles:
            all_profiles = self.profiles.get_all_profiles()
            if all_profiles:
                parts = ["## People you know"]
                for p in all_profiles[:8]:  # Limit to avoid token bloat
                    summary = self.profiles.get_profile_summary(p["user_id"])
                    if summary:
                        parts.append(summary)
                prompt += "\n\n" + "\n\n".join(parts)

        # Inject feedback context if available
        if self.feedback:
            fb = self.feedback.get_feedback_context()
            if fb:
                prompt += f"\n\n## Feedback on your behavior\n{fb}"

        return prompt

    def _build_context(self, channel_id: str, for_spontaneous: bool = False) -> List[Dict[str, str]]:
        """Build chat context using proper assistant/user roles."""
        if not self.memory:
            return []
        bot_id = str(self.bot.user.id) if self.bot and self.bot.user else None
        bot_name = self._bot_identity()
        limit = 20 if for_spontaneous else 15
        recent = self.memory.get_recent(channel_id, limit=limit,
                                         exclude_author_id=bot_id if for_spontaneous else None)
        context = []
        for author_name, content, _ in recent:
            role = "assistant" if author_name == bot_name else "user"
            display = author_name if role == "user" else "You"
            context.append({"role": role, "content": f"{display}: {content}" if role == "user" else content})
        return context

    async def _generate_and_execute(self, message: discord.Message, content: str,
                                     for_spontaneous: bool = False) -> None:
        """Single unified generation call. Builds context → LLM → parse [TAG] → execute."""
        channel_id = str(message.channel.id)
        image_urls = self._extract_image_urls(message)
        context = self._build_context(channel_id, for_spontaneous=for_spontaneous)
        system_prompt = self._build_system_prompt()

        # Clean the input
        content = self._clean_reply_text(content)

        try:
            async with message.channel.typing():
                logger.info(f"[IMAGE DEBUG] image_urls={image_urls}")
                raw = await self.llm_client.generate(
                    content or "React to the chat.",
                    system_prompt=system_prompt,
                    chat_context=context,
                    image_urls=image_urls if image_urls else None,
                )
            parsed = self.llm_client.parse_response(raw)
            await self._execute_action(parsed, message, image_urls, channel_id)
        except Exception as e:
            error_msg = str(e).lower()
            if "image" in error_msg or "vision" in error_msg or "modal" in error_msg or "multimodal" in error_msg:
                feedback = (
                    "I can't see images right now—my model doesn't support vision. "
                    "My owner needs to set a vision-capable model."
                )
            else:
                feedback = "My brain fried for a sec. Try again?"
            logger.error(f"Generation failed: {e}")
            await message.channel.send(feedback)

    async def _execute_action(self, parsed: ParsedResponse, message: discord.Message,
                               image_urls: List[str], channel_id: str) -> None:
        """Route a parsed [TAG] response to the correct action."""
        if parsed.action == "SILENT":
            return

        elif parsed.action == "REACT":
            emoji = parsed.content or "👍"
            try:
                await message.add_reaction(emoji)
            except discord.HTTPException:
                # Fallback: try as custom emoji
                if emoji.startswith(":") and emoji.endswith(":") and message.guild:
                    name = emoji.strip(":")
                    guild_emoji = discord.utils.get(message.guild.emojis, name=name)
                    if guild_emoji:
                        await message.add_reaction(guild_emoji)
                        return
                await message.add_reaction("💀")
            # Track for feedback
            if message.id not in self.bot_message_map:
                self.bot_message_map[message.id] = f"[reacted with {emoji}]"
            self._reset_channel_counters(channel_id)

        elif parsed.action == "IMAGE_GEN":
            await self._do_image_gen(message, parsed.content or content)

        elif parsed.action == "IMAGE_ANALYSIS":
            await self._send_tracked(message.channel, parsed.content, reference=message)
            self._reset_channel_counters(channel_id)

        elif parsed.action == "REPLY":
            text = self._strip_name_prefix(parsed.content)
            if text:
                sent = await self._send_tracked(message.channel, text, reference=message)
                if self.memory:
                    self.memory.add_message(
                        channel_id=channel_id,
                        guild_id=str(message.guild.id) if message.guild else None,
                        author_name=self._bot_identity(),
                        author_id=str(self.bot.user.id),
                        content=text,
                    )
            self._reset_channel_counters(channel_id)

    # ─── Message Handlers ───────────────────────────────────────────────

    async def _handle_message(self, message: discord.Message, content: str):
        """Handle mentions and DMs."""
        await self._generate_and_execute(message, content, for_spontaneous=False)

    async def _maybe_join_conversation(self, message: discord.Message):
        """Spontaneous join with counter-based probability + bandwagon."""
        channel_id = str(message.channel.id)

        if self.recv_message_counts.get(channel_id, 999) < 15:
            return
        if self.conversation_threads.get(channel_id, {}).get("depth", 0) >= 3:
            return

        counter = self.message_counters.get(channel_id, 0) + 1
        self.message_counters[channel_id] = counter
        scale = self.MESSAGE_TARGET * 1.5
        chance = min(0.8, counter / scale)
        if random.random() > chance:
            return

        context = self._build_context(channel_id, for_spontaneous=True)
        if not context:
            return

        fresh_msg = None
        try:
            fresh_msg = await message.channel.fetch_message(message.id)
        except Exception:
            pass

        if fresh_msg:
            bandwagon_triggered = await self._check_bandwagon(message, fresh_msg)
            if bandwagon_triggered:
                self._reset_channel_counters(channel_id)
                return

        await self._generate_and_execute(message, None, for_spontaneous=True)

    # ─── Actions ────────────────────────────────────────────────────────

    async def _do_image_gen(self, message: discord.Message, content: str):
        """Generate an image and send it."""
        if not self.image_gen:
            await message.channel.send("Image gen isn't configured.")
            return

        async with message.channel.typing():
            refined = await self.llm_client.generate(
                f"Convert this into a detailed image generation prompt. ONLY the prompt:\n{content}",
                system_prompt="You write image generation prompts. Be specific. Return ONLY the prompt.",
            )
            path = await self.image_gen.generate_and_download(refined)

        if not path:
            await message.channel.send("Couldn't generate that image.")
            return

        await message.channel.send(f"{message.author.mention} here:")
        await message.channel.send(file=discord.File(path))
        try:
            os.remove(path)
        except (OSError, TypeError):
            pass

    # ─── Helpers ────────────────────────────────────────────────────────

    def _bot_identity(self) -> str:
        return self.bot.user.display_name if self.bot and self.bot.user else "bot"

    def _reset_channel_counters(self, channel_id: str):
        self.message_counters[channel_id] = 0
        self.recv_message_counts[channel_id] = 0
        self.conversation_threads.setdefault(channel_id, {"depth": 0})["depth"] += 1

    async def _send_tracked(self, channel, content: str, **kwargs) -> discord.Message:
        sent = await channel.send(content, **kwargs)
        self.feedback.register_message(str(sent.id), content)
        self.bot_message_map[sent.id] = content
        return sent

    def _extract_image_urls(self, message: discord.Message) -> List[str]:
        """Extract image URLs from message attachments."""
        urls = []
        for attachment in message.attachments:
            logger.info(f"Attachment: {attachment.filename} type={attachment.content_type} url={attachment.url[:80]}")
            if attachment.content_type and attachment.content_type.startswith("image/"):
                urls.append(attachment.url)
            elif attachment.url and any(attachment.url.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
                urls.append(attachment.url)
        if urls:
            logger.info(f"Extracted {len(urls)} image URL(s)")
        return urls

    def _clean_reply_text(self, content: str) -> str:
        lines = content.split("\n") if content else []
        if len(lines) <= 1:
            return content.strip() if content else ""
        for line in lines:
            s = line.strip()
            if s and not s.startswith("<@") and ":" not in s[:30]:
                return "\n".join(lines[lines.index(line):]).strip()
        return lines[-1].strip() if lines else ""

    def _strip_name_prefix(self, text: str) -> str:
        import re
        name = self._bot_identity()
        pattern = rf"^(?:{re.escape(name)}|[A-Za-z]+)(?:\s*:\s*|\s+(?:says:?)\s*)"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # ─── Profile Learning ───────────────────────────────────────────────

    async def _run_profile_learning(self, channel_id: str, guild_id: str):
        if not self.profiles or not self.memory:
            return
        recent = self.memory.get_recent(channel_id, limit=25)
        if not recent:
            return

        convo = "\n".join(f"{name}: {content}" for name, content, _ in recent)
        prompt = (
            "Analyze this conversation. Extract observations about users.\n"
            "Format: USER_NAME|FACT|PREFERENCE\n"
            "FACT or - if none. PREFERENCE: likes:X, dislikes:X, or -.\n"
            f"Conversation:\n{convo}\n\n"
            "Only extract genuinely noteworthy stuff. Respond 'nothing' if none."
        )
        try:
            result = await self.llm_client.generate(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
            result = result.strip()
            if result.lower() == "nothing" or not result:
                return
            for line in result.split("\n"):
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                user_name, fact, pref = [p.strip() for p in parts[:3]]
                all_profiles = self.profiles.get_all_profiles()
                target = None
                for p in all_profiles:
                    d = p["display_name"].lower()
                    if user_name in d or d in user_name:
                        target = p
                        break
                if not target:
                    continue
                uid = target["user_id"]
                if fact and fact != "-":
                    self.profiles.add_fact(uid, fact, target["display_name"])
                if pref and pref != "-":
                    if pref.startswith("likes:") and pref[6:] != "-":
                        self.profiles.add_preference(uid, "like", pref[6:], target["display_name"])
                    elif pref.startswith("dislikes:") and pref[9:] != "-":
                        self.profiles.add_preference(uid, "dislike", pref[9:], target["display_name"])
            logger.info(f"Profile learning done for channel {channel_id}")
        except Exception as e:
            logger.error(f"Profile learning failed: {e}")

    # ─── Bandwagon ──────────────────────────────────────────────────────

    async def _check_bandwagon(self, message: discord.Message, fresh_msg: discord.Message) -> bool:
        if not message.guild:
            return False
        try:
            total = message.guild.member_count
            if not total or total <= 1:
                return False
        except Exception:
            return False

        counts = {}
        for r in fresh_msg.reactions:
            emoji = str(r.emoji)
            if emoji in self.BANDWAGON_EMOJI:
                try:
                    n = sum(1 async for u in r.users() if u.id != self.bot.user.id)
                except Exception:
                    n = max(0, r.count - 1)
                if n > 0:
                    counts[emoji] = n

        if not counts:
            return False

        top_emoji, reactors = max(counts.items(), key=lambda x: x[1])
        chance = self.BANDWAGON_MAX * (reactors / total)
        if random.random() < chance:
            try:
                await message.add_reaction(top_emoji)
                if reactors >= 3:
                    short = random.choice(["this", "real", "yes", "facts", "agreed", "yeah", "true", "100%", "👆"])
                    await self._send_tracked(message.channel, short, reference=message)
                return True
            except discord.HTTPException:
                pass
        return False

    async def on_guild_join(self, guild: discord.Guild):
        logger.info(f"Joined new guild: {guild.name}")

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.bot.user.id:
            return
        mid = str(payload.message_id)
        if mid not in self.bot_message_map:
            return
        emoji = str(payload.emoji)
        self.feedback.add_reaction(mid, emoji, str(payload.user_id))
        logger.debug(f"Feedback: {emoji} on message {mid}")

    def run(self):
        if not self.setup_complete:
            asyncio.run(self.setup())
        port = int(os.getenv("PORT", "8080"))
        start_health_server(port)
        try:
            self.bot.run(self.config.discord_token)
        except discord.errors.LoginFailure:
            logger.error("Invalid Discord token!")
            sys.exit(1)
        except KeyboardInterrupt:
            pass


class MainCommands(commands.Cog):
    """Command handlers."""

    def __init__(self, bot, llm_client, memory, feedback=None, profiles=None):
        self.bot = bot
        self.llm_client = llm_client
        self.memory = memory
        self.feedback = feedback
        self.profiles = profiles
        dev_ids = os.getenv("DEV_USER_IDS", "")
        self.dev_user_ids = set(x.strip() for x in dev_ids.split(",") if x.strip())

    def _is_dev(self, ctx) -> bool:
        return str(ctx.author.id) in self.dev_user_ids

    @commands.command(name="ping")
    async def ping(self, ctx):
        lat = round(self.bot.latency * 1000)
        if lat < 100:
            await ctx.send(f"{lat}ms. Alive and kicking.")
        elif lat < 300:
            await ctx.send(f"{lat}ms. Still here.")
        elif lat < 600:
            await ctx.send(f"{lat}ms. Running but feeling it.")
        else:
            await ctx.send(f"{lat}ms. Alive, barely. Don't ask me to run a marathon.")

    @commands.command(name="help")
    async def help_command(self, ctx):
        name = self.bot.user.display_name if self.bot.user else "bot"
        embed = discord.Embed(
            title=name,
            description="I'm the bot. I'm here to hang out.",
            color=discord.Color.blue(),
        )
        embed.add_field(name="How to talk to me", value=f"Mention me (@{name}) or DM me.", inline=False)
        embed.add_field(name="Commands", value=(
            "`!ping` - Check if I'm alive\n"
            "`!help` - Show this message\n"
            "`!whoami` - Ask me who I am\n"
            "`!feedback` - Show reaction feedback stats\n"
            "`!learn` - Force re-analyze feedback lessons\n"
            "`!improve <text>` - Suggest an improvement\n"
        ), inline=False)
        embed.add_field(name="Memory", value="I remember recent messages in each channel.", inline=False)
        await ctx.send(embed=embed)

    @commands.command(name="whoami")
    async def whoami(self, ctx):
        name = self.bot.user.display_name if self.bot.user else "bot"
        await ctx.send(f"I'm {name}. I hang out here with you folks. Participate in conversations, react, occasionally share my two cents.")

    @commands.command(name="feedback")
    async def feedback(self, ctx):
        if not self.feedback:
            await ctx.send("Feedback system not initialized.")
            return
        stats = self.feedback.get_stats()
        if stats["total_positive"] + stats["total_negative"] == 0:
            await ctx.send("No reactions to learn from yet.")
            return
        lines = [
            f"**Feedback**",
            f"Positive: {stats['total_positive']} | Negative: {stats['total_negative']}",
            f"Tracked messages: {stats['total_tracked_messages']}",
        ]
        if self.feedback.lessons:
            lines.append(f"\nLessons ({len(self.feedback.lessons)}):")
            for l in self.feedback.lessons:
                lines.append(f"  ▸ {l}")
        await ctx.send("\n".join(lines))

    @commands.command(name="learn")
    async def learn(self, ctx):
        if not self.feedback:
            return
        old = list(self.feedback.lessons)
        self.feedback.update_lessons()
        new = self.feedback.lessons
        if not new and not old:
            await ctx.send("Not enough data yet.")
            return
        changes = [f"  ▸ NEW: {l}" for l in new if l not in old]
        changes += [f"  ▸ DROPPED: {l}" for l in old if l not in new]
        msg = "Lessons updated:\n" + "\n".join(changes) if changes else "No changes."
        await ctx.send(msg)

    @commands.command(name="improve")
    async def improve(self, ctx, *, suggestion: str = ""):
        if not suggestion:
            await ctx.send(f"Tell {self.bot.user.display_name if self.bot.user else 'me'} what to improve.\nUsage: `!improve your suggestion here`")
            return
        ts = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        author = f"{ctx.author.display_name} (ID: {ctx.author.id})"
        channel = f"#{ctx.channel.name}" if ctx.guild else "DM"
        entry = f"[{ts}] {author} in {channel}: {suggestion}\n"
        try:
            with open(IMPROVEMENTS_LOG, "a") as f:
                f.write(entry)
            await ctx.send("Suggestion noted. I'll work on that.")
        except Exception as e:
            logger.error(f"Failed to log improvement: {e}")
            await ctx.send("Something went wrong.")

    @commands.command(name="admin")
    async def admin(self, ctx, action: str = "status"):
        if not self._is_dev(ctx):
            return
        if action == "status":
            stats = self.feedback.get_stats() if self.feedback else {}
            msg = [
                "**Internal Status**",
                f"Messages tracked: {stats.get('total_tracked_messages', 0)}",
                f"Positive: {stats.get('total_positive', 0)} | Negative: {stats.get('total_negative', 0)}",
                f"User profiles: {len(self.profiles.get_all_profiles()) if self.profiles else 0}",
                f"Dev IDs: {', '.join(self.dev_user_ids) or 'none'}",
            ]
            await ctx.send("\n".join(msg))

        elif action == "profiles" and self.profiles:
            summary = self.profiles.get_all_summaries(limit=20)
            if not summary:
                await ctx.send("No profiles yet.")
                return
            for i in range(0, len(summary), 1900):
                await ctx.send(f"```\n{summary[i:i+1900]}\n```")

        elif action == "profile" and self.profiles and ctx.message.mentions:
            uid = str(ctx.message.mentions[0].id)
            p = self.profiles.get_profile(uid)
            if not p:
                await ctx.send(f"No profile for {uid}.")
                return
            import json
            lines = [f"**{p['display_name']}** (uid: {uid})", f"Messages: {p['message_count']}"]
            facts = json.loads(p["known_facts"] or "[]")
            if facts:
                lines.append("Facts:")
                for f in facts:
                    lines.append(f"  - {f}")
            notes = p["personality_notes"].strip()
            if notes:
                lines.append("Personality:")
                for n in notes.split("\n"):
                    if n.strip():
                        lines.append(f"  - {n.strip()}")
            likes = json.loads(p["likes"] or "[]")
            if likes:
                lines.append(f"Likes: {', '.join(likes)}")
            dislikes = json.loads(p["dislikes"] or "[]")
            if dislikes:
                lines.append(f"Dislikes: {', '.join(dislikes)}")
            await ctx.send("\n".join(lines))

        elif action == "logs":
            count = 10
            parts = ctx.message.content.split()
            if len(parts) > 2:
                try:
                    count = int(parts[-1])
                except ValueError:
                    pass
            try:
                with open(IMPROVEMENTS_LOG, "r") as f:
                    lines = f.readlines()[-count:]
                if not lines:
                    await ctx.send("No improvement requests.")
                    return
                msg = "".join(lines)
                for i in range(0, len(msg), 1900):
                    await ctx.send(f"```\n{msg[i:i+1900]}\n```")
            except FileNotFoundError:
                await ctx.send("No improvement log file.")

        elif action == "memories" and self.memory:
            cid = str(ctx.channel.id)
            recent = self.memory.get_recent(cid, limit=30)
            if not recent:
                await ctx.send("No memories.")
                return
            lines = [f"{name}: {content}" for name, content, _ in recent]
            msg = "\n".join(lines)
            for i in range(0, len(msg), 1900):
                await ctx.send(f"```\n{msg[i:i+1900]}\n```")

        elif action == "clear":
            sub = ctx.message.content.split()[-1] if len(ctx.message.content.split()) > 2 else ""
            if sub == "memory" and self.memory:
                self.memory.cleanup_old(days=0)
                await ctx.send("Memory wiped.")
            elif sub == "profiles" and self.profiles:
                for p in self.profiles.get_all_profiles():
                    self.profiles.delete_profile(p["user_id"])
                await ctx.send("Profiles cleared.")
            else:
                await ctx.send("Usage: `!admin clear memory` or `!admin clear profiles`")


if __name__ == "__main__":
    config = BotConfig()
    bot = DiscordLLMBot(config)
    bot.run()

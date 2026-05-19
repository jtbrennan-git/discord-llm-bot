#!/usr/bin/env python3
"""
Main bot application for Discord LLM Bot.
Unified architecture: every response goes through a single LLM call with [TAG] output.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import threading
import random
import re
from typing import Optional, Dict, List

import discord
from discord.ext import commands

from config.config import BotConfig
from utils.llm import LLMClient, LLMConfig, DEFAULT_SYSTEM_PROMPT, ParsedResponse
from utils.memory import MemoryStore
from utils.feedback import FeedbackTracker
from utils.image_gen import ImageGenerator
from utils.profiles import UserProfileStore
from utils.channel_state import ChannelStateStore
from utils.prompts import build_system_prompt
from utils.action_audit import ActionAuditStore
from utils.style_guide import StyleGuideLearner, StyleGuideStore
from utils.topic_log import TopicLearner, TopicLogStore

IMPROVEMENTS_LOG = os.getenv("IMPROVEMENTS_LOG", "/tmp/bot_improvements.log")
BOT_LOG_PATH = os.getenv("BOT_LOG_PATH", "/tmp/discord_llm_bot.log")

def configure_logging() -> None:
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if BOT_LOG_PATH:
        try:
            os.makedirs(os.path.dirname(BOT_LOG_PATH) or ".", exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    BOT_LOG_PATH,
                    maxBytes=int(os.getenv("BOT_LOG_MAX_BYTES", "5242880")),
                    backupCount=int(os.getenv("BOT_LOG_BACKUP_COUNT", "5")),
                    encoding="utf-8",
                )
            )
        except OSError:
            pass
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)


configure_logging()
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
        self.channel_states = ChannelStateStore()
        self.bot_message_map: Dict[str, str] = {}
        self.active_followups: Dict[str, dict] = {}
        self.bot_topic_map: Dict[str, int] = {}
        self.channel_topic_starters: Dict[str, dict] = {}
        self.learning_message_counts: Dict[str, int] = {}
        self.style_guides: Optional[StyleGuideStore] = None
        self.style_learner: Optional[StyleGuideLearner] = None
        self.topic_log: Optional[TopicLogStore] = None
        self.topic_learner: Optional[TopicLearner] = None
        self.action_audit: Optional[ActionAuditStore] = None
        self.learning_queue: Optional[asyncio.Queue] = None
        self.learning_worker_task: Optional[asyncio.Task] = None
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
        self.style_guides = StyleGuideStore()
        self.style_learner = StyleGuideLearner(
            self.style_guides,
            interval=self.config.style_learning_interval,
            context_limit=self.config.style_learning_context_limit,
        )
        self.topic_log = TopicLogStore()
        self.topic_learner = TopicLearner(
            self.topic_log,
            interval=self.config.topic_learning_interval,
            context_limit=self.config.topic_learning_context_limit,
            starter_cooldown_seconds=self.config.topic_starter_cooldown_seconds,
        )
        self.action_audit = ActionAuditStore()
        self.learning_queue = asyncio.Queue(maxsize=self.config.learning_queue_maxsize)
        self.image_gen = ImageGenerator(
            api_key=self.config.llm_api_key,
            model=self.config.image_model,
            base_url=self.config.llm_base_url,
        )

        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self.bot.event(self.on_guild_join)
        self.bot.event(self.on_raw_reaction_add)

        await self.bot.add_cog(
            MainCommands(
                self.bot,
                self.llm_client,
                self.memory,
                self.feedback,
                self.profiles,
                self.style_guides,
                self.topic_log,
                self.action_audit,
                self.config,
                self.learning_queue,
            )
        )
        await self.bot.add_cog(
            ControlCommands(
                self.bot,
                self.config,
                self.memory,
                self.style_guides,
                self.topic_log,
                self.action_audit,
                self.learning_queue,
            )
        )

        self.setup_complete = True
        logger.info("Bot setup complete")

    async def on_ready(self):
        logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        logger.info(f"Connected to {len(self.bot.guilds)} guilds")
        if not self.learning_worker_task or self.learning_worker_task.done():
            self.learning_worker_task = asyncio.create_task(self._learning_worker())

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not message.guild:
            return
        if self._is_command_message(message.content):
            await self.bot.process_commands(message)
            return

        channel_id = str(message.channel.id)
        guild_id = str(message.guild.id)
        if guild_id in {self.config.control_guild_id, self.config.target_guild_id}:
            logger.info(
                "Message received guild_id=%s channel_id=%s channel_name=%s author_id=%s content_len=%s",
                guild_id,
                channel_id,
                getattr(message.channel, "name", "dm"),
                str(message.author.id),
                len(message.content or ""),
            )
        controls = self._channel_controls(channel_id)

        tracking_enabled = controls.get("tracking_enabled", True)
        remember_enabled = self._user_remember_enabled(str(message.author.id))

        if self.memory and tracking_enabled and remember_enabled:
            self.memory.add_message(
                channel_id=channel_id,
                guild_id=guild_id,
                author_name=self._server_display_name(message.author),
                author_id=str(message.author.id),
                content=message.content,
            )

        if self.profiles and tracking_enabled and remember_enabled:
            self.profiles.upsert_user(str(message.author.id), self._server_display_name(message.author))

        if tracking_enabled and remember_enabled and self._learning_allowed(message) and controls["learning_enabled"]:
            self.learning_message_counts[channel_id] = self.learning_message_counts.get(channel_id, 0) + 1
            self._enqueue_group_learning(channel_id, str(message.guild.id))
            self._expire_topic_starters()
            self._record_topic_followup(channel_id)

        if self.profiles and tracking_enabled and remember_enabled and message.guild:
            self._learning_counter += 1
            if self._learning_counter % 30 == 0:
                await self._run_profile_learning(channel_id, str(message.guild.id))

        channel_state = self.channel_states.get(channel_id)
        user_response_mode = self._user_response_mode(str(message.author.id))

        bot_was_mentioned = self.bot.user in message.mentions
        is_reply_to_bot = (
            bot_was_mentioned
            or (
                message.reference
                and message.reference.message_id
                and await self._is_bot_message(message.channel, message.reference.message_id)
            )
        )

        if is_reply_to_bot:
            content = self._strip_bot_mentions(message.content) if bot_was_mentioned else message.content
            self.channel_states.mark_direct_interaction(channel_id)
            await self._handle_message(message, content)
            return

        await self.bot.process_commands(message)

        if self._message_names_bot(message.content):
            if user_response_mode == "strict":
                self._record_action_audit(message, action_type="skip", reason="user requires @ mention or reply")
                return
            self.channel_states.mark_direct_interaction(channel_id)
            await self._handle_name_call(message)
            return

        if user_response_mode in {"prompted", "strict"}:
            self._record_action_audit(message, action_type="skip", reason=f"user response mode is {user_response_mode}")
            return

        channel_state = self.channel_states.record_inbound(channel_id)
        self.channel_states.decay_thread_depth(channel_id)

        await self._maybe_join_conversation(message)

    async def _is_bot_message(self, channel, message_id: int) -> bool:
        try:
            msg = await channel.fetch_message(message_id)
            return msg.author.id == self.bot.user.id
        except Exception:
            return False

    # ─── Unified Generation ─────────────────────────────────────────────

    def _build_system_prompt(self, channel_id: Optional[str] = None) -> str:
        """Assemble full system prompt with identity, profiles, and feedback."""
        base_prompt = (
            self.llm_client.config.system_prompt
            if self.llm_client and self.llm_client.config.system_prompt
            else DEFAULT_SYSTEM_PROMPT
        )
        style_context = (
            self.style_guides.get_prompt_context(channel_id)
            if channel_id
            and self.style_guides
            and self._channel_controls(channel_id)["style_enabled"]
            and self._channel_controls(channel_id).get("tracking_enabled", True)
            else None
        )
        return build_system_prompt(
            bot_name=self._bot_identity(),
            base_prompt=base_prompt,
            profiles=self.profiles if getattr(self.config, "profile_context_enabled", False) else None,
            feedback=self.feedback,
            style_context=style_context,
        ) + self._guild_emoji_prompt_context()

    def _build_context(self, channel_id: str, for_spontaneous: bool = False) -> List[Dict[str, str]]:
        """Build chat context using proper assistant/user roles."""
        if not self.memory:
            return []
        if not self._channel_controls(channel_id).get("tracking_enabled", True):
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
        system_prompt = self._build_system_prompt(channel_id)

        # Clean the input
        content = self._clean_reply_text(content)

        try:
            async with message.channel.typing():
                logger.info(
                    "LLM generation requested channel_id=%s guild_id=%s images=%s spontaneous=%s",
                    channel_id,
                    str(message.guild.id) if message.guild else None,
                    bool(image_urls),
                    for_spontaneous,
                )
                raw = await self._llm_generate(
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
            logger.exception("Generation failed channel_id=%s", channel_id)
            await message.channel.send(feedback)

    async def _llm_generate(self, *args, **kwargs) -> str:
        if not self.llm_client:
            return "[REPLY] LLM is not configured."
        return await asyncio.wait_for(
            self.llm_client.generate(*args, **kwargs),
            timeout=self.config.llm_timeout_seconds,
        )

    async def _execute_action(self, parsed: ParsedResponse, message: discord.Message,
                               image_urls: List[str], channel_id: str) -> None:
        """Route a parsed [TAG] response to the correct action."""
        if parsed.action == "SILENT":
            self._record_action_audit(message, action_type="silent", reason="model chose SILENT", final_action="silent")
            return

        elif parsed.action == "REACT":
            emoji = parsed.content or "👍"
            if self.config.dry_run_actions:
                logger.info("DRY_RUN would react with %s in channel_id=%s", emoji, channel_id)
                self._record_action_audit(message, action_type="dry_run_react", reason=emoji)
                self._reset_channel_counters(channel_id)
                return
            actual_emoji = await self._add_reaction(message, emoji)
            # Track for feedback
            if str(message.id) not in self.bot_message_map:
                self.bot_message_map[str(message.id)] = f"[reacted with {actual_emoji}]"
            self._record_action_audit(message, action_type="react", reason=str(actual_emoji), final_action="react")
            self._reset_channel_counters(channel_id)

        elif parsed.action == "IMAGE_GEN":
            await self._do_image_gen(message, parsed.content or "Create an image from the current chat.")

        elif parsed.action == "IMAGE_ANALYSIS":
            if self.config.dry_run_actions:
                logger.info("DRY_RUN would send image analysis in channel_id=%s: %s", channel_id, parsed.content)
                self._record_action_audit(message, action_type="dry_run_image_analysis", reason=parsed.content[:200])
                self._reset_channel_counters(channel_id)
                return
            await self._send_tracked_response(message, parsed.content)
            await self._maybe_apply_supplemental_reaction(message, parsed.reaction, channel_id)
            self._record_action_audit(message, action_type="image_analysis", reason=parsed.content[:200], final_action="reply")
            self._reset_channel_counters(channel_id)

        elif parsed.action == "REPLY":
            text = self._strip_name_prefix(parsed.content)
            if text:
                if self.config.dry_run_actions:
                    logger.info("DRY_RUN would reply in channel_id=%s: %s", channel_id, text)
                    self._record_action_audit(message, action_type="dry_run_reply", reason=text[:200])
                    self._reset_channel_counters(channel_id)
                    return
                sent = await self._send_tracked_response(message, text)
                if self.memory:
                    self.memory.add_message(
                        channel_id=channel_id,
                        guild_id=str(message.guild.id) if message.guild else None,
                        author_name=self._bot_identity(),
                        author_id=str(self.bot.user.id),
                        content=text,
                    )
                await self._maybe_apply_supplemental_reaction(message, parsed.reaction, channel_id)
                self._record_action_audit(message, action_type="reply", reason=text[:200], message_id=str(sent.id), final_action="reply")
            self._reset_channel_counters(channel_id)

    # ─── Message Handlers ───────────────────────────────────────────────

    async def _handle_message(self, message: discord.Message, content: str):
        """Handle mentions and DMs."""
        await self._generate_and_execute(message, content, for_spontaneous=False)

    async def _handle_name_call(self, message: discord.Message):
        """Ask the LLM whether an exact name call merits a response."""
        controls = self._channel_controls(str(message.channel.id))
        if controls["quiet_enabled"]:
            self._record_action_audit(message, action_type="skip", reason="name call while channel quiet")
            return
        prompt = (
            f"Someone may have referred to you by name as {self._bot_identity()}.\n"
            "Decide whether the message is actually inviting you into the conversation.\n"
            "If it is just talking about you, quoting your name, or does not need you, return [SILENT].\n"
            "If a reply would help, return [REPLY]. Only return standalone [REACT: emoji] when words would add nothing.\n\n"
            f"Message: {message.content}"
        )
        await self._generate_and_execute(message, prompt, for_spontaneous=True)
        self._record_action_audit(message, action_type="name_call", reason="exact bot name detected")

    async def _maybe_handle_implicit_followup(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return False
        channel_id = str(message.channel.id)
        active = self.active_followups.get(channel_id)
        if not active:
            return False

        import time
        if time.time() > float(active.get("expires_at") or 0):
            self.active_followups.pop(channel_id, None)
            return False

        remaining = int(active.get("remaining") or 0) - 1
        active["remaining"] = remaining
        if remaining < 0:
            self.active_followups.pop(channel_id, None)
            return False

        if self._is_low_signal_followup(message.content):
            self.active_followups.pop(channel_id, None)
            self._record_action_audit(
                message,
                action_type="skip",
                reason="implicit followup ended by low-signal message",
                message_id=str(active.get("message_id") or ""),
            )
            return False

        max_messages = max(1, self.config.followup_window_messages)
        chance = min(0.85, max(0.05, ((remaining + 1) / max_messages) ** 1.5))
        roll = random.random()
        if roll > chance:
            self._record_action_audit(
                message,
                action_type="skip",
                reason=f"implicit followup probability roll failed remaining={remaining}",
                probability=chance,
                roll=roll,
                message_id=str(active.get("message_id") or ""),
            )
            if remaining <= 0:
                self.active_followups.pop(channel_id, None)
            return False

        controls = self._channel_controls(channel_id)
        if controls["quiet_enabled"] or not controls["spontaneous_enabled"]:
            return False

        prompt = (
            "You recently sent a message in this Discord channel. A newer message arrived without using Discord's reply feature.\n"
            "Decide whether this newer message is naturally replying to you or inviting you to keep talking.\n"
            "If it is likely addressed to someone else, the room moved on, or no response is needed, return [SILENT].\n"
            "If it is naturally responding to you, continue briefly with [REPLY]. Use [REACT] only when words add nothing.\n\n"
            f"Your recent message: {active.get('content') or '(unknown)'}\n"
            f"New message from {self._server_display_name(message.author)}: {message.content}"
        )
        self.channel_states.mark_direct_interaction(channel_id)
        await self._generate_and_execute(message, prompt, for_spontaneous=True)
        self._record_action_audit(
            message,
            action_type="implicit_followup",
            reason=f"active bot message {active.get('message_id')} remaining={remaining}",
            probability=chance,
            roll=roll,
            message_id=str(active.get("message_id") or ""),
        )
        if remaining <= 0:
            self.active_followups.pop(channel_id, None)
        return True

    @staticmethod
    def _is_low_signal_followup(content: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]+", "", (content or "").lower()).strip()
        if not normalized:
            return True
        normalized = re.sub(r"\s+", " ", normalized)
        low_signal = {
            "ok",
            "okay",
            "k",
            "kk",
            "lol",
            "lmao",
            "haha",
            "ha",
            "yeah",
            "yea",
            "yep",
            "nah",
            "no",
            "nope",
            "sure",
            "cool",
            "alright",
            "aight",
            "bet",
            "true",
            "real",
            "fair",
            "word",
            "ok bro",
            "okay bro",
            "thanks",
            "ty",
            "nvm",
            "oh nvm",
        }
        return normalized in low_signal

    async def _maybe_join_conversation(self, message: discord.Message):
        """Spontaneous join with counter-based probability + time decay."""
        import time
        channel_id = str(message.channel.id)
        controls = self._channel_controls(channel_id)
        channel_state = self.channel_states.get(channel_id)

        if controls["quiet_enabled"] or not controls["spontaneous_enabled"]:
            self._record_action_audit(message, action_type="skip", reason="channel quiet/spontaneous disabled")
            return
        rate = max(0.0, min(2.0, float(controls.get("spontaneous_rate", 0.0))))
        if rate <= 0:
            self._record_action_audit(message, action_type="skip", reason="spontaneous rate is zero")
            return
        min_messages = self.config.spontaneous_min_messages_since_action
        now = time.time()
        last = channel_state.last_action_time
        idle_seconds = now - last
        idle_eligible = (
            channel_state.received_since_action >= self.config.spontaneous_idle_min_messages
            and idle_seconds >= self.config.spontaneous_idle_trigger_seconds
        )
        if channel_state.received_since_action < min_messages and not idle_eligible:
            self._record_action_audit(
                message,
                action_type="skip",
                reason=(
                    f"not enough messages since action ({channel_state.received_since_action}/{min_messages}); "
                    f"idle={int(idle_seconds)}s"
                ),
            )
            return
        if channel_state.thread_depth >= self.config.spontaneous_max_thread_depth:
            self._record_action_audit(message, action_type="skip", reason="thread depth too high")
            return

        counter = channel_state.message_count
        scale = self.config.spontaneous_message_target * 1.5
        chance_cap = max(0.0, min(1.0, self.config.spontaneous_chance_cap))
        if chance_cap <= 0:
            self._record_action_audit(message, action_type="skip", reason="spontaneous chance cap is zero")
            return
        chance = min(chance_cap, counter / scale)

        # Time-based modifier ramps to the same cap over the configured idle window.
        time_modifier = min(
            chance_cap,
            (idle_seconds / self.config.spontaneous_idle_ramp_seconds) * chance_cap,
        )
        chance = min(chance_cap, (chance + time_modifier) * rate)

        topic_started = await self._maybe_start_topic(message, channel_state, idle_seconds)
        if topic_started:
            return

        roll = random.random()
        if roll > chance:
            self._record_action_audit(
                message,
                action_type="skip",
                reason="spontaneous probability roll failed",
                probability=chance,
                roll=roll,
            )
            return

        context = self._build_context(channel_id, for_spontaneous=True)
        if not context:
            self._record_action_audit(message, action_type="skip", reason="no context for spontaneous response")
            return

        fresh_msg = None
        try:
            fresh_msg = await message.channel.fetch_message(message.id)
        except Exception:
            pass

        if fresh_msg:
            bandwagon_triggered = await self._check_bandwagon(message, fresh_msg)
            if bandwagon_triggered:
                self._record_action_audit(
                    message,
                    action_type="bandwagon",
                    reason="matched channel reaction pattern",
                    probability=chance,
                    roll=roll,
                )
                self._reset_channel_counters(channel_id)
                return

        await self._generate_and_execute(message, None, for_spontaneous=True)
        self._record_action_audit(
            message,
            action_type="spontaneous",
            reason="counter/time probability passed",
            probability=chance,
            roll=roll,
        )

    # ─── Actions ────────────────────────────────────────────────────────

    async def _do_image_gen(self, message: discord.Message, content: str):
        """Generate an image and send it."""
        if not self.image_gen:
            await message.channel.send("Image gen isn't configured.")
            return
        if self.config.dry_run_actions:
            logger.info("DRY_RUN would generate image: %s", content)
            self._record_action_audit(message, action_type="dry_run_image_gen", reason=content[:200])
            return

        async with message.channel.typing():
            refined = await self._llm_generate(
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

    def _guild_emoji_prompt_context(self) -> str:
        if not self.bot:
            return ""
        names = []
        for guild in getattr(self.bot, "guilds", []) or []:
            for emoji in getattr(guild, "emojis", []) or []:
                name = getattr(emoji, "name", "")
                if name and name not in names:
                    names.append(name)
                if len(names) >= 60:
                    break
            if len(names) >= 60:
                break
        if not names:
            return ""
        listed = ", ".join(f":{name}:" for name in names[:60])
        return (
            "\n\n## Server custom reactions\n"
            "You may use these server custom emoji in [REACT] by writing the exact colon name. "
            "Only use one when it fits better than a standard emoji.\n"
            f"{listed}"
        )

    @staticmethod
    def _server_display_name(user) -> str:
        return getattr(user, "display_name", None) or getattr(user, "name", None) or str(user)

    def _bot_name_candidates(self) -> List[str]:
        if not self.bot or not self.bot.user:
            return ["bot"]
        user = self.bot.user
        names = [
            getattr(user, "display_name", None),
            getattr(user, "global_name", None),
            getattr(user, "name", None),
        ]
        seen = set()
        candidates = []
        for name in names:
            if not name:
                continue
            normalized = re.sub(r"\s+", " ", str(name).strip().lower())
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
        return candidates or ["bot"]

    def _is_command_message(self, content: str) -> bool:
        text = (content or "").lstrip()
        if not text:
            return False
        prefixes = [self.config.command_prefix] if isinstance(self.config.command_prefix, str) else list(self.config.command_prefix)
        prefixes.append("/")
        return any(prefix and text.startswith(prefix) for prefix in prefixes)

    def _message_names_bot(self, content: str) -> bool:
        text = (content or "").lower()
        if not text.strip():
            return False
        for candidate in self._bot_name_candidates():
            candidate = candidate.strip().lower()
            if len(candidate) < 3:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(candidate) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        return False

    def _reset_channel_counters(self, channel_id: str):
        self.channel_states.mark_bot_action(channel_id)

    async def _send_tracked(self, channel, content: str, **kwargs) -> discord.Message:
        sent = await channel.send(content, **kwargs)
        self.feedback.register_message(str(sent.id), content)
        self.bot_message_map[str(sent.id)] = content
        return sent

    def _activate_followup_window(self, channel_id: str, message_id: str, content: str) -> None:
        import time
        if self.config.followup_window_messages <= 0 or self.config.followup_window_seconds <= 0:
            self.active_followups.pop(channel_id, None)
            return
        self.active_followups[channel_id] = {
            "message_id": str(message_id),
            "content": content,
            "remaining": self.config.followup_window_messages,
            "expires_at": time.time() + self.config.followup_window_seconds,
        }

    async def _send_tracked_response(self, trigger_message: discord.Message, content: str) -> discord.Message:
        kwargs = {}
        if await self._channel_has_newer_user_message(trigger_message):
            kwargs["reference"] = trigger_message
        sent = await self._send_tracked(trigger_message.channel, content, **kwargs)
        self._activate_followup_window(str(trigger_message.channel.id), str(sent.id), content)
        return sent

    async def _channel_has_newer_user_message(self, trigger_message: discord.Message) -> bool:
        if not trigger_message.guild:
            return False
        try:
            async for msg in trigger_message.channel.history(
                limit=5,
                after=trigger_message.created_at,
                oldest_first=False,
            ):
                if msg.id == trigger_message.id:
                    continue
                if self.bot and self.bot.user and msg.author.id == self.bot.user.id:
                    continue
                return True
        except Exception:
            logger.debug("Could not inspect channel history for delayed reply check", exc_info=True)
        return False

    async def _maybe_apply_supplemental_reaction(self, message: discord.Message, emoji: Optional[str], channel_id: str) -> None:
        if not emoji:
            return
        if self.config.dry_run_actions:
            logger.info("DRY_RUN would add supplemental reaction %s in channel_id=%s", emoji, channel_id)
            return
        await self._add_reaction(message, emoji, suppress_errors=True)

    def _resolve_custom_emoji(self, guild, emoji: str):
        if not guild or not emoji:
            return None
        text = emoji.strip()
        if text.startswith("<") and text.endswith(">"):
            try:
                return discord.PartialEmoji.from_str(text)
            except Exception:
                return None
        if text.startswith(":") and text.endswith(":"):
            name = text.strip(":")
        else:
            name = text
        if not name or any(ch.isspace() for ch in name):
            return None
        exact = discord.utils.get(getattr(guild, "emojis", []) or [], name=name)
        if exact:
            return exact
        lowered = name.lower()
        return next((e for e in getattr(guild, "emojis", []) or [] if getattr(e, "name", "").lower() == lowered), None)

    async def _add_reaction(self, message: discord.Message, emoji: str, *, suppress_errors: bool = False):
        candidate = (emoji or "👍").strip()
        custom = self._resolve_custom_emoji(getattr(message, "guild", None), candidate)
        try:
            actual = custom or candidate
            await message.add_reaction(actual)
            return actual
        except discord.HTTPException:
            if suppress_errors:
                logger.debug("Reaction failed channel_id=%s emoji=%s", getattr(message.channel, "id", None), emoji)
                return None
            await message.add_reaction("💀")
            return "💀"

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

    def _strip_bot_mentions(self, content: str) -> str:
        if not self.bot or not self.bot.user:
            return content.strip()

        cleaned = content
        for mention in (self.bot.user.mention, f"<@!{self.bot.user.id}>"):
            cleaned = cleaned.replace(mention, "")
        return cleaned.strip()

    def _strip_name_prefix(self, text: str) -> str:
        import re
        name = self._bot_identity()
        pattern = rf"^(?:{re.escape(name)}|[A-Za-z]+)(?:\s*:\s*|\s+(?:says:?)\s*)"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # ─── Profile Learning ───────────────────────────────────────────────

    def _learning_allowed(self, message: discord.Message) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return self.config.allow_dm_learning
        return True

    def _channel_controls(self, channel_id: str) -> dict:
        if not self.action_audit:
            return {
                "learning_enabled": True,
                "style_enabled": True,
                "topics_enabled": True,
                "starters_enabled": True,
                "spontaneous_enabled": True,
                "quiet_enabled": False,
                "tracking_enabled": True,
                "spontaneous_rate": 0.0,
                "mode": "normal",
            }
        return self.action_audit.get_channel_controls(channel_id)

    def _user_response_mode(self, user_id: str) -> str:
        if not self.action_audit:
            return "normal"
        return self.action_audit.get_user_response_mode(user_id)

    def _user_remember_enabled(self, user_id: str) -> bool:
        if not self.action_audit:
            return True
        return bool(self.action_audit.get_user_privacy(user_id).get("remember_enabled", True))

    def _enqueue_group_learning(self, channel_id: str, guild_id: Optional[str]) -> None:
        if not self.learning_queue:
            return
        controls = self._channel_controls(channel_id)
        count = self.learning_message_counts.get(channel_id, 0)
        needs_style = (
            self.config.style_learning_enabled
            and controls["style_enabled"]
            and self.style_learner
            and self.style_learner.should_learn(count)
        )
        needs_topics = (
            self.config.topic_learning_enabled
            and controls["topics_enabled"]
            and self.topic_learner
            and self.topic_learner.should_learn(count)
        )
        if not needs_style and not needs_topics:
            return
        try:
            self.learning_queue.put_nowait({
                "channel_id": channel_id,
                "guild_id": guild_id,
                "style": needs_style,
                "topics": needs_topics,
            })
        except asyncio.QueueFull:
            logger.warning("Learning queue full; dropped learning job for channel_id=%s", channel_id)

    async def _learning_worker(self) -> None:
        if not self.learning_queue:
            return
        while True:
            job = await self.learning_queue.get()
            if self.config.dry_run_actions:
                logger.info("DRY_RUN would react with %s in channel_id=%s", emoji, channel_id)
                self._record_action_audit(message, action_type="dry_run_react", reason=emoji)
                self._reset_channel_counters(channel_id)
                return
            try:
                await self._run_group_learning_job(job)
            except Exception:
                logger.exception("Learning job failed")
            finally:
                self.learning_queue.task_done()

    async def _run_group_learning_job(self, job: dict) -> None:
        if not self.memory:
            return
        channel_id = job["channel_id"]
        recent_limit = max(self.config.style_learning_context_limit, self.config.topic_learning_context_limit)
        recent = self.memory.get_recent(
            channel_id,
            limit=recent_limit,
            exclude_author_id=str(self.bot.user.id) if self.bot and self.bot.user else None,
        )
        if job.get("style") and self.style_learner:
            await self.style_learner.learn_from_recent(channel_id, job.get("guild_id"), recent, self._llm_generate)
        if job.get("topics") and self.topic_learner:
            await self.topic_learner.learn_from_recent(channel_id, job.get("guild_id"), recent, self._llm_generate)

    async def _maybe_run_group_learning(self, channel_id: str, guild_id: Optional[str]) -> None:
        if not self.memory:
            return
        count = self.learning_message_counts.get(channel_id, 0)
        recent_limit = max(self.config.style_learning_context_limit, self.config.topic_learning_context_limit)
        recent = self.memory.get_recent(
            channel_id,
            limit=recent_limit,
            exclude_author_id=str(self.bot.user.id) if self.bot and self.bot.user else None,
        )
        if (
            self.config.style_learning_enabled
            and self.style_learner
            and self.style_learner.should_learn(count)
        ):
            await self.style_learner.learn_from_recent(channel_id, guild_id, recent, self._llm_generate)
        if (
            self.config.topic_learning_enabled
            and self.topic_learner
            and self.topic_learner.should_learn(count)
        ):
            await self.topic_learner.learn_from_recent(channel_id, guild_id, recent, self._llm_generate)

    async def _maybe_start_topic(self, message: discord.Message, channel_state, idle_seconds: float) -> bool:
        if isinstance(message.channel, discord.DMChannel):
            return False
        if not self.config.topic_starter_enabled or not self.topic_learner or not self.topic_log:
            return False
        if self.config.topic_starter_chance <= 0:
            self._record_action_audit(message, action_type="skip", reason="topic starter chance is zero")
            return False
        controls = self._channel_controls(str(message.channel.id))
        if controls["quiet_enabled"] or not controls["starters_enabled"] or not controls["topics_enabled"]:
            self._record_action_audit(message, action_type="skip", reason="topic starters disabled by channel controls")
            return False
        if channel_state.received_since_action < self.config.topic_starter_min_messages_since_action:
            self._record_action_audit(message, action_type="skip", reason="not enough messages for topic starter")
            return False
        if idle_seconds < self.config.topic_starter_min_idle_seconds:
            self._record_action_audit(message, action_type="skip", reason="topic starter idle threshold not met")
            return False
        if channel_state.thread_depth >= 2:
            self._record_action_audit(message, action_type="skip", reason="thread depth too high for topic starter")
            return False
        starter_roll = random.random()
        if starter_roll > self.config.topic_starter_chance:
            self._record_action_audit(
                message,
                action_type="skip",
                reason="topic starter probability roll failed",
                probability=self.config.topic_starter_chance,
                roll=starter_roll,
            )
            return False

        channel_id = str(message.channel.id)
        topic = self.topic_learner.choose_starter_topic(channel_id)
        if not topic:
            self._record_action_audit(message, action_type="skip", reason="no eligible topic starter candidates")
            return False

        style_context = self.style_guides.get_prompt_context(channel_id) if self.style_guides else ""
        context = self._build_context(channel_id, for_spontaneous=True)
        prompt = (
            "Start a natural Discord conversation based on this internal topic.\n"
            "Do not mention that this came from a topic log.\n"
            "Keep it short. One sentence is preferred.\n"
            "Return only [REPLY] or [SILENT].\n\n"
            f"Recent conversation:\n{self._context_to_text(context) or '(no recent messages)'}\n\n"
            f"Local channel style:\n{style_context or '(none)'}\n\n"
            f"Topic tag:\n{topic['label']}\n\n"
            f"Topic summary:\n{topic['summary']}\n\n"
            f"Starter angle:\n{topic['seed_prompt']}"
        )
        raw = await self._llm_generate(
            prompt,
            system_prompt=self._build_system_prompt(channel_id),
            chat_context=context,
        )
        parsed = self.llm_client.parse_response(raw)
        if parsed.action != "REPLY":
            return False

        text = self._strip_name_prefix(parsed.content)
        if not text:
            return False
        if self.config.dry_run_actions:
            topic_id = int(topic["id"])
            self._record_action_audit(
                message,
                action_type="dry_run_topic_starter",
                reason=f"would start topic {topic['label']}: {text[:120]}",
                topic_id=topic_id,
                probability=self.config.topic_starter_chance,
                roll=starter_roll,
            )
            self._reset_channel_counters(channel_id)
            return True
        sent = await self._send_tracked(message.channel, text)
        if self.memory:
            self.memory.add_message(
                channel_id=channel_id,
                guild_id=str(message.guild.id) if message.guild else None,
                author_name=self._bot_identity(),
                author_id=str(self.bot.user.id),
                content=text,
            )
        topic_id = int(topic["id"])
        self.topic_log.mark_started(topic_id)
        self.bot_topic_map[str(sent.id)] = topic_id
        self.channel_topic_starters[channel_id] = {
            "topic_id": topic_id,
            "message_id": sent.id,
            "reply_count": 0,
            "started_at": __import__("time").time(),
            "marked_success": False,
            "marked_ignored": False,
        }
        self._record_action_audit(
            message,
            action_type="topic_starter",
            reason=f"selected topic {topic['label']}",
            topic_id=topic_id,
            probability=self.config.topic_starter_chance,
            roll=starter_roll,
            message_id=str(sent.id),
        )
        self._reset_channel_counters(channel_id)
        return True

    def _record_topic_followup(self, channel_id: str) -> None:
        if not self.topic_log:
            return
        active = self.channel_topic_starters.get(channel_id)
        if not active or active.get("marked_success"):
            return
        now = __import__("time").time()
        if now - float(active.get("started_at") or 0) > 600:
            if not active.get("marked_ignored"):
                self.topic_log.mark_ignored(int(active["topic_id"]))
                active["marked_ignored"] = True
            return
        active["reply_count"] = int(active.get("reply_count") or 0) + 1
        if active["reply_count"] >= 3:
            self.topic_log.mark_success(int(active["topic_id"]))
            active["marked_success"] = True

    def _expire_topic_starters(self) -> None:
        if not self.topic_log:
            return
        now = __import__("time").time()
        for active in self.channel_topic_starters.values():
            if active.get("marked_success") or active.get("marked_ignored"):
                continue
            if now - float(active.get("started_at") or 0) > 600 and int(active.get("reply_count") or 0) == 0:
                self.topic_log.mark_ignored(int(active["topic_id"]))
                active["marked_ignored"] = True

    def _context_to_text(self, context: List[Dict[str, str]]) -> str:
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in context)

    def _record_action_audit(
        self,
        message: discord.Message,
        *,
        action_type: str,
        reason: str = "",
        topic_id: Optional[int] = None,
        probability: Optional[float] = None,
        roll: Optional[float] = None,
        message_id: Optional[str] = None,
        trigger_type: Optional[str] = None,
        final_action: Optional[str] = None,
    ) -> None:
        if not self.action_audit:
            return
        controls = self._channel_controls(str(message.channel.id))
        self.action_audit.record(
            guild_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            action_type=action_type,
            reason=reason,
            topic_id=topic_id,
            probability=probability,
            roll=roll,
            message_id=message_id,
            trigger_type=trigger_type,
            user_response_mode=self._user_response_mode(str(message.author.id)),
            channel_mode=str(controls.get("mode") or "normal"),
            final_action=final_action or action_type,
        )

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
            result = await self._llm_generate(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
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
        self.feedback.update_lessons()
        if self.topic_log and mid in self.bot_topic_map:
            sentiment = self.feedback.store.classify_reaction(emoji) if self.feedback else "neutral"
            if sentiment == "positive":
                topic_id = self.bot_topic_map[mid]
                active = self.channel_topic_starters.get(str(payload.channel_id))
                if not active or not active.get("marked_success"):
                    self.topic_log.mark_success(topic_id)
                    if active:
                        active["marked_success"] = True
        logger.debug(f"Feedback: {emoji} on message {mid}")

    def run(self):
        # Start health server FIRST so Fly smoke checks pass during setup
        port = int(os.getenv("PORT", "8080"))
        start_health_server(port)

        if not self.setup_complete:
            asyncio.run(self.setup())

        try:
            self.bot.run(self.config.discord_token)
        except discord.errors.LoginFailure:
            logger.error("Invalid Discord token!")
            sys.exit(1)
        except KeyboardInterrupt:
            pass


class MainCommands(commands.Cog):
    """Command handlers."""

    def __init__(
        self,
        bot,
        llm_client,
        memory,
        feedback=None,
        profiles=None,
        style_guides=None,
        topic_log=None,
        action_audit=None,
        config=None,
        learning_queue=None,
    ):
        self.bot = bot
        self.llm_client = llm_client
        self.memory = memory
        self.feedback = feedback
        self.profiles = profiles
        self.style_guides = style_guides
        self.topic_log = topic_log
        self.action_audit = action_audit
        self.config = config
        self.learning_queue = learning_queue
        dev_ids = os.getenv("DEV_USER_IDS", "")
        self.dev_user_ids = set(x.strip() for x in dev_ids.split(",") if x.strip())

    def _is_dev(self, ctx) -> bool:
        return str(ctx.author.id) in self.dev_user_ids

    @staticmethod
    def _display_name(user) -> str:
        return getattr(user, "display_name", None) or getattr(user, "name", None) or str(user)

    def _is_command_message(self, content: str) -> bool:
        text = (content or "").lstrip()
        if not text:
            return False
        prefix = getattr(self.config, "command_prefix", "!") if self.config else "!"
        prefixes = [prefix] if isinstance(prefix, str) else list(prefix)
        prefixes.append("/")
        return any(item and text.startswith(item) for item in prefixes)

    def _format_controls(self, controls: dict) -> str:
        labels = {
            "learning_enabled": "learning",
            "style_enabled": "style",
            "topics_enabled": "topics",
            "starters_enabled": "starters",
            "spontaneous_enabled": "spontaneous",
            "quiet_enabled": "quiet",
            "tracking_enabled": "tracking",
        }
        parts = [f"{label}={'on' if controls[key] else 'off'}" for key, label in labels.items()]
        parts.insert(0, f"mode={controls.get('mode', 'normal')}")
        parts.append(f"spont-rate={float(controls.get('spontaneous_rate', 0.0)):.2g}x")
        return ", ".join(parts)

    def _command_help(self) -> dict:
        return {
            "ping": {
                "usage": "!ping",
                "summary": "Check whether the bot is online and see current Discord latency.",
                "details": "Returns a short latency message in milliseconds.",
            },
            "help": {
                "usage": "!help [command]",
                "summary": "Show the command list or detailed help for one command.",
                "details": "Examples: `!help`, `!help control`, `!help feedback`.",
            },
            "control": {
                "usage": "!control status|mute|unmute|prompted|strict|normal|remember|forget|privacy",
                "summary": "Set your personal bot response and memory preferences.",
                "details": (
                    "`!control status` shows your current setting.\n"
                    "`!control mute` makes the bot respond to you only when you @mention it or use Discord reply.\n"
                    "`!control unmute` restores normal behavior.\n"
                    "`!control prompted` prevents spontaneous replies to you, but still allows name prompts.\n"
                    "`!control strict` and `!control normal` set those modes directly.\n"
                    "`!control remember off` stops storing your future messages for memory/profile learning.\n"
                    "`!control forget me` deletes your stored messages/profile and turns remembering off.\n"
                    "`!control privacy` shows response and memory settings.\n"
                    "Admins also use `!control` in the control channel for channel, user, logging, delete, react, and dashboard controls."
                ),
            },
            "whoami": {
                "usage": "!whoami",
                "summary": "Ask the bot for a short description of itself.",
                "details": "This is just an identity/status command. It does not change settings.",
            },
            "feedback": {
                "usage": "!feedback",
                "summary": "Show what reaction feedback the bot has collected.",
                "details": (
                    "The bot tracks reactions on its own messages as signal. Positive and negative reaction totals feed the lessons "
                    "shown here, which are used to tune future responses."
                ),
            },
            "highlights": {
                "usage": "!highlights [count] [scan_limit]",
                "summary": "Show recent funny/high-signal messages with the most reactions in this channel.",
                "details": (
                    "Scans recent visible history in the current channel, ranks messages by reaction count, "
                    "and skips commands, bot messages, and messages that look sensitive or serious. "
                    "`count` defaults to 5 and `scan_limit` defaults to 500."
                ),
            },
            "improve": {
                "usage": "!improve <suggestion>",
                "summary": "Log a human suggestion for improving the bot.",
                "details": (
                    "This writes your suggestion to the improvement log with your server display name and channel. "
                    "It is for manual review and future tuning, not an immediate behavior change."
                ),
            },
            "admin": {
                "usage": "!admin <status|profiles|profile|logs|mark|labels|memories|style|topics|dashboard|topic|channel|learn|lastaction|why|clear>",
                "summary": "Developer-only diagnostics and maintenance commands.",
                "details": (
                    "Only configured dev users can use it. It exposes internal status, profile/style/topic diagnostics, "
                    "quality labels, channel controls, learning jobs, recent action logs, and cleanup tools."
                ),
            },
        }

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
    async def help_command(self, ctx, command_name: str = ""):
        name = self.bot.user.display_name if self.bot.user else "bot"
        help_items = self._command_help()
        command_name = (command_name or "").strip().lower().lstrip("!")
        if command_name:
            item = help_items.get(command_name)
            if not item:
                await ctx.send(f"No help for `{command_name}`. Use `!help` to see commands.")
                return
            embed = discord.Embed(
                title=f"!{command_name}",
                description=item["summary"],
                color=discord.Color.blue(),
            )
            embed.add_field(name="Usage", value=f"`{item['usage']}`", inline=False)
            embed.add_field(name="Details", value=item["details"], inline=False)
            await ctx.send(embed=embed)
            return

        embed = discord.Embed(
            title=name,
            description="I'm the bot. I'm here to hang out.",
            color=discord.Color.blue(),
        )
        embed.add_field(name="How to talk to me", value=f"Mention me or @{name} in this server.", inline=False)
        commands_text = "\n".join(
            f"`!{command}` - {item['summary']}"
            for command, item in help_items.items()
        )
        embed.add_field(name="Commands", value=commands_text, inline=False)
        embed.add_field(name="Details", value="Use `!help <command>` for details, like `!help feedback`.", inline=False)
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
            f"Quality labels: {stats.get('quality_labels', 0)}",
        ]
        if self.feedback.lessons:
            lines.append(f"\nLessons ({len(self.feedback.lessons)}):")
            for l in self.feedback.lessons:
                lines.append(f"  ▸ {l}")
        await ctx.send("\n".join(lines))

    @staticmethod
    def _reaction_score(message) -> int:
        return sum(int(getattr(reaction, "count", 0) or 0) for reaction in getattr(message, "reactions", []) or [])

    @staticmethod
    def _looks_sensitive_highlight(content: str) -> bool:
        text = re.sub(r"\s+", " ", (content or "").lower()).strip()
        if not text:
            return True
        sensitive_terms = {
            "suicide", "kill myself", "kms", "self harm", "self-harm", "died", "death", "funeral",
            "cancer", "hospital", "surgery", "diagnosis", "depression", "anxiety", "panic attack",
            "abuse", "assault", "rape", "harassment", "dox", "doxx", "address", "phone number",
            "fired", "laid off", "breakup", "divorce", "police", "court", "lawsuit", "legal",
            "password", "token", "secret", "private", "confidential",
        }
        if any(term in text for term in sensitive_terms):
            return True
        serious_patterns = [
            r"\b(my|our|his|her|their) (mom|dad|mother|father|brother|sister|friend|dog|cat) (died|passed away)\b",
            r"\bnot funny\b",
            r"\bserious(ly)?\b",
        ]
        return any(re.search(pattern, text) for pattern in serious_patterns)

    def _highlight_candidate(self, message) -> Optional[dict]:
        content = (getattr(message, "content", "") or "").strip()
        if not content or self._is_command_message(content):
            return None
        author = getattr(message, "author", None)
        if getattr(author, "bot", False):
            return None
        score = self._reaction_score(message)
        if score <= 0 or self._looks_sensitive_highlight(content):
            return None
        return {
            "message": message,
            "score": score,
            "content": content,
            "author": self._display_name(author),
        }

    @commands.command(name="highlights")
    async def highlights(self, ctx, count: int = 5, scan_limit: int = 500):
        count = max(1, min(10, count))
        scan_limit = max(count, min(2000, scan_limit))
        candidates = []

        try:
            async for message in ctx.channel.history(limit=scan_limit):
                candidate = self._highlight_candidate(message)
                if candidate:
                    candidates.append(candidate)
        except discord.Forbidden:
            await ctx.send("I can't read enough channel history here.")
            return
        except discord.HTTPException:
            await ctx.send("Discord rejected the history lookup.")
            return

        if not candidates:
            await ctx.send("No good highlights found in recent visible history.")
            return

        candidates.sort(
            key=lambda item: (item["score"], str(getattr(item["message"], "created_at", ""))),
            reverse=True,
        )
        lines = [f"**Highlights from recent #{ctx.channel.name}**"]
        for item in candidates[:count]:
            message = item["message"]
            content = item["content"].replace("\n", " ")
            if len(content) > 180:
                content = content[:177].rstrip() + "..."
            jump_url = getattr(message, "jump_url", "")
            suffix = f" {jump_url}" if jump_url else ""
            lines.append(f"- {item['score']} reacts | {item['author']}: {content}{suffix}")
        await ctx.send("\n".join(lines)[:1900])

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
            cid = str(ctx.channel.id)
            style = self.style_guides.get_channel_style(cid) if self.style_guides else None
            topic_count = len(self.topic_log.get_candidate_topics(cid, limit=100)) if self.topic_log else 0
            queue_depth = self.learning_queue.qsize() if hasattr(self, "learning_queue") and self.learning_queue else 0
            msg = [
                "**Internal Status**",
                f"Messages tracked: {stats.get('total_tracked_messages', 0)}",
                f"Positive: {stats.get('total_positive', 0)} | Negative: {stats.get('total_negative', 0)}",
                f"User profiles: {len(self.profiles.get_all_profiles()) if self.profiles else 0}",
                f"Style learning: {getattr(self.config, 'style_learning_enabled', False)}",
                f"Topic learning: {getattr(self.config, 'topic_learning_enabled', False)}",
                f"Topic starters: {getattr(self.config, 'topic_starter_enabled', False)} @ {getattr(self.config, 'topic_starter_chance', 0)}",
                f"Channel style confidence: {style['confidence'] if style else 'none'}",
                f"Channel topics: {topic_count}",
                f"Learning queue depth: {queue_depth}",
                f"Channel controls: {self._format_controls(self.action_audit.get_channel_controls(cid)) if self.action_audit else 'default'}",
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

        elif action == "mark" and self.feedback:
            parts = ctx.message.content.split(maxsplit=4)
            allowed = {"good", "bad", "too-much", "too-friendly", "missed-opportunity", "wrong-tone"}
            if len(parts) < 4 or parts[3].lower() not in allowed:
                await ctx.send("Usage: `!admin mark <message_id> <good|bad|too-much|too-friendly|missed-opportunity|wrong-tone> [note]`")
                return
            message_id = parts[2]
            label = parts[3].lower()
            note = parts[4] if len(parts) > 4 else ""
            try:
                self.feedback.add_quality_label(message_id, label, str(ctx.author.id), note)
            except ValueError:
                await ctx.send("Unknown quality label.")
                return
            await ctx.send(f"Marked `{message_id}` as `{label}`.")

        elif action == "labels" and self.feedback:
            rows = self.feedback.store.get_quality_labels(limit=10)
            if not rows:
                await ctx.send("No quality labels yet.")
                return
            lines = ["**Quality Labels**"]
            for row in rows:
                note = f" | {row['note']}" if row.get("note") else ""
                lines.append(f"{row['created_at']} {row['message_id']} -> {row['label']}{note}")
            await ctx.send("\n".join(lines)[:1900])

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

        elif action == "style" and self.style_guides:
            cid = str(ctx.channel.id)
            style = self.style_guides.get_channel_style(cid)
            if not style:
                await ctx.send("No style guide for this channel yet.")
                return
            lines = [
                "**Channel Style**",
                f"Confidence: {style['confidence']}",
                f"Samples: {style['sample_count']}",
                f"Energy: {style['energy_level']}",
                f"Summary: {style['style_summary'] or '-'}",
            ]
            if style["do_patterns"]:
                lines.append("Do: " + "; ".join(style["do_patterns"]))
            if style["avoid_patterns"]:
                lines.append("Avoid: " + "; ".join(style["avoid_patterns"]))
            if style["common_phrases"]:
                lines.append("Phrases: " + "; ".join(style["common_phrases"]))
            if style["humor_notes"]:
                lines.append("Humor: " + style["humor_notes"])
            await ctx.send("\n".join(lines)[:1900])

        elif action == "topics" and self.topic_log:
            cid = str(ctx.channel.id)
            topics = self.topic_log.get_candidate_topics(cid, limit=10)
            if not topics:
                await ctx.send("No topics for this channel yet.")
                return
            lines = ["**Channel Topics**"]
            for t in topics:
                lines.append(
                    f"{t['id']}: {t['label']} | score {t['score']:.2f} | "
                    f"seen {t['seen_count']} | started {t['started_count']} | success {t['success_count']}"
                )
            await ctx.send("\n".join(lines)[:1900])

        elif action == "dashboard":
            cid = str(ctx.channel.id)
            controls = self.action_audit.get_channel_controls(cid) if self.action_audit else {}
            style = self.style_guides.get_channel_style(cid) if self.style_guides else None
            topics = self.topic_log.get_candidate_topics(cid, limit=5) if self.topic_log else []
            last = self.action_audit.get_last(cid) if self.action_audit else None
            topic_summary = ", ".join(
                f"{t['id']}:{t['label']}({t['score']:.1f})" for t in topics
            ) or "none"
            lines = [
                "**Channel Dashboard**",
                f"Controls: {self._format_controls(controls) if controls else 'default'}",
                f"Style confidence: {style['confidence'] if style else 'none'}",
                f"Top topics: {topic_summary}",
                f"Learning queue: {self.learning_queue.qsize() if self.learning_queue else 0}",
            ]
            if last:
                lines.append(f"Last action: {last['created_at']} {last['action_type']} - {last['reason']}")
            await ctx.send("\n".join(lines)[:1900])

        elif action == "topic" and self.topic_log:
            parts = ctx.message.content.split()
            if len(parts) >= 4 and parts[2] in {"boost", "mute", "unmute", "delete", "rename", "summary", "starter"}:
                sub = parts[2]
                try:
                    topic_id = int(parts[3])
                except ValueError:
                    await ctx.send("Topic ID must be a number.")
                    return
                if sub == "boost":
                    amount = 1.0
                    if len(parts) >= 5:
                        try:
                            amount = float(parts[4])
                        except ValueError:
                            pass
                    self.topic_log.boost_topic(topic_id, amount)
                    await ctx.send(f"Topic {topic_id} boosted by {amount}.")
                elif sub == "mute":
                    self.topic_log.set_topic_muted(topic_id, True)
                    await ctx.send(f"Topic {topic_id} muted.")
                elif sub == "unmute":
                    self.topic_log.set_topic_muted(topic_id, False)
                    await ctx.send(f"Topic {topic_id} unmuted.")
                elif sub == "delete":
                    self.topic_log.delete_topic(topic_id)
                    await ctx.send(f"Topic {topic_id} deleted.")
                elif sub == "rename":
                    text = " ".join(parts[4:]).strip()
                    self.topic_log.update_topic(topic_id, label=text)
                    await ctx.send(f"Topic {topic_id} renamed.")
                elif sub == "summary":
                    text = " ".join(parts[4:]).strip()
                    self.topic_log.update_topic(topic_id, summary=text)
                    await ctx.send(f"Topic {topic_id} summary updated.")
                elif sub == "starter":
                    text = " ".join(parts[4:]).strip()
                    self.topic_log.update_topic(topic_id, seed_prompt=text)
                    await ctx.send(f"Topic {topic_id} starter updated.")
                return
            if len(parts) < 3:
                await ctx.send("Usage: `!admin topic <id>`")
                return
            try:
                topic_id = int(parts[2])
            except ValueError:
                await ctx.send("Topic ID must be a number.")
                return
            topic = self.topic_log.get_topic(topic_id)
            if not topic:
                await ctx.send("No topic with that ID.")
                return
            lines = [
                f"**Topic {topic['id']}: {topic['label']}**",
                f"Score: {topic['score']:.2f}",
                f"Seen: {topic['seen_count']} | Started: {topic['started_count']} | Success: {topic['success_count']} | Muted: {bool(topic.get('muted'))}",
                f"Summary: {topic['summary']}",
                f"Starter: {topic['seed_prompt']}",
            ]
            await ctx.send("\n".join(lines)[:1900])

        elif action == "channel" and self.action_audit:
            parts = ctx.message.content.split()
            if len(parts) < 4:
                controls = self.action_audit.get_channel_controls(str(ctx.channel.id))
                await ctx.send("Channel controls: " + self._format_controls(controls))
                return
            control = parts[2].lower()
            value = parts[3].lower()
            aliases = {
                "learning": "learning",
                "style": "style",
                "topics": "topics",
                "starters": "starters",
                "spontaneous": "spontaneous",
                "quiet": "quiet",
                "tracking": "tracking",
            }
            if control not in aliases or value not in {"on", "off"}:
                await ctx.send("Usage: `!admin channel <learning|style|topics|starters|spontaneous|quiet|tracking> <on|off>`")
                return
            self.action_audit.set_channel_control(str(ctx.channel.id), aliases[control], value == "on")
            controls = self.action_audit.get_channel_controls(str(ctx.channel.id))
            await ctx.send("Channel controls: " + self._format_controls(controls))

        elif action == "learn":
            parts = ctx.message.content.split()
            target = parts[2] if len(parts) > 2 else "all"
            cid = str(ctx.channel.id)
            gid = str(ctx.guild.id) if ctx.guild else None
            if self.learning_queue:
                try:
                    self.learning_queue.put_nowait({
                        "channel_id": cid,
                        "guild_id": gid,
                        "style": target in {"style", "all"},
                        "topics": target in {"topics", "all"},
                    })
                    await ctx.send(f"Learning job queued for `{target}`.")
                    return
                except asyncio.QueueFull:
                    await ctx.send("Learning queue is full. Try again in a bit.")
                    return
            if target in {"style", "all"} and self.style_guides and self.memory:
                recent = self.memory.get_recent(cid, limit=getattr(self.config, "style_learning_context_limit", 80))
                learner = StyleGuideLearner(self.style_guides, interval=1, context_limit=getattr(self.config, "style_learning_context_limit", 80))
                await learner.learn_from_recent(cid, gid, recent, self.llm_client.generate)
            if target in {"topics", "all"} and self.topic_log and self.memory:
                recent = self.memory.get_recent(cid, limit=getattr(self.config, "topic_learning_context_limit", 60))
                learner = TopicLearner(self.topic_log, interval=1, context_limit=getattr(self.config, "topic_learning_context_limit", 60))
                await learner.learn_from_recent(cid, gid, recent, self.llm_client.generate)
            await ctx.send(f"Learning run completed for `{target}`.")

        elif action == "lastaction" and self.action_audit:
            rows = self.action_audit.get_recent(str(ctx.channel.id), limit=5)
            if not rows:
                await ctx.send("No autonomous actions logged for this channel.")
                return
            lines = ["**Recent Actions**"]
            for row in rows:
                bits = [f"{row['created_at']} {row['action_type']}"]
                if row.get("topic_id"):
                    bits.append(f"topic={row['topic_id']}")
                if row.get("probability") is not None:
                    bits.append(f"p={row['probability']:.3f}")
                if row.get("roll") is not None:
                    bits.append(f"roll={row['roll']:.3f}")
                if row.get("trigger_type"):
                    bits.append(f"trigger={row['trigger_type']}")
                if row.get("user_response_mode"):
                    bits.append(f"user={row['user_response_mode']}")
                if row.get("channel_mode"):
                    bits.append(f"channel={row['channel_mode']}")
                if row.get("final_action"):
                    bits.append(f"final={row['final_action']}")
                if row.get("reason"):
                    bits.append(row["reason"])
                lines.append(" | ".join(bits))
            await ctx.send("\n".join(lines)[:1900])

        elif action == "why" and self.action_audit:
            last = self.action_audit.get_last(str(ctx.channel.id))
            if not last:
                await ctx.send("No decision logged for this channel yet.")
                return
            bits = [
                f"Action: {last['action_type']}",
                f"Reason: {last['reason'] or '-'}",
                f"At: {last['created_at']}",
            ]
            if last.get("topic_id"):
                bits.append(f"Topic: {last['topic_id']}")
            if last.get("probability") is not None:
                bits.append(f"Probability: {last['probability']:.3f}")
            if last.get("roll") is not None:
                bits.append(f"Roll: {last['roll']:.3f}")
            if last.get("trigger_type"):
                bits.append(f"Trigger: {last['trigger_type']}")
            if last.get("user_response_mode"):
                bits.append(f"User mode: {last['user_response_mode']}")
            if last.get("channel_mode"):
                bits.append(f"Channel mode: {last['channel_mode']}")
            if last.get("final_action"):
                bits.append(f"Final action: {last['final_action']}")
            await ctx.send("\n".join(bits))

        elif action == "clear":
            sub = ctx.message.content.split()[-1] if len(ctx.message.content.split()) > 2 else ""
            if sub == "memory" and self.memory:
                deleted = self.memory.delete_channel_messages(str(ctx.channel.id))
                await ctx.send(f"Channel memory wiped ({deleted} messages).")
            elif sub == "profiles" and self.profiles:
                for p in self.profiles.get_all_profiles():
                    self.profiles.delete_profile(p["user_id"])
                await ctx.send("Profiles cleared.")
            elif sub == "style" and self.style_guides:
                self.style_guides.clear_channel_style(str(ctx.channel.id))
                await ctx.send("Channel style cleared.")
            elif sub == "topics" and self.topic_log:
                self.topic_log.clear_channel_topics(str(ctx.channel.id))
                await ctx.send("Channel topics cleared.")
            else:
                await ctx.send("Usage: `!admin clear memory`, `!admin clear profiles`, `!admin clear style`, or `!admin clear topics`")


class ControlCommands(commands.Cog):
    """Cross-server control plane for managing target server channels."""

    def __init__(self, bot, config, memory=None, style_guides=None, topic_log=None, action_audit=None, learning_queue=None):
        self.bot = bot
        self.config = config
        self.memory = memory
        self.style_guides = style_guides
        self.topic_log = topic_log
        self.action_audit = action_audit
        self.learning_queue = learning_queue

    def _is_control_admin(self, ctx) -> bool:
        if not self.config.control_guild_id or not self.config.control_channel_id:
            return False
        if not ctx.guild or str(ctx.guild.id) != self.config.control_guild_id:
            return False
        if str(ctx.channel.id) != self.config.control_channel_id:
            return False
        return str(ctx.author.id) in set(self.config.control_admin_ids)

    def _target_guild(self):
        if not self.config.target_guild_id:
            return None
        return self.bot.get_guild(int(self.config.target_guild_id))

    def _target_text_channels(self):
        guild = self._target_guild()
        if not guild:
            return []
        return [
            c for c in guild.channels
            if isinstance(c, (discord.TextChannel, discord.Thread))
        ]

    def _resolve_target(self, alias_or_id: str) -> Optional[str]:
        value = (alias_or_id or "").strip()
        if not value:
            return None
        if self.action_audit:
            resolved = self.action_audit.resolve_alias(value)
            if resolved:
                return resolved
        if value.startswith("<#") and value.endswith(">"):
            value = value[2:-1]
        if value.isdigit():
            return value

        wanted = value.lower().lstrip("#")
        matches = []
        for channel in self._target_text_channels():
            if channel.name.lower() == wanted:
                matches.append(channel)
        if len(matches) == 1:
            return str(matches[0].id)
        return None

    def _target_label(self, channel_id: str) -> str:
        channel = self.bot.get_channel(int(channel_id)) if channel_id.isdigit() else None
        if channel:
            return f"#{channel.name}"
        return channel_id

    @staticmethod
    def _resolve_user_id(value: str) -> Optional[str]:
        value = (value or "").strip()
        if value.startswith("<@") and value.endswith(">"):
            value = value[2:-1].lstrip("!")
        return value if value.isdigit() else None

    @staticmethod
    def _self_control_mode(action: str, target: str = "") -> Optional[str]:
        action = (action or "").strip().lower()
        target = (target or "").strip().lower()
        if action == "mute":
            return "strict"
        if action == "unmute":
            return "normal"
        if action in {"normal", "prompted", "strict"}:
            return action
        if action == "me" and target in {"normal", "prompted", "strict"}:
            return target
        return None

    @staticmethod
    def _self_mode_description(mode: str) -> str:
        descriptions = {
            "normal": "Normal: I can respond to your messages under the server's current settings.",
            "prompted": "Prompted: I will not jump into your messages unless you prompt me, including by name.",
            "strict": "Muted: I will only respond to your @mentions or Discord replies to me.",
        }
        return descriptions.get(mode, descriptions["normal"])

    def _self_control_usage(self) -> str:
        return (
            "Your controls: `!control status`, `!control mute`, `!control unmute`, "
            "`!control prompted`, `!control strict`, `!control normal`, "
            "`!control remember on|off`, or `!control forget me`."
        )

    async def _handle_self_control(self, ctx, action: str, target: str = "") -> None:
        if not self.action_audit:
            await ctx.send("Control settings are not initialized.")
            return

        action = (action or "status").strip().lower()
        user_id = str(ctx.author.id)

        if action in {"status", "me"} and not target:
            mode = self.action_audit.get_user_response_mode(user_id)
            privacy = self.action_audit.get_user_privacy(user_id)
            remember = "on" if privacy.get("remember_enabled", True) else "off"
            await ctx.send(
                f"{self._self_mode_description(mode)}\nRemember me: `{remember}`.\n{self._self_control_usage()}"
            )
            return

        if action == "remember":
            value = target.strip().lower()
            if value not in {"on", "off"}:
                await ctx.send("Usage: `!control remember <on|off>`")
                return
            self.action_audit.set_user_remember_enabled(user_id, value == "on")
            await ctx.send(f"Remember me set to `{value}`.")
            return

        if action == "privacy":
            privacy = self.action_audit.get_user_privacy(user_id)
            mode = self.action_audit.get_user_response_mode(user_id)
            await ctx.send(
                f"Response mode: `{mode}`. Remember me: `{'on' if privacy.get('remember_enabled', True) else 'off'}`."
            )
            return

        if action == "forget" and target.strip().lower() in {"me", "myself"}:
            deleted = self.memory.delete_user_messages(user_id) if self.memory else 0
            if self.profiles:
                self.profiles.delete_profile(user_id)
            self.action_audit.set_user_remember_enabled(user_id, False)
            await ctx.send(f"Forgot your stored messages ({deleted}) and disabled future remembering.")
            return

        mode = self._self_control_mode(action, target)
        if not mode:
            await ctx.send(self._self_control_usage())
            return

        self.action_audit.set_user_response_mode(user_id, mode)
        await ctx.send(f"Set your bot response mode. {self._self_mode_description(mode)}")

    def _format_controls(self, controls: dict) -> str:
        labels = {
            "learning_enabled": "learning",
            "style_enabled": "style",
            "topics_enabled": "topics",
            "starters_enabled": "starters",
            "spontaneous_enabled": "spontaneous",
            "quiet_enabled": "quiet",
            "tracking_enabled": "tracking",
        }
        parts = [f"{label}={'on' if controls[key] else 'off'}" for key, label in labels.items()]
        parts.insert(0, f"mode={controls.get('mode', 'normal')}")
        parts.append(f"spont-rate={float(controls.get('spontaneous_rate', 0.0)):.2g}x")
        return ", ".join(parts)

    async def _send_target_message(self, channel_id: str, text: str) -> bool:
        channel = self.bot.get_channel(int(channel_id))
        if not channel:
            try:
                channel = await self.bot.fetch_channel(int(channel_id))
            except Exception:
                return False
        await channel.send(text)
        return True

    async def _fetch_target_message(self, channel_id: str, message_id: str):
        channel = self.bot.get_channel(int(channel_id))
        if not channel:
            channel = await self.bot.fetch_channel(int(channel_id))
        return await channel.fetch_message(int(message_id))

    def _guild_dashboard_lines(self) -> List[str]:
        guild = self._target_guild()
        if not guild:
            return ["**Target Guild Dashboard**", "Target guild is not visible to this bot."]

        channels = self._target_text_channels()
        aliases = self.action_audit.list_aliases() if self.action_audit else []
        alias_by_channel = {item["channel_id"]: item["alias"] for item in aliases}
        lines = [
            f"**Target Guild Dashboard: {guild.name}**",
            f"Visible text channels: {len(channels)}",
            f"Aliases: {len(aliases)}",
        ]
        for channel in channels[:20]:
            cid = str(channel.id)
            controls = self.action_audit.get_channel_controls(cid) if self.action_audit else {}
            style = self.style_guides.get_channel_style(cid) if self.style_guides else None
            topics = self.topic_log.get_candidate_topics(cid, limit=3) if self.topic_log else []
            topic_count = self.topic_log.count_channel_topics(cid) if self.topic_log else 0
            status = []
            if controls.get("quiet_enabled"):
                status.append("quiet")
            if not controls.get("spontaneous_enabled", True):
                status.append("no-spont")
            if not controls.get("starters_enabled", True):
                status.append("no-starters")
            topic_names = ", ".join(t["label"] for t in topics) or "no topics"
            alias = f" `{alias_by_channel[cid]}`" if cid in alias_by_channel else ""
            lines.append(
                f"#{channel.name}{alias} | style={style['confidence'] if style else 'none'} | "
                f"{'/'.join(status) or 'active'} | topics={topic_count} | {topic_names}"
            )
        if len(channels) > 20:
            lines.append(f"...and {len(channels) - 20} more")
        return lines

    def _coverage_lines(self) -> List[str]:
        guild = self._target_guild()
        if not guild:
            return ["**Coverage**", "Target guild is not visible to this bot."]
        activity = {}
        if self.memory:
            activity = {
                item["channel_id"]: item
                for item in self.memory.get_channel_activity(str(guild.id), limit=200)
            }
        channels = self._target_text_channels()
        seen_count = sum(1 for c in channels if str(c.id) in activity)
        lines = [
            f"**Coverage: {guild.name}**",
            f"Target guild ID: {guild.id}",
            f"Visible text channels: {len(channels)}",
            f"Channels seen since current storage started: {seen_count}",
        ]
        member = guild.me
        for channel in channels[:40]:
            item = activity.get(str(channel.id))
            perms = channel.permissions_for(member) if member else None
            access = []
            if perms:
                access.append("view" if perms.view_channel else "no-view")
                access.append("read-history" if perms.read_message_history else "no-history")
                access.append("send" if perms.send_messages else "no-send")
            access_text = f" | {'/'.join(access)}" if access else ""
            if item:
                lines.append(f"#{channel.name} | seen={item['count']} | last={item['last_seen']}{access_text}")
            else:
                lines.append(f"#{channel.name} | seen=0 | last=never{access_text}")
        if len(channels) > 40:
            lines.append(f"...and {len(channels) - 40} more")
        return lines

    def _guilds_lines(self) -> List[str]:
        lines = ["**Connected Guilds**"]
        for guild in sorted(self.bot.guilds, key=lambda g: g.name.lower()):
            marker = []
            if str(guild.id) == self.config.control_guild_id:
                marker.append("control")
            if str(guild.id) == self.config.target_guild_id:
                marker.append("target")
            channels = [c for c in guild.channels if isinstance(c, (discord.TextChannel, discord.Thread))]
            suffix = f" ({', '.join(marker)})" if marker else ""
            lines.append(f"{guild.name} | id={guild.id}{suffix} | text_channels={len(channels)}")
        if self.config.target_guild_id and not any(str(g.id) == self.config.target_guild_id for g in self.bot.guilds):
            lines.append(f"Configured target guild {self.config.target_guild_id} is not in bot guild cache.")
        return lines

    def _activity_lines(self) -> List[str]:
        lines = ["**Stored Activity Across Guilds**"]
        if not self.memory:
            return lines + ["Memory is not initialized."]
        activity = self.memory.get_channel_activity(limit=50)
        if not activity:
            return lines + ["No messages stored yet."]
        for item in activity:
            channel = self.bot.get_channel(int(item["channel_id"])) if str(item["channel_id"]).isdigit() else None
            guild_name = channel.guild.name if channel and getattr(channel, "guild", None) else "unknown guild"
            channel_name = f"#{channel.name}" if channel else item["channel_id"]
            lines.append(f"{guild_name} / {channel_name} | seen={item['count']} | last={item['last_seen']}")
        return lines

    def _log_lines(self, count: int = 40) -> List[str]:
        count = max(5, min(80, count))
        if not BOT_LOG_PATH:
            return ["**Persistent Logs**", "BOT_LOG_PATH is not configured."]
        if not os.path.exists(BOT_LOG_PATH):
            return ["**Persistent Logs**", f"No log file at {BOT_LOG_PATH} yet."]
        with open(BOT_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-count:]
        trimmed = [line.rstrip()[-300:] for line in lines if line.strip()]
        return [f"**Persistent Logs: last {len(trimmed)}**", f"`{BOT_LOG_PATH}`"] + trimmed

    @commands.command(name="control")
    async def control(self, ctx, action: str = "status", target: str = "", *, rest: str = ""):
        if not self._is_control_admin(ctx):
            await self._handle_self_control(ctx, action, target)
            return
        if not self.action_audit:
            await ctx.send("Control plane storage is not initialized.")
            return

        action = action.lower()

        if action == "status":
            aliases = self.action_audit.list_aliases()
            guild = self._target_guild()
            lines = [
                "**Control Plane**",
                f"Target guild: {guild.name if guild else self.config.target_guild_id or 'not set'}",
                f"Visible target channels: {len(self._target_text_channels())}",
                f"Aliases: {len(aliases)}",
            ]
            for item in aliases[:10]:
                lines.append(f"{item['alias']} -> {item['channel_id']}")
            await ctx.send("\n".join(lines))
            return

        if action == "channels":
            channels = self._target_text_channels()
            if not channels:
                await ctx.send("No target channels visible.")
                return
            lines = ["**Target Channels**"]
            for channel in channels[:40]:
                lines.append(f"#{channel.name} -> {channel.id}")
            if len(channels) > 40:
                lines.append(f"...and {len(channels) - 40} more")
            await ctx.send("\n".join(lines)[:1900])
            return

        if action == "coverage":
            await ctx.send("\n".join(self._coverage_lines())[:1900])
            return

        if action == "guilds":
            await ctx.send("\n".join(self._guilds_lines())[:1900])
            return

        if action == "activity":
            await ctx.send("\n".join(self._activity_lines())[:1900])
            return

        if action == "logs":
            try:
                count = int(target or "40")
            except ValueError:
                count = 40
            await ctx.send("\n".join(self._log_lines(count))[:1900])
            return

        if action == "bind":
            parts = rest.split()
            if not target or not parts:
                await ctx.send("Usage: `!control bind <alias> <target_channel_id|#channel-name|channel-name>`")
                return
            channel_id = self._resolve_target(parts[0])
            if not channel_id:
                await ctx.send("Could not resolve target channel.")
                return
            self.action_audit.bind_alias(target, channel_id, self.config.target_guild_id or None)
            await ctx.send(f"Bound `{target.lower()}` to `{self._target_label(channel_id)}` (`{channel_id}`).")
            return

        if action == "unbind":
            if not target:
                await ctx.send("Usage: `!control unbind <alias>`")
                return
            self.action_audit.unbind_alias(target)
            await ctx.send(f"Unbound `{target.lower()}`.")
            return

        if action == "aliases":
            aliases = self.action_audit.list_aliases()
            if not aliases:
                await ctx.send("No aliases bound.")
                return
            lines = ["**Control Aliases**"]
            for item in aliases:
                lines.append(f"{item['alias']} -> {item['channel_id']}")
            await ctx.send("\n".join(lines)[:1900])
            return

        if action == "users":
            rows = self.action_audit.list_user_response_modes()
            if not rows:
                await ctx.send("No user response controls set.")
                return
            lines = ["**User Response Controls**"]
            for row in rows[:40]:
                lines.append(f"{row['user_id']} -> {row['mode']}")
            await ctx.send("\n".join(lines)[:1900])
            return

        if action in {"user", "user_mode", "user-response"}:
            user_id = self._resolve_user_id(target)
            mode = rest.strip().lower()
            if not user_id or mode not in {"normal", "prompted", "strict"}:
                await ctx.send("Usage: `!control user <@user|user_id> <normal|prompted|strict>`")
                return
            self.action_audit.set_user_response_mode(user_id, mode)
            await ctx.send(f"User `{user_id}` response mode set to `{mode}`.")
            return

        if action in {"privacy", "remember"}:
            user_id = self._resolve_user_id(target)
            value = rest.strip().lower()
            if not user_id or value not in {"on", "off"}:
                await ctx.send("Usage: `!control remember <@user|user_id> <on|off>`")
                return
            self.action_audit.set_user_remember_enabled(user_id, value == "on")
            await ctx.send(f"User `{user_id}` remember set to `{value}`.")
            return

        if action == "dashboard" and not target:
            await ctx.send("\n".join(self._guild_dashboard_lines())[:1900])
            return

        channel_id = self._resolve_target(target)
        if not channel_id:
            await ctx.send("Unknown target. Use a channel ID, channel name, #channel mention/name, or bound alias.")
            return

        if action in {"quiet", "learning", "style", "topics", "starters", "spontaneous", "tracking"}:
            value = rest.strip().lower()
            if value not in {"on", "off"}:
                await ctx.send(f"Usage: `!control {action} <target> <on|off>`")
                return
            self.action_audit.set_channel_control(channel_id, action, value == "on")
            controls = self.action_audit.get_channel_controls(channel_id)
            await ctx.send(f"`{target}` controls: {self._format_controls(controls)}")
            return

        if action in {"mode", "channel-mode", "channel_mode"}:
            mode = rest.strip().lower()
            if mode not in {"normal", "quiet", "observe-only", "ignore", "no-learning"}:
                await ctx.send("Usage: `!control mode <target> <normal|quiet|observe-only|ignore|no-learning>`")
                return
            self.action_audit.set_channel_mode(channel_id, mode)
            controls = self.action_audit.get_channel_controls(channel_id)
            await ctx.send(f"`{target}` controls: {self._format_controls(controls)}")
            return

        if action in {"spontaneous_rate", "spont-rate", "rate"}:
            value = rest.strip().lower()
            presets = {
                "off": 0.0,
                "none": 0.0,
                "verylow": 0.15,
                "very-low": 0.15,
                "low": 0.35,
                "medium": 0.65,
                "normal": 1.0,
                "high": 1.35,
            }
            try:
                rate = presets[value] if value in presets else float(value.rstrip("x"))
            except ValueError:
                await ctx.send("Usage: `!control spontaneous_rate <target> <off|very-low|low|medium|normal|high|0..2>`")
                return
            if rate < 0 or rate > 2:
                await ctx.send("Spontaneous rate must be between `0` and `2`.")
                return
            self.action_audit.set_spontaneous_rate(channel_id, rate)
            controls = self.action_audit.get_channel_controls(channel_id)
            await ctx.send(f"`{target}` controls: {self._format_controls(controls)}")
            return

        if action == "dashboard":
            controls = self.action_audit.get_channel_controls(channel_id)
            style = self.style_guides.get_channel_style(channel_id) if self.style_guides else None
            topics = self.topic_log.get_candidate_topics(channel_id, limit=5) if self.topic_log else []
            last = self.action_audit.get_last(channel_id)
            topic_summary = ", ".join(f"{t['id']}:{t['label']}({t['score']:.1f})" for t in topics) or "none"
            lines = [
                f"**Dashboard: {target}**",
                f"Channel: {self._target_label(channel_id)} (`{channel_id}`)",
                f"Controls: {self._format_controls(controls)}",
                f"Style confidence: {style['confidence'] if style else 'none'}",
                f"Top topics: {topic_summary}",
            ]
            if last:
                lines.append(f"Last action: {last['created_at']} {last['action_type']} - {last['reason']}")
            await ctx.send("\n".join(lines)[:1900])
            return

        if action == "seen":
            recent = self.memory.get_recent(channel_id, limit=5) if self.memory else []
            activity = self.memory.get_channel_activity(self.config.target_guild_id or None, limit=200) if self.memory else []
            summary = next((item for item in activity if item["channel_id"] == channel_id), None)
            lines = [
                f"**Seen: {self._target_label(channel_id)}**",
                f"Messages stored: {summary['count'] if summary else 0}",
                f"Last seen: {summary['last_seen'] if summary else 'never'}",
            ]
            if recent:
                lines.append("Recent:")
                for author, content, created_at in recent:
                    short = content[:80] + ("..." if len(content) > 80 else "")
                    lines.append(f"{created_at} {author}: {short}")
            await ctx.send("\n".join(lines)[:1900])
            return

        if action == "topics":
            if not self.topic_log:
                await ctx.send("Topic log is not initialized.")
                return
            topics = self.topic_log.get_candidate_topics(channel_id, limit=10)
            if not topics:
                await ctx.send("No topics for that target.")
                return
            lines = [f"**Topics: {target}**"]
            for t in topics:
                lines.append(
                    f"{t['id']}: {t['label']} | score {t['score']:.2f} | "
                    f"seen {t['seen_count']} | started {t['started_count']} | success {t['success_count']}"
                )
            await ctx.send("\n".join(lines)[:1900])
            return

        if action == "topic":
            parts = rest.split(maxsplit=2)
            if len(parts) < 2:
                await ctx.send("Usage: `!control topic <target> <mute|unmute|boost|delete|rename|summary|starter> <id> [text]`")
                return
            sub = parts[0].lower()
            try:
                topic_id = int(parts[1])
            except ValueError:
                await ctx.send("Topic ID must be numeric.")
                return
            text = parts[2] if len(parts) > 2 else ""
            if sub == "mute":
                self.topic_log.set_topic_muted(topic_id, True)
            elif sub == "unmute":
                self.topic_log.set_topic_muted(topic_id, False)
            elif sub == "boost":
                self.topic_log.boost_topic(topic_id, float(text or 1.0))
            elif sub == "delete":
                self.topic_log.delete_topic(topic_id)
            elif sub == "rename":
                self.topic_log.update_topic(topic_id, label=text)
            elif sub == "summary":
                self.topic_log.update_topic(topic_id, summary=text)
            elif sub == "starter":
                self.topic_log.update_topic(topic_id, seed_prompt=text)
            else:
                await ctx.send("Unknown topic action.")
                return
            await ctx.send(f"Topic {topic_id} updated.")
            return

        if action == "learn":
            if not self.learning_queue:
                await ctx.send("Learning queue is not initialized.")
                return
            target_kind = rest.strip().lower() or "all"
            try:
                self.learning_queue.put_nowait({
                    "channel_id": channel_id,
                    "guild_id": self.config.target_guild_id or None,
                    "style": target_kind in {"style", "all"},
                    "topics": target_kind in {"topics", "all"},
                })
            except asyncio.QueueFull:
                await ctx.send("Learning queue is full. Try again in a bit.")
                return
            await ctx.send(f"Queued `{target_kind}` learning for `{target}`.")
            return

        if action == "say":
            if not rest.strip():
                await ctx.send("Usage: `!control say <target> <message>`")
                return
            ok = await self._send_target_message(channel_id, rest.strip())
            if ok:
                self.action_audit.record(
                    channel_id=channel_id,
                    guild_id=self.config.target_guild_id or None,
                    action_type="control_say",
                    reason=f"sent by {ctx.author.id}",
                )
                await ctx.send("Sent.")
            else:
                await ctx.send("Could not find or send to that target channel.")
            return

        if action == "delete":
            message_id = rest.strip().split()[0] if rest.strip() else ""
            if not message_id.isdigit():
                await ctx.send("Usage: `!control delete <target> <message_id>`")
                return
            try:
                msg = await self._fetch_target_message(channel_id, message_id)
                await msg.delete()
            except discord.NotFound:
                await ctx.send("Target message was not found.")
                return
            except discord.Forbidden:
                await ctx.send("I do not have permission to delete that message.")
                return
            except discord.HTTPException as exc:
                await ctx.send(f"Discord rejected the delete: {exc}")
                return
            self.action_audit.record(
                channel_id=channel_id,
                guild_id=self.config.target_guild_id or None,
                action_type="control_delete",
                reason=f"message={message_id} by {ctx.author.id}",
                message_id=message_id,
            )
            await ctx.send("Deleted.")
            return

        if action == "react":
            parts = rest.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[0].isdigit():
                await ctx.send("Usage: `!control react <target> <message_id> <emoji>`")
                return
            message_id, emoji = parts[0], parts[1].strip()
            try:
                msg = await self._fetch_target_message(channel_id, message_id)
                await msg.add_reaction(emoji)
            except discord.NotFound:
                await ctx.send("Target message was not found.")
                return
            except discord.Forbidden:
                await ctx.send("I do not have permission to react to that message.")
                return
            except discord.HTTPException as exc:
                await ctx.send(f"Discord rejected the reaction: {exc}")
                return
            self.action_audit.record(
                channel_id=channel_id,
                guild_id=self.config.target_guild_id or None,
                action_type="control_react",
                reason=f"message={message_id} emoji={emoji} by {ctx.author.id}",
                message_id=message_id,
            )
            await ctx.send("Reacted.")
            return

        await ctx.send(
            "Usage: `!control status`, `channels`, `bind`, `aliases`, `dashboard [target]`, "
            "`quiet|learning|style|topics|starters|spontaneous|tracking <target> on|off`, "
            "`mode <target> <normal|quiet|observe-only|ignore|no-learning>`, "
            "`spontaneous_rate <target> <off|very-low|low|medium|normal|high|0..2>`, "
            "`user <@user|user_id> <normal|prompted|strict>`, `remember <@user|user_id> <on|off>`, `users`, "
            "`topics <target>`, `topic <target> ...`, `learn <target> [style|topics|all]`, "
            "`coverage`, `guilds`, `activity`, `logs [count]`, `seen <target>`, `say <target> <message>`, "
            "`delete <target> <message_id>`, `react <target> <message_id> <emoji>`"
        )


if __name__ == "__main__":
    config = BotConfig()
    bot = DiscordLLMBot(config)
    bot.run()

"""
Test suite for Discord LLM Bot.
"""

import pytest
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from config.config import BotConfig
from utils.llm import LLMClient, LLMConfig, ParsedResponse
from utils.feedback import FeedbackTracker
from utils.profiles import UserProfileStore
from utils.channel_state import ChannelState, ChannelStateStore
from utils.prompts import build_system_prompt
from utils.action_audit import ActionAuditStore
from utils.learning_safety import sanitize_learning_text
from utils.style_guide import StyleGuideStore
from utils.topic_log import TopicLearner, TopicLogStore
from utils.memory import MemoryStore
from bot.main import ControlCommands, DiscordLLMBot, MainCommands
from tests.fixtures.personality_regressions import PERSONALITY_REGRESSIONS


# ─── LLM Parsing ────────────────────────────────────────────────────────

class TestParsedResponse:
    """Test the [TAG] response parser."""

    def setup_method(self):
        config = LLMConfig(api_key="test")
        self.client = LLMClient(config)

    def test_parse_reply(self):
        r = self.client.parse_response("[REPLY] hey there")
        assert r.action == "REPLY"
        assert r.content == "hey there"
        assert r.reaction is None

    def test_parse_reply_with_supplemental_reaction(self):
        r = self.client.parse_response("[REPLY] yeah that tracks\n[REACT: 👍]")
        assert r.action == "REPLY"
        assert r.content == "yeah that tracks"
        assert r.reaction == "👍"

    def test_parse_react(self):
        r = self.client.parse_response("[REACT: 💀]")
        assert r.action == "REACT"
        assert r.content == "💀"

    def test_parse_react_custom_emoji(self):
        r = self.client.parse_response("[REACT: :pepega:]")
        assert r.action == "REACT"
        assert r.content == ":pepega:"

    def test_parse_image_gen(self):
        r = self.client.parse_response("[IMAGE_GEN: a cat riding a bicycle, cartoon style]")
        assert r.action == "IMAGE_GEN"
        assert "cat" in r.content

    def test_parse_image_analysis(self):
        r = self.client.parse_response("[IMAGE_ANALYSIS] This is a photo of a sunset")
        assert r.action == "IMAGE_ANALYSIS"
        assert r.content == "This is a photo of a sunset"

    def test_parse_silent(self):
        r = self.client.parse_response("[SILENT]")
        assert r.action == "SILENT"
        assert r.content == ""

    def test_parse_silent_with_reason(self):
        r = self.client.parse_response("[SILENT]\nNope")
        assert r.action == "SILENT"

    def test_parse_silent_after_leaked_commentary(self):
        r = self.client.parse_response("No clear reply needed.\n\n[SILENT]")
        assert r.action == "SILENT"
        assert r.content == ""

    def test_parse_fallback_no_tag(self):
        r = self.client.parse_response("just some text without a tag")
        assert r.action == "REPLY"
        assert "just some text" in r.content

    def test_parse_lowercase_tags(self):
        r = self.client.parse_response("[reply] hello")
        assert r.action == "REPLY"
        r = self.client.parse_response("[react: 👍]")
        assert r.action == "REACT"


class TestPersonalityFixtures:
    def test_personality_regression_fixture_is_structured(self):
        cases = PERSONALITY_REGRESSIONS

        assert len(cases) >= 3
        for case in cases:
            assert case["id"]
            assert case["conversation"]
            assert case["expected"]["preferred_action"] in {"SILENT", "REACT", "REPLY"}
            assert "avoid" in case["expected"]


class TestCommandHelp:
    def test_public_learn_command_is_not_advertised(self):
        commands = MainCommands(MagicMock(), MagicMock(), MagicMock())._command_help()

        assert "learn" not in commands

    def test_highlights_command_is_advertised(self):
        commands = MainCommands(MagicMock(), MagicMock(), MagicMock())._command_help()

        assert "highlights" in commands


class TestHighlights:
    def setup_method(self):
        config = MagicMock()
        config.command_prefix = "!"
        self.commands = MainCommands(MagicMock(), MagicMock(), MagicMock(), config=config)

    def _message(self, content, counts, *, bot=False):
        message = MagicMock()
        message.content = content
        message.author.bot = bot
        message.author.display_name = "Alice"
        message.reactions = [MagicMock(count=count) for count in counts]
        return message

    def test_highlight_candidate_scores_reactions(self):
        candidate = self.commands._highlight_candidate(self._message("that track goes hard", [2, 3]))

        assert candidate["score"] == 5
        assert candidate["content"] == "that track goes hard"

    def test_highlight_candidate_skips_commands_bots_and_sensitive_topics(self):
        assert self.commands._highlight_candidate(self._message("!help", [5])) is None
        assert self.commands._highlight_candidate(self._message("funny", [5], bot=True)) is None
        assert self.commands._highlight_candidate(self._message("my dad died", [10])) is None
        assert self.commands._highlight_candidate(self._message("no reacts", [])) is None


# ─── Probability Curve ─────────────────────────────────────────────────

class TestProbabilityCurve:
    """Test the spontaneous message counter probability curve."""

    def test_probability_rises(self):
        MESSAGE_TARGET = 18
        CHANCE_CAP = 0.30
        results = []
        for n in range(1, 60):
            scale = MESSAGE_TARGET * 1.5
            chance = min(CHANCE_CAP, n / scale)
            results.append(chance)

        # Should start near zero
        assert results[0] < 0.1
        # Should hit the configured 30% cap around n=9
        assert results[8] == CHANCE_CAP
        assert results[-1] == CHANCE_CAP
        # Should be strictly increasing until cap
        for i in range(len(results) - 1):
            if results[i] < CHANCE_CAP:
                assert results[i + 1] >= results[i]


# ─── Bandwagon Scaling ─────────────────────────────────────────────────

class TestBandwagonScaling:
    """Test the bandwagon probability scaling."""

    def test_linear_scaling(self):
        BANDWAGON_MAX = 0.65
        # 1 reactor out of 10 members = 6.5%
        assert abs(0.65 * (1 / 10) - 0.065) < 0.001
        # 5 reactors out of 10 = 32.5%
        assert abs(0.65 * (5 / 10) - 0.325) < 0.001
        # 10 reactors out of 10 = 65% (max)
        assert abs(0.65 * (10 / 10) - 0.65) < 0.001


class TestChannelState:
    """Test per-channel runtime state transitions."""

    def test_record_inbound(self):
        state = ChannelState()
        state.record_inbound()
        state.record_inbound()
        assert state.message_count == 2
        assert state.received_since_action == 2

    def test_direct_interaction_tracks_thread_depth(self):
        state = ChannelState()
        state.record_inbound()
        state.mark_direct_interaction()
        assert state.received_since_action == 0
        assert state.thread_depth == 1

    def test_decay_thread_depth_after_quiet_period(self):
        state = ChannelState(thread_depth=2, received_since_action=11)
        state.decay_thread_depth()
        assert state.thread_depth == 1

    def test_mark_bot_action_resets_counters(self):
        state = ChannelState(message_count=10, received_since_action=20, thread_depth=1)
        state.mark_bot_action(now=123.0)
        assert state.message_count == 0
        assert state.received_since_action == 0
        assert state.thread_depth == 2
        assert state.last_action_time == 123.0

    def test_store_reuses_channel_state(self):
        store = ChannelStateStore()
        state = store.get("123")
        state.record_inbound()
        assert store.get("123").message_count == 1

    def test_store_persists_channel_state(self, tmp_path):
        path = str(tmp_path / "channel_state.db")
        store = ChannelStateStore(path)
        store.record_inbound("123")
        store.record_inbound("123")
        store.mark_bot_action("456", now=100.0)

        reloaded = ChannelStateStore(path)

        assert reloaded.get("123").message_count == 2
        assert reloaded.get("123").received_since_action == 2
        assert reloaded.get("456").last_action_time == 100.0


class TestBotNameDetection:
    def setup_method(self):
        self.bot = object.__new__(DiscordLLMBot)
        user = MagicMock()
        user.display_name = "testbot"
        user.global_name = None
        user.name = "testbot"
        self.bot.bot = MagicMock()
        self.bot.bot.user = user

    def test_detects_exact_name_case_insensitive(self):
        assert self.bot._message_names_bot("hey TESTBOT what do you think")

    def test_ignores_spaced_near_match(self):
        assert not self.bot._message_names_bot("test bot get in here")

    def test_ignores_small_typo(self):
        assert not self.bot._message_names_bot("tesbot you seeing this")

    def test_detects_exact_name_with_punctuation_boundary(self):
        assert self.bot._message_names_bot("testbot, you seeing this")

    def test_ignores_unrelated_text(self):
        assert not self.bot._message_names_bot("the channel is quiet today")

    def test_server_display_name_prefers_nickname(self):
        user = MagicMock()
        user.display_name = "Server Nick"
        user.name = "account_name"
        assert DiscordLLMBot._server_display_name(user) == "Server Nick"


class TestCustomEmoji:
    def setup_method(self):
        self.bot = object.__new__(DiscordLLMBot)
        emoji = MagicMock()
        emoji.name = "thonk"
        guild = MagicMock()
        guild.emojis = [emoji]
        self.bot.bot = MagicMock()
        self.bot.bot.guilds = [guild]
        self.guild = guild
        self.emoji = emoji

    def test_guild_emoji_prompt_context_lists_colon_names(self):
        context = self.bot._guild_emoji_prompt_context()

        assert ":thonk:" in context
        assert "Server custom reactions" in context

    def test_resolves_colon_custom_emoji_case_insensitive(self):
        assert self.bot._resolve_custom_emoji(self.guild, ":ThOnK:") is self.emoji

    def test_plain_unicode_is_not_custom_emoji(self):
        assert self.bot._resolve_custom_emoji(self.guild, "💀") is None


class TestResponseSending:
    class FakeHistory:
        def __init__(self, messages):
            self.messages = messages

        def __aiter__(self):
            self._iter = iter(self.messages)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    class FakeChannel:
        def __init__(self, newer_messages=None):
            self.id = "c1"
            self.newer_messages = newer_messages or []
            self.sent = []

        def history(self, **kwargs):
            return TestResponseSending.FakeHistory(self.newer_messages)

        async def send(self, content, **kwargs):
            self.sent.append((content, kwargs))
            sent = MagicMock()
            sent.id = 999
            return sent

    def setup_method(self):
        self.bot = object.__new__(DiscordLLMBot)
        self.bot.feedback = MagicMock()
        self.bot.bot_message_map = {}
        self.bot.active_followups = {}
        self.bot.config = MagicMock()
        self.bot.config.followup_window_messages = 4
        self.bot.config.followup_window_seconds = 300
        self.bot.config.dry_run_actions = False
        self.bot.bot = MagicMock()
        self.bot.bot.user = MagicMock()
        self.bot.bot.user.id = 42
        self.trigger = MagicMock()
        self.trigger.id = 123
        self.trigger.guild = MagicMock()
        self.trigger.created_at = datetime.now(timezone.utc)

    async def _send_with_newer_messages(self, newer_messages):
        self.trigger.channel = self.FakeChannel(newer_messages)
        await self.bot._send_tracked_response(self.trigger, "hello")
        return self.trigger.channel.sent[0][1]

    @pytest.mark.asyncio
    async def test_plain_send_when_no_new_messages(self):
        kwargs = await self._send_with_newer_messages([])
        assert "reference" not in kwargs

    @pytest.mark.asyncio
    async def test_threaded_reply_when_chat_moved_on(self):
        newer = MagicMock()
        newer.id = 456
        newer.author.id = 7
        kwargs = await self._send_with_newer_messages([newer])
        assert kwargs["reference"] is self.trigger

    def test_send_response_activates_followup_window(self):
        self.bot._activate_followup_window("c1", "m1", "hello")

        assert self.bot.active_followups["c1"]["message_id"] == "m1"
        assert self.bot.active_followups["c1"]["remaining"] == 4

    @pytest.mark.asyncio
    async def test_implicit_followup_probability_can_fade_out(self):
        self.bot.active_followups = {
            "c1": {
                "message_id": "m1",
                "content": "hello",
                "remaining": 1,
                "expires_at": 9999999999,
            }
        }
        self.bot.channel_states = MagicMock()
        self.bot.action_audit = None
        self.bot._generate_and_execute = AsyncMock()
        message = MagicMock()
        message.channel.id = "c1"
        message.channel.__class__ = MagicMock()
        message.content = "yeah"
        message.author.display_name = "Alice"

        with patch("bot.main.random.random", return_value=0.99):
            handled = await self.bot._maybe_handle_implicit_followup(message)

        assert handled is False
        self.bot._generate_and_execute.assert_not_called()
        assert "c1" not in self.bot.active_followups

    @pytest.mark.asyncio
    async def test_low_signal_followup_ends_window(self):
        self.bot.active_followups = {
            "c1": {
                "message_id": "m1",
                "content": "hello",
                "remaining": 4,
                "expires_at": 9999999999,
            }
        }
        self.bot.channel_states = MagicMock()
        self.bot.action_audit = None
        self.bot._generate_and_execute = AsyncMock()
        message = MagicMock()
        message.channel.id = "c1"
        message.channel.__class__ = MagicMock()
        message.content = "ok bro"

        handled = await self.bot._maybe_handle_implicit_followup(message)

        assert handled is False
        self.bot._generate_and_execute.assert_not_called()
        assert "c1" not in self.bot.active_followups

    def test_low_signal_followup_detection(self):
        assert DiscordLLMBot._is_low_signal_followup("ok bro")
        assert DiscordLLMBot._is_low_signal_followup("oh nvm")
        assert not DiscordLLMBot._is_low_signal_followup("wait what do you mean")

    def test_command_message_detection(self):
        self.bot.config.command_prefix = "!"

        assert self.bot._is_command_message("!help")
        assert self.bot._is_command_message("   !control status")
        assert self.bot._is_command_message("/control mute")
        assert not self.bot._is_command_message("testbot help")
        assert not self.bot._is_command_message("that was !weird")

    @pytest.mark.asyncio
    async def test_custom_trigger_sends_stored_response(self, tmp_path):
        self.bot.profiles = UserProfileStore(str(tmp_path / "profiles.db"))
        self.bot.profiles.set_trigger("good bot", "thanks boss", guild_id="g1")
        self.bot.action_audit = None
        self.bot._send_custom_trigger_response = AsyncMock()
        message = MagicMock()
        message.content = "GOOD BOT"
        message.channel.id = "c1"

        handled = await self.bot._maybe_apply_custom_trigger(message, "g1")

        assert handled is True
        self.bot._send_custom_trigger_response.assert_awaited_once_with(message, "thanks boss")

    def test_legacy_image_response_url_strips_groupme_image_flag(self):
        url = "https://i.groupme.com/600x600.jpeg.example -i"

        assert DiscordLLMBot._legacy_image_response_url(url) == "https://i.groupme.com/600x600.jpeg.example"

    def test_legacy_image_response_url_ignores_non_groupme_or_plain_url(self):
        assert DiscordLLMBot._legacy_image_response_url("https://example.com/a.jpg -i") is None
        assert DiscordLLMBot._legacy_image_response_url("https://i.groupme.com/a.jpg") is None

    def test_legacy_video_response_url_strips_groupme_video_flag(self):
        url = "https://v.groupme.com/15629174/2022-02-04T21:14:46Z/70601751.640x1138r.mp4 -v"

        assert DiscordLLMBot._legacy_video_response_url(url) == (
            "https://v.groupme.com/15629174/2022-02-04T21:14:46Z/70601751.640x1138r.mp4"
        )

    def test_legacy_video_response_url_ignores_non_groupme_or_plain_url(self):
        assert DiscordLLMBot._legacy_video_response_url("https://example.com/a.mp4 -v") is None
        assert DiscordLLMBot._legacy_video_response_url("https://v.groupme.com/a.mp4") is None

    @pytest.mark.asyncio
    async def test_custom_trigger_embeds_legacy_groupme_image(self):
        self.trigger.channel = self.FakeChannel()
        self.bot._download_external_file = AsyncMock(return_value=(b"image-bytes", "image/jpeg"))
        self.bot._channel_has_newer_user_message = AsyncMock(return_value=False)

        await self.bot._send_custom_trigger_response(
            self.trigger,
            "https://i.groupme.com/600x600.jpeg.example -i",
        )

        content, kwargs = self.trigger.channel.sent[0]
        assert content == ""
        assert "file" in kwargs
        assert "embed" in kwargs
        assert kwargs["embed"].image.url.startswith("attachment://trigger-image")

    @pytest.mark.asyncio
    async def test_custom_trigger_sends_clean_legacy_groupme_video_url(self):
        self.bot._send_tracked_response = AsyncMock()
        message = MagicMock()

        await self.bot._send_custom_trigger_response(
            message,
            "https://v.groupme.com/15629174/2022-02-04T21:14:46Z/70601751.640x1138r.mp4 -v",
        )

        self.bot._send_tracked_response.assert_awaited_once_with(
            message,
            "https://v.groupme.com/15629174/2022-02-04T21:14:46Z/70601751.640x1138r.mp4",
        )


class TestMemoryStore:
    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        self.store = MemoryStore(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_channel_activity_counts_messages(self):
        self.store.add_message("c1", "g1", "alice", "u1", "hello")
        self.store.add_message("c1", "g1", "bob", "u2", "hi")
        self.store.add_message("c2", "g1", "alice", "u1", "elsewhere")

        activity = self.store.get_channel_activity("g1")
        counts = {item["channel_id"]: item["count"] for item in activity}

        assert counts["c1"] == 2
        assert counts["c2"] == 1

    def test_delete_user_messages(self):
        self.store.add_message("c1", "g1", "alice", "u1", "hello")
        self.store.add_message("c1", "g1", "bob", "u2", "hi")
        self.store.add_message("c2", "g1", "alice", "u1", "elsewhere")

        deleted = self.store.delete_user_messages("u1")

        assert deleted == 2
        assert self.store.get_user_history("u1") == []
        assert len(self.store.get_user_history("u2")) == 1

    def test_delete_channel_messages(self):
        self.store.add_message("c1", "g1", "alice", "u1", "hello")
        self.store.add_message("c1", "g1", "bob", "u2", "hi")
        self.store.add_message("c2", "g1", "alice", "u1", "elsewhere")

        deleted = self.store.delete_channel_messages("c1")

        assert deleted == 2
        assert self.store.get_recent("c1") == []
        assert len(self.store.get_recent("c2")) == 1


class TestPromptBuilder:
    """Test system prompt assembly without Discord objects."""

    def test_replaces_bot_name_placeholder(self):
        prompt = build_system_prompt("marvin", base_prompt="You are {name}.")
        assert prompt == "You are marvin."

    def test_adds_profile_context(self, tmp_path):
        store = UserProfileStore(str(tmp_path / "profiles.db"))
        store.upsert_user("123", "alice")
        store.add_fact("123", "likes coding", "alice")

        prompt = build_system_prompt("bot", base_prompt="You are {name}.", profiles=store)

        assert "## People you know" in prompt
        assert "alice" in prompt
        assert "likes coding" in prompt

    def test_adds_feedback_context(self):
        feedback = MagicMock()
        feedback.get_feedback_context.return_value = "Users like concise replies."

        prompt = build_system_prompt("bot", base_prompt="You are {name}.", feedback=feedback)

        assert "## Feedback on your behavior" in prompt
        assert "Users like concise replies." in prompt

    def test_adds_style_context(self):
        prompt = build_system_prompt(
            "bot",
            base_prompt="You are {name}.",
            style_context="Keep it short and dry.",
        )

        assert "## Local channel style" in prompt
        assert "Keep it short and dry." in prompt


class TestStyleGuideStore:
    """Test channel style storage and prompt formatting."""

    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        self.store = StyleGuideStore(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_missing_channel(self):
        assert self.store.get_channel_style("missing") is None

    def test_prompt_context_requires_confidence(self):
        self.store.upsert_channel_style("c1", "g1", {
            "style_summary": "quiet and direct",
            "confidence": 0.2,
        })

        assert self.store.get_prompt_context("c1") == ""

    def test_prompt_context_includes_high_confidence_style(self):
        self.store.upsert_channel_style("c1", "g1", {
            "style_summary": "quiet and direct",
            "do_patterns": ["short replies"],
            "avoid_patterns": ["big speeches"],
            "common_phrases": ["fair"],
            "humor_notes": "dry",
            "energy_level": "low",
            "confidence": 0.8,
            "sample_count": 30,
        })

        context = self.store.get_prompt_context("c1")

        assert "quiet and direct" in context
        assert "short replies" in context
        assert "big speeches" in context

    def test_truncates_overlong_summary(self):
        self.store.upsert_channel_style("c1", "g1", {
            "style_summary": "x" * 1000,
            "confidence": 1.0,
        })

        style = self.store.get_channel_style("c1")
        assert len(style["style_summary"]) == 600


class TestTopicLogStore:
    """Test recurring topic storage and scoring."""

    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        self.store = TopicLogStore(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_upsert_topic_creates_and_updates_seen_count(self):
        topic = {"label": "games", "summary": "game chat", "seed_prompt": "ask about games", "score": 1.0}
        self.store.upsert_topics("c1", "g1", [topic])
        self.store.upsert_topics("c1", "g1", [topic])

        topics = self.store.get_candidate_topics("c1")

        assert len(topics) == 1
        assert self.store.count_channel_topics("c1") == 1
        assert topics[0]["seen_count"] == 2
        assert topics[0]["score"] == 1.5

    def test_mark_started_and_success_adjust_scores(self):
        self.store.upsert_topics("c1", "g1", [
            {"label": "music", "summary": "music chat", "seed_prompt": "ask about music", "score": 2.0}
        ])
        topic_id = self.store.get_candidate_topics("c1")[0]["id"]

        self.store.mark_started(topic_id)
        after_start = self.store.get_topic(topic_id)
        self.store.mark_success(topic_id)
        after_success = self.store.get_topic(topic_id)

        assert after_start["started_count"] == 1
        assert after_start["score"] == 1.25
        assert after_success["success_count"] == 1
        assert after_success["score"] == 2.25

    def test_topic_learner_avoids_recently_started_topic(self):
        self.store.upsert_topics("c1", "g1", [
            {"label": "music", "summary": "music chat", "seed_prompt": "ask about music", "score": 2.0}
        ])
        topic_id = self.store.get_candidate_topics("c1")[0]["id"]
        self.store.mark_started(topic_id)
        learner = TopicLearner(self.store, starter_cooldown_seconds=7200)

        assert learner.choose_starter_topic("c1") is None

    def test_muted_topics_are_not_candidates(self):
        self.store.upsert_topics("c1", "g1", [
            {"label": "music", "summary": "music chat", "seed_prompt": "ask about music", "score": 2.0}
        ])
        topic_id = self.store.get_candidate_topics("c1")[0]["id"]

        self.store.set_topic_muted(topic_id, True)

        assert self.store.get_candidate_topics("c1") == []

    def test_boost_and_delete_topic(self):
        self.store.upsert_topics("c1", "g1", [
            {"label": "music", "summary": "music chat", "seed_prompt": "ask about music", "score": 2.0}
        ])
        topic_id = self.store.get_candidate_topics("c1")[0]["id"]

        self.store.boost_topic(topic_id, 2.5)
        assert self.store.get_topic(topic_id)["score"] == 4.5

        self.store.delete_topic(topic_id)
        assert self.store.get_topic(topic_id) is None

    def test_update_topic_fields_and_mark_ignored(self):
        self.store.upsert_topics("c1", "g1", [
            {"label": "music", "summary": "music chat", "seed_prompt": "ask about music", "score": 2.0}
        ])
        topic_id = self.store.get_candidate_topics("c1")[0]["id"]

        self.store.update_topic(topic_id, label="albums", summary="album talk", seed_prompt="favorite album?")
        self.store.mark_ignored(topic_id)
        topic = self.store.get_topic(topic_id)

        assert topic["label"] == "albums"
        assert topic["summary"] == "album talk"
        assert topic["seed_prompt"] == "favorite album?"
        assert topic["score"] == 1.5


class TestLearningSafety:
    def test_sanitize_redacts_identifiers(self):
        text = sanitize_learning_text(
            "hi <@123> email me at a@example.com https://example.com abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        )

        assert "<@123>" not in text
        assert "a@example.com" not in text
        assert "https://example.com" not in text
        assert "[mention]" in text
        assert "[email]" in text
        assert "[url]" in text


class TestActionAuditStore:
    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        self.store = ActionAuditStore(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_records_recent_action(self):
        self.store.record(
            guild_id="g1",
            channel_id="c1",
            action_type="topic_starter",
            reason="selected topic",
            topic_id=12,
            probability=0.08,
            roll=0.01,
            message_id="99",
        )

        rows = self.store.get_recent("c1")

        assert len(rows) == 1
        assert rows[0]["action_type"] == "topic_starter"
        assert rows[0]["topic_id"] == 12

    def test_records_decision_metadata(self):
        self.store.record(
            guild_id="g1",
            channel_id="c1",
            action_type="skip",
            reason="probability failed",
            trigger_type="spontaneous",
            user_response_mode="normal",
            channel_mode="quiet",
            final_action="silent",
        )

        row = self.store.get_last("c1")

        assert row["trigger_type"] == "spontaneous"
        assert row["user_response_mode"] == "normal"
        assert row["channel_mode"] == "quiet"
        assert row["final_action"] == "silent"

    def test_channel_controls_defaults_and_updates(self):
        defaults = self.store.get_channel_controls("c1")
        assert defaults["learning_enabled"] is True
        assert defaults["quiet_enabled"] is False
        assert defaults["tracking_enabled"] is True
        assert defaults["spontaneous_rate"] == 0.0
        assert defaults["mode"] == "normal"

        self.store.set_channel_control("c1", "quiet", True)
        self.store.set_channel_control("c1", "starters", False)
        self.store.set_channel_control("c1", "tracking", False)
        self.store.set_spontaneous_rate("c1", 0.35)
        controls = self.store.get_channel_controls("c1")

        assert controls["quiet_enabled"] is True
        assert controls["starters_enabled"] is False
        assert controls["tracking_enabled"] is False
        assert controls["spontaneous_rate"] == 0.35

    def test_channel_mode_presets(self):
        self.store.set_channel_mode("c1", "ignore")
        controls = self.store.get_channel_controls("c1")

        assert controls["mode"] == "ignore"
        assert controls["tracking_enabled"] is False
        assert controls["learning_enabled"] is False
        assert controls["spontaneous_enabled"] is False

        self.store.set_channel_mode("c1", "no-learning")
        controls = self.store.get_channel_controls("c1")

        assert controls["mode"] == "no-learning"
        assert controls["tracking_enabled"] is True
        assert controls["learning_enabled"] is False
        assert controls["spontaneous_enabled"] is True

    def test_channel_controls_migrates_added_columns(self, tmp_path):
        path = str(tmp_path / "actions.db")
        first = ActionAuditStore(path)
        with sqlite3.connect(path) as conn:
            conn.execute("ALTER TABLE channel_controls DROP COLUMN tracking_enabled")
            conn.execute("ALTER TABLE channel_controls DROP COLUMN spontaneous_rate")
            conn.execute("ALTER TABLE channel_controls DROP COLUMN mode")
            conn.commit()

        migrated = ActionAuditStore(path)

        assert migrated.get_channel_controls("c1")["tracking_enabled"] is True
        assert migrated.get_channel_controls("c1")["spontaneous_rate"] == 0.0
        assert migrated.get_channel_controls("c1")["mode"] == "normal"

    def test_spontaneous_rate_validates_range(self):
        with pytest.raises(ValueError):
            self.store.set_spontaneous_rate("c1", 2.1)

    def test_user_response_modes(self):
        assert self.store.get_user_response_mode("u1") == "normal"

        self.store.set_user_response_mode("u1", "prompted")
        self.store.set_user_response_mode("u2", "strict")

        assert self.store.get_user_response_mode("u1") == "prompted"
        assert self.store.get_user_response_mode("u2") == "strict"
        assert {row["user_id"]: row["mode"] for row in self.store.list_user_response_modes()} == {
            "u1": "prompted",
            "u2": "strict",
        }

    def test_user_response_mode_validates(self):
        with pytest.raises(ValueError):
            self.store.set_user_response_mode("u1", "weird")

    def test_user_privacy_controls(self):
        assert self.store.get_user_privacy("u1")["remember_enabled"] is True

        self.store.set_user_remember_enabled("u1", False)

        assert self.store.get_user_privacy("u1")["remember_enabled"] is False

    def test_self_control_mode_aliases(self):
        assert ControlCommands._self_control_mode("mute") == "strict"
        assert ControlCommands._self_control_mode("unmute") == "normal"
        assert ControlCommands._self_control_mode("prompted") == "prompted"
        assert ControlCommands._self_control_mode("strict") == "strict"
        assert ControlCommands._self_control_mode("me", "normal") == "normal"
        assert ControlCommands._self_control_mode("me", "weird") is None

    def test_control_aliases_bind_resolve_and_unbind(self):
        self.store.bind_alias("main-general", "123", "guild")

        assert self.store.resolve_alias("main-general") == "123"
        assert self.store.resolve_alias("123") == "123"
        assert self.store.list_aliases()[0]["alias"] == "main-general"

        self.store.unbind_alias("main-general")
        assert self.store.resolve_alias("main-general") is None


# ─── Feedback Tracker ──────────────────────────────────────────────────

class TestFeedbackTracker:
    """Test reaction classification and lesson derivation."""

    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".json")
        self.tracker = FeedbackTracker(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_positive_reaction(self):
        self.tracker.register_message("1", "test message")
        self.tracker.add_reaction("1", "💀", "user1")
        assert self.tracker.total_positive == 1

    def test_negative_reaction(self):
        self.tracker.register_message("1", "test message")
        self.tracker.add_reaction("1", "👎", "user1")
        assert self.tracker.total_negative == 1

    def test_sentiment_calculation(self):
        self.tracker.register_message("1", "msg")
        self.tracker.add_reaction("1", "🔥", "u1")
        self.tracker.add_reaction("1", "💀", "u2")
        assert self.tracker.compute_sentiment("1") == 1.0

    def test_no_duplicate_reactions(self):
        self.tracker.register_message("1", "msg")
        self.tracker.add_reaction("1", "👍", "u1")
        self.tracker.add_reaction("1", "👍", "u1")  # same user
        assert self.tracker.total_positive == 1

    def test_persistence(self):
        self.tracker.register_message("1", "msg")
        self.tracker.add_reaction("1", "👍", "u1")
        tracker2 = FeedbackTracker(self.path)
        assert tracker2.total_positive == 1


# ─── User Profile Store ────────────────────────────────────────────────

    def test_quality_labels(self):
        self.tracker.register_message("1", "too eager")
        self.tracker.add_quality_label("1", "too-friendly", "admin", "sounds forced")

        rows = self.tracker.store.get_quality_labels()

        assert rows[0]["message_id"] == "1"
        assert rows[0]["label"] == "too-friendly"
        assert rows[0]["note"] == "sounds forced"
        assert self.tracker.get_stats()["quality_labels"] == 1


class TestUserProfileStore:
    """Test CRUD operations on user profiles."""

    def setup_method(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".db")
        self.store = UserProfileStore(self.path)

    def teardown_method(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_upsert_user(self):
        self.store.upsert_user("123", "alice")
        p = self.store.get_profile("123")
        assert p is not None
        assert p["display_name"] == "alice"
        assert p["message_count"] == 1

    def test_upsert_increments_count(self):
        self.store.upsert_user("123", "alice")
        self.store.upsert_user("123", "alice")
        p = self.store.get_profile("123")
        assert p["message_count"] == 2

    def test_add_fact_deduplication(self):
        self.store.upsert_user("123", "alice")
        self.store.add_fact("123", "likes hiking")
        self.store.add_fact("123", "likes hiking")  # should not duplicate
        facts = self.store.get_facts("123")
        assert facts.count("likes hiking") == 1

    def test_add_preference(self):
        self.store.upsert_user("123", "alice")
        self.store.add_preference("123", "like", "pizza")
        self.store.add_preference("123", "dislike", "pineapple")
        profile = self.store.get_profile("123")
        assert "pizza" in json.loads(profile["likes"])
        assert "pineapple" in json.loads(profile["dislikes"])

    def test_get_profile_missing(self):
        assert self.store.get_profile("999") is None

    def test_get_all_summaries(self):
        self.store.upsert_user("123", "alice")
        self.store.upsert_user("456", "bob")
        self.store.add_fact("123", "likes coding")
        summary = self.store.get_all_summaries()
        assert "alice" in summary
        assert "coding" in summary

    def test_delete_profile(self):
        self.store.upsert_user("123", "alice")
        self.store.delete_profile("123")
        assert self.store.get_profile("123") is None

    def test_reset_profile(self):
        self.store.upsert_user("123", "alice")
        self.store.add_fact("123", "likes coding")
        self.store.reset_profile("123")
        facts = self.store.get_facts("123")
        assert len(facts) == 0

    def test_custom_trigger_matches_exact_message_case_insensitive(self):
        self.store.set_trigger("Good Bot", "👍", guild_id="g1", set_by="u1")

        match = self.store.find_trigger_match("GOOD   BOT", guild_id="g1")

        assert match["trigger_text"] == "Good Bot"
        assert match["reaction"] == "👍"

    def test_custom_trigger_does_not_match_inside_longer_message(self):
        self.store.set_trigger("good bot", "👍", guild_id="g1")

        assert self.store.find_trigger_match("that was a good bot moment", guild_id="g1") is None

    def test_custom_trigger_forget_is_scoped_to_guild(self):
        self.store.set_trigger("good bot", "👍", guild_id="g1")
        self.store.set_trigger("good bot", "💀", guild_id="g2")

        assert self.store.forget_trigger("GOOD BOT", guild_id="g1") is True

        assert self.store.find_trigger_match("good bot", guild_id="g1") is None
        assert self.store.find_trigger_match("good bot", guild_id="g2")["reaction"] == "💀"

    def test_import_triggers_csv(self, tmp_path):
        path = tmp_path / "triggers.csv"
        path.write_text("trigger,response,set_by\nhello,👋,seed\n", encoding="utf-8")

        imported = self.store.import_triggers_csv(str(path), guild_id="g1")

        assert imported == 1
        assert self.store.find_trigger_match("HELLO", guild_id="g1")["reaction"] == "👋"

    def test_import_triggers_csv_once_does_not_overwrite_later_changes(self, tmp_path):
        path = tmp_path / "triggers.csv"
        path.write_text("trigger,response,set_by\nhello,👋,seed\n", encoding="utf-8")

        assert self.store.import_triggers_csv_once(str(path), guild_id="g1") == 1
        self.store.set_trigger("hello", "👍", guild_id="g1")
        assert self.store.import_triggers_csv_once(str(path), guild_id="g1") == 0

        assert self.store.find_trigger_match("hello", guild_id="g1")["reaction"] == "👍"

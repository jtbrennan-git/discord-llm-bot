"""
Test suite for Discord LLM Bot.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

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

    def test_parse_fallback_no_tag(self):
        r = self.client.parse_response("just some text without a tag")
        assert r.action == "REPLY"
        assert "just some text" in r.content

    def test_parse_lowercase_tags(self):
        r = self.client.parse_response("[reply] hello")
        assert r.action == "REPLY"
        r = self.client.parse_response("[react: 👍]")
        assert r.action == "REACT"


# ─── Probability Curve ─────────────────────────────────────────────────

class TestProbabilityCurve:
    """Test the spontaneous message counter probability curve."""

    def test_probability_rises(self):
        MESSAGE_TARGET = 30
        BANDWAGON_MAX = 0.8
        results = []
        for n in range(1, 60):
            scale = MESSAGE_TARGET * 1.5
            chance = min(0.8, n / scale)
            results.append(chance)

        # Should start near zero
        assert results[0] < 0.1
        # Should hit ~50% around n=22
        assert 0.4 < results[21] < 0.6
        # Should cap at 0.8
        assert results[-1] == 0.8
        # Should be strictly increasing until cap
        for i in range(len(results) - 1):
            if results[i] < 0.8:
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

    def test_channel_controls_defaults_and_updates(self):
        defaults = self.store.get_channel_controls("c1")
        assert defaults["learning_enabled"] is True
        assert defaults["quiet_enabled"] is False

        self.store.set_channel_control("c1", "quiet", True)
        self.store.set_channel_control("c1", "starters", False)
        controls = self.store.get_channel_controls("c1")

        assert controls["quiet_enabled"] is True
        assert controls["starters_enabled"] is False

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

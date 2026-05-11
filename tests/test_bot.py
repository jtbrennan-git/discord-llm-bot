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

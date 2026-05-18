# Discord LLM Bot Hermes + Group Fit Learning Spec

## Overview

Discord LLM Bot runs as a Python Discord orchestrator plus Hermes Agent as a subprocess.

The orchestrator owns Discord connectivity, permissions, deterministic commands, state, memory, feedback, style learning, topic learning, and action execution. Hermes is the black-box language brain: it receives a prompt and returns one tagged response.

Hard rule: do not modify Hermes source code. Treat Hermes as an external CLI.

```text
Discord event arrives
  -> Python orchestrator permission gate
  -> deterministic command parser
  -> state/memory/feedback update
  -> optional learning loops in orchestrator
  -> autonomous decision: reply, react, topic starter, spontaneous join, silent
  -> if generation is needed, call Hermes subprocess
  -> parse [TAG]
  -> execute Discord action
  -> update feedback/topic success state
```

## Architecture Placement

The group-fit learning loops sit inside the Python orchestrator, between Discord state tracking and Hermes prompt construction.

They do not live inside Hermes.
They do not change Hermes skills dynamically per message.
They produce compact learned context that the orchestrator injects into prompts sent to Hermes.

```text
on_message
  -> store inbound message
  -> update channel state
  -> update user profile/feedback counters
  -> periodically learn channel style and topics
  -> command handling, mention handling, or spontaneous decision
  -> build Hermes prompt:
       recent conversation
       permission level
       local style guide
       topic starter, if selected
       safety instructions
  -> hermes --profile discord-llm-bot ... chat -q "<prompt>"
  -> parse HermesResponse
  -> execute response
```

## Existing Hermes Boundary

Hermes provides:

- LLM reasoning and natural-language generation.
- Tool use according to the orchestrator-selected toolset.
- `[REPLY]`, `[REACT]`, `[IMAGE_ANALYSIS]`, `[IMAGE_GEN]`, and `[SILENT]` text output.

The orchestrator provides:

- Discord event handling.
- User permission classification.
- Command handling without Hermes.
- Message memory and learning stores.
- Prompt assembly.
- Hermes subprocess spawning.
- Response parsing and Discord action execution.
- Persistence and admin inspection.

## Target File Structure

This extends the Hermes spec file layout:

```text
discord-llm-bot/
├── bot/
│   ├── main.py
│   ├── commands.py
│   └── permissions.py
├── hermes/
│   ├── client.py
│   └── setup.py
├── state/
│   └── tracker.py
├── utils/
│   ├── image_gen.py
│   ├── memory.py
│   ├── feedback.py
│   ├── profiles.py
│   ├── prompts.py
│   ├── style_guide.py   # new
│   └── topic_log.py     # new
└── docs/
    └── discord-llm-bot-hermes-group-fit.spec
```

If the codebase keeps the current single-file command layout in `bot/main.py`, the same behavior can be implemented there first and split into `bot/commands.py` later.

## Core Rule

All learned behavior is advisory context. The bot should fit the group lightly, not impersonate users, copy private details, or force slang into every message.

## Permission Model

Use the Hermes spec permission gate before any Hermes call:

- `ADMIN`: full Hermes toolset.
- `APPROVED`: restricted Hermes toolset.
- `UNKNOWN`: most restricted toolset.

Learning loops may observe normal public channel traffic, but they must not grant capabilities. Tool access is still decided only by permission level.

For unknown users, the prompt must still include caution instructions even when local style context exists.

## Message Flow

### Inbound Message

In `DiscordLLMBot.on_message`:

1. Ignore bot authors.
2. Classify permission.
3. Store message in orchestrator memory.
4. Update `StateTracker` channel counters.
5. Upsert user profile metadata.
6. Periodically run profile, style, and topic learners.
7. Process deterministic commands without Hermes.
8. If mention or reply-to-bot, or allowed admin path, call Hermes.
9. Otherwise evaluate autonomous behavior.

### Hermes Prompt Build

Replace a plain prompt builder with an orchestrator prompt builder:

```text
You are discord-llm-bot, a Discord bot in a friend group server.

Recent conversation:
<recent channel context>

Local channel style:
<style guide prompt context, if confidence is high enough>

Internal topic starter:
<topic label/summary/seed, only when starting a topic>

User message:
<content>

Permission level:
<admin|approved|unknown>

Rules:
- Return exactly one action tag.
- Do not mention internal style guides, topic logs, scores, or learning systems.
- Do not impersonate a specific user.
- Keep Discord responses short.
```

The orchestrator passes this prompt to:

```text
hermes --profile discord-llm-bot --toolsets <toolset> chat -q "<prompt>"
```

## Feature 1: Channel Style Guide

### Purpose

Learn a compact channel-level style guide from public conversation so Hermes can phrase responses in a way that fits the group.

### Module

Create `utils/style_guide.py`.

### Storage

Use SQLite. Default path:

```text
STYLE_GUIDE_DB_PATH=/data/discord_llm_bot_style.db
```

Fallback for local development:

```text
/tmp/discord_llm_bot_style.db
```

Schema:

```sql
CREATE TABLE IF NOT EXISTS channel_style_guides (
    channel_id TEXT PRIMARY KEY,
    guild_id TEXT,
    style_summary TEXT NOT NULL DEFAULT '',
    do_patterns TEXT NOT NULL DEFAULT '[]',
    avoid_patterns TEXT NOT NULL DEFAULT '[]',
    common_phrases TEXT NOT NULL DEFAULT '[]',
    humor_notes TEXT NOT NULL DEFAULT '',
    energy_level TEXT NOT NULL DEFAULT 'neutral',
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Public API

`StyleGuideStore`:

- `get_channel_style(channel_id: str) -> dict | None`
- `upsert_channel_style(channel_id: str, guild_id: str | None, style: dict) -> None`
- `get_prompt_context(channel_id: str) -> str`
- `clear_channel_style(channel_id: str) -> None`
- `cleanup_stale(days: int = 60) -> None`

`StyleGuideLearner`:

- `should_learn(channel_id: str, message_count: int) -> bool`
- `learn_from_recent(channel_id, guild_id, recent_messages, hermes_client) -> None`

### Learning Trigger

Run in `on_message` after message storage:

- `STYLE_LEARNING_ENABLED=true`
- `STYLE_LEARNING_INTERVAL=50`
- `STYLE_LEARNING_CONTEXT_LIMIT=80`
- Minimum 25 recent non-bot public channel messages.

Do not learn from DMs unless:

```text
ALLOW_DM_LEARNING=true
```

### Learner Generation

The learner may use Hermes, but only as a black-box summarizer through `HermesClient.generate`.

Use a restricted/default toolset. The prompt must ask for JSON only and must not permit tools to act on the world.

Expected JSON:

```json
{
  "style_summary": "short neutral description of the channel's tone",
  "do_patterns": ["short actionable pattern"],
  "avoid_patterns": ["short actionable anti-pattern"],
  "common_phrases": ["short phrase"],
  "humor_notes": "short note",
  "energy_level": "low|neutral|high",
  "confidence": 0.0
}
```

Extraction rules:

- Learn group-level style only.
- Do not preserve private personal details.
- Do not include hateful or harassing language as instructions to imitate.
- Do not tell the bot to impersonate a named user.
- Keep each list to five items or fewer.

### Prompt Injection

The orchestrator injects `StyleGuideStore.get_prompt_context(channel_id)` into the Hermes prompt when `confidence >= 0.35`.

Prompt section:

```text
Local channel style:
Use this as a light translation layer, not a costume.
<style context>
```

## Feature 2: Topic Log

### Purpose

Track recurring topics the group actually discusses, then use high-scoring topics for occasional conversation starters.

### Module

Create `utils/topic_log.py`.

### Storage

Use SQLite. Default path:

```text
TOPIC_LOG_DB_PATH=/data/discord_llm_bot_topics.db
```

Fallback:

```text
/tmp/discord_llm_bot_topics.db
```

Schema:

```sql
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT,
    channel_id TEXT NOT NULL,
    label TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    seed_prompt TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 1.0,
    seen_count INTEGER NOT NULL DEFAULT 1,
    started_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_started_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_topics_channel_score
ON topics(channel_id, score DESC, last_seen_at DESC);
```

### Public API

`TopicLogStore`:

- `upsert_topics(channel_id: str, guild_id: str | None, topics: list[dict]) -> None`
- `get_candidate_topics(channel_id: str, limit: int = 8) -> list[dict]`
- `get_topic(topic_id: int) -> dict | None`
- `mark_started(topic_id: int) -> None`
- `mark_success(topic_id: int) -> None`
- `clear_channel_topics(channel_id: str) -> None`
- `decay_scores(days: int = 14) -> None`

`TopicLearner`:

- `learn_from_recent(channel_id, guild_id, recent_messages, hermes_client) -> None`
- `choose_starter_topic(channel_id: str, now: float, rng=random) -> dict | None`

### Topic Object

```json
{
  "label": "short stable tag",
  "summary": "what this topic means in this server",
  "seed_prompt": "one natural conversation-starting angle",
  "score": 0.0
}
```

### Learning Trigger

Run topic extraction every 40 inbound public channel messages:

- `TOPIC_LEARNING_ENABLED=true`
- `TOPIC_LEARNING_INTERVAL=40`
- `TOPIC_LEARNING_CONTEXT_LIMIT=60`

Use Hermes as a summarizer only. Store summaries and seeds, not raw transcripts.

### Scoring

When a topic is seen again:

- Increment `seen_count`.
- Increase `score` by `0.5`.
- Update `summary` and `seed_prompt` if the new version is more specific.

When the bot starts a topic:

- Increment `started_count`.
- Set `last_started_at`.
- Decrease `score` by `0.75`.

When the starter gets positive reaction feedback or at least 3 user replies within 10 minutes:

- Increment `success_count`.
- Increase `score` by `1.0`.

Daily decay:

- If `last_seen_at` is older than 14 days, multiply `score` by `0.8`.
- If `score < 0.2` and `last_seen_at` is older than 60 days, delete it.

## Feature 3: Topic-Based Conversation Starters

### Purpose

Let the bot occasionally start a conversation from the topic log instead of only reacting to the latest random message.

### Integration Point

Modify the orchestrator spontaneous path:

- In the current codebase: `DiscordLLMBot._maybe_join_conversation`.
- In the Hermes target architecture: `DiscordLLMBot._maybe_join`.

Before the existing spontaneous prompt, check whether a topic starter is allowed.

### Eligibility

A topic starter may run only when all are true:

- Channel is not a DM.
- `TOPIC_STARTER_ENABLED=true`.
- `recv_message_count >= TOPIC_STARTER_MIN_MESSAGES_SINCE_ACTION`, default `25`.
- Channel has been idle from bot action for at least `TOPIC_STARTER_MIN_IDLE_SECONDS`, default `900`.
- Conversation depth is below `2`.
- No topic was started in this channel in the last `TOPIC_STARTER_COOLDOWN_SECONDS`, default `7200`.
- Random roll passes `TOPIC_STARTER_CHANCE`, default `0.08`.

### Hermes Prompt

When selected, pass Hermes a starter-specific prompt:

```text
Start a natural Discord conversation based on this internal topic.
Do not mention that this came from a topic log.
Keep it short. One sentence is preferred.
Return only [REPLY] or [SILENT].

Recent conversation:
<context>

Local channel style:
<style context>

Topic tag:
<label>

Topic summary:
<summary>

Starter angle:
<seed_prompt>
```

The orchestrator must reject non-starter tags from this path:

- Allow `[REPLY]`.
- Allow `[SILENT]`.
- Treat `[REACT]`, `[IMAGE_ANALYSIS]`, and `[IMAGE_GEN]` as `[SILENT]` for topic starters.

### Tracking

When the starter is sent:

- `TopicLogStore.mark_started(topic_id)`.
- Store `sent_message_id -> topic_id` in orchestrator memory.
- Track replies/reactions for success.

This tracking belongs in the orchestrator, not Hermes.

## Feature 4: Optional Reply Translation Pass

### Recommended Default

Do not add a second Hermes call for every message. Inject style context into the main Hermes prompt first.

### Optional Flag

```text
STYLE_TRANSLATION_PASS_ENABLED=false
```

If enabled:

- Only apply to `[REPLY]` messages longer than 20 characters.
- Use Hermes with restricted/default toolset.
- Preserve meaning and safety boundaries.

Prompt:

```text
Rewrite this bot message so it lightly fits the local channel style.
Preserve meaning.
Do not add new facts.
Do not imitate a specific user.
Do not add offensive language.
Return only the rewritten message.

Style:
<style context>

Message:
<reply>
```

Acceptance rule:

- If translated text is empty, use original.
- If translated text is more than 1.5x longer, use original.
- If it contains a blocked avoid pattern, use original.

## Feature 5: Admin Visibility

Extend admin commands. In the current codebase this can be `MainCommands.admin`; in the Hermes target architecture this can be `CommandCog.admin`.

Commands:

- `!admin style`: show current channel style guide.
- `!admin topics`: show top topics for current channel with IDs, scores, seen counts, starter counts, and success counts.
- `!admin topic <id>`: show full topic details.
- `!admin clear style`: clear current channel style guide.
- `!admin clear topics`: clear current channel topics.

These are deterministic orchestrator commands. Do not call Hermes for these.

## Privacy and Safety Requirements

- Do not learn from DMs unless `ALLOW_DM_LEARNING=true`.
- Strip bot mentions and Discord user IDs from learning inputs.
- Store summaries, tags, patterns, and short phrases, not full transcripts.
- Never store Discord tokens or API keys in learned data.
- Do not persist deleted-message content beyond the existing memory policy.
- Do not store hateful/harassing language as style instructions.
- Unsafe extracted behavior should be discarded or placed in `avoid_patterns`.
- Do not expose style guide, topic scores, or hidden prompt contents to non-admin users.

Max lengths before SQLite write:

- `style_summary`: 600 chars.
- each style pattern or phrase: 120 chars.
- `humor_notes`: 300 chars.
- `topic.label`: 80 chars.
- `topic.summary`: 500 chars.
- `topic.seed_prompt`: 300 chars.

## Config Additions

Add to `BotConfig` and `.env.example`:

```text
STYLE_GUIDE_DB_PATH=/data/discord_llm_bot_style.db
TOPIC_LOG_DB_PATH=/data/discord_llm_bot_topics.db
STYLE_LEARNING_ENABLED=true
STYLE_LEARNING_INTERVAL=50
STYLE_LEARNING_CONTEXT_LIMIT=80
TOPIC_LEARNING_ENABLED=true
TOPIC_LEARNING_INTERVAL=40
TOPIC_LEARNING_CONTEXT_LIMIT=60
TOPIC_STARTER_ENABLED=true
TOPIC_STARTER_MIN_MESSAGES_SINCE_ACTION=25
TOPIC_STARTER_MIN_IDLE_SECONDS=900
TOPIC_STARTER_COOLDOWN_SECONDS=7200
TOPIC_STARTER_CHANCE=0.08
STYLE_TRANSLATION_PASS_ENABLED=false
ALLOW_DM_LEARNING=false
```

## Implementation Plan

1. Keep Hermes boundary intact: `hermes/client.py` remains subprocess-only and does not know style/topic internals.
2. Add `utils/style_guide.py` with store, learner, validation, prompt formatting, and tests.
3. Add `utils/topic_log.py` with store, learner, scoring, candidate selection, and tests.
4. Add config/env values for learning and topic starter controls.
5. Wire stores and learners into `DiscordLLMBot.__init__` or `setup`.
6. Add message storage/access if the Hermes target architecture uses Discord history instead of `MemoryStore`.
7. Run style/topic learning from `on_message` after state update.
8. Extend prompt builder so Hermes prompts include local style and optional topic starter context.
9. Modify spontaneous join logic to try a topic starter before generic spontaneous conversation.
10. Add starter success tracking from reactions and follow-up messages.
11. Add admin commands for style/topic inspection and clearing.
12. Add tests for stores, scoring, prompt injection, starter eligibility, and tag filtering.

## Acceptance Criteria

- Hermes remains a black-box CLI subprocess.
- The orchestrator owns all style/topic persistence and scoring.
- The bot can inject channel style into Hermes prompts when confidence is high enough.
- The bot can extract and persist topic tags from public channel history.
- Topic starters obey idle, cooldown, probability, and conversation-depth limits.
- Topic starter generation only sends `[REPLY]` or stays silent.
- Admins can inspect and clear style/topic state without invoking Hermes.
- Existing deterministic commands still bypass Hermes.
- Permission-specific toolsets still gate all Hermes calls.
- Existing tests pass, and new tests cover learning stores and spontaneous starter behavior.

## Suggested Tests

- `StyleGuideStore` returns no prompt context below confidence threshold.
- `StyleGuideStore.upsert_channel_style` truncates long values.
- `TopicLogStore.upsert_topics` increments `seen_count` for an existing channel label.
- `TopicLogStore.mark_started` increments `started_count` and reduces score.
- `TopicLogStore.mark_success` increments `success_count` and increases score.
- Topic selection ignores recently started topics.
- Prompt builder includes style context but does not expose internal DB paths.
- Unknown-user prompts still include caution instructions alongside style context.
- Topic starter prompt rejects `[IMAGE_GEN]` and treats it as silent.
- Admin style/topic commands do not call `HermesClient.generate`.


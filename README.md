# Discord LLM Bot

A Python Discord bot that can participate in server conversations, react, learn lightweight channel style/topic context, and expose admin controls for tuning behavior.

## Features

- Discord server message handling through `discord.py`
- LLM-backed replies with tagged actions: reply, react, image analysis, image generation, or silent
- Per-channel memory, style learning, topic learning, and spontaneous reply controls
- Per-user response and memory/privacy controls
- Custom exact-match text triggers with local SQLite persistence and optional one-time CSV seeding
- Profile prompt context is disabled by default for safer public multi-server deployments
- Admin/control-plane commands for diagnostics, channel modes, delete/react, logs, and learning jobs
- Reaction feedback and manual quality labels for personality tuning
- Recent channel highlights based on reaction counts with basic sensitive-topic filtering

## Requirements

- Python 3.11+
- A Discord bot token
- An LLM provider compatible with the OpenAI chat completions client

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in secrets locally:

```bash
cp .env.example .env
```

Required:

```env
DISCORD_TOKEN=
LLM_API_KEY=
```

Useful optional settings:

```env
COMMAND_PREFIX=!
LLM_BASE_URL=
LLM_MODEL=openrouter/owl-alpha
PROFILE_CONTEXT_ENABLED=false
TRIGGER_DEFAULTS_CSV=fellasbot_triggers_active.csv
TRIGGER_DEFAULTS_IMPORT_ENABLED=true
DEV_USER_IDS=
CONTROL_GUILD_ID=
CONTROL_CHANNEL_ID=
CONTROL_ADMIN_IDS=
TARGET_GUILD_ID=
```

Do not commit `.env`, database files, logs, or copied production secrets.

## Run Locally

```bash
python -m bot.main
```

## Deployment

`fly.toml` is a template. Before deploying to Fly.io, change:

- `app`
- mounted volume `source`
- any persistent paths you want to customize

Set runtime secrets through your deployment platform, not in the repo.

## Commands

User-facing:

- `!help [command]`
- `!ping`
- `!whoami`
- `!feedback`
- `!highlights [count] [scan_limit]`
- `!trigger <text> <response>`
- `!forget <text>`
- `!improve <suggestion>`
- `!control status|mute|unmute|prompted|strict|normal|remember|forget|privacy`

Developer/admin:

- `!admin ...` for local diagnostics and maintenance
- Control-plane `!control ...` commands when configured with `CONTROL_*` environment variables

## Security Notes

- The bot ignores DMs by default/runtime behavior.
- The bot can only read channels where Discord permissions allow it to view the channel and read message history.
- Admin/control commands must be protected by `DEV_USER_IDS` and/or `CONTROL_ADMIN_IDS`.
- Message learning tries to redact obvious secrets before sending learning prompts, but this is not a complete DLP system.
- User profile context is not injected into model prompts unless `PROFILE_CONTEXT_ENABLED=true`.
- Custom trigger defaults are imported into the local `PROFILES_DB`; keep seed CSV exports out of git.
- Image generation uses an external service by default; prompts are sent to that provider.
- Public deployment should review channel permissions and disable tracking in sensitive channels.

## Tests

```bash
pytest -q
```

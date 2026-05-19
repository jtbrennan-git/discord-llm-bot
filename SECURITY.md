# Security Policy

## Reporting

Please report security issues privately to the repository maintainer. Do not open public issues containing tokens, private Discord IDs, database exports, or message logs.

## Sensitive Data

This bot may process Discord messages, display names, channel names, message IDs, reactions, and generated model prompts. Production operators should treat runtime databases and logs as sensitive.

Never commit:

- `.env`
- Discord bot tokens
- LLM API keys
- Fly.io tokens or deployment secrets
- SQLite databases
- persistent logs
- exported Discord message data
- trigger seed CSV exports

## Deployment Guidance

- Grant the bot only the Discord permissions it needs.
- Disable tracking or set `ignore` mode in sensitive channels.
- Keep admin controls limited to trusted Discord user IDs.
- Store secrets in the deployment platform's secret manager.
- Keep `PROFILE_CONTEXT_ENABLED=false` for public multi-server deployments unless user profiles are scoped appropriately for your use case.
- Keep `PROFILES_DB` on local or private persistent storage; custom reaction triggers are stored there.
- Review external providers used for LLM and image generation before enabling them.

## Known Limitations

The learning sanitizer redacts common identifiers and token-like strings, but it is not a complete privacy or data-loss-prevention system. Sensitive channels should be excluded at the Discord permission level or through bot channel controls.

"""
Sanitizers for text that is sent to learning prompts.
"""

import re
from typing import List, Tuple

MESSAGE_LIMIT = 500

_MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_TOKENISH_RE = re.compile(r"\b(?:[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}|[A-Za-z0-9_\-]{40,})\b")
_URL_RE = re.compile(r"https?://\S+")


def sanitize_learning_text(content: str, *, keep_urls: bool = False) -> str:
    """Remove obvious identifiers/secrets and cap message length before learning."""
    text = content or ""
    text = _MENTION_RE.sub("[mention]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _TOKENISH_RE.sub("[redacted]", text)
    if not keep_urls:
        text = _URL_RE.sub("[url]", text)
    text = " ".join(text.split())
    return text[:MESSAGE_LIMIT]


def sanitize_recent_messages(
    recent_messages: List[Tuple[str, str, str]], *, keep_urls: bool = False
) -> List[Tuple[str, str, str]]:
    sanitized = []
    for author, content, created_at in recent_messages:
        cleaned = sanitize_learning_text(content, keep_urls=keep_urls)
        if cleaned:
            sanitized.append((author, cleaned, created_at))
    return sanitized


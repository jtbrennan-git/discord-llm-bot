"""
Prompt construction helpers for the Discord bot.
"""

from typing import Optional

from utils.feedback import FeedbackTracker
from utils.llm import DEFAULT_SYSTEM_PROMPT
from utils.profiles import UserProfileStore


def build_system_prompt(
    bot_name: str,
    base_prompt: Optional[str] = None,
    profiles: Optional[UserProfileStore] = None,
    feedback: Optional[FeedbackTracker] = None,
    style_context: Optional[str] = None,
    profile_limit: int = 8,
) -> str:
    """Assemble the full system prompt with identity, profiles, and feedback."""
    prompt = (base_prompt or DEFAULT_SYSTEM_PROMPT).replace("{name}", bot_name)

    if profiles:
        all_profiles = profiles.get_all_profiles()
        if all_profiles:
            parts = ["## People you know"]
            for profile in all_profiles[:profile_limit]:
                summary = profiles.get_profile_summary(profile["user_id"])
                if summary:
                    parts.append(summary)
            prompt += "\n\n" + "\n\n".join(parts)

    if feedback:
        feedback_context = feedback.get_feedback_context()
        if feedback_context:
            prompt += f"\n\n## Feedback on your behavior\n{feedback_context}"

    if style_context:
        prompt += (
            "\n\n## Local channel style\n"
            "Use this as a light translation layer, not a costume. "
            "Do not mention the style guide or imitate a specific person.\n"
            f"{style_context}"
        )

    return prompt

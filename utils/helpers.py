"""
Helper functions for the Discord bot.
"""

from typing import List, Optional
from datetime import datetime

import discord
from discord.ext import commands


def get_guild_channel(guild: discord.Guild, channel_name: str) -> Optional[discord.TextChannel]:
    """Get a channel by name from a guild."""
    for channel in guild.text_channels:
        if channel.name == channel_name:
            return channel
    return None


def is_mod(user: discord.Member) -> bool:
    """Check if a user is a moderator."""
    return any(role.name.lower() == "moderator" for role in user.roles) or user.guild_permissions.manage_messages


def format_cooldown(cooldown: int) -> str:
    """Format cooldown time in a user-friendly way."""
    minutes, seconds = divmod(cooldown, 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def escape_mentions(content: str) -> str:
    """Escape Discord mentions to prevent pinging."""
    # Escape @everyone and @here
    content = content.replace("@everyone", "\\@everyone")
    content = content.replace("@here", "\\@here")
    # Escape role mentions
    # (This is more complex and would need regex)
    return content


def is_bot_mentioned(message: discord.Message) -> bool:
    """Check if the bot is mentioned in a message."""
    return message.guild and message.author != message.guild.me and message.guild.me in message.mentions


def get_user_name(member: discord.Member) -> str:
    """Get a user's display name."""
    return member.display_name


def get_uptime(start_time: datetime) -> str:
    """Get bot uptime."""
    delta = datetime.utcnow() - start_time
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


# Decorator for error handling
def error_handler(coro):
    """Decorator to handle command errors."""
    async def wrapper(*args, **kwargs):
        try:
            await coro(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Command error: {e}", exc_info=True)
            # Don't send error to user unless it's a specific case
    return wrapper


# Simple command cooldown
class CommandCooldown:
    """Simple cooldown manager."""
    def __init__(self):
        self.cooldowns = {}

    def is_on_cooldown(self, command_name: str, cooldown: int) -> bool:
        """Check if a command is on cooldown."""
        if command_name in self.cooldowns:
            if self.cooldowns[command_name] > datetime.utcnow():
                return True
            else:
                del self.cooldowns[command_name]
        return False

    def set_cooldown(self, command_name: str, cooldown: int):
        """Set cooldown for a command."""
        self.cooldowns[command_name] = datetime.utcnow() + cooldown
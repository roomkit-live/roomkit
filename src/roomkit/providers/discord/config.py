"""Discord bot provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr


class DiscordConfig(BaseModel):
    """Discord bot configuration.

    The ``message_content`` intent is a *privileged* gateway intent: it must
    be enabled in the Discord Developer Portal (Bot → Privileged Gateway
    Intents) or every inbound ``message.content`` arrives empty.
    """

    bot_token: SecretStr
    # Request the privileged Message Content intent so inbound messages carry
    # their text. Disable only for bots that rely solely on slash commands.
    intents_message_content: bool = True
    # Drop inbound messages authored by other bots (the bot's own messages are
    # always dropped by the parser to avoid echo loops).
    ignore_bots: bool = True
    timeout: float = 30.0

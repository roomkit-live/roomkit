"""Mock Discord provider for testing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.discord.base import DiscordProvider


class MockDiscordProvider(DiscordProvider):
    """Records sent messages and reactions for verification in tests.

    Carries no ``discord`` dependency and no gateway client, so it can drive
    the delivery path without a live connection.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.reactions: list[dict[str, str]] = []

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        self.sent.append({"event": event, "to": to})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        self.reactions.append({"channel_id": channel_id, "message_id": message_id, "emoji": emoji})

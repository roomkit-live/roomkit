"""Mock Buzz provider for testing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.buzz.base import BuzzRelayProvider


class MockBuzzProvider(BuzzRelayProvider):
    """Records sent messages for verification in tests.

    Carries no ``buzzkit`` dependency and no relay client, so it can drive the
    delivery path without a live connection.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        self.sent.append({"event": event, "to": to})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

    async def send_reaction(self, target_event_id: str, emoji: str) -> ProviderResult:
        self.reactions.append({"action": "add", "target": target_event_id, "emoji": emoji})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

    async def remove_reaction(self, reaction_event_id: str) -> ProviderResult:
        self.reactions.append({"action": "remove", "target": reaction_event_id})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

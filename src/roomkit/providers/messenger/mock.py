"""Mock Messenger provider for testing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.messenger.base import MessengerProvider


class MockMessengerProvider(MessengerProvider):
    """Records sent messages for verification in tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        self.sent.append({"event": event, "to": to})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

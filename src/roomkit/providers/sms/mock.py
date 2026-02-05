"""Mock SMS provider for testing."""

from __future__ import annotations

from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.sms.base import SMSProvider


class MockSMSProvider(SMSProvider):
    """Records sent messages for verification in tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, str | RoomEvent]] = []

    @property
    def from_number(self) -> str:
        return "+15550001234"

    async def send(self, event: RoomEvent, to: str, from_: str | None = None) -> ProviderResult:
        self.sent.append({"event": event, "to": to, "from": from_ or ""})
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

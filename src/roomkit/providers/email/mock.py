"""Mock email provider for testing."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent
from roomkit.providers.email.base import EmailProvider


class MockEmailProvider(EmailProvider):
    """Records sent emails for verification in tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        event: RoomEvent,
        to: str,
        from_: str | None = None,
        subject: str | None = None,
    ) -> ProviderResult:
        self.sent.append(
            {
                "event": event,
                "to": to,
                "from": from_ or "",
                "subject": subject or "",
            }
        )
        return ProviderResult(success=True, provider_message_id=uuid4().hex)

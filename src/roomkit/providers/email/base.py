"""Abstract base class for email providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent


class EmailProvider(ABC):
    """Email delivery provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'sendgrid', 'ses')."""
        return self.__class__.__name__

    @abstractmethod
    async def send(
        self,
        event: RoomEvent,
        to: str,
        from_: str | None = None,
        subject: str | None = None,
    ) -> ProviderResult:
        """Send an email message.

        Args:
            event: The room event containing the message content.
            to: Recipient email address.
            from_: Sender email address override.
            subject: Email subject line.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    async def parse_inbound(self, payload: dict[str, Any]) -> InboundMessage:
        """Parse an inbound email payload into an InboundMessage."""
        raise NotImplementedError

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

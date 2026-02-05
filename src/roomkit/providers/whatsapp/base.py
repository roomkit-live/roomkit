"""Abstract base class for WhatsApp providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent


class WhatsAppProvider(ABC):
    """WhatsApp delivery provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'meta', 'twilio_wa')."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a WhatsApp message.

        Args:
            event: The room event containing the message content.
            to: Recipient WhatsApp ID or phone number.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    async def parse_webhook(self, payload: dict[str, Any]) -> InboundMessage:
        """Parse an inbound webhook payload into an InboundMessage."""
        raise NotImplementedError

    async def send_template(
        self, to: str, template_name: str, params: dict[str, Any] | None = None
    ) -> ProviderResult:
        """Send a template message."""
        raise NotImplementedError

    async def send_reaction(
        self,
        chat: str,
        sender: str,
        message_id: str,
        emoji: str,
    ) -> None:
        """Send a reaction to a message.

        Args:
            chat: Chat identifier (JID or phone).
            sender: Sender of the message being reacted to.
            message_id: External message ID to react to.
            emoji: Emoji reaction (empty string to remove).
        """
        raise NotImplementedError

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

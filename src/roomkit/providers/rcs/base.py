"""Abstract base class for RCS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent


class RCSDeliveryResult(ProviderResult):
    """Extended result for RCS delivery with fallback information."""

    channel_used: str = "rcs"  # "rcs" or "sms" if fallback occurred
    fallback: bool = False


class RCSProvider(ABC):
    """RCS delivery provider for rich communication services."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'twilio', 'sinch')."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def sender_id(self) -> str:
        """RCS sender/agent identifier."""
        ...

    @property
    def supports_fallback(self) -> bool:
        """Whether this provider supports automatic SMS fallback."""
        return True

    @abstractmethod
    async def send(
        self,
        event: RoomEvent,
        to: str,
        *,
        fallback: bool = True,
    ) -> RCSDeliveryResult:
        """Send an RCS message.

        Args:
            event: The room event containing the message content.
            to: Recipient phone number (E.164 format).
            fallback: If True, allow fallback to SMS when RCS unavailable.

        Returns:
            Result with delivery info including whether fallback occurred.
        """
        ...

    async def check_capability(self, phone_number: str) -> bool:
        """Check if a phone number supports RCS.

        Args:
            phone_number: Phone number to check (E.164 format).

        Returns:
            True if the number supports RCS, False otherwise.

        Raises:
            NotImplementedError: If the provider doesn't support capability check.
        """
        raise NotImplementedError(f"{self.name} does not support RCS capability checking")

    async def parse_webhook(self, payload: dict[str, Any]) -> InboundMessage:
        """Parse an inbound webhook payload into an InboundMessage."""
        raise NotImplementedError

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: str | None = None,
    ) -> bool:
        """Verify that a webhook payload was signed by the provider.

        Args:
            payload: Raw request body bytes.
            signature: Signature header value from the webhook request.
            timestamp: Timestamp header value (required by some providers).

        Returns:
            True if the signature is valid, False otherwise.
        """
        raise NotImplementedError(f"{self.name} does not support webhook signature verification")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

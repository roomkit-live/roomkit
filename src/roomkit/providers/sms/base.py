"""Abstract base class for SMS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.delivery import InboundMessage, ProviderResult
from roomkit.models.event import RoomEvent


class SMSProvider(ABC):
    """SMS delivery provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'twilio', 'sinch')."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def from_number(self) -> str:
        """Default sender phone number."""
        ...

    @abstractmethod
    async def send(self, event: RoomEvent, to: str, from_: str | None = None) -> ProviderResult:
        """Send an SMS message.

        Args:
            event: The room event containing the message content.
            to: Recipient phone number (E.164 format).
            from_: Sender phone number override. Defaults to ``from_number``.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

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

        Raises:
            NotImplementedError: If the provider does not support signature
                verification.
        """
        raise NotImplementedError(f"{self.name} does not support webhook signature verification")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

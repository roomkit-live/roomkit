"""Abstract base class for Telegram providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent


class TelegramProvider(ABC):
    """Telegram Bot delivery provider."""

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a Telegram message.

        Args:
            event: The room event containing the message content.
            to: Recipient Telegram chat ID.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify that a webhook request was sent by Telegram.

        Args:
            payload: Raw request body bytes (unused — Telegram uses a
                shared secret token, not payload signing).
            signature: Value of the ``X-Telegram-Bot-Api-Secret-Token`` header.

        Returns:
            True if the signature matches the configured webhook secret.

        Raises:
            NotImplementedError: If the provider does not support signature
                verification.
        """
        raise NotImplementedError(f"{self.name} does not support webhook signature verification")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

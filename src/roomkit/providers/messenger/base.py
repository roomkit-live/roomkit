"""Abstract base class for Messenger providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent


class MessengerProvider(ABC):
    """Facebook Messenger delivery provider."""

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a Facebook Messenger message.

        Args:
            event: The room event containing the message content.
            to: Recipient Messenger user ID.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

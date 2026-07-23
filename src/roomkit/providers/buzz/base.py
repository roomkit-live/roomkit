"""Abstract base class for Buzz (Nostr relay) providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent


class BuzzRelayProvider(ABC):
    """Buzz relay delivery provider."""

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a message to a Buzz channel.

        Args:
            event: The room event containing the message content.
            to: Recipient Buzz channel UUID.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

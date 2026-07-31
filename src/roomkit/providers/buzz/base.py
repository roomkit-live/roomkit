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

    async def send_reaction(self, target_event_id: str, emoji: str) -> ProviderResult:
        """Add an emoji reaction (NIP-25 kind 7) to a relay event.

        Args:
            target_event_id: Nostr event id (hex) of the message to react to.
            emoji: Unicode emoji (relay caps it at 64 chars).

        Raises:
            NotImplementedError: If the provider does not support reactions.
        """
        raise NotImplementedError(f"{self.name} does not support reactions")

    async def remove_reaction(self, reaction_event_id: str) -> ProviderResult:
        """Retract one of our own reactions by deleting its event (kind 5).

        Args:
            reaction_event_id: Nostr event id (hex) of OUR reaction event —
                the ``provider_message_id`` a successful ``send_reaction``
                returned, not the reacted-to message.

        Raises:
            NotImplementedError: If the provider does not support reactions.
        """
        raise NotImplementedError(f"{self.name} does not support reactions")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

"""Abstract base class for Discord providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent


class DiscordProvider(ABC):
    """Discord bot delivery provider."""

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a Discord message.

        Args:
            event: The room event containing the message content.
            to: Recipient Discord channel ID (the numeric snowflake as a
                string).

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add an emoji reaction to a message.

        Args:
            channel_id: Discord channel ID containing the message.
            message_id: Target message ID to react to.
            emoji: Unicode emoji or custom emoji (``name:id``).

        Raises:
            NotImplementedError: If the provider does not support reactions.
        """
        raise NotImplementedError(f"{self.name} does not support reactions")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

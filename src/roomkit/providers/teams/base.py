"""Abstract base class for Teams providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import RoomEvent


class TeamsProvider(ABC):
    """Microsoft Teams delivery provider."""

    @property
    def name(self) -> str:
        """Provider name."""
        return self.__class__.__name__

    @abstractmethod
    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a Microsoft Teams message.

        Args:
            event: The room event containing the message content.
            to: Recipient Teams conversation ID.

        Returns:
            Result with provider-specific delivery metadata.
        """
        ...

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify that a webhook payload was signed by the Bot Framework.

        Args:
            payload: Raw request body bytes.
            signature: Value of the ``Authorization`` header (``Bearer <token>``).

        Returns:
            True if the signature is valid, False otherwise.

        Raises:
            NotImplementedError: If the provider does not support signature
                verification.
        """
        raise NotImplementedError(f"{self.name} does not support webhook signature verification")

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses that hold connections."""

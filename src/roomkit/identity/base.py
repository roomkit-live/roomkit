"""Abstract base class for identity resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.identity import IdentityResult


class IdentityResolver(ABC):
    """Resolves user identity from inbound messages."""

    @abstractmethod
    async def resolve(self, message: InboundMessage, context: RoomContext) -> IdentityResult:
        """Resolve the identity of an inbound message sender.

        Args:
            message: The inbound message with sender information.
            context: Current room context (room, bindings, participants).

        Returns:
            An identity result indicating identified, ambiguous, pending,
            unknown, or rejected status.
        """
        ...

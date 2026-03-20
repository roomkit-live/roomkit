"""Base class for orchestration strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.core.framework import RoomKit


class Orchestration(ABC):
    """Abstract base for orchestration strategies.

    Orchestration strategies compose existing primitives
    (``ConversationPipeline``, ``ConversationRouter``, ``HandoffHandler``)
    into declarative patterns that can be passed to ``RoomKit`` or
    ``create_room``.

    Subclasses must implement:

    - :meth:`agents` — which agents participate in the room.
    - :meth:`install` — wire hooks, tools, and state into a room.
    """

    @abstractmethod
    def agents(self) -> list[Agent]:
        """Return agents to register and attach to the room.

        The framework calls this to determine which agents should be
        registered on the kit and attached to the room at creation time.
        """

    @abstractmethod
    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire hooks, tools, and state into the room.

        Called after agents are registered and attached. Implementations
        should install room-scoped hooks, set up handoff tools, and
        initialise conversation state.
        """

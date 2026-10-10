"""Base class for orchestration strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.channels.ai import AIChannel
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
    def agents(self) -> Sequence[AIChannel]:
        """Return agents to register and attach to the room.

        The framework calls this to determine which agents should be
        registered on the kit and attached to the room at creation time.
        """

    @abstractmethod
    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Wire hooks, tools, and state into the room.

        Called after agents are registered and attached. Implementations
        should install room-scoped hooks, set up handoff tools, and
        initialise conversation state. Install through
        :meth:`RoomKit.install_strategy` (or ``create_room(orchestration=...)``),
        which claims the room for one strategy and records what the install
        adds, so :meth:`RoomKit.uninstall_strategy` can take it back.
        """

    async def uninstall(self, kit: RoomKit, room_id: str) -> None:  # noqa: B027
        """Undo what :meth:`install` set up that the kit does not record.

        :meth:`RoomKit.uninstall_strategy` calls it first, then removes what it
        recorded of the install: the room hooks, the tools and turn runners set
        up on the agents for the room, the agents the install attached, and the
        room metadata the install wrote (RFC §19.7). A strategy with state of
        its own elsewhere (a discussion's turns) stops it here. Default: nothing.
        """

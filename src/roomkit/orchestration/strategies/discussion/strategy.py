"""The Discussion orchestration strategy (RFC §19.7.5)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from roomkit.channels.ai import AIChannel
from roomkit.orchestration.base import Orchestration

from ._room import DiscussionRoom

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

DoneFn = Callable[[str], bool | Awaitable[bool]]
"""``done(room_id)``: whether the discussion is over, sync or async."""


class Discussion(Orchestration):
    """Several agents and people hold one conversation; one agent speaks at a time.

    Any agent may address any other by ``@channel_id``, a person may address
    any agent, and who speaks next follows from who was addressed: the
    strategy queues the agents each message asks for and gives them their
    turns one at a time, each turn reading the room as it is when it starts.
    The room is the discussion's: its intelligence channels are the
    discussion's agents and no other, and no router or other strategy shares
    it. Text only in this version.

    Example::

        kit = RoomKit()
        room = await kit.create_room(
            orchestration=Discussion(agents=[investigator, dev, sre]),
        )
        # "@dev can you check the deploy?" asks dev; dev's "@sre please
        # roll back" asks sre once dev's turn has ended.

    Args:
        agents: The agents of the discussion: AI channels, Agents among them,
            with distinct channel ids.
        people: The names agents address people by. ``None``: the names of
            the room's participants that are neither agents nor bots.
        addressed_only: A person's message asks only the agents it names;
            one that names nobody asks no agent.
        everyone: The agents a person's message that answers no one asks, in
            this order (the one that should open first). ``None``: every agent,
            in ``agents`` order.
        max_turns: Turns given in the room's lifetime, after which the
            discussion is over. ``None``: no bound but ``max_depth``.
        max_depth: The depth limit of the turns it gives, in place of the
            kit's ``max_chain_depth``. ``None``: the kit's.
        done: ``done(room_id)``, checked before each turn: once it holds,
            the discussion is over.
        silent_token: The answer of an agent with nothing to add.
    """

    def __init__(
        self,
        agents: Sequence[AIChannel],
        *,
        people: Sequence[str] | None = None,
        addressed_only: bool = False,
        everyone: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_depth: int | None = None,
        done: DoneFn | None = None,
        silent_token: str = "(silent)",
    ) -> None:
        _check(agents, everyone, max_turns, max_depth, silent_token)
        self._agents = list(agents)
        self.people = list(people) if people is not None else None
        self.addressed_only = addressed_only
        self.everyone = list(everyone) if everyone is not None else None
        self.max_turns = max_turns
        self.max_depth = max_depth
        self.done = done
        self.silent_token = silent_token

    def agents(self) -> list[AIChannel]:
        """The agents of the discussion."""
        return list(self._agents)

    async def install(self, kit: RoomKit, room_id: str) -> None:
        """Give the room to the discussion: its agents attached, its stored
        queue read, its turns started.

        Raises:
            ValueError: the room binds another intelligence channel or a voice
                or realtime channel, an agent thinks while it listens, or a
                router, another strategy or a discussion is installed.
        """
        await kit._install_discussion(room_id, DiscussionRoom.installed(kit, room_id, self))

    async def uninstall(self, kit: RoomKit, room_id: str) -> None:
        """Give the room back to its ``agent_response_policy``: the turns
        stopped, the queue dropped."""
        await kit._uninstall_discussion(room_id)


def _check(
    agents: Sequence[AIChannel],
    everyone: Sequence[str] | None,
    max_turns: int | None,
    max_depth: int | None,
    silent_token: str,
) -> None:
    if not agents:
        raise ValueError("A discussion needs at least one agent")
    for agent in agents:
        if not isinstance(agent, AIChannel):
            raise TypeError(
                f"{getattr(agent, 'channel_id', agent)!r} is not an AI channel: a discussion "
                "holds AI channels only (an external agent runs its own loop)"
            )
    ids = [a.channel_id for a in agents]
    if len(set(ids)) != len(ids):
        raise ValueError("A discussion's agents need distinct channel ids")
    unknown = [a for a in everyone or () if a not in ids]
    if unknown:
        raise ValueError(f"everyone names channels that are not agents: {unknown}")
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_depth is not None and max_depth < 2:
        # A person's message is at depth 0 and the first turn answering it at 1.
        raise ValueError("max_depth must be at least 2")
    if not silent_token.strip():
        raise ValueError("silent_token must not be blank")

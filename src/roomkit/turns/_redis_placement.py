"""Where a room's turns of a channel go on the Redis turn backend (RFC §10.6.7).

The room owner chooses: its own executor alone for a channel that executor
hosts (the agents it holds live), closed or not; else the executor that holds the room's
state for the channel, while it lives and takes the turn, else one that takes
it, one of the owner's own process first (carried in the process), else one
alive elsewhere, the rooms no executor holds yet spread over them. The holder is a
key every owner process reads, so a room's turns follow its session state
whichever owner submits them. An executor elsewhere is one process's: its id
restarted in another process is another executor.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote

from roomkit.turns._organizations import RoomNeeds, concerns
from roomkit.turns._redis_presence import Declaration, member_of
from roomkit.turns.base import no_executor

if TYPE_CHECKING:
    from roomkit.turns._redis_beat import LiveExecutors
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelReleased, RoomAttached, RoomDetached

# How long a room's holder is kept once no turn renews it.
_HOLDER_TTL_S = 7 * 24 * 3600


class Elsewhere(NamedTuple):
    """An executor of another process (of this one, in the tests' loop back)."""

    executor_id: str
    node: str

    @property
    def member(self) -> str:
        return member_of(self.node, self.executor_id)


class Placement:
    """The executors of this process, and where each turn goes.

    Args:
        client: The ``redis.asyncio`` client.
        prefix: The keys' prefix.
        node: This process's node id.
        live: The processes alive, as this one reads them.
        loop_back: Tests only: this process's executors are reached across
            Redis too, as executors elsewhere would be.
    """

    def __init__(
        self, client: Any, prefix: str, node: str, live: LiveExecutors, *, loop_back: bool
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._node = node
        self._live = live
        self._loop_back = loop_back
        self.here: list[TurnExecutor] = []
        """The executors this process serves."""
        self.own: TurnExecutor | None = None
        """The executor of the room owner this process attends, which runs
        the channels the owner holds live (RFC §10.6.7)."""
        self._next = 0

    async def place(
        self,
        channel_id: str,
        room_id: str,
        needs: RoomNeeds,
        *,
        excluded: Collection[str] = (),
    ) -> TurnExecutor | Elsewhere:
        """The executor a room's turn or channel call of *channel_id* goes
        to, but the members *excluded* (refused it), recorded as the one
        holding the room's state from now."""
        placed = await self._chosen(channel_id, room_id, needs, excluded, spread=True)
        holder = placed.member if isinstance(placed, Elsewhere) else self._here_member(placed)
        await self._client.set(self._holder_key(channel_id, room_id), holder, ex=_HOLDER_TTL_S)
        return placed

    async def hold(
        self,
        channel_id: str,
        room_id: str,
        needs: RoomNeeds,
        *,
        excluded: Collection[str] = (),
    ) -> TurnExecutor | Elsewhere:
        """The executor holding the room's state of *channel_id* while it
        lives and takes it, else the one the room's next turn would go to;
        the holder is left as it was (RFC §10.6.3)."""
        return await self._chosen(channel_id, room_id, needs, excluded, spread=False)

    async def _chosen(
        self,
        channel_id: str,
        room_id: str,
        needs: RoomNeeds,
        excluded: Collection[str],
        *,
        spread: bool,
    ) -> TurnExecutor | Elsewhere:
        """The owner's own executor alone for a channel it hosts, else the
        holder while it lives and takes the turn, else one that takes it, one
        of this process first, else one elsewhere: a room no executor holds
        yet goes to the next of them, the spread moving on only when *spread*
        (a turn or call placed, not an event held)."""
        own = self._owning(channel_id)
        if own is not None:
            return self._own_taken(own, channel_id, needs, excluded)
        here = {
            self._here_member(e): e
            for e in self.here
            if e._meets(channel_id, needs) and self._here_member(e) not in excluded
        }
        elsewhere = await self._takers_elsewhere(channel_id, needs, set(here) | set(excluded))
        stored = await self._client.get(self._holder_key(channel_id, room_id))
        holder = _text(stored) if stored is not None else None
        if self._loop_back and here:
            chosen = here[holder] if holder in here else next(iter(here.values()))
            return Elsewhere(chosen.executor_id, self._node)
        if holder in here:
            return here[holder]
        if holder in elsewhere:
            return elsewhere[holder]
        if here:
            return next(iter(here.values()))
        if not elsewhere:
            raise no_executor(channel_id, needs)
        # A room no executor holds yet: its turns spread over the takers.
        members = sorted(elsewhere)
        following = (self._next + 1) % len(members)
        if spread:
            self._next = following
        return elsewhere[members[following]]

    async def forget(self, channel_id: str, room_id: str) -> None:
        """The room detached the channel: nothing holds its state any more."""
        await self._client.delete(self._holder_key(channel_id, room_id))

    async def told(
        self, message: RoomAttached | RoomDetached | ChannelReleased
    ) -> tuple[list[TurnExecutor], list[Elsewhere]]:
        """The executors a room's attach or detach, or a channel's release,
        reaches, here and elsewhere: those that may run the channel's turns,
        serving the room's organization for an attach or a detach; the
        owner's own alone for a channel it hosts (RFC §10.6.3)."""
        channel_id = message.channel_id
        own = self._owning(channel_id)
        if own is not None:
            told = own.declares(channel_id) and concerns(own.organizations, message)
            return ([own] if told else []), []
        here = [
            e for e in self.here if e.declares(channel_id) and concerns(e.organizations, message)
        ]
        return here, await self._declaring_elsewhere(message)

    async def _declaring_elsewhere(
        self, message: RoomAttached | RoomDetached | ChannelReleased
    ) -> list[Elsewhere]:
        """The executors of other processes alive that declare *message*'s
        channel, and that it concerns."""
        return [
            Elsewhere(declared.executor_id, declared.node)
            for declared in (await self._live.current()).values()
            if declared.node != self._node
            and declared.channels.covers(message.channel_id)
            and concerns(declared.organizations, message)
        ]

    async def _takers_elsewhere(
        self, channel_id: str, needs: RoomNeeds, skipped: set[str]
    ) -> dict[str, Elsewhere]:
        """The executors of other processes alive that take the turn, by
        member: read afresh, so one just started counts and one whose
        process withdrew it on closing does not."""
        alive = await self._live.current()
        return _takers(alive, channel_id, needs, self._node, skipped)

    def _owning(self, channel_id: str) -> TurnExecutor | None:
        """The owner's own executor when it hosts *channel_id*, closed or not:
        it runs the channels the owner holds live, and no other executor
        takes their turns, even once it closed (RFC §10.6.7)."""
        own = self.own
        return own if own is not None and own.hosts_channel(channel_id) else None

    def _own_taken(
        self,
        own: TurnExecutor,
        channel_id: str,
        needs: RoomNeeds,
        excluded: Collection[str],
    ) -> TurnExecutor | Elsewhere:
        """*own*, the owner's executor, for a turn of a channel it runs (across
        Redis in the tests' loop back); ``no_executor`` when it refused the
        turn or does not define the room's hooks."""
        if self._here_member(own) in excluded or not own._meets(channel_id, needs):
            raise no_executor(channel_id, needs)
        return Elsewhere(own.executor_id, self._node) if self._loop_back else own

    def _here_member(self, executor: TurnExecutor) -> str:
        return member_of(self._node, executor.executor_id)

    def _holder_key(self, channel_id: str, room_id: str) -> str:
        return f"{self._prefix}:holder:{quote(channel_id, safe='')}:{quote(room_id, safe='')}"


def _takers(
    alive: dict[str, Declaration],
    channel_id: str,
    needs: RoomNeeds,
    node: str,
    skipped: set[str],
) -> dict[str, Elsewhere]:
    return {
        member: Elsewhere(declared.executor_id, declared.node)
        for member, declared in alive.items()
        if declared.node != node and member not in skipped and declared.takes(channel_id, needs)
    }


def _text(value: Any) -> str:
    return value if isinstance(value, str) else bytes(value).decode()

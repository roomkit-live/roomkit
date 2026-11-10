"""The turn backend: what carries a turn between the room owner and an
executor (RFC §10.6).

A backend hands the owner a :class:`TurnLink` per turn it submits, and an
executor that serves it an :class:`ExecutorLink` per turn it takes. Each link
carries numbered messages both ways; :class:`MessageOrder` is the rule every
backend applies on arrival: a duplicate is dropped, a gap fails the turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError

logger = logging.getLogger("roomkit.turns")

if TYPE_CHECKING:
    from roomkit.models.event import RoomEvent
    from roomkit.realtime.base import EphemeralEvent
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.presence import ExecutorInfo
    from roomkit.turns.wire import (
        ChannelControl,
        ChannelResult,
        Control,
        Frame,
        HostResult,
        OwnerTool,
        RoomCall,
        RoomPublication,
        ToolCall,
        TurnRequest,
    )


@dataclass(frozen=True, slots=True)
class TurnTimeouts:
    """The bounds both sides of a turn keep (RFC §10.6.8), in seconds."""

    accept: float = 10.0
    """How long the owner waits for an executor to take the turn."""
    heartbeat_interval: float = 2.0
    """How often each side tells the other it is still there."""
    heartbeat_timeout: float = 10.0
    """How long a silence lasts before its side is taken for lost."""
    cancel: float = 5.0
    """How long the owner waits for a cancelled turn's end."""
    input_hold_timeout: float = 60.0
    """How long the owner holds an answer for a human-input request it does
    not know yet (RFC §10.6.5)."""

    @property
    def silent_intervals(self) -> int:
        """How many intervals in a row with no message make a side lost:
        counted per interval, so a stall of the counting side's own loop
        counts once."""
        return max(1, round(self.heartbeat_timeout / self.heartbeat_interval))


def response_depth(request: TurnRequest) -> int:
    """The chain depth of the turn's response (RFC §8.3): what the turn does
    on its own behalf through the owner is at least that deep, so a chain of
    turns stops at the chain-depth limit (RFC §10.6.5)."""
    return request.event.chain_depth + 1


def no_executor(channel_id: str, room_hooks: Sequence[str] = ()) -> TurnLinkError:
    """The refusal of a turn, channel call or committed event of *channel_id*
    that no executor takes, naming the room hooks *room_hooks* it needs
    (RFC §10.6.7)."""
    missing = f" with room hooks {list(room_hooks)}" if room_hooks else ""
    return TurnLinkError("no_executor", f"No executor serves channel {channel_id!r}{missing}")


def builds_nothing(channel_id: str, executor_id: str) -> TurnLinkError:
    """The failure of a turn or channel call of *channel_id* placed on an
    executor whose factory builds nothing for it: no executor serves it, and
    it is placed on no other (RFC §10.6.7)."""
    return TurnLinkError("no_executor", f"Executor {executor_id} builds no channel {channel_id!r}")


async def heartbeat_tick(interval: float) -> None:
    """Wait one heartbeat interval on the loop's own timer, so code that
    patches ``asyncio.sleep`` (a test, a host's clock) cannot make the
    heartbeats spin."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.Event().wait(), interval)


class MessageOrder:
    """The receiving end of one direction of a link: messages numbered from 1
    arrive exactly once and in order."""

    __slots__ = ("_last",)

    def __init__(self) -> None:
        self._last = 0

    def admit(self, seq: int) -> bool:
        """Whether the message *seq* is the next one; ``False`` for a
        duplicate, :class:`TurnLinkError` for a gap."""
        if seq <= self._last:
            return False
        if seq != self._last + 1:
            raise TurnLinkError("frame_lost", f"Message {self._last + 1} of the turn was lost")
        self._last = seq
        return True


class TurnLink(ABC):
    """The owner's side of one turn."""

    request: TurnRequest

    @property
    @abstractmethod
    def taken(self) -> bool:
        """Whether an executor took the request."""

    @abstractmethod
    async def receive(self) -> tuple[int, Frame]:
        """The next frame and its sequence number."""

    @abstractmethod
    async def send(self, message: Control) -> None:
        """Send a control message to the executor."""

    @abstractmethod
    def release(self) -> None:
        """Forget the turn: later frames are dropped."""

    def beat_heard(self) -> bool:
        """Whether the executor that took the turn beat since this was last
        asked, outside the turn's messages: an executor in a process of its
        own beats from a thread its turn's work cannot stall (RFC §10.6.8).
        Default: it beats by messages alone."""
        return False


class ExecutorLink(ABC):
    """The executor's side of one turn."""

    request: TurnRequest

    @abstractmethod
    async def receive(self) -> Control:
        """The next control message from the owner."""

    @abstractmethod
    async def send(self, frame: Frame) -> int:
        """Send a frame to the owner; its sequence number."""

    def turn_context(self) -> contextvars.Context:
        """The execution context the turn's work runs in: a fresh one, unless
        the backend shares the owner's process (RFC §10.6.5)."""
        return contextvars.Context()

    def turn_ended(self) -> None:  # noqa: B027
        """Called by the executor once the turn's task ended, however it
        ended (its end sent or not): the link lets go of the turn. Default:
        nothing to let go of."""

    def stream_context(self) -> contextvars.Context | None:
        """The execution context the turn's stream advances in, once known;
        ``None`` means a fresh one, kept for the stream's life."""
        return None


class OwnerTools(ABC):
    """The tools a room owner serves its turns (RFC §10.6.5): what it lists in
    a turn's request, and how it runs a call's handler."""

    @abstractmethod
    def declared(self, channel_id: str, room_id: str) -> list[OwnerTool]:
        """The tools the owner serves *channel_id*'s turns in *room_id*."""

    @abstractmethod
    async def serve(self, request: TurnRequest, call: ToolCall) -> Any:
        """Run *call*'s handler, under the call's context already installed:
        its answer. Raises what the handler raises, and
        ``UnservedToolCallError`` when nothing serves the call."""


class OwnerHost(ABC):
    """What a room owner runs for the ``TurnHost`` of its turns (RFC §10.6.5)."""

    @abstractmethod
    async def call(self, request: TurnRequest, verb: str, arguments: dict[str, Any]) -> Any:
        """Run *verb* for the turn of *request*: its result, as data JSON
        carries. Raises what the verb raises; ``ValueError`` for a verb or
        arguments it does not know."""

    async def call_between_turns(self, message: RoomCall) -> Any:
        """Run the verb of *message*, made after its turn ended, for its room
        (RFC §10.6.5): its result, as data JSON carries. Default: refused.

        Raises:
            ValueError: no verb is served between turns here.
        """
        raise ValueError(f"No verb {message.verb!r} is served between turns")

    async def channel_rows(
        self, room_id: str, channel_id: str, first: int | None
    ) -> list[RoomEvent]:
        """What a turn of *channel_id* on an executor reads of the room
        (RFC §10.6.5), *first* being the index its history starts at.
        Default: nothing."""
        return []


class OwnerEvents(ABC):
    """Where a room owner publishes what its turns published on their
    executors (RFC §10.6.3): ephemeral events on its realtime backend,
    framework events on its own bus."""

    @abstractmethod
    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        """Publish *event* on the realtime channel *channel*."""

    @abstractmethod
    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None,
        channel_id: str | None,
        event_id: str | None,
        data: dict[str, Any],
    ) -> None:
        """Emit the framework event *event_type*."""

    async def reaches_room(self, room_id: str, channel_id: str) -> bool:
        """Whether *channel_id*'s binding in *room_id*, as the room holds it
        now, lets its answers reach the whole room (RFC §8.4): what decides,
        at each publish, whether a turn's ephemerals are published. An owner
        that cannot read its rooms answers ``False``: none is published."""
        return False


class RoomOwner(ABC):
    """What reaches a room owner outside a turn (RFC §10.6.3): the owner
    implements it, and an executor holds one its backend gave it."""

    executor: TurnExecutor | None = None
    """The owner's own executor, the one that runs the channels it holds live
    (the kit's), or ``None``: a backend the owner attends places the turns of
    the channels it hosts (``hosts_channel``) on it alone, never on another
    executor, and once it closed fails them ``no_executor`` (RFC §10.6.7)."""

    @abstractmethod
    async def room_call(self, message: RoomCall) -> HostResult:
        """Run a host verb made after its turn ended: the verb's result, or
        its refusal as an error record."""

    @abstractmethod
    async def room_publication(self, message: RoomPublication) -> None:
        """Publish what a channel published after its turn ended."""


class TurnBackend(ABC):
    """Carries turns between room owners and executors (RFC §10.6)."""

    timeouts: TurnTimeouts | None = None
    """The bounds every side of its turns keeps, when the backend sets them:
    its processes must agree on them (RFC §10.6.8)."""

    _presence_claims: dict[tuple[str, str], bool] | None = None
    _executor_watchers: list[Callable[[], Awaitable[None]]] | None = None

    @abstractmethod
    async def submit(self, request: TurnRequest) -> TurnLink:
        """Start a turn: an executor that declares its channel takes it. The
        backend marks *request* handed (``request.hand_over()``) as it gives
        it to an executor, before any wait on the executor's answer."""

    @abstractmethod
    async def serve(self, executor: TurnExecutor) -> None:
        """Have *executor* take the turns of the channels it declares, from
        now until the backend closes or the executor does. A backend that
        carries work after a turn gives it a port to the owner
        (``executor.reach_owner``, RFC §10.6.3)."""

    @abstractmethod
    async def control(self, message: ChannelControl) -> ChannelResult | None:
        """Carry a channel control message (RFC §10.6.3): ``room_attached``
        and ``room_detached`` to the executors that may run the channel's
        turns (the owner's own alone for a channel it hosts, else every
        executor that declares the channel), answered by none; a
        ``committed_event`` to the executor that holds the room's state for
        the channel, else one that would take the room's turns, never moving
        the holder, answered by none; a ``channel_call``
        to the executor that holds the room's state for the channel, else one
        that declares it, answered by its result, marked handed
        (``message.hand_over()``) as the backend gives it to that executor.

        Raises:
            TurnLinkError: ``no_executor`` for a ``channel_call`` or a
                ``committed_event`` no executor takes.
        """

    async def attend(self, owner: RoomOwner) -> None:  # noqa: B027
        """Have what executors send outside a turn reach *owner* (RFC
        §10.6.3), and place the turns of the channels its own executor
        (``owner.executor``) hosts on that executor alone, closed or not (RFC
        §10.6.7).
        Default: nothing reaches an owner outside a turn, and an executor's
        verbs made after their turn are refused."""

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        """Record that *channel_id* is now *online* (or not) in *room_id*:
        ``True`` when it is a change, for the one caller that then publishes
        it there, ``False`` when it was already so (RFC §10.6.7). ``None``
        forgets it, the channel detached from the room: its next attach
        there is a change.
        Default: kept in this process, which fits one room owner process.
        A backend shared by several keeps it where they all read it."""
        if self._presence_claims is None:
            self._presence_claims = {}
        key = (channel_id, room_id)
        if online is None:
            self._presence_claims.pop(key, None)
            return False
        if self._presence_claims.get(key) == online:
            return False
        self._presence_claims[key] = online
        return True

    def _watch_executors(self, listener: Callable[[], Awaitable[None]]) -> None:
        """Have *listener* run when this process serves an executor or one it
        serves closes; changes elsewhere are read at the next look."""
        if self._executor_watchers is None:
            self._executor_watchers = []
        self._executor_watchers.append(listener)

    async def _executors_changed(self) -> None:
        for listener in list(self._executor_watchers or ()):
            try:
                await listener()
            except Exception:
                logger.exception("A watcher of the executors served failed")

    async def executors(self) -> list[ExecutorInfo]:
        """The executors this backend carries turns to, the owner's own
        executor aside, with what each declares and whether it is online
        (RFC §10.6.7, §10.6.8); a closed one is not listed.
        Default: none known, so the kit reads every agent it holds by its
        declaration alone as offline."""
        return []

    @abstractmethod
    async def close(self) -> None:
        """Fail the turns in flight (``backend_closed``) and release the
        transport."""


def lists_executors(backend: TurnBackend) -> bool:
    """Whether *backend* lists the executors it carries turns to: one that
    keeps :meth:`TurnBackend.executors` as it is tells nothing of which
    agents are served (RFC §10.6.7)."""
    return type(backend).executors is not TurnBackend.executors

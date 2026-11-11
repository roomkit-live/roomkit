"""The in-memory turn backend: the default, in the owner's process (RFC §10.6).

Every message is serialized to JSON and parsed on the other side, as any
backend carries it, so the in-memory path is the distributed one on another
transport; a request's events alone cross as they are, records nothing
changes once made (:class:`CarriedRequest`). What it alone carries besides
is the owner's execution context: a turn's
work runs in a copy of the context of the task that submitted it, and its
stream advances in a copy of the context of the task that first read it, so
host code that calls the kit from a turn finds the reentrancy guards it finds
in the owner's own process (RFC §10.6.5).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.core.exceptions import TurnLinkError
from roomkit.turns._codec import CarriedRequest, control_json
from roomkit.turns._error_codec import error_record
from roomkit.turns._organizations import RoomNeeds, concerns
from roomkit.turns._rooms import RoomRecords
from roomkit.turns.base import (
    ExecutorLink,
    MessageOrder,
    RoomOwner,
    TurnBackend,
    TurnLink,
    no_executor,
)
from roomkit.turns.presence import ExecutorInfo, local_info
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControlEnvelope,
    ChannelReleased,
    ChannelResult,
    CommittedEvent,
    ControlEnvelope,
    FrameEnvelope,
    HostResult,
    OwnerMessageEnvelope,
    Pull,
    RoomAttached,
    RoomCall,
    RoomDetached,
    RoomPublication,
    TurnRequest,
)

if TYPE_CHECKING:
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.wire import ChannelControl, Control, Frame

# Put on a pipe when the backend closes: whoever waits on it learns the turn
# will not end.
_CLOSED = "\x00closed"


logger = logging.getLogger("roomkit.turns")


class _Pipe:
    """One direction of a turn's link: numbered JSON envelopes on a queue."""

    def __init__(self, turn_id: str, envelope: type[FrameEnvelope | ControlEnvelope]) -> None:
        self._turn_id = turn_id
        self._envelope = envelope
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._order = MessageOrder()
        self._sent = 0
        self.open = True

    def put(self, message: Frame | Control) -> int:
        """Queue *message*; its sequence number. A closed pipe drops it."""
        if not self.open:
            return self._sent
        # Serialized before it is numbered: a message that cannot cross
        # raises here and takes no number, so it is never a lost frame.
        envelope = self._envelope.model_validate(
            {"turn_id": self._turn_id, "seq": self._sent + 1, "message": message}
        )
        raw = envelope.model_dump_json()
        self._sent += 1
        self._queue.put_nowait(raw)
        return self._sent

    async def get(self) -> tuple[int, Any]:
        """The next message in order, parsed anew (a frame or a control
        message, by the pipe's envelope)."""
        while True:
            raw = await self._queue.get()
            if raw == _CLOSED:
                self._queue.put_nowait(_CLOSED)
                raise TurnLinkError("backend_closed")
            envelope = self._envelope.model_validate_json(raw)
            if self._order.admit(envelope.seq):
                return envelope.seq, envelope.message

    def shut(self) -> None:
        """Stop the pipe: its waiters raise, later messages are dropped."""
        self.open = False
        self._queue.put_nowait(_CLOSED)


class _MemoryTurn:
    """What both sides of one in-memory turn share: its two pipes, the
    request as it is carried, and the owner's execution contexts."""

    def __init__(self, request: TurnRequest) -> None:
        self.request = CarriedRequest.of(request)
        self.channel_id = request.channel_id
        self.frames = _Pipe(request.turn_id, FrameEnvelope)
        self.controls = _Pipe(request.turn_id, ControlEnvelope)
        self.turn_context = contextvars.copy_context()
        self.stream_context: contextvars.Context | None = None
        self.taken = False

    def shut(self) -> None:
        self.frames.shut()
        self.controls.shut()


class _OwnerLink(TurnLink):
    def __init__(self, turn: _MemoryTurn, request: TurnRequest, backend: InMemoryTurnBackend):
        self._turn = turn
        self._backend = backend
        self.request = request

    @property
    def taken(self) -> bool:
        return self._turn.taken

    async def receive(self) -> tuple[int, Frame]:
        seq, frame = await self._turn.frames.get()
        return seq, frame

    async def send(self, message: Control) -> None:
        if isinstance(message, Pull) and self._turn.stream_context is None:
            # The task that first reads the stream lends it its context.
            self._turn.stream_context = contextvars.copy_context()
        self._turn.controls.put(message)

    def release(self) -> None:
        self._turn.shut()
        self._backend._forget(self._turn)


class _ExecutorLink(ExecutorLink):
    def __init__(self, turn: _MemoryTurn) -> None:
        self._turn = turn
        self.request = turn.request.request()

    async def receive(self) -> Control:
        _, message = await self._turn.controls.get()
        return message

    async def send(self, frame: Frame) -> int:
        return self._turn.frames.put(frame)

    def turn_context(self) -> contextvars.Context:
        return self._turn.turn_context

    def stream_context(self) -> contextvars.Context | None:
        return self._turn.stream_context


class InMemoryTurnBackend(TurnBackend):
    """Turns carried in this process, serialized like any other but a
    request's events (:class:`CarriedRequest`, RFC §10.6).

    The turns of a channel the owner's own executor hosts go to it alone,
    and fail ``no_executor`` once it closed.
    Another channel's turns in a room go to the executor that took its last
    turn or channel call while it serves and declares the channel, else to
    the first that declares it; only to one serving the room's organization
    (RFC §10.6.7). Every executor is in this
    process, so a turn none declares fails at once with ``no_executor``.
    """

    def __init__(self) -> None:
        self._executors: list[TurnExecutor] = []
        # Where each room's state of a channel is: the executor that took its
        # last turn or channel call.
        self._holders: RoomRecords[TurnExecutor] = RoomRecords()
        self._turns: set[_MemoryTurn] = set()
        self._closed = False
        # The owner what executors send outside a turn reaches (RFC §10.6.3).
        self._owner: RoomOwner | None = None
        self._port = _OwnerPort(self)

    async def submit(self, request: TurnRequest) -> TurnLink:
        if self._closed:
            raise TurnLinkError("backend_closed")
        executor = self._place(request.channel_id, request.room_id, RoomNeeds.of_turn(request))
        return self._submit_to(executor, request)

    def _submit_to(self, executor: TurnExecutor, request: TurnRequest) -> TurnLink:
        """Start a turn on *executor*, an executor of this process another
        backend placed it on; carried here, as :class:`CarriedRequest` carries
        a request, in the owner's execution context."""
        if self._closed:
            raise TurnLinkError("backend_closed")
        turn = _MemoryTurn(request)
        self._turns.add(turn)
        turn.taken = True
        request.hand_over()
        executor.take(_ExecutorLink(turn))
        return _OwnerLink(turn, request, self)

    async def serve(self, executor: TurnExecutor) -> None:
        own = self._owner.executor if self._owner is not None else None
        if executor not in self._executors:
            self._executors.append(executor)
            if executor is not own:
                executor._listen_closed(self._executors_changed)
        executor.reach_owner(self._port)
        if executor is not own:
            await self._executors_changed()

    async def attend(self, owner: RoomOwner) -> None:
        if self._owner is not None and self._owner is not owner:
            # Which owner a turn's work belongs to is a distributed backend's
            # to address; here, the first one keeps it.
            logger.warning(
                "A second room owner attends this in-memory turn backend: the work of "
                "a turn after it ended reaches the first one only"
            )
            return
        self._owner = owner

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        if self._closed:
            raise TurnLinkError("backend_closed")
        raw = control_json(message)
        if isinstance(message, ChannelCall):
            needs = RoomNeeds(organization_id=message.organization_id)
            executor = self._place(message.channel_id, message.room_id, needs)
            message.hand_over()
            answer = await executor.control(_crossed(raw))
            if answer is None:
                return ChannelResult()
            return ChannelResult.model_validate_json(answer.model_dump_json())
        if isinstance(message, CommittedEvent):
            needs = RoomNeeds(tuple(message.room_hooks), message.organization_id)
            holder = self._hold(message.channel_id, message.room_id, needs)
            await holder.control(_crossed(raw))
            return None
        if isinstance(message, RoomDetached):
            self._holders.pop(message.channel_id, message.room_id)
        for executor in self._told(message):
            await executor.control(_crossed(raw))
        return None

    async def close(self) -> None:
        self._closed = True
        for turn in list(self._turns):
            turn.shut()
        self._turns.clear()

    def _hold(self, channel_id: str, room_id: str, needs: RoomNeeds) -> TurnExecutor:
        """The executor that would take the room's turns of *channel_id*, one
        that meets what the room *needs*; the holder is left as it was (RFC
        §10.6.3)."""
        taker = self._taker(channel_id, room_id, needs)
        if taker is None:
            raise no_executor(channel_id, needs)
        return taker

    def _place(self, channel_id: str, room_id: str, needs: RoomNeeds) -> TurnExecutor:
        """The executor a room's turn or channel call of *channel_id* goes to,
        one that meets what the room *needs* (its hooks, its organization),
        recorded as the one holding the room's state from now."""
        holder = self._taker(channel_id, room_id, needs)
        if holder is None:
            raise no_executor(channel_id, needs)
        self._holders.put(channel_id, room_id, holder)
        return holder

    def _taker(self, channel_id: str, room_id: str, needs: RoomNeeds) -> TurnExecutor | None:
        """The executor that takes a room's turn of *channel_id*: the owner's
        own alone when it hosts the channel, closed or not, since it runs the
        channels the owner holds live; else the one holding the room's state while it
        takes it, else the first that takes it (RFC §10.6.7); ``None`` when
        none does."""
        own = self._owning(channel_id)
        if own is not None:
            # Its live channels never run on a copy, even once it closed.
            return own if own in self._executors and own._meets(channel_id, needs) else None
        holder = self._holders.get(channel_id, room_id)
        if holder is not None and holder._meets(channel_id, needs):
            return holder
        return next((e for e in self._executors if e._meets(channel_id, needs)), None)

    def _told(self, message: RoomAttached | RoomDetached | ChannelReleased) -> list[TurnExecutor]:
        """The executors a room's attach or detach, or a channel's release,
        reaches: those that may run the channel's turns, serving the room's
        organization for an attach or a detach; the owner's own alone for a
        channel it hosts (RFC §10.6.3)."""
        channel_id = message.channel_id
        own = self._owning(channel_id)
        if own is not None:
            candidates = [own] if own in self._executors else []
        else:
            candidates = self._executors
        return [
            e for e in candidates if e.declares(channel_id) and concerns(e.organizations, message)
        ]

    async def executors(self) -> list[ExecutorInfo]:
        own = self._owner.executor if self._owner is not None else None
        return [local_info(e) for e in self._executors if e is not own and not e.closed]

    def _owning(self, channel_id: str) -> TurnExecutor | None:
        """The owner's own executor when it hosts *channel_id*, closed or not:
        no other executor runs the channels the owner holds live (RFC §10.6.7)."""
        own = self._owner.executor if self._owner is not None else None
        return own if own is not None and own.hosts_channel(channel_id) else None

    def _forget(self, turn: _MemoryTurn) -> None:
        self._turns.discard(turn)


def _crossed(raw: str) -> ChannelControl:
    """A channel control message as the executor parses it."""
    return ChannelControlEnvelope.model_validate_json(raw).message


class _OwnerPort(RoomOwner):
    """What an executor served here reaches the owner by outside a turn: each
    message serialized and parsed again, as on any backend, the owner's side
    run in a context of its own, never the executor's."""

    def __init__(self, backend: InMemoryTurnBackend) -> None:
        self._backend = backend

    async def room_call(self, message: RoomCall) -> HostResult:
        owner = self._backend._owner
        if owner is None:
            refusal = RuntimeError("No room owner is reached outside a turn on this backend")
            return HostResult(call_id=message.call_id, error=error_record(refusal))
        crossed = _crossed_to_owner(message)
        assert isinstance(crossed, RoomCall)
        answer = await _apart(owner.room_call(crossed))
        return HostResult.model_validate_json(answer.model_dump_json())

    async def room_publication(self, message: RoomPublication) -> None:
        owner = self._backend._owner
        if owner is None:
            return
        crossed = _crossed_to_owner(message)
        assert isinstance(crossed, RoomPublication)
        await _apart(owner.room_publication(crossed))


def _crossed_to_owner(message: RoomCall | RoomPublication) -> RoomCall | RoomPublication:
    raw = OwnerMessageEnvelope(message=message).model_dump_json()
    return OwnerMessageEnvelope.model_validate_json(raw).message


async def _apart[T](work: Coroutine[Any, Any, T]) -> T:
    """Run *work* as a task of its own context: the owner's side shares no
    context variable with the executor's."""
    return await asyncio.get_running_loop().create_task(work, context=contextvars.Context())

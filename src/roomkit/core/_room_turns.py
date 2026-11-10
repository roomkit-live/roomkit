"""A channel's turns in a room, one after the other (RFC §6.4).

A channel that declares ``serializes_room_turns`` has its turns in a room run
in the order they were asked, each once the one before it is over. The room's
owner keeps the order (:class:`RoomTurns`), per process, as an ACP agent's
channel keeps its own.

A turn takes its place in its channel's line when it is asked, in the room's
delivery lane, so the line follows the order the lane delivers in. It waits
for its place in its stream (:class:`QueuedTurn`), never in the lane: the
previous turn's answer is delivered through that lane, so a turn waiting there
would wait on itself. It starts when its delivery first reads it, on the room
as it is then, less what others wrote after it was asked
(:func:`room_as_asked`), and gives its place up once its reader has written
what it said. A cancelled delivery closes its streams, so a turn still waiting
never starts.
"""

from __future__ import annotations

import asyncio
import contextvars
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import Any

from roomkit._media import text_form
from roomkit.channels._speaker import labelled_lines
from roomkit.channels.base import Channel
from roomkit.core._changes import for_channels, opens_no_turn
from roomkit.core._requester import asking_label
from roomkit.core.lanes import anchor_index
from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.steering import InjectMessage, SteeringDirective
from roomkit.providers.utils import _aclose_stream
from roomkit.telemetry.context import get_current_span, restored_span
from roomkit.turns.owner import drop_unread, read_as_trace

Reaction = Callable[[Channel, RoomEvent, ChannelBinding, RoomContext], Awaitable[ChannelOutput]]
"""How the room's owner runs a channel's turn: ``EventRouter.react``."""

Steer = Callable[..., Awaitable[int]]
"""How the room's owner steers a channel's running turns in a room:
``TurnClient.steer(room_id, channel_id, directive)``, how many it reached."""

_HELD: contextvars.ContextVar[tuple[_Place, ...]] = contextvars.ContextVar(
    "roomkit_room_turn_places", default=()
)
"""The places of the turns the current task reads, and the work they run in
(their tools, their hooks) sees: what a call made from them must not wait on."""


class _Place:
    """A turn's place in its channel's line in a room: taken when the turn is
    asked, reached once the turns asked before it are over, given up once
    its reader is done with it."""

    __slots__ = ("_turns", "_key", "_reached", "_left")

    def __init__(self, turns: RoomTurns, key: tuple[str, str]) -> None:
        self._turns = turns
        self._key = key
        self._reached = asyncio.Event()
        self._left = False

    async def until_reached(self) -> None:
        """Wait until the place is reached, or given up."""
        await self._reached.wait()

    async def hold(self) -> None:
        """Wait until the place is reached, then hold it in the current
        task, which the work of its turn sees (:data:`_HELD`)."""
        await self._reached.wait()
        _HELD.set((*(place for place in _HELD.get() if not place._left), self))

    def behind_one_held_here(self) -> bool:
        """Whether the place waits behind a turn the current task holds."""
        return any(
            place._key == self._key and not place._left and place is not self
            for place in _HELD.get()
        )

    def leave(self) -> None:
        """Give the place up, once however often it is called; a reader
        still waiting for it goes on, to a turn that is closed."""
        self._left = True
        self._turns._leave(self._key, self)
        self._reached.set()


class RoomTurns:
    """The lines of the channels that take a room's turns one at a time: one
    line per channel and room, kept while a turn of it waits or runs."""

    def __init__(
        self,
        react: Reaction,
        context_of: Callable[..., Awaitable[RoomContext]] | None,
        telemetry: Any,
        steer: Steer | None = None,
    ) -> None:
        self._react = react
        self._context_of = context_of
        self._telemetry = telemetry
        self._steer = steer
        self._lines: dict[tuple[str, str], list[_Place]] = {}

    def join(self, room_id: str, channel_id: str) -> _Place:
        """A place in *channel_id*'s line in *room_id*, after the turns asked before it."""
        key = (room_id, channel_id)
        line = self._lines.setdefault(key, [])
        place = _Place(self, key)
        line.append(place)
        if len(line) == 1:
            place._reached.set()
        return place

    def _leave(self, key: tuple[str, str], place: _Place) -> None:
        """*place* leaves its line; the next one is reached if it was first."""
        line = self._lines.get(key)
        if line is None or place not in line:
            return
        first = line[0] is place
        line.remove(place)
        if not line:
            del self._lines[key]
        elif first:
            line[0]._reached.set()

    async def ask(
        self, channel: Channel, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """*channel*'s answer to *event*: a turn that takes its place now and
        starts once the turns asked before it are over, when its stream is
        first read; none when the turn *channel* runs in the room reads
        *event* instead (:meth:`_steered`)."""
        if await self._steered(channel, event, context):
            return ChannelOutput.empty()
        place = self.join(context.room.id, channel.channel_id)
        record = ResponseMetadata()
        start = partial(self._start, channel, event, binding, context, get_current_span())
        turn = QueuedTurn(place, start, record)
        return ChannelOutput(responded=True, response_stream=turn, response_metadata=record)

    async def _steered(self, channel: Channel, event: RoomEvent, context: RoomContext) -> bool:
        """Whether *event* was handed to the turn *channel* runs in the room,
        which reads it at its next stop rather than a turn of its own (RFC
        §6.4, §21.3): for a channel that steers its room turns, when its line
        in the room holds that one turn alone, so no turn asked before
        *event* is passed over, and the turn takes it (``steer`` reached it)."""
        if self._steer is None or not channel.steers_room_turns:
            return False
        line = self._lines.get((context.room.id, channel.channel_id))
        if line is None or len(line) != 1:
            return False
        message = steered_message(event, context, channel.channel_id)
        if message is None:
            return False
        reached = await self._steer(context.room.id, channel.channel_id, message)
        return reached > 0

    async def _start(
        self,
        channel: Channel,
        event: RoomEvent,
        binding: ChannelBinding,
        asked: RoomContext,
        span_id: str | None,
    ) -> ChannelOutput:
        """Run a turn that reached its place, under the span of the delivery
        that asked it; no turn when its channel left the room meanwhile."""
        now = await self._room_now(asked)
        if now.get_binding(channel.channel_id) is None:
            return ChannelOutput.empty()
        context = room_as_asked(asked, now, channel.channel_id)
        telemetry = self._telemetry
        telemetry_ctx = telemetry.get_span_context(span_id) if telemetry and span_id else None
        with restored_span(span_id, telemetry_ctx=telemetry_ctx):
            output = await self._react(channel, event, binding, context)
        _refuse_unstreamed(channel, output)
        return output

    async def _room_now(self, asked: RoomContext) -> RoomContext:
        """The room as it is now, holding as much of its history up to the
        turn's horizon as the turn was asked with: the events committed while
        it waited would otherwise push that history out of the window."""
        if self._context_of is None:
            return asked
        room_id = asked.room.id
        now = await self._context_of(room_id)
        horizon = _horizon(asked)
        kept = sum(1 for event in now.recent_events if event.index <= horizon)
        latest = now.room.latest_index
        if kept < len(asked.recent_events) and latest is not None and latest > horizon:
            # Never past the ceiling a context holds, however many came.
            limit = min(len(asked.recent_events) + latest - horizon, _RECENT_EVENTS_LIMIT)
            now = await self._context_of(room_id, recent_limit=limit)
        return for_channels(now)


def steered_message(
    event: RoomEvent, context: RoomContext, channel_id: str
) -> SteeringDirective | None:
    """*event* as the turn it is handed to reads it (RFC §6.4): the text its
    content reads as, a participant's words under their label when the room
    has several speakers, as the turn's history labels them; none for an
    event that reads as no text, which waits a turn of its own, nor for an
    event that opens no turn (a notice, a reaction: RFC §6.4; an edit or a
    delete: RFC §10.3), whose own turn the agent does not take."""
    if opens_no_turn(event):
        return None
    text = text_form(event.content)
    if not text:
        return None
    # Labelled as the turn's history labels it: the speakers the agent may
    # read, its own turns and the application's instructions aside.
    label = asking_label(event, context, channel_id)
    if label:
        text = labelled_lines(text, label)
    return InjectMessage(content=text)


def _refuse_unstreamed(channel: Channel, output: ChannelOutput) -> None:
    """A turn that waited its place answers through its stream: what it
    returned beside it would reach nobody, so it fails instead."""
    if output.response_events or output.tasks or output.observations or output.metadata_updates:
        raise TypeError(
            f"Channel {channel.channel_id!r} takes a room's turns one at a time "
            "(serializes_room_turns) and answered beside its response stream"
        )


def room_as_asked(asked: RoomContext, now: RoomContext, channel_id: str) -> RoomContext:
    """The room a turn that waited reads: the room *now*, less the events
    committed after the turn was asked (*asked*) that *channel_id* did not
    write. Its previous turns' answers are in it, and so is what changed
    meanwhile (an edit, a deletion, RFC §10.3); a message that came after it
    waits a turn of its own (RFC §6.4)."""
    horizon = _horizon(asked)
    recent = [
        event
        for event in now.recent_events
        if event.index <= horizon or event.source.channel_id == channel_id
    ]
    return now.model_copy(update={"recent_events": recent})


def _horizon(asked: RoomContext) -> int:
    """The index of the room's last event when the turn was asked."""
    return max(
        (event.index for event in asked.recent_events),
        default=anchor_index(asked.room.latest_index),
    )


async def _no_items() -> AsyncIterator[Any]:
    return
    yield  # pragma: no cover - makes this an async generator


class QueuedTurn:
    """The stream of a channel's turn that waits its place in the room.

    Nothing runs before it is read. Its first read waits until the place is
    reached, starts the turn, then hands the turn's stream on item by item.
    :meth:`stop` ends the turn and keeps the place, for a reader that still
    has to write what the turn said; :meth:`aclose` ends it and gives the
    place up, which its reader does once it is done, so the next turn reads
    that answer. *record* is the turn's response metadata as its reader reads
    it, brought in line with the turn's own record at each item.
    """

    def __init__(
        self,
        place: _Place,
        start: Callable[[], Awaitable[ChannelOutput]],
        record: ResponseMetadata,
    ) -> None:
        self._place = place
        self._start = start
        self._record = record
        self._turn_record: ResponseMetadata | None = None
        self._stream: AsyncIterator[Any] | None = None
        self._as_trace = False
        self._closed = False
        self._reading = False
        # A turn nobody closes still gives its place up when it is collected;
        # never at the interpreter's exit, whose loop is gone.
        weakref.finalize(self, place.leave).atexit = False

    def __aiter__(self) -> QueuedTurn:
        return self

    async def until_reached(self) -> None:
        """Wait until the place is reached, as the first read does: a reader
        waiting here is reading the turn, so :meth:`drop` leaves it to it."""
        if self._closed or self._stream is not None:
            return
        self._reading = True
        try:
            await self._place.until_reached()
        finally:
            self._reading = False

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        self._reading = True
        try:
            if self._stream is None:
                self._stream = await self._begin()
            return await self._stream.__anext__()
        finally:
            self._reading = False
            self._mirror()

    async def _begin(self) -> AsyncIterator[Any]:
        """Wait until the place is reached, then start the turn; its stream."""
        await self._place.hold()
        try:
            output = await self._start()
        except BaseException:
            self._place.leave()
            raise
        if output.error is not None:
            await _aclose_stream(output.response_stream)
            self._place.leave()
            raise output.error
        self._turn_record = output.response_metadata
        if output.response_stream is None:
            return _no_items()
        if self._as_trace:
            read_as_trace(output.response_stream)
        return output.response_stream

    def _mirror(self) -> None:
        """Bring the record the reader reads in line with the turn's own."""
        turn = self._turn_record
        if turn is None or turn is self._record:
            return
        for key in [key for key in self._record if key not in turn]:
            del self._record[key]
        self._record.update(turn)

    @property
    def waits_on_caller(self) -> bool:
        """Whether reading it here would wait behind the current task's own
        turn: a message a turn sends into its own room, asking its own
        channel, is answered after that turn, so its caller does not wait for
        the answer (RFC §6.4)."""
        return self._place.behind_one_held_here()

    @property
    def executor_id(self) -> str | None:
        """The executor that took the turn, once it started."""
        return getattr(self._stream, "executor_id", None)

    @property
    def end_owed(self) -> bool:
        """Whether the turn's executor still owes its end (RFC §10.6.8)."""
        return bool(getattr(self._stream, "end_owed", False))

    async def stop(self) -> None:
        """End the turn, what it has not said yet discarded, keeping the
        place until its reader has written what it did say."""
        self._closed = True
        if self._stream is not None:
            await _aclose_stream(self._stream)

    async def aclose(self) -> None:
        """End the turn and give its place up."""
        try:
            await self.stop()
        finally:
            self._place.leave()

    def drop(self) -> None:
        """Give the turn up unread: one never started holds a place, which
        goes to the next turn. A turn being read is its reader's to close."""
        if self._reading:
            return
        self._closed = True
        if self._stream is not None:
            drop_unread(self._stream)
        self._place.leave()

    def read_as_trace(self) -> None:
        """Read the turn as a delegated task's trace (RFC §23.3 step 5), from
        its start when it has not started yet."""
        self._as_trace = True
        if self._stream is not None:
            read_as_trace(self._stream)


async def until_turn_starts(stream: Any) -> None:
    """Wait until the turn *stream* carries may start: a turn that waits its
    place, until it is reached; any other stream starts when read."""
    if isinstance(stream, QueuedTurn):
        await stream.until_reached()


async def stop_turn(stream: Any) -> None:
    """End the turn *stream* carries, its reader still to write what it said:
    a turn that waited its place keeps it until its reader closes it."""
    if isinstance(stream, QueuedTurn):
        await stream.stop()
    else:
        await _aclose_stream(stream)

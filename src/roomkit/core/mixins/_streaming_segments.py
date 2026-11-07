"""The writer of a streamed turn's timeline rows."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from roomkit.channels._tool_event_result import tool_event_payload
from roomkit.core._closing import KIT_CLOSED, ended_by_close
from roomkit.core._instructions import response_metadata
from roomkit.core._observed import Observed, observed_once_over
from roomkit.core._queued_injections import QueuedInjections
from roomkit.core.event_router import stream_record
from roomkit.core.exceptions import KitClosedError, RoomClosedError, RoomNotFoundError
from roomkit.core.mixins._stream_reach import reads_the_stream
from roomkit.core.mixins.helpers import _refuses_writes, _source_block_reason
from roomkit.core.mixins.inbound_locked import _Blocked, _Refused, within_answer_scope
from roomkit.core.mixins.lane_execution import DeliverySource
from roomkit.core.task_utils import cancel_reason
from roomkit.core.visibility import answer_scope, answers_reach, with_committed_visibility
from roomkit.models.enums import EventDirection, EventStatus, EventType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.response_metadata import recorded_turn_end
from roomkit.models.streaming import (
    InstructionReadMarker,
    LoopEndMarker,
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
    UsageMarker,
    add_usage,
    reports_counters,
)

if TYPE_CHECKING:
    from roomkit.core.event_router import StreamingResponse
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.core.mixins._response_reader import ResponseReader
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.inbound")

_HOLD_LIMIT = 256
"""The most text a segment's start is held back for, in case it is a
discussion's silence token (RFC §19.7.5 rule 13): the token is short."""


class RowSink(Protocol):
    """How a streamed turn's rows are committed, the one thing writers differ in."""

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        """Commit *event*: the stored row, or ``None`` when it was refused or excluded.

        *exclude* names the channels the row must not be delivered to.
        """
        ...

    def reads(self, channel_id: str) -> bool:
        """Whether *channel_id* may still read the stream, as the last
        commit read the room."""
        ...


@dataclass(frozen=True)
class TurnScope:
    """Where a streamed turn writes its rows in a room: one deeper than the
    event it answers, in that event's answer scope and thread (RFC §8.3)."""

    chain_depth: int
    visibility: str
    response_visibility: str | None
    parent_event_id: str | None

    @classmethod
    def answering(cls, trigger: RoomEvent) -> TurnScope:
        """The scope of a turn that answers *trigger*.

        The trigger's thread parent is inherited, so the reply lands in the
        same thread; the caller hands the trigger with its parent rooted
        (RFC §10.1 step 8), and a top-level trigger keeps the reply
        top-level. A scope of ``internal``,
        which only the framework writes, commits the rows with ``none``,
        with a warning (RFC §7.3, §8.5). The rows carry the trigger's answer
        scope, so what answers a row stays within the perimeter of the
        message the cascade derives from (§5.12).
        """
        visibility = answers_reach(trigger)
        if visibility == Visibility.INTERNAL:
            logger.warning(
                "The answer to event %s is scoped 'internal', which only the framework "
                "writes: its stream's rows are committed with 'none'",
                trigger.id,
                extra={"room_id": trigger.room_id, "event_id": trigger.id},
            )
            visibility = Visibility.NONE
        return cls(
            chain_depth=trigger.chain_depth + 1,
            visibility=visibility,
            response_visibility=answer_scope(trigger.response_visibility),
            parent_event_id=trigger.parent_event_id,
        )


def _ended_record(marker: LoopEndMarker) -> dict[str, Any]:
    """The record of a turn whose loop reached its end, as its *marker* tells
    it (RFC §6.4): how it ended and what it used, then what a channel that
    says so said of its generations."""
    record: dict[str, Any] = {"ai_usage": dict(marker.usage), "loop_end_reason": marker.reason}
    if marker.usage_complete is not None:
        record["ai_last_usage"] = marker.last_usage
        record["ai_usage_complete"] = marker.usage_complete
    if marker.model is not None:
        record["ai_model"] = marker.model
    if marker.provider is not None:
        record["ai_provider"] = marker.provider
    return record


class _GenerationsRead:
    """What a turn's reader was told of its generations, for the record of a
    turn cut before its loop's end could tell it (RFC §6.4).

    A generation told its usage ends one. The next counts as in flight from the
    end of the calls that followed it, or from a segment break, until it tells:
    the safe side, a reader that stops right there reading incomplete. A
    provider's own calls run within a generation and change nothing.
    """

    def __init__(self) -> None:
        self.usage: dict[str, int] = {}
        self.last_usage: dict[str, Any] | None = None
        self.complete = True
        self.model: str | None = None
        self.provider: str | None = None
        self._in_flight = True
        self._open_calls = 0

    def read(self, item: Any) -> None:
        """Take in one item of the stream."""
        if isinstance(item, UsageMarker):
            add_usage(self.usage, item.usage)
            self.last_usage = item.usage
            self.complete = self.complete and reports_counters(item.usage)
            self.model, self.provider = item.model, item.provider
            self._in_flight = False
        elif isinstance(item, ToolCallStartMarker):
            self._open_calls += 1
        elif isinstance(item, ToolCallEndMarker):
            self._open_calls -= 1
            if self._open_calls <= 0:
                self._in_flight = True
        elif isinstance(item, SegmentBreakMarker):
            self._in_flight = True

    def cut_record(self) -> dict[str, Any]:
        """The record of a turn cut here, as its loop's end would have written
        it from what the generations told; with no counter told, no usage.

        A stream that told nothing has no more to say than its end: an ACP
        agent's or a text-only channel's reports no usage at all, and a first
        generation in flight cannot be told from them.
        """
        if self.last_usage is None:
            return {"loop_end_reason": "cancelled"}
        record = _ended_record(
            LoopEndMarker(
                reason="cancelled",
                usage=dict(self.usage),
                last_usage=self.last_usage,
                usage_complete=self.complete and not self._in_flight,
                model=self.model,
                provider=self.provider,
            )
        )
        if not self.usage:
            del record["ai_usage"]
        return record


class SegmentWriter:
    """The one writer of a streamed turn's timeline rows.

    A stream produces three kinds of row — a text segment, a tool call's
    start, its end — and they are verbs of one writer, whose commit is its
    own step, through its sink: a hook's decision (a display label, a PII
    rewrite, a refusal) reaches every kind of row alike, and a fourth kind
    cannot be written past it.

    Each verb returns the row it committed, or ``None`` when there was
    nothing to write, a hook refused it, or the persistence policy excluded
    it (RFC §14.3) — which is exactly what the caller yields into the stream.

    How a row is committed is the *sink*'s: a room's rows cross the hooks
    and ride its lane (:class:`LaneSink`), a delegated turn's trace is
    committed as is (RFC §23.3). Everything else is this writer's, for both.
    """

    def __init__(
        self,
        kit: Any,
        sr: StreamingResponse,
        sink: RowSink,
        *,
        room_id: str,
        chain_depth: int,
        visibility: str = Visibility.ALL,
        response_visibility: str | None = None,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        streamed_to: set[str] | None = None,
        response_events: list[RoomEvent] | None = None,
        silent: Any = None,
        hands_on_record: bool = False,
    ) -> None:
        self._kit = kit
        self._sr = sr
        self._sink = sink
        self._room_id = room_id
        self._chain_depth = chain_depth
        self._visibility = visibility
        # The trigger's answer scope rides every row, so what another agent
        # answers to a row is scoped as the turn itself is (RFC §5.12, §8.3).
        self._response_visibility = response_visibility
        self._correlation_id = correlation_id
        self._parent_event_id = parent_event_id
        # The channel a text segment reaches *as it is produced* — the stream
        # itself is its delivery, so the lane must not send it again. Only the
        # first target streams (V1); any other streaming-capable channel is an
        # ordinary recipient.
        self._streamed_to = streamed_to if streamed_to is not None else set()
        self._response_events = response_events
        # The turn's record (``loop_end_reason``, ``ai_usage``), once its
        # ``LoopEndMarker`` is read; ``_record_owed`` while no row carries it.
        self._turn_record: dict[str, Any] | None = None
        self._record_owed = False
        # A turn whose text is kept out of the room hands its record on: it
        # rides none of its rows (RFC §19.7.3).
        self._hands_on_record = hands_on_record
        self._accumulated: list[str] = []
        self._writing: set[asyncio.Task[RoomEvent | None]] = set()
        # A discussion's silence token (RFC §19.7.5): the start of each text
        # segment is held back from the rendering channel while it may still
        # become the token, since a delivered segment is not taken back (§6.4).
        self._silent = silent
        self._held: list[str] = []
        self._held_chars = 0
        self._released = False
        # The starts whose row was handed to the writer, by identity: two
        # calls under one id are two calls (RFC §12.4).
        self._started: list[ToolCallStartMarker] = []
        # Whether the reader was handed anything, a thinking delta included:
        # from there the turn began, and a stop cuts it.
        self._began = False
        self._generations = _GenerationsRead()
        # The rows handed to the target in the stream, by id; whether the
        # room stopped letting it read the stream (``_stopping``, set by
        # :meth:`_hand_to_target`, the read handing it nothing more); and
        # whether the stream ended there, on its own (:attr:`withdrawn`), not
        # on its reader's stop, a barge-in cutting the read mid-commit.
        self._streamed_rows: set[str] = set()
        self._stopping = False
        self.withdrawn = False
        self.persisted: list[RoomEvent] = []
        # How :meth:`stream` ended: read to its end, or on the response's failure.
        self.read_to_end = False
        self.failure: Exception | None = None

    # -- what the stream hands in ------------------------------------------

    def add_text(self, delta: str) -> None:
        """Accumulate a text delta; a segment is written when something ends it."""
        self._accumulated.append(delta)

    def end_turn(self, marker: LoopEndMarker) -> None:
        """The loop reached its end: its record rides the turn's last message.

        The marker comes last, so the next flush is the turn's final text and
        carries it; a turn with no final text has it written on the message it
        already wrote, or on its last row when it wrote none, by
        :meth:`write_owed_record` (RFC §6.4). It rides the stream too, where a
        reader of the turn finds how it ended.
        """
        self._turn_record = _ended_record(marker)
        self._record_owed = True
        self._sr.turn_record = self._turn_record

    def end_stopped(self) -> None:
        """A turn its reader stopped once it began ends ``cancelled``: its
        record rides the stream and its kept text, never read as a turn that
        completed (RFC §6.4, §12.2 step 13s). An end its loop's record names
        holds (the reader was handed it, so it read the whole turn), and so
        does one an ACP agent's names when it says how the turn was cut
        (``interrupted``, a stop reason); its ``completed`` says the agent
        finished, not that the reader read it. A turn never read names none."""
        began = self._began or bool(self._accumulated or self.persisted or self._started)
        if not began or self._turn_record is not None:
            return
        if recorded_turn_end(stream_record(self._sr)) not in (None, "completed"):
            return
        self._turn_record = self._generations.cut_record()
        self._record_owed = True
        self._sr.turn_record = self._turn_record

    async def write_owed_record(self) -> None:
        """Write a record no final text carried on the last MESSAGE already
        stored, or on the turn's last row when it wrote no message.

        Run once the turn's deliveries are done, so no delivery record is
        written to the same row meanwhile. Only the row's metadata changes,
        through ``update_event``, so ON_EVENT_UPDATED sees it; it is not
        delivered again. Best effort: the turn's outcome stands without it.
        """
        if not self._record_owed or self._turn_record is None or self._hands_on_record:
            return
        self._record_owed = False
        last = next((e for e in reversed(self.persisted) if e.type == EventType.MESSAGE), None)
        if last is None and self.persisted:
            last = self.persisted[-1]
        if last is None:
            return
        try:
            stored = await self._kit.store.get_event(last.id)
            if stored is None:
                return
            updated = await self._kit.update_event(
                self._room_id, last.id, metadata={**stored.metadata, **self._turn_record}
            )
        except Exception as exc:
            if ended_by_close(self._kit, exc):
                # The kit's close refused it: the row keeps the end its
                # commit carried (RFC §12.10.4).
                logger.debug("The turn's end on row %s not rewritten: the kit closed", last.id)
                return
            logger.warning(
                "Could not record the turn's end on row %s (room %s)",
                last.id,
                self._room_id,
                exc_info=True,
            )
            return
        if updated is None:
            return
        for rows in (self.persisted, self._response_events):
            if rows is not None and last in rows:
                rows[rows.index(last)] = updated

    def stream_lost(self) -> None:
        """The streaming channel failed: what it was handed may not all have
        been rendered, so the text still to write goes out like any other event."""
        self._streamed_to.clear()

    async def read(
        self, reader: ResponseReader
    ) -> AsyncGenerator[str | ThinkingDeltaMarker | UsageMarker | RoomEvent, None]:
        """Read the response through *reader*, writing each row as it ends.

        Yields every text delta, thinking and usage marker as it arrives, for
        the channel rendering the stream, and every row committed. Stops at the
        stream's end, before the final text is written: the caller writes it.
        """
        while True:
            try:
                delta = await reader.next()
            except StopAsyncIteration:
                if held := self._end_held():
                    yield held
                return
            self._began = True
            self._generations.read(delta)
            if isinstance(delta, str):
                self.add_text(delta)
                if self._silent is None:
                    yield delta
                elif released := self._release_text(delta):
                    yield released
            elif isinstance(delta, ThinkingDeltaMarker | UsageMarker):
                yield delta
            else:
                if held := self._end_held():
                    yield held
                for row in await self.take(delta):
                    if row.id in self._streamed_rows:
                        yield row
                if self._stopping:
                    return
                if not self._accumulated:
                    self._released = False

    def _release_text(self, delta: str) -> str:
        """What of *delta* reaches the rendering channel now: all of it, unless
        the segment's start may still become the silence token, which holds it
        back; the held start goes out in one piece once it cannot."""
        if self._silent is None or self._released:
            return delta
        self._held.append(delta)
        self._held_chars += len(delta)
        # A start this long is no bare token: released, and read no further.
        if self._held_chars <= _HOLD_LIMIT and self._silent.may_become("".join(self._accumulated)):
            return ""
        self._released = True
        held, self._held, self._held_chars = "".join(self._held), [], 0
        return held

    def _end_held(self) -> str:
        """The held start of a segment that ends: what goes out, nothing when
        the segment is the silence token (its row is stored, never delivered)."""
        held, self._held, self._held_chars = "".join(self._held), [], 0
        if held and self._silent.is_silent("".join(self._accumulated)):
            return ""
        return held

    async def stream(
        self, reader: ResponseReader
    ) -> AsyncGenerator[str | ThinkingDeltaMarker | UsageMarker | RoomEvent, None]:
        """The response as the channel rendering it reads it, its last text row included.

        Yields what :meth:`read` yields, then the row of the text it ends on.
        A response that fails ends the same way, then raises: the channel was
        handed every delta of that text, so the row reaches it here, as a
        completed response's does, and its delivery excludes it (RFC §12.2
        step 13s). Its open calls are the caller's to close (:meth:`end_failed`),
        outside a read the channel may cancel.
        """
        try:
            async with aclosing(self.read(reader)) as items:
                async for item in items:
                    yield item
        except Exception as exc:
            self.failure = exc
            if (row := await self.flush_text()) is not None and row.id in self._streamed_rows:
                yield row
            raise
        if self._stopping:
            # The room no longer lets the channel read the response: its
            # stream ends here, the rest read by :meth:`read_rest`.
            self.withdrawn = True
            return
        self.read_to_end = True
        if (row := await self.flush_text()) is not None and row.id in self._streamed_rows:
            yield row

    async def read_rest(self, reader: ResponseReader) -> None:
        """Read on a response the room stopped letting its channel read
        (:attr:`withdrawn`): its rows are written and delivered as each one's
        plan says, nothing of it handed to that channel any more (RFC §7.2,
        §7.5). A failure or a cancellation ends it as the stream's would.
        """
        self._stopping = False
        async with aclosing(self.stream(reader)) as items:
            async for _ in items:
                pass

    async def drain(self, reader: ResponseReader) -> None:
        """Read the response to its end, writing its rows, those of a cut-short turn included.

        For a response nobody renders as it streams (a headless turn, a
        delegated one): a cancellation or a failure writes what the turn
        leaves (:meth:`end_cancelled`, :meth:`end_failed`), then propagates.
        """
        try:
            async with aclosing(self.stream(reader)) as items:
                async for _ in items:
                    pass
        except asyncio.CancelledError as cut:
            await self.end_cancelled(reader, cancel_reason(cut))
            raise
        except Exception:
            await self.end_failed(reader)
            raise

    async def end_cancelled(self, reader: ResponseReader, reason: str | None = None) -> None:
        """Write what a cancelled turn leaves: open calls closed, text kept as
        cancelled, and its record ``cancelled``, as a turn its reader stopped.

        A running tool is aborted, with the cancellation's *reason*, and its
        call closed ``failed``, so no start row stays pending (RFC §12.2 step
        13s): its error ``kit_closed`` when the kit's close cut it (RFC
        §12.10.4), else ``cancelled``.
        """
        error = KIT_CLOSED if reason == KIT_CLOSED else "cancelled"
        await self.close_calls(await reader.abandon(error, reason=reason))
        self.end_stopped()
        await self.flush_text(cancelled=True)
        # Text already written at a call's boundary carries no record of its
        # own: the turn's end goes on the last message now, or its last row, as
        # a stopped read's does, since no later step of a cancelled read writes it.
        await self.write_owed_record()

    async def end_failed(self, reader: ResponseReader) -> None:
        """Write what a failed turn leaves: open calls closed as ``turn failed``, text kept."""
        await self.close_calls(await reader.abandon("turn failed"))
        await self.flush_text()

    # -- the three rows ----------------------------------------------------

    async def flush_text(self, *, cancelled: bool = False) -> RoomEvent | None:
        """Write the accumulated text as a MESSAGE row.

        ``sr.response_metadata`` (the turn's ``AIContext.response_metadata``,
        the same live record) rides every MESSAGE segment as it stands when
        the segment is committed, so turn-level attribution, including what a
        tool handler wrote mid-loop, lands in the stored row (and, in a room,
        in the stream_end frame) without any post-hoc rewrite.

        ``cancelled`` marks a segment cut short by an interrupted turn, so a
        reader can tell a finished answer from one the user stopped. The final
        text of a loop that ended cancelled (a steering Cancel) is marked
        alike (RFC §12.2 step 13s).
        """
        # Joined even with nothing to write: the caller's last flush is what
        # lets a write cut off by a stop land before the turn returns.
        await self._settle()
        if not self._accumulated:
            return None
        body = "".join(self._accumulated)
        self._accumulated.clear()
        # A blank segment says nothing: no row, so no empty SMS. A record it
        # would have carried goes to the turn's last message, else its last row.
        if not body.strip():
            return None
        metadata = dict(self._sr.response_metadata or {})
        # Read after the loop's end: this is the turn's final text
        record = self._turn_record
        carries_record = record is not None
        if record is not None:
            metadata.update(record)
        if cancelled or (record is not None and record.get("loop_end_reason") == "cancelled"):
            metadata["cancelled"] = True
        event = self._build(EventType.MESSAGE, TextContent(body=body), metadata=metadata)
        # The streaming channel already rendered this text chunk by chunk —
        # only the others get it as an event.
        row = await self._write(event, exclude=set(self._streamed_to))
        if carries_record and row is not None:
            self._record_owed = False
        return row

    def started(self, marker: ToolCallStartMarker) -> bool:
        """Whether this call's start row was handed to the writer."""
        return any(start is marker for start in self._started)

    async def take(self, marker: Any) -> list[RoomEvent]:
        """Handle a stream marker; the rows it committed, in order.

        A call's start ends the text before it, which is its own segment, and
        so does a break the loop marks when it goes on after a round without
        a call. The loop's end is recorded, and written with the turn's last
        message.
        """
        rows: list[RoomEvent | None] = []
        if isinstance(marker, InstructionReadMarker):
            rows = [await self.instruction_read(marker)]
        elif isinstance(marker, ToolCallStartMarker):
            rows = [await self.flush_text(), await self.tool_start(marker)]
        elif isinstance(marker, SegmentBreakMarker):
            rows = [await self.flush_text()]
        elif isinstance(marker, ToolCallEndMarker):
            rows = [await self.tool_end(marker)]
        elif isinstance(marker, LoopEndMarker):
            self.end_turn(marker)
        return [row for row in rows if row is not None]

    async def instruction_read(self, marker: InstructionReadMarker) -> RoomEvent | None:
        """The instruction the turn read, as the channel's own row, stored
        before anything else of the turn: a record no member receives (RFC
        §10.5), as the instruction is none of the turn's answer (§10.1.1). It
        takes the room's next index; no hook reads it, no channel is delivered
        it, no reader of the stream is handed it, and it is no activity. Its
        visibility is the channel alone, so no other agent's history reads it;
        it is none of the turn's response events, and the turn's record never
        lands on it."""
        row = self._build(EventType.INSTRUCTION_READ, TextContent(body=marker.text))
        own = row.model_copy(update={"visibility": self._sr.source_channel_id})
        await self._settle()
        try:
            await self._kit.commit_event(self._room_id, own)
        except (RoomClosedError, RoomNotFoundError, KitClosedError):
            # As a row the room refuses: not kept, and the turn goes on, its
            # next row meeting the same refusal its own way.
            logger.debug("Room %s took no write; the instruction read is not kept", self._room_id)
        return None

    async def tool_start(self, marker: ToolCallStartMarker) -> RoomEvent | None:
        return await self._write(self._start_row(marker), exclude=set(self._streamed_to))

    async def tool_end(self, marker: ToolCallEndMarker) -> RoomEvent | None:
        # Excluded like a text segment, and for the same reason: the channel
        # consuming the stream is handed every persisted event inline, so
        # laning it there too sent the same event id twice.
        return await self._write(self._end_row(marker), exclude=set(self._streamed_to))

    async def close_calls(
        self, closed: list[tuple[ToolCallStartMarker, ToolCallEndMarker]]
    ) -> None:
        """Write the end of each call a stop or an abandon closed.

        The stream no longer carries rows, so each goes to every channel, the
        one that streamed included: it saw the start inline and would keep
        the call running. A call whose start row was cut off gets it first
        (RFC §12.2 step 13s).
        """
        for start, end in closed:
            if not self.started(start):
                await self._write(self._start_row(start), exclude=None)
            await self._write(self._end_row(end), exclude=None)

    def _start_row(self, marker: ToolCallStartMarker) -> RoomEvent:
        self._started.append(marker)
        return self._build(
            EventType.TOOL_CALL_START,
            ToolCallContent(
                tool_name=marker.tool_name,
                tool_id=marker.tool_id,
                arguments=marker.arguments,
                status="pending",
                intent=marker.intent,
            ),
        )

    def _end_row(self, marker: ToolCallEndMarker) -> RoomEvent:
        result, structured = tool_event_payload(marker.result, marker.structured_content)
        return self._build(
            EventType.TOOL_CALL_END,
            ToolCallContent(
                tool_name=marker.tool_name,
                tool_id=marker.tool_id,
                arguments=marker.arguments,
                result=result,
                status=marker.status,
                duration_ms=marker.duration_ms,
                error=marker.error,
                structured_content=structured,
                outcome=marker.outcome,
                refused_but_ran=marker.refused_but_ran,
                intent=marker.intent,
            ),
        )

    # -- how any of them is written ----------------------------------------

    def _build(
        self,
        event_type: EventType,
        content: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RoomEvent:
        return RoomEvent(
            room_id=self._room_id,
            source=EventSource(
                channel_id=self._sr.source_channel_id,
                channel_type=self._sr.source_channel_type,
                direction=EventDirection.OUTBOUND,
            ),
            type=event_type,
            content=content,
            status=EventStatus.DELIVERED,
            chain_depth=self._chain_depth,
            visibility=self._visibility,
            response_visibility=self._response_visibility,
            correlation_id=self._correlation_id,
            parent_event_id=self._parent_event_id,
            # Every row of the turn names the event it answers (RFC §8.5),
            # and the instruction it answers, if it answers one (§10.1.1).
            responds_to=self._sr.trigger_event.id,
            metadata=response_metadata(metadata or {}, self._sr.trigger_event),
        )

    async def _settle(self) -> None:
        """Wait for every row whose write outlived the read that started it.

        ``asyncio.wait`` rather than ``gather``: a waiter cancelled here must
        not cancel the writes it was waiting on.
        """
        while self._writing:
            await asyncio.wait(set(self._writing))

    async def _write(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        """Commit a row through the sink, even when the read that produced it is cancelled.

        A barge-in cancels the stream's in-flight read wherever it stands,
        possibly halfway through a commit. The row's text has already left
        the buffer, so a cancelled commit would lose it: the write finishes
        on its own. Every row first waits for the writes before it, so rows
        commit in the order the stream produced them.
        """
        await self._settle()
        task = asyncio.ensure_future(self._write_now(event, exclude=exclude))
        self._writing.add(task)
        task.add_done_callback(self._writing.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Nobody awaits the write any more: its failure is logged here.
            task.add_done_callback(self._log_orphan_failure)
            raise

    def _log_orphan_failure(self, task: asyncio.Task[RoomEvent | None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            if ended_by_close(self._kit, exc):
                logger.debug("Streamed row of room %s not written: the kit closed", self._room_id)
                return
            logger.error(
                "Streamed row write failed after its read was cancelled (room %s)",
                self._room_id,
                exc_info=exc,
            )

    async def _write_now(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        # A row is kept from the channel reading the stream only while it
        # still reads it: the rows before may have withdrawn it.
        if exclude:
            exclude = exclude & self._streamed_to
        stored = await self._sink.commit(event, exclude=exclude)
        if exclude:
            self._hand_to_target(stored, exclude)
        if stored is not None:
            self.persisted.append(stored)
            if self._response_events is not None:
                self._response_events.append(stored)
        return stored

    def _hand_to_target(self, stored: RoomEvent | None, target: set[str]) -> None:
        """Whether the channel reading the stream is handed *stored*, a row
        committed for it: it is while the room, as the row's commit read it,
        still lets it read the response. Once it no longer does (the agent
        muted, made read-only or detached, the channel made write-only or
        detached), it reads nothing more of the stream, neither text nor
        rows, and the rows after are delivered as their plans say (RFC §7.2,
        §7.5)."""
        if all(self._sink.reads(channel_id) for channel_id in target):
            if stored is not None:
                self._streamed_rows.add(stored.id)
            return
        self._streamed_to.clear()
        self._stopping = True


class LaneSink:
    """A room's streamed rows: refused once the room refuses writes, gated on
    ``BEFORE_BROADCAST``, committed on its lane with the room's context as it
    stands at each row's commit."""

    def __init__(
        self,
        kit: Any,
        *,
        room_id: str,
        context: RoomContext,
        cascade: DeliveryCascade,
    ) -> None:
        self._kit = kit
        self._room_id = room_id
        self._context = context
        self._cascade = cascade
        # The room's status as last read, and the kit's close count then:
        # None until the first row reads it.
        self._refusing = False
        self._close_epoch: int | None = None
        # The channels the room still lets read the stream, as the last
        # row's commit read it.
        self._readers: set[str] = set()

    def reads(self, channel_id: str) -> bool:
        """Whether the room, as the last row's commit read it, still lets
        *channel_id* read the response as it streams (RFC §7.2, §7.5)."""
        return channel_id in self._readers

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        self._readers = set()
        if await self._room_refuses_writes():
            logger.debug("Room %s refuses writes; %s not committed", self._room_id, event.type)
            return None
        # The row's BEFORE_BROADCAST observers start once it is laned, or
        # refused (RFC §10.1).
        async with observed_once_over(
            self._kit._hook_engine, self._room_id, closing=lambda: self._kit._closed
        ) as observed:
            gated = await self._gate(event, observed, exclude)
            if gated is None:
                return None
            event, hook_result = gated
            committed = await self._lane(event, exclude=exclude, hook_result=hook_result)
            if isinstance(committed, _Blocked):
                await self._record_blocked(committed, observed)
                return None
            return committed

    async def _room_refuses_writes(self) -> bool:
        """Whether the room refuses this row (RFC §5.1).

        The status gate holds at every point the timeline grows: a room closed
        mid-stream takes no further row. Around a row's hooks, which it spares
        a room that refuses it, the run reads the status at its first row, and
        again only once the kit has closed or archived a room since; the
        row's commit reads the room under its lock (:meth:`_lane`). The count
        is taken before the read and kept only once the read succeeded: a
        close landing meanwhile, or a read that failed, is read again at the
        next row.
        """
        epoch = self._kit._room_close_epoch
        if epoch != self._close_epoch:
            self._refusing = await self._kit._room_refuses_writes(self._room_id)
            self._close_epoch = epoch
        return self._refusing

    async def _gate(
        self, event: RoomEvent, observed: Observed, exclude: set[str] | None
    ) -> tuple[RoomEvent, SyncPipelineResult] | None:
        """Pass a row through the commit gate before it commits: its
        BEFORE_BROADCAST sync hooks, then its source's right to write.

        The gate every commit path shares (``_gate_commit``, RFC §10.1 steps
        9 to 11), its hooks run off the room lock here. The live chunks
        already piped to streaming channels are outside a hook's reach by
        construction; this lands any hook modification (e.g. PII
        de-anonymisation, a display label on a tool call) on the persisted
        row and on the delivery to the non-streaming channels.

        ``None`` when the gate refused the row: a hook blocked it, or its
        source cannot write (a read-only agent's stream, RFC §7.5 rule 2). A
        refusal is not a drop: the row is committed with status BLOCKED as
        the audit record (:meth:`_record_blocked`). The source's right to
        write is checked again at the commit (:meth:`_lane`).
        """
        room_id = self._room_id
        # The row carries its cascade's scope (TurnScope), before its hooks.
        scope = event.response_visibility
        decision = await self._kit._gate_commit(
            room_id, event, self._context, under_lock=False, observed=observed
        )
        # The hooks ran without the room lock: a close meanwhile refuses the
        # row, which is then neither committed nor recorded as blocked.
        if await self._room_refuses_writes():
            logger.debug("Room %s closed during the hooks; %s not committed", room_id, event.type)
            return None
        if isinstance(decision, _Refused):
            # Only a change is refused here, and a streamed row is text.
            return None
        # Its hooks may have widened it: back within the cascade's scope.
        decision = within_answer_scope(decision, scope)
        if isinstance(decision, _Blocked):
            # Who still reads the stream: a hook's block withdraws no one.
            bindings = await self._kit._store.list_bindings(room_id)
            self._readers = _readers(bindings, event.source.channel_id, scope, exclude)
            await self._record_blocked(decision, observed)
            return None
        return decision.event, decision.sync_result

    async def _record_blocked(self, decision: _Blocked, observed: Observed) -> None:
        """Store a refused row BLOCKED, announced as ``event_blocked``; what
        its hooks decided still stands, their tasks and observations
        persisted and their injected events laned."""
        room_id = self._room_id
        # A row commits without the room lock: what its hooks injected
        # runs its passes now, each taking the lock (RFC §9.5.1), and
        # whatever becomes of the stream's reader (RFC §9.4).
        queued = QueuedInjections()
        await self._kit._store_blocked(room_id, decision, queued)
        # Its pass is over: observed before its injected events' passes.
        await observed.start(self._kit._hook_engine, room_id)
        handed = self._kit._hand_off_queued(queued, room_id, self._cascade)
        await self._kit._finish_queued(handed)
        logger.info(
            "Streamed %s blocked (room %s): %s",
            decision.event.type.value,
            room_id,
            decision.reason,
        )

    async def _lane(
        self,
        event: RoomEvent,
        *,
        exclude: set[str] | None,
        hook_result: SyncPipelineResult,
    ) -> RoomEvent | _Blocked | None:
        """Commit a row and queue its delivery on the room's lane; the row's
        block when its source can no longer write.

        The room's context is read under its lock at the row's commit, as any
        event's (RFC §10.1 step 12), and the row planned with it (§7.5 rule
        1, §10.2): its status, a binding changed during the stream (a mute,
        an access, a detach) and the events committed since hold from the
        next row, the source's right to write included (rule 2), and it tells
        who may still read the stream.

        Deliberately not awaited to completion: this runs inside the
        streaming channel's ``deliver_stream``, and blocking the generator on
        a transport round trip would stall the stream. The cascade collects
        every row's unit instead, and the caller waits on it once the stream
        is done.
        """
        kit, room_id = self._kit, self._room_id
        scope = event.response_visibility
        async with kit._lock_manager.locked(room_id):
            context = await kit._existing_room_context(room_id)
            # A close that held the lock meanwhile refuses the row (RFC §5.1).
            if context is None or _refuses_writes(context.room):
                logger.debug("Room %s closed before %s's commit", room_id, event.type)
                # A status changed elsewhere moves no close count: the next
                # row's hooks are spared all the same.
                self._refusing = True
                return None
            bindings = context.bindings
            source = context.get_binding(event.source.channel_id)
            reason = _source_block_reason(source)
            if source is None or reason is not None:
                stamped = with_committed_visibility(event, source, bindings)
                return _Blocked.after_hooks(
                    stamped, context, hook_result, reason=reason, blocked_by=reason
                )
            resolved = DeliverySource(binding=source, context=context)
            stored = await kit._commit_resolved(
                room_id,
                event,
                resolved,
                self._cascade,
                kit._plan_factory(resolved, exclude, hook_result),
                policy_aware=True,
            )
        self._readers = _readers(bindings, event.source.channel_id, scope, exclude)
        if hook_result.injected_events:
            await kit._lane_injected_events(
                hook_result.injected_events, room_id, self._cascade, cause=stored or event
            )
        return stored


def _readers(
    bindings: list[ChannelBinding], source_id: str, scope: str | None, streamed: set[str] | None
) -> set[str]:
    """The channels of *streamed* the room, bound by *bindings*, still lets
    read the stream of *source_id*'s answer scoped by *scope*."""
    by_id = {binding.channel_id: binding for binding in bindings}
    source = by_id.get(source_id)
    return {
        channel_id
        for channel_id in streamed or ()
        if (binding := by_id.get(channel_id)) is not None
        and reads_the_stream(source, binding, scope)
    }

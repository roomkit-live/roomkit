"""The writer of a streamed turn's timeline rows."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from roomkit.channels._tool_event_result import tool_event_payload
from roomkit.core.event_router import stream_record
from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT
from roomkit.core.mixins.inbound_locked import _Blocked
from roomkit.models.enums import EventStatus, EventType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.response_metadata import recorded_turn_end
from roomkit.models.streaming import (
    LoopEndMarker,
    SegmentBreakMarker,
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)

if TYPE_CHECKING:
    from roomkit.core.event_router import StreamingResponse
    from roomkit.core.hooks import SyncPipelineResult
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.core.mixins._response_reader import ResponseReader
    from roomkit.core.mixins.lane_execution import DeliverySource
    from roomkit.models.context import RoomContext

logger = logging.getLogger("roomkit.inbound")


class RowSink(Protocol):
    """How a streamed turn's rows are committed, the one thing writers differ in."""

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        """Commit *event*: the stored row, or ``None`` when it was refused or excluded.

        *exclude* names the channels the row must not be delivered to.
        """
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

        The trigger's thread root is inherited, so the reply lands in the same
        thread (already normalised to a root by the locked pipeline); a
        top-level trigger keeps the reply top-level.
        """
        return cls(
            chain_depth=trigger.chain_depth + 1,
            visibility=trigger.response_visibility or Visibility.ALL,
            response_visibility=trigger.response_visibility,
            parent_event_id=trigger.parent_event_id,
        )


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
    ) -> None:
        self._kit = kit
        self._sr = sr
        self._sink = sink
        self._room_id = room_id
        self._chain_depth = chain_depth
        self._visibility = visibility
        # The trigger's answer scope rides every row, so what another agent
        # answers to a row is scoped as the turn itself is (RFC §8.3).
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
        self._accumulated: list[str] = []
        self._writing: set[asyncio.Task[RoomEvent | None]] = set()
        # The starts whose row was handed to the writer, by identity: two
        # calls under one id are two calls (RFC §12.4).
        self._started: list[ToolCallStartMarker] = []
        # Whether the reader was handed anything, a thinking delta included:
        # from there the turn began, and a stop cuts it.
        self._began = False
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
        already wrote, by :meth:`record_on_last_message` (RFC §6.4). It rides
        the stream too, where a reader of the turn finds how it ended when it
        wrote no message at all.
        """
        self._turn_record = {"ai_usage": dict(marker.usage), "loop_end_reason": marker.reason}
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
        self._turn_record = {"loop_end_reason": "cancelled"}
        self._record_owed = True
        self._sr.turn_record = self._turn_record

    async def record_on_last_message(self) -> None:
        """Write a record no final text carried on the last MESSAGE already stored.

        Run once the turn's deliveries are done, so no delivery record is
        written to the same row meanwhile. Only the row's metadata changes,
        through ``update_event``, so ON_EVENT_UPDATED sees it; it is not
        delivered again. Best effort: the turn's outcome stands without it.
        """
        if not self._record_owed or self._turn_record is None:
            return
        self._record_owed = False
        last = next((e for e in reversed(self.persisted) if e.type == EventType.MESSAGE), None)
        if last is None:
            return
        try:
            stored = await self._kit.store.get_event(last.id)
            if stored is None:
                return
            updated = await self._kit.update_event(
                self._room_id, last.id, metadata={**stored.metadata, **self._turn_record}
            )
        except Exception:
            logger.warning(
                "Could not record the turn's end on message %s (room %s)",
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
    ) -> AsyncGenerator[str | ThinkingDeltaMarker | RoomEvent, None]:
        """Read the response through *reader*, writing each row as it ends.

        Yields every text delta and thinking marker as it arrives, for the
        channel rendering the stream, and every row committed. Stops at the
        stream's end, before the final text is written: the caller writes it.
        """
        while True:
            try:
                delta = await reader.next()
            except StopAsyncIteration:
                return
            self._began = True
            if isinstance(delta, str):
                self.add_text(delta)
                yield delta
            elif isinstance(delta, ThinkingDeltaMarker):
                yield delta
            else:
                for row in await self.take(delta):
                    yield row

    async def stream(
        self, reader: ResponseReader
    ) -> AsyncGenerator[str | ThinkingDeltaMarker | RoomEvent, None]:
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
            if (row := await self.flush_text()) is not None:
                yield row
            raise
        self.read_to_end = True
        if (row := await self.flush_text()) is not None:
            yield row

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
        except asyncio.CancelledError:
            await self.end_cancelled(reader)
            raise
        except Exception:
            await self.end_failed(reader)
            raise

    async def end_cancelled(self, reader: ResponseReader) -> None:
        """Write what a cancelled turn leaves: open calls closed, text kept as
        cancelled, and its record ``cancelled``, as a turn its reader stopped.

        A running tool is aborted and its call closed ``failed``, so no start
        row stays pending (RFC §12.2 step 13s).
        """
        await self.close_calls(await reader.abandon())
        self.end_stopped()
        await self.flush_text(cancelled=True)
        # Text already written at a call's boundary carries no record of its
        # own: the turn's end goes on the last message now, as a stopped
        # read's does, since no later step of a cancelled read writes it.
        await self.record_on_last_message()

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
        reader can tell a finished answer from one the user stopped.
        """
        # Joined even with nothing to write: the caller's last flush is what
        # lets a write cut off by a stop land before the turn returns.
        await self._settle()
        if not self._accumulated:
            return None
        body = "".join(self._accumulated)
        self._accumulated.clear()
        metadata = dict(self._sr.response_metadata or {})
        # Read after the loop's end: this is the turn's final text
        carries_record = self._turn_record is not None
        if self._turn_record is not None:
            metadata.update(self._turn_record)
        if cancelled:
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
        if isinstance(marker, ToolCallStartMarker):
            rows = [await self.flush_text(), await self.tool_start(marker)]
        elif isinstance(marker, SegmentBreakMarker):
            rows = [await self.flush_text()]
        elif isinstance(marker, ToolCallEndMarker):
            rows = [await self.tool_end(marker)]
        elif isinstance(marker, LoopEndMarker):
            self.end_turn(marker)
        return [row for row in rows if row is not None]

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
            ),
            type=event_type,
            content=content,
            status=EventStatus.DELIVERED,
            chain_depth=self._chain_depth,
            visibility=self._visibility,
            response_visibility=self._response_visibility,
            correlation_id=self._correlation_id,
            parent_event_id=self._parent_event_id,
            metadata=metadata or {},
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
            logger.error(
                "Streamed row write failed after its read was cancelled (room %s)",
                self._room_id,
                exc_info=exc,
            )

    async def _write_now(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        stored = await self._sink.commit(event, exclude=exclude)
        if stored is not None:
            self.persisted.append(stored)
            if self._response_events is not None:
                self._response_events.append(stored)
        return stored


class LaneSink:
    """A room's streamed rows: refused once the room refuses writes, gated on
    ``BEFORE_BROADCAST``, committed on its lane."""

    def __init__(
        self,
        kit: Any,
        *,
        room_id: str,
        context: RoomContext,
        cascade: DeliveryCascade,
        plan_source: DeliverySource | str,
    ) -> None:
        self._kit = kit
        self._room_id = room_id
        self._context = context
        self._cascade = cascade
        self._plan_source = plan_source
        # The room's status as last read, and the kit's close count then:
        # None until the first row reads it.
        self._refusing = False
        self._close_epoch: int | None = None

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        if await self._room_refuses_writes():
            logger.debug("Room %s refuses writes; %s not committed", self._room_id, event.type)
            return None
        gated = await self._gate(event)
        if gated is None:
            return None
        event, hook_result = gated
        return await self._lane(event, exclude=exclude, hook_result=hook_result)

    async def _room_refuses_writes(self) -> bool:
        """Whether the room refuses this row (RFC §5.1).

        The status gate holds at every point the timeline grows: a room closed
        mid-stream takes no further row. The run reads the status at its first
        row, and again only once the kit has closed or archived a room since,
        so a stream does not cost a store read per row. The count is taken
        before the read and kept only once the read succeeded: a close landing
        meanwhile, or a read that failed, is read again at the next row.
        """
        epoch = self._kit._room_close_epoch
        if epoch != self._close_epoch:
            self._refusing = await self._kit._room_refuses_writes(self._room_id)
            self._close_epoch = epoch
        return self._refusing

    async def _gate(self, event: RoomEvent) -> tuple[RoomEvent, SyncPipelineResult] | None:
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
        the audit record, announced as ``event_blocked``, and what the hooks
        decided still stands, their tasks and observations persisted and
        their injected events laned.
        """
        room_id = self._room_id
        decision = await self._kit._gate_commit(room_id, event, self._context)
        # The hooks ran without the room lock: a close meanwhile refuses the
        # row, which is then neither committed nor recorded as blocked.
        if await self._room_refuses_writes():
            logger.debug("Room %s closed during the hooks; %s not committed", room_id, event.type)
            return None
        if isinstance(decision, _Blocked):
            await self._kit._store_blocked(room_id, decision, self._cascade)
            logger.info(
                "Streamed %s blocked (room %s): %s",
                event.type.value,
                room_id,
                decision.reason,
            )
            return None
        return decision.event, decision.sync_result

    async def _lane(
        self,
        event: RoomEvent,
        *,
        exclude: set[str] | None,
        hook_result: SyncPipelineResult,
    ) -> RoomEvent | None:
        """Commit a row and queue its delivery on the room's lane.

        Deliberately not awaited to completion: this runs inside the
        streaming channel's ``deliver_stream``, and blocking the generator on
        a transport round trip would stall the stream. The cascade collects
        every row's unit instead, and the caller waits on it once the stream
        is done.
        """
        stored = await self._kit._commit_and_deliver(
            self._room_id,
            event,
            self._plan_source,
            exclude_delivery=exclude,
            cascade=self._cascade,
            hook_result=hook_result,
            gate_status=False,
        )
        if stored is not None:
            self._plan_source = _with_row(self._plan_source, stored)
        return stored


def _with_row(source: DeliverySource | str, row: RoomEvent) -> DeliverySource | str:
    """The run's planning inputs with *row* in their history.

    The context is resolved once for the whole stream; without this, the
    agents a later row reaches would answer it without the rows before it,
    where a buffered response's segments each re-enter with the ones before
    them in their context.
    """
    if isinstance(source, str):
        return source
    recent = [*source.context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :], row]
    return replace(source, context=source.context.model_copy(update={"recent_events": recent}))

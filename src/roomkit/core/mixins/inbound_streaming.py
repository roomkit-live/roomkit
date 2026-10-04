"""InboundStreamingMixin — streaming response handling outside the room lock."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._failure_log import log_failure
from roomkit.core.event_router import stream_record, unanswered
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins._response_reader import ResponseReader
from roomkit.core.mixins._streaming_segments import LaneSink, SegmentWriter, TurnScope
from roomkit.core.mixins.helpers import HelpersMixin, _source_block_reason
from roomkit.core.mixins.lane_execution import DeliverySource
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import (
    ResponseMetadata,
    add_turn_entry,
    merge_channel_record,
    turn_summary,
)
from roomkit.providers.utils import _aclose_stream

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter, StreamingResponse
    from roomkit.core.hooks import HookEngine
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.hook import InjectedEvent
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@dataclass
class _StreamingResult:
    """Result of handling a streaming response.

    ``error`` is the exception raised while consuming the response stream
    (provider/transport failure), captured so the inbound pipeline can surface
    it to a headless caller. ``None`` when the stream completed.
    """

    events: list[RoomEvent] = field(default_factory=list)
    error: Exception | None = None


@runtime_checkable
class InboundStreamingHost(Protocol):
    """Contract: capabilities a host class must provide for InboundStreamingMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for event persistence.
        _channels: Channel registry.
        _hook_engine: Hook engine for AFTER_BROADCAST / ON_ERROR hooks.
        _max_chain_depth: Maximum chain depth to prevent infinite loops.

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter`` for broadcast.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    def _get_router(self) -> EventRouter: ...

    async def _lane_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: RoomContext,
        cascade: DeliveryCascade,
    ) -> None: ...


async def _close_handed_stream(segments: Any, room_id: str) -> None:
    """Close the stream a channel was handed, wherever it stopped reading it.

    Closing a stream already ended is a no-op. One a channel left a read of
    running cannot be closed from here: that is logged, not raised.
    """
    try:
        await segments.aclose()
    except Exception:
        logger.warning(
            "Could not close the stream handed to the channel (room %s)", room_id, exc_info=True
        )


class InboundStreamingMixin(HelpersMixin):
    """Streaming response handling extracted from the inbound pipeline.

    These methods run outside the room lock so that streaming delivery
    (e.g. TTS playback) does not block other ``process_inbound`` calls.

    Host contract: :class:`InboundStreamingHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    # Cross-mixin method — attribute annotation avoids MRO shadowing
    _commit_and_deliver: Any  # LaneExecutionMixin
    _lane_injected_events: Any  # LaneExecutionMixin
    _store_past_reentry_cap: Any  # LaneExecutionMixin
    _handle_block: Any  # InboundLockedMixin

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _handle_streaming_response(
        self,
        router: EventRouter,
        sr: StreamingResponse,
        room_id: str,
        context: RoomContext,
        *,
        cascade: DeliveryCascade,
        response_events: list[RoomEvent] | None = None,
    ) -> _StreamingResult | None:
        """Consume a streaming response, pipe to streaming channels, store segments."""
        streaming_targets = self._find_streaming_targets(router, sr, context)

        logger.debug(
            "Streaming targets for room %s: %d found",
            room_id,
            len(streaming_targets),
        )

        # One cascade for the whole response: each segment's delivery is
        # enqueued without waiting (blocking the generator on an SMS round
        # trip would stall the stream), and the run is awaited once, after
        # the stream. It is the caller's cascade, so a stream another agent
        # starts in answer to a segment joins the caller's reading.
        # Only the first target streams (V1, below); any other
        # streaming-capable channel is an ordinary recipient.
        streamed_to: set[str] = (
            {streaming_targets[0][1].channel_id} if streaming_targets else set()
        )
        correlation_id = uuid4().hex
        scope = TurnScope.answering(sr.trigger_event)
        chain_depth = scope.chain_depth
        visibility = scope.visibility
        parent_event_id = scope.parent_event_id

        # Planning inputs for the whole run, resolved once. Every segment has
        # the same sender and the same delivery set, so re-resolving per
        # segment would buy nothing and cost a room lock plus a context read
        # each time — on a tool-heavy turn, tens of them on the hot path the
        # delivery lanes exist to keep clear. The binding is already in the
        # context the caller built; the batch broadcast this replaced planned
        # off that same snapshot.
        plan_source = DeliverySource.of(sr.source_channel_id, context)

        writer = SegmentWriter(
            self,
            sr,
            LaneSink(
                self, room_id=room_id, context=context, cascade=cascade, plan_source=plan_source
            ),
            room_id=room_id,
            chain_depth=chain_depth,
            visibility=visibility,
            response_visibility=scope.response_visibility,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            streamed_to=streamed_to,
            response_events=response_events,
        )

        reader = ResponseReader(sr.stream)

        stream_error: Exception | None = None
        if streaming_targets:
            channel, binding = streaming_targets[0]  # V1: single target
            stream_error = await self._stream_to_target(
                channel,
                binding,
                self._stream_placeholder(sr, room_id, scope, correlation_id),
                sr,
                writer,
                reader,
                context,
                correlation_id=correlation_id,
            )
        else:
            # No streaming targets (e.g. a PII-locked / edge agent whose stream
            # send fn was withheld, or a headless one-shot call whose only
            # transport is also the source) — still consume the stream to drive
            # persistence via markers. Under the SAME error contract as the
            # streaming branch above: a failure (context overflow, provider
            # error) must fire ON_ERROR so the error reaches the ON_ERROR hooks
            # (which classify + surface it) AND be returned to the caller via
            # ``_StreamingResult.error``, instead of vanishing with no card.
            try:
                await writer.drain(reader)
            except Exception as exc:
                stream_error = exc
                await self._report_stream_failure(
                    exc,
                    f"stream consumption (no targets) of {sr.source_channel_id} "
                    f"for room {room_id}",
                    sr,
                    context,
                    correlation_id=correlation_id,
                    # A chained stream's failure is not the caller's.
                    caller_logs=cascade.caller_logs and not sr.chained,
                )

        # Every segment's delivery set, awaited once now that the stream is
        # done — the run's completion is what the caller's turn waits on.
        await cascade.wait()
        await writer.record_on_last_message()

        if not writer.persisted and stream_error is None:
            return None

        return _StreamingResult(events=writer.persisted, error=stream_error)

    @staticmethod
    def _stream_placeholder(
        sr: StreamingResponse, room_id: str, scope: TurnScope, correlation_id: str
    ) -> RoomEvent:
        """The empty event a streaming channel renders the response under."""
        return RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id=sr.source_channel_id,
                channel_type=sr.source_channel_type,
            ),
            content=TextContent(body=""),
            chain_depth=scope.chain_depth,
            visibility=scope.visibility,
            correlation_id=correlation_id,
            parent_event_id=scope.parent_event_id,
        )

    async def _stream_to_target(
        self,
        channel: Channel,
        binding: ChannelBinding,
        placeholder: RoomEvent,
        sr: StreamingResponse,
        writer: SegmentWriter,
        reader: ResponseReader,
        context: RoomContext,
        *,
        correlation_id: str,
    ) -> Exception | None:
        """Hand the response to the channel that streams it; the failure it ended on, if any.

        Text deltas drive the channel's live rendering; the rows the writer
        commits reach it inline, interleaved between the chunks, the text the
        response ends on included, failed or not (RFC §12.2 step 13s).
        """
        room_id = placeholder.room_id
        # Rows ride it besides text: the ABC's AsyncIterator[str] names only the text.
        segments: Any = writer.stream(reader)
        try:
            channel_error = await self._deliver_segments(
                channel, binding, placeholder, segments, sr, writer, reader, context
            )
            failure = await self._end_failed_stream(writer, reader, channel_error)
        finally:
            # On every exit: a channel that failed, or stopped reading at the
            # failed text's row, leaves it suspended, and nothing else closes it.
            await _close_handed_stream(segments, room_id)
        if failure is not None:
            if channel_error is not None and channel_error is not failure:
                log_failure(
                    logger,
                    channel_error,
                    f"streaming channel {binding.channel_id} handling the failed response "
                    f"of {sr.source_channel_id} in room {room_id}",
                )
            await self._report_stream_failure(
                failure,
                f"streaming delivery of {sr.source_channel_id} to {binding.channel_id} "
                f"for room {room_id}",
                sr,
                context,
                correlation_id=correlation_id,
            )
        return failure

    async def _deliver_segments(
        self,
        channel: Channel,
        binding: ChannelBinding,
        placeholder: RoomEvent,
        segments: Any,
        sr: StreamingResponse,
        writer: SegmentWriter,
        reader: ResponseReader,
        context: RoomContext,
    ) -> Exception | None:
        """Run the channel's deliver_stream over *segments*; the error it raised, if any."""
        try:
            await channel.deliver_stream(segments, placeholder, binding, context)
            # A transport that hands back early (every voice session barged
            # in, RFC §12.2 step 13s; or one that never reads it at all) left
            # the response unread: it is closed and stored cancelled.
            if not writer.read_to_end and writer.failure is None:
                await self._stop_unread_stream(segments, sr, reader, writer, placeholder.room_id)
        except asyncio.CancelledError:
            # A turn interrupted on purpose (the console's Esc). What was
            # already streamed is on the user's screen, so the timeline
            # MUST hold it too: dropping it would leave the room
            # disagreeing with what the human read, and the agent's next
            # context missing what it already said. Not an error — nobody
            # failed — so ON_ERROR stays silent and the cancellation
            # propagates untouched.
            await writer.end_cancelled(reader)
            raise
        except Exception as exc:
            return exc
        return None

    @staticmethod
    async def _end_failed_stream(
        writer: SegmentWriter, reader: ResponseReader, channel_error: Exception | None
    ) -> Exception | None:
        """Write what a failed stream leaves; the failure that is the turn's, if any.

        The response's own failure is the turn's, even when its channel
        swallowed it or raised another error on top of it. Its open calls are
        closed here, outside the read the channel may have cancelled
        (RFC §12.2 step 13s).
        """
        if writer.failure is not None:
            await writer.end_failed(reader)
            return writer.failure
        if channel_error is None:
            return None
        # The channel failed, not the response: what it was handed may not
        # all have been rendered, so the text it leaves goes out as an
        # ordinary event, to everyone.
        writer.stream_lost()
        await writer.end_failed(reader)
        return channel_error

    async def _report_stream_failure(
        self,
        exc: Exception,
        what: str,
        sr: StreamingResponse,
        context: RoomContext,
        *,
        correlation_id: str,
        caller_logs: bool = False,
    ) -> None:
        """Log a response stream's failure once, then fire ON_ERROR as its source."""
        room_id = context.room.id
        log_failure(
            logger,
            exc,
            what,
            caller_logs=caller_logs,
            extra={"room_id": room_id, "channel_id": sr.source_channel_id},
        )
        await self._fire_stream_error_hook(exc, room_id, context, sr, correlation_id)

    @staticmethod
    async def _stop_unread_stream(
        segments: Any,
        sr: StreamingResponse,
        reader: ResponseReader,
        writer: SegmentWriter,
        room_id: str,
    ) -> None:
        """Close a response its transport stopped reading, keep what it produced.

        The open tool round is settled first: a round already executing is let
        finish and its results stored, one that had not started never does
        (RFC §12.2 step 13s). Then the generation is closed, so no token is
        produced and no tool call starts past this point, and the text already
        produced is stored, marked like any interrupted turn. A failure while closing is
        the provider's finalizer, not the response's: it is logged and the
        text is still stored as cancelled.
        """
        await writer.close_calls(await reader.stop())
        try:
            await segments.aclose()
            await _aclose_stream(sr.stream)
        except Exception:
            logger.exception("Closing an unread response stream failed for room %s", room_id)
        writer.end_stopped()
        await writer.flush_text(cancelled=True)

    async def _fire_stream_error_hook(
        self,
        exc: Exception,
        room_id: str,
        context: RoomContext,
        sr: StreamingResponse,
        correlation_id: str,
    ) -> None:
        """Fire ON_ERROR for a response stream that failed, as its source."""
        await self._fire_error_hook(
            room_id,
            context,
            EventSource(channel_id=sr.source_channel_id, channel_type=sr.source_channel_type),
            error=str(exc),
            error_type=type(exc).__name__,
            error_category="streaming",
            chain_depth=sr.trigger_event.chain_depth + 1,
            visibility=sr.trigger_event.response_visibility or "all",
            correlation_id=correlation_id,
            parent_event_id=sr.trigger_event.parent_event_id,
        )

    def _find_streaming_targets(
        self,
        router: Any,
        sr: Any,
        context: RoomContext,
    ) -> list[Any]:
        """Find transport channels that support streaming delivery.

        No target for a source that cannot write (RFC §7.5 rule 2): a read-only
        agent's stream is read to its end, each row stored BLOCKED by the
        commit gate, and nothing of it is piped live.
        """
        if _source_block_reason(context.get_binding(sr.source_channel_id)) is not None:
            return []
        response_vis = sr.trigger_event.response_visibility
        targets: list[Any] = []
        for binding in context.bindings:
            if binding.category != ChannelCategory.TRANSPORT:
                continue
            if binding.channel_id == sr.source_channel_id:
                continue
            if binding.access in (Access.WRITE_ONLY, Access.NONE):
                continue
            if binding.direction == ChannelDirection.OUTBOUND:
                continue
            if response_vis is not None and not visibility_allows(response_vis, binding):
                continue
            channel = router.get_channel(binding.channel_id)
            # Asked per room: a channel can hold streaming clients for one room
            # and none for another, and only the room being delivered counts.
            supports = (
                channel.supports_streaming_delivery_for(binding.room_id) if channel else False
            )
            if channel and supports:
                targets.append((channel, binding))
        return targets

    async def _process_streaming_responses(
        self,
        cascade: DeliveryCascade,
        room_id: str,
        *,
        response_events: list[RoomEvent] | None = None,
    ) -> tuple[Exception | None, ResponseMetadata]:
        """Read every stream of *cascade*, the ones added while reading included.

        Handles streaming responses outside the room lock.

        Streaming delivery (TTS playback) can take seconds. Running it outside
        the lock allows other process_inbound calls to proceed concurrently,
        preventing continuous STT echo from being queued behind the lock.

        Each segment commits and reaches the non-streaming channels through
        the room's delivery lane as it is produced (RFC §10.2 — the lane is
        the room's single ordering authority, and its executor fires each
        segment's AFTER_BROADCAST once that segment's delivery set has run,
        step 16). Broadcasting the run in one batch after the stream would let
        the cursor run ahead of the deliveries.

        Returns the first response-stream failure encountered (so the inbound
        pipeline can surface it to a headless caller), or ``None`` when every
        stream completed, plus the turn's response-metadata record.

        The record is handed back rather than left to be read off a persisted
        segment: a turn that ends on a tool call persists no segment after it,
        and one that ends before writing any text persists none at all, so the
        room is not a place where "how did that turn end" can always be asked.

        A stream a segment's delivery or a reentry pass started (``chained``)
        is read after the caller's, each in the order it joined, until none
        is left (RFC §8.3: a started response is read, never discarded). Its
        failure fires ON_ERROR like any stream's, but neither it nor its
        record is the caller's, as for a buffered answer to an answer. The
        context is rebuilt for each stream: a chained one answers events
        committed after the caller's read began.
        """
        router = self._get_router()
        first_error: Exception | None = None
        record = ResponseMetadata()
        read = 0
        try:
            while read < len(cascade.streams) and cascade.cancelled is None:
                sr = cascade.streams[read]
                read += 1
                if sr.chained and not await self._admit_chained_stream(sr, cascade, room_id):
                    continue
                context = await self._build_context(room_id)
                try:
                    sr_result = await self._handle_streaming_response(
                        router,
                        sr,
                        room_id,
                        context,
                        cascade=cascade,
                        response_events=response_events,
                    )
                except asyncio.CancelledError:
                    # A turn cancelled from outside still tells its caller how
                    # it ended, as one its reader stopped does (RFC §6.4).
                    if not sr.chained:
                        _add_turn(cascade.response_metadata, sr)
                    raise
                if sr.chained:
                    continue
                if sr_result and sr_result.error and first_error is None:
                    first_error = sr_result.error
                # Several streams answer one inbound only when several channels
                # replied; each writes under its own key, so merging keeps them
                # all rather than letting the last one win, and its end goes
                # under its channel in ``turns`` (RFC §6.4).
                merge_channel_record(record, sr.response_metadata or {})
                _add_turn(record, sr)
        finally:
            # A transport can stop reading between two yields (or fail while
            # rendering one). Async-for alone does not close its generator;
            # finalizers must run before the delivery handle reports cleanup.
            for sr in cascade.streams:
                await _aclose_stream(sr.stream)

        return first_error, record

    async def _admit_chained_stream(
        self, sr: StreamingResponse, cascade: DeliveryCascade, room_id: str
    ) -> bool:
        """Whether a chained stream is read, within the cascade's reentry budget.

        Past the budget it is closed unread, so nothing is generated, and its
        BLOCKED record keeps the trace a buffered answer past the budget
        leaves.
        """
        if cascade.consume_reentry_budget():
            return True
        await _aclose_stream(sr.stream)
        await self._store_past_reentry_cap(
            room_id, unanswered(sr.trigger_event, sr.source_channel_id, sr.source_channel_type)
        )
        return False


def _add_turn(record: MutableMapping[str, Any], sr: StreamingResponse) -> None:
    """Put how *sr*'s turn ended under its channel in *record*'s ``turns``."""
    add_turn_entry(record, sr.source_channel_id, turn_summary(stream_record(sr)))

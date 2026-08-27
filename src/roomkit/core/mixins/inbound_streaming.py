"""InboundStreamingMixin — streaming response handling outside the room lock."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.lane_execution import DeliverySource
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import ProviderError

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter, StreamingResponse
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
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

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _handle_streaming_response(
        self,
        router: EventRouter,
        sr: StreamingResponse,
        room_id: str,
        context: RoomContext,
    ) -> _StreamingResult | None:
        """Consume a streaming response, pipe to streaming channels, store segments."""
        from roomkit.models.event import EventSource, TextContent

        response_vis = sr.trigger_event.response_visibility
        streaming_targets = self._find_streaming_targets(router, sr, context)

        logger.debug(
            "Streaming targets for room %s: %d found",
            room_id,
            len(streaming_targets),
        )

        # Shared state for the segment persistence logic.
        accumulated_text: list[str] = []
        persisted_events: list[RoomEvent] = []
        # One cascade for the whole response: each segment's delivery is
        # enqueued without waiting (blocking the generator on an SMS round
        # trip would stall the stream), and the run is awaited once, after
        # the stream, by the caller.
        cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        # The channel a text segment is reaching *as it is produced* — the
        # stream itself is its delivery, so the lane must not send it again.
        # Only the first target streams (V1, below); any other streaming-capable
        # channel is an ordinary recipient. Cleared when the stream fails: text
        # accumulated past the failure never reached it and must go out like
        # any other event.
        streamed_to: set[str] = (
            {streaming_targets[0][1].channel_id} if streaming_targets else set()
        )
        correlation_id = uuid4().hex
        chain_depth = sr.trigger_event.chain_depth + 1
        visibility = response_vis or "all"
        # Inherit the trigger's thread root so the AI reply lands in the same
        # thread (already normalised to a root by the locked pipeline). None
        # when the trigger is top-level — the reply stays top-level too.
        parent_event_id = sr.trigger_event.parent_event_id

        def _make_source() -> EventSource:
            return EventSource(
                channel_id=sr.source_channel_id,
                channel_type=sr.source_channel_type,
            )

        # Planning inputs for the whole run, resolved once. Every segment has
        # the same sender and the same delivery set, so re-resolving per
        # segment would buy nothing and cost a room lock plus a context read
        # each time — on a tool-heavy turn, tens of them on the hot path the
        # delivery lanes exist to keep clear. The binding is already in the
        # context the caller built; the batch broadcast this replaced planned
        # off that same snapshot.
        source_binding = next(
            (b for b in context.bindings if b.channel_id == sr.source_channel_id), None
        )
        plan_source: DeliverySource | str = (
            DeliverySource(binding=source_binding, context=context)
            if source_binding is not None
            else sr.source_channel_id
        )

        async def _lane_segment(event: RoomEvent, *, exclude: set[str] | None) -> None:
            """Commit a segment and queue its delivery on the room's lane.

            Deliberately not awaited to completion: this runs inside the
            streaming channel's ``deliver_stream``, and blocking the
            generator on a transport round trip would stall the stream.
            ``cascade`` collects every segment's unit instead, and the
            caller waits on it once the stream is done.
            """
            stored = await self._commit_and_deliver(
                room_id,
                event,
                plan_source,
                exclude_delivery=exclude,
                cascade=cascade,
            )
            if stored is not None:
                persisted_events.append(stored)

        async def _persist_text_segment(*, cancelled: bool = False) -> None:
            """Persist the accumulated text as a MESSAGE event.

            ``sr.response_metadata`` (the turn's ``AIContext.response_metadata``,
            the same live record) rides every MESSAGE segment as it stands
            when the segment is persisted — persisted before broadcast, so
            turn-level attribution, including what a tool handler wrote
            mid-loop, lands in the stored row and in the stream_end frame
            without any post-hoc rewrite.

            ``cancelled`` marks a segment cut short by an interrupted turn, so
            a reader can tell a finished answer from one the user stopped.
            """
            if not accumulated_text:
                return
            text = "".join(accumulated_text)
            accumulated_text.clear()
            metadata = dict(sr.response_metadata or {})
            if cancelled:
                metadata["cancelled"] = True
            event = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                type=EventType.MESSAGE,
                content=TextContent(body=text),
                status=EventStatus.DELIVERED,
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
                metadata=metadata,
            )
            # Run BEFORE_BROADCAST sync hooks on the assembled segment before it
            # is stored and re-broadcast, mirroring the locked path. The live
            # chunks already piped to streaming channels are outside a hook's
            # reach by construction; this lands any hook modification (e.g. PII
            # de-anonymisation) on the persisted row and the re-broadcast to
            # non-streaming channels, and drops a segment a hook blocks.
            #
            # Scope: only the allow/modify/block decision is honoured here. The
            # hook side effects the locked path also applies — injected_events,
            # tasks, observations — are not yet replayed on the streaming path.
            # No streaming hook uses them today; wiring full parity is a
            # follow-up if one ever does.
            sync_result = await self._hook_engine.run_sync_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, event, context
            )
            if sync_result.hook_errors:
                logger.warning(
                    "BEFORE_BROADCAST hook error on streamed segment (room %s): %s",
                    room_id,
                    sync_result.hook_errors,
                )
            if not sync_result.allowed:
                logger.info(
                    "Streamed segment blocked by BEFORE_BROADCAST hook (room %s): %s",
                    room_id,
                    sync_result.reason,
                )
                return
            if isinstance(sync_result.event, RoomEvent):
                event = sync_result.event
            # The streaming channels already rendered this text chunk by
            # chunk — only the others get it as an event.
            await _lane_segment(event, exclude=set(streamed_to))

        async def _persist_tool_start(marker: ToolCallStartMarker) -> None:
            event = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                type=EventType.TOOL_CALL_START,
                content=ToolCallContent(
                    tool_name=marker.tool_name,
                    tool_id=marker.tool_id,
                    arguments=marker.arguments,
                    status="pending",
                ),
                status=EventStatus.DELIVERED,
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            # Tool-call events are delivered to every channel, streaming ones
            # included: a stream renders text, not tool cards.
            await _lane_segment(event, exclude=None)

        async def _persist_tool_end(marker: ToolCallEndMarker) -> None:
            event = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                type=EventType.TOOL_CALL_END,
                content=ToolCallContent(
                    tool_name=marker.tool_name,
                    tool_id=marker.tool_id,
                    arguments=marker.arguments,
                    result=marker.result,
                    status=marker.status,
                    duration_ms=marker.duration_ms,
                    error=marker.error,
                    structured_content=marker.structured_content,
                ),
                status=EventStatus.DELIVERED,
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            await _lane_segment(event, exclude=None)

        # Generator that yields text deltas and persisted events.
        # Text deltas drive the streaming bubble; RoomEvents are delivered
        # as regular events interleaved between stream chunks.
        async def segment_stream() -> Any:
            """Yield str for text deltas, RoomEvent for persisted segments.

            Thinking markers pass straight through to the channel — they
            carry transient display info only and are not persisted as
            RoomEvents (the realtime bus still publishes a buffered
            ``THINKING_END`` for out-of-band observers).
            """
            async for delta in sr.stream:
                if isinstance(delta, str):
                    accumulated_text.append(delta)
                    yield delta
                elif isinstance(delta, ThinkingDeltaMarker):
                    yield delta
                elif isinstance(delta, ToolCallStartMarker):
                    # Persist text before the tool call and yield if new
                    count = len(persisted_events)
                    await _persist_text_segment()
                    if len(persisted_events) > count:
                        yield persisted_events[-1]
                    # Persist and yield tool call start
                    count = len(persisted_events)
                    await _persist_tool_start(delta)
                    if len(persisted_events) > count:
                        yield persisted_events[-1]
                elif isinstance(delta, ToolCallEndMarker):
                    count = len(persisted_events)
                    await _persist_tool_end(delta)
                    if len(persisted_events) > count:
                        yield persisted_events[-1]

            # Persist final text segment and yield it
            count = len(persisted_events)
            await _persist_text_segment()
            if len(persisted_events) > count:
                yield persisted_events[-1]

        stream_error: Exception | None = None
        if streaming_targets:
            channel, binding = streaming_targets[0]  # V1: single target
            placeholder = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                content=TextContent(body=""),
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
            )
            try:
                await channel.deliver_stream(segment_stream(), placeholder, binding, context)
            except asyncio.CancelledError:
                # A turn interrupted on purpose (the console's Esc). What was
                # already streamed is on the user's screen, so the timeline
                # MUST hold it too: dropping it would leave the room
                # disagreeing with what the human read, and the agent's next
                # context missing what it already said. Not an error — nobody
                # failed — so ON_ERROR stays silent and the cancellation
                # propagates untouched.
                await _persist_text_segment(cancelled=True)
                raise
            except Exception as exc:
                stream_error = exc
                self._log_stream_failure(
                    exc, f"streaming delivery to {binding.channel_id}", room_id
                )
                # Persist any text accumulated before the error. The stream is
                # gone, so this text never reached its channels — it goes out
                # as an ordinary event, to everyone.
                streamed_to.clear()
                await _persist_text_segment()
                await self._fire_error_hook(
                    room_id,
                    context,
                    _make_source(),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_category="streaming",
                    chain_depth=chain_depth,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
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
                async for _ in segment_stream():
                    pass
            except asyncio.CancelledError:
                await _persist_text_segment(cancelled=True)
                raise
            except Exception as exc:
                stream_error = exc
                self._log_stream_failure(
                    exc, "stream consumption (no targets)", room_id, headless=True
                )
                await _persist_text_segment()
                await self._fire_error_hook(
                    room_id,
                    context,
                    _make_source(),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_category="streaming",
                    chain_depth=chain_depth,
                    visibility=visibility,
                    correlation_id=correlation_id,
                    parent_event_id=parent_event_id,
                )

        # Every segment's delivery set, awaited once now that the stream is
        # done — the run's completion is what the caller's turn waits on.
        await cascade.wait()

        if not persisted_events and stream_error is None:
            return None

        return _StreamingResult(events=persisted_events, error=stream_error)

    @staticmethod
    def _log_stream_failure(
        exc: Exception, what: str, room_id: str, *, headless: bool = False
    ) -> None:
        """Log a streaming-response failure at the right verbosity.

        A ``ProviderError`` (backend unreachable, 5xx, timeout, context
        overflow) is a transient/expected condition, not a code defect — no
        traceback, and the error is also returned to the caller and delivered to
        ``ON_ERROR`` hooks. When there is no streaming target (``headless`` — a
        one-shot programmatic caller that owns its own logging), a framework
        WARNING would just duplicate the caller's line, so it drops to DEBUG;
        with a streaming target the framework WARNING is the operational record.
        Any other exception is unexpected and keeps its full traceback.
        """
        if isinstance(exc, ProviderError):
            level = logging.DEBUG if headless else logging.WARNING
            logger.log(level, "%s failed for room %s: %s", what, room_id, exc)
        else:
            logger.exception("%s failed for room %s", what, room_id)

    def _find_streaming_targets(
        self,
        router: Any,
        sr: Any,
        context: RoomContext,
    ) -> list[Any]:
        """Find transport channels that support streaming delivery."""
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
        pending_streams: list[Any],
        room_id: str,
    ) -> tuple[Exception | None, ResponseMetadata]:
        """Handle streaming responses outside the room lock.

        Streaming delivery (TTS playback) can take seconds. Running it outside
        the lock allows other process_inbound calls to proceed concurrently,
        preventing continuous STT echo from being queued behind the lock.

        Each segment commits and reaches the non-streaming channels through
        the room's delivery lane as it is produced (RFC §10.2 — the lane is
        the room's single ordering authority, and its executor fires each
        segment's AFTER_BROADCAST once that segment's delivery set has run,
        step 16). Broadcasting the run in one batch after the stream is what
        used to let the cursor run ahead of the deliveries.

        Returns the first response-stream failure encountered (so the inbound
        pipeline can surface it to a headless caller), or ``None`` when every
        stream completed, plus the turn's response-metadata record.

        The record is handed back rather than left to be read off a persisted
        segment: a turn that ends on a tool call persists no segment after it,
        and one that ends before writing any text persists none at all, so the
        room is not a place where "how did that turn end" can always be asked.
        """
        router = self._get_router()
        context = await self._build_context(room_id)

        first_error: Exception | None = None
        record = ResponseMetadata()
        for sr in pending_streams:
            sr_result = await self._handle_streaming_response(router, sr, room_id, context)
            if sr_result and sr_result.error and first_error is None:
                first_error = sr_result.error
            # Several streams answer one inbound only when several channels
            # replied; each writes under its own key, so merging keeps them
            # all rather than letting the last one win.
            record.update(sr.response_metadata or {})

        return first_error, record

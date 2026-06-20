"""InboundStreamingMixin — streaming response handling outside the room lock."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT, HelpersMixin
from roomkit.core.visibility import visibility_allows
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    EventType,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter, StreamingResponse
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@dataclass
class _StreamingResult:
    """Result of handling a streaming response."""

    events: list[RoomEvent] = field(default_factory=list)
    delivered_to: set[str] = field(default_factory=set)


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
        correlation_id = uuid4().hex
        chain_depth = sr.trigger_event.chain_depth + 1
        visibility = response_vis or "all"

        def _make_source() -> EventSource:
            return EventSource(
                channel_id=sr.source_channel_id,
                channel_type=sr.source_channel_type,
            )

        async def _persist_text_segment() -> None:
            """Persist the accumulated text as a MESSAGE event.

            ``sr.response_metadata`` (AIContext.response_metadata captured at
            stream start) rides every MESSAGE segment — persisted before
            broadcast, so turn-level attribution lands in the stored row
            and in the stream_end frame without any post-hoc rewrite.
            """
            if not accumulated_text:
                return
            text = "".join(accumulated_text)
            accumulated_text.clear()
            event = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                type=EventType.MESSAGE,
                content=TextContent(body=text),
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
                metadata=dict(sr.response_metadata or {}),
            )
            stored = await self._persist_event_auto_index(room_id, event)
            if stored is not None:
                persisted_events.append(stored)

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
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
            )
            stored = await self._persist_event_auto_index(room_id, event)
            if stored is not None:
                persisted_events.append(stored)

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
                ),
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
            )
            stored = await self._persist_event_auto_index(room_id, event)
            if stored is not None:
                persisted_events.append(stored)

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

        delivered_to: set[str] = set()
        if streaming_targets:
            channel, binding = streaming_targets[0]  # V1: single target
            placeholder = RoomEvent(
                room_id=room_id,
                source=_make_source(),
                content=TextContent(body=""),
                chain_depth=chain_depth,
                visibility=visibility,
                correlation_id=correlation_id,
            )
            try:
                await channel.deliver_stream(segment_stream(), placeholder, binding, context)
                delivered_to.add(binding.channel_id)
            except Exception as exc:
                logger.exception("Streaming delivery to %s failed", binding.channel_id)
                # Persist any text accumulated before the error
                await _persist_text_segment()
                error_event = RoomEvent(
                    room_id=room_id,
                    source=_make_source(),
                    content=TextContent(body=str(exc)),
                    metadata={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "error_category": "streaming",
                    },
                    chain_depth=chain_depth,
                    visibility=visibility,
                    correlation_id=correlation_id,
                )
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.ON_ERROR, error_event, context
                )
        else:
            # No streaming targets — consume stream (triggers persistence via markers)
            async for _ in segment_stream():
                pass

        if not persisted_events:
            return None

        return _StreamingResult(events=persisted_events, delivered_to=delivered_to)

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
            supports = getattr(channel, "supports_streaming_delivery", False) if channel else False
            if channel and supports:
                targets.append((channel, binding))
        return targets

    async def _process_streaming_responses(
        self,
        pending_streams: list[Any],
        room_id: str,
    ) -> None:
        """Handle streaming responses outside the room lock.

        Streaming delivery (TTS playback) can take seconds. Running it outside
        the lock allows other process_inbound calls to proceed concurrently,
        preventing continuous STT echo from being queued behind the lock.
        """
        router = self._get_router()
        context = await self._build_context(room_id)

        for sr in pending_streams:
            sr_result = await self._handle_streaming_response(router, sr, room_id, context)
            if sr_result and sr_result.events:
                # Broadcast all persisted segments to non-streaming channels.
                # Tool call events are safe to broadcast — the AI channel's
                # self-loop guard (ai.py:283) skips events from its own
                # channel_id, so no AI re-response is triggered.
                first_event = sr_result.events[0]
                binding = await self._store.get_binding(room_id, first_event.source.channel_id)
                if binding:
                    for seg_event in sr_result.events:
                        reentry_ctx = context.model_copy(
                            update={
                                "recent_events": [
                                    *context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                                    seg_event,
                                ]
                            }
                        )
                        # Text segments were already streamed to streaming
                        # channels — exclude them. Tool call events were NOT
                        # streamed, so deliver to all channels.
                        is_text = seg_event.type == EventType.MESSAGE
                        reentry_result = await router.broadcast(
                            seg_event,
                            binding,
                            reentry_ctx,
                            exclude_delivery=(sr_result.delivered_to if is_text else None),
                        )
                        for blocked in reentry_result.blocked_events:
                            await self._store.add_event(blocked)

                    # Run AFTER_BROADCAST hooks for each segment so hook
                    # authors see the complete timeline.
                    for seg_event in sr_result.events:
                        await self._hook_engine.run_async_hooks(
                            room_id,
                            HookTrigger.AFTER_BROADCAST,
                            seg_event,
                            context,
                        )

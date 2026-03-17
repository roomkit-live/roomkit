"""InboundStreamingMixin — streaming response handling outside the room lock."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    HookTrigger,
)
from roomkit.models.event import RoomEvent

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

    event: RoomEvent
    delivered_to: set[str] = field(default_factory=set)


class InboundStreamingMixin(HelpersMixin):
    """Streaming response handling extracted from the inbound pipeline.

    These methods run outside the room lock so that streaming delivery
    (e.g. TTS playback) does not block other ``process_inbound`` calls.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    async def _handle_streaming_response(
        self,
        router: EventRouter,
        sr: StreamingResponse,
        room_id: str,
        context: RoomContext,
    ) -> _StreamingResult | None:
        """Consume a streaming response, pipe to streaming channels, store result."""
        from roomkit.models.event import EventSource, TextContent

        # Find streaming delivery targets (transport channels that support it).
        # Apply the same permission checks as _filter_targets() in broadcast():
        # access, direction, and response_visibility.
        response_vis = sr.trigger_event.response_visibility
        streaming_targets: list[Any] = []
        for binding in context.bindings:
            if binding.category != ChannelCategory.TRANSPORT:
                continue
            if binding.channel_id == sr.source_channel_id:
                continue
            if binding.access in (Access.WRITE_ONLY, Access.NONE):
                continue
            if binding.direction == ChannelDirection.OUTBOUND:
                continue
            # Apply response_visibility filter (same logic as _check_visibility)
            if response_vis is not None and response_vis != "all":
                if response_vis == "none":
                    continue
                if response_vis == "transport":
                    pass  # already filtered to TRANSPORT above
                elif response_vis == "intelligence":
                    continue  # skip all transport targets
                elif "," in response_vis:
                    allowed = {cid.strip() for cid in response_vis.split(",") if cid.strip()}
                    if binding.channel_id not in allowed:
                        continue
                elif binding.channel_id != response_vis:
                    continue
            channel = router.get_channel(binding.channel_id)
            supports = getattr(channel, "supports_streaming_delivery", False) if channel else False
            if channel and supports:
                streaming_targets.append((channel, binding))

        logger.debug(
            "Streaming targets for room %s: %d found",
            room_id,
            len(streaming_targets),
        )

        accumulated: list[str] = []
        stream_exhausted = asyncio.Event()

        async def accumulated_stream() -> Any:
            try:
                async for delta in sr.stream:
                    accumulated.append(delta)
                    yield delta
            finally:
                stream_exhausted.set()

        async def _store_when_ready() -> RoomEvent | None:
            """Store the response as soon as the text stream exhausts.

            This runs concurrently with audio playback so the response
            is visible in the conversation store before TTS finishes.
            """
            await stream_exhausted.wait()
            full_text = "".join(accumulated)
            if not full_text:
                return None
            event = RoomEvent(
                room_id=room_id,
                source=EventSource(
                    channel_id=sr.source_channel_id,
                    channel_type=sr.source_channel_type,
                ),
                content=TextContent(body=full_text),
                chain_depth=sr.trigger_event.chain_depth + 1,
                # visibility mirrors response_visibility so the subsequent
                # broadcast of this stored event is filtered correctly by
                # the existing _check_visibility() logic in the router.
                visibility=response_vis or "all",
            )
            return await self._store.add_event_auto_index(room_id, event)

        store_task = asyncio.create_task(_store_when_ready())

        delivered_to: set[str] = set()
        if streaming_targets:
            channel, binding = streaming_targets[0]  # V1: single target
            placeholder = RoomEvent(
                room_id=room_id,
                source=EventSource(
                    channel_id=sr.source_channel_id,
                    channel_type=sr.source_channel_type,
                ),
                content=TextContent(body=""),
                chain_depth=sr.trigger_event.chain_depth + 1,
                visibility=response_vis or "all",
            )
            try:
                await channel.deliver_stream(accumulated_stream(), placeholder, binding, context)
                delivered_to.add(binding.channel_id)
            except Exception as exc:
                logger.exception("Streaming delivery to %s failed", binding.channel_id)
                # Fire ON_ERROR hook so consumers can react
                error_event = RoomEvent(
                    room_id=room_id,
                    source=EventSource(
                        channel_id=sr.source_channel_id,
                        channel_type=sr.source_channel_type,
                    ),
                    content=TextContent(body="".join(accumulated)),
                    metadata={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "error_category": "streaming",
                        "stream_partial_text": "".join(accumulated),
                    },
                    chain_depth=sr.trigger_event.chain_depth + 1,
                    visibility=response_vis or "all",
                )
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.ON_ERROR, error_event, context
                )
        else:
            # No streaming targets — just consume the stream
            async for delta in sr.stream:
                accumulated.append(delta)
            stream_exhausted.set()

        response_event = await store_task
        if response_event is None:
            return None

        return _StreamingResult(event=response_event, delivered_to=delivered_to)

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
        router = self._get_router()  # type: ignore[attr-defined]
        context = await self._build_context(room_id)

        for sr in pending_streams:
            sr_result = await self._handle_streaming_response(router, sr, room_id, context)
            if sr_result:
                # Broadcast complete text to non-streaming channels
                binding = await self._store.get_binding(room_id, sr_result.event.source.channel_id)
                if binding:
                    reentry_ctx = context.model_copy(
                        update={
                            "recent_events": [
                                *context.recent_events[-49:],
                                sr_result.event,
                            ]
                        }
                    )
                    reentry_result = await router.broadcast(
                        sr_result.event,
                        binding,
                        reentry_ctx,
                        exclude_delivery=sr_result.delivered_to,
                    )
                    for blocked in reentry_result.blocked_events:
                        await self._store.add_event(blocked)
                    # Run AFTER_BROADCAST hooks for the AI response
                    await self._hook_engine.run_async_hooks(
                        room_id,
                        HookTrigger.AFTER_BROADCAST,
                        sr_result.event,
                        reentry_ctx,
                    )

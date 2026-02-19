"""Event routing with permission enforcement and transcoding."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels.base import Channel
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.core.retry import retry_with_backoff
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    EventStatus,
)
from roomkit.models.event import (
    AudioContent,
    DeleteContent,
    EditContent,
    LocationContent,
    MediaContent,
    RichContent,
    RoomEvent,
    TemplateContent,
    TextContent,
    VideoContent,
)
from roomkit.models.task import Observation, Task

logger = logging.getLogger("roomkit.event_router")


@dataclass
class StreamingResponse:
    """A streaming response from an intelligence channel."""

    stream: Any  # AsyncIterator[str]
    source_channel_id: str
    source_channel_type: Any  # ChannelType
    trigger_event: RoomEvent


@dataclass
class BroadcastResult:
    """Result of broadcasting an event to channels."""

    outputs: dict[str, ChannelOutput] = field(default_factory=dict)
    delivery_outputs: dict[str, ChannelOutput] = field(default_factory=dict)
    reentry_events: list[RoomEvent] = field(default_factory=list)
    streaming_responses: list[StreamingResponse] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    metadata_updates: dict[str, Any] = field(default_factory=dict)
    blocked_events: list[RoomEvent] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class _TargetResult:
    """Per-target result collected during concurrent broadcast."""

    channel_id: str
    output: ChannelOutput | None = None
    delivery_output: ChannelOutput | None = None
    streaming_response: StreamingResponse | None = None
    error: str | None = None
    reentry_events: list[RoomEvent] = field(default_factory=list)
    blocked_events: list[RoomEvent] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


class EventRouter:
    """Routes events to target channels with access control and transcoding."""

    def __init__(
        self,
        channels: dict[str, Channel],
        transcoder: DefaultContentTranscoder | None = None,
        max_chain_depth: int = 5,
        rate_limiter: TokenBucketRateLimiter | None = None,
        telemetry: Any = None,
    ) -> None:
        self._channels = channels
        self._transcoder = transcoder or DefaultContentTranscoder()
        self._max_chain_depth = max_chain_depth
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._telemetry = telemetry

    def _get_breaker(self, channel_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a channel."""
        if channel_id not in self._circuit_breakers:
            self._circuit_breakers[channel_id] = CircuitBreaker()
        return self._circuit_breakers[channel_id]

    def get_channel(self, channel_id: str) -> Channel | None:
        """Look up a channel by ID."""
        return self._channels.get(channel_id)

    async def broadcast(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        context: RoomContext,
        *,
        exclude_delivery: set[str] | None = None,
    ) -> BroadcastResult:
        """Broadcast an event to all eligible channels in the room.

        RFC §3.8: For each target channel:
        - on_event(): all channels react (intelligence generates, observers analyze)
        - deliver(): only transport channels push to external recipients

        Args:
            exclude_delivery: Channel IDs to skip delivery for (already
                received content via streaming).
        """
        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = self._telemetry or NoopTelemetryProvider()
        session_id = (event.metadata or {}).get("voice_session_id")
        span_id = telemetry.start_span(
            SpanKind.BROADCAST,
            "framework.broadcast",
            parent_id=get_current_span(),
            room_id=event.room_id,
            session_id=session_id,
            attributes={Attr.CHANNEL_ID: source_binding.channel_id},
        )
        broadcast_token = set_current_span(span_id)

        result = BroadcastResult()

        # Check source can write
        if source_binding.access in (Access.READ_ONLY, Access.NONE):
            logger.debug(
                "Source %s has no write access",
                source_binding.channel_id,
                extra={"room_id": event.room_id, "channel_id": source_binding.channel_id},
            )
            reset_span(broadcast_token)
            telemetry.end_span(span_id, attributes={"target_count": 0})
            return result

        # Check source is not muted
        if source_binding.muted:
            logger.debug(
                "Source %s is muted",
                source_binding.channel_id,
                extra={"room_id": event.room_id, "channel_id": source_binding.channel_id},
            )
            reset_span(broadcast_token)
            telemetry.end_span(span_id, attributes={"target_count": 0})
            return result

        # Stamp visibility from source binding
        if event.visibility == "all" and source_binding.visibility != "all":
            event = event.model_copy(update={"visibility": source_binding.visibility})

        # Determine target bindings (includes muted channels — they can still read)
        targets = self._filter_targets(event, source_binding, context.bindings)

        if not targets:
            reset_span(broadcast_token)
            telemetry.end_span(span_id, attributes={"target_count": 0})
            return result

        # Collect per-target results to avoid concurrent mutation
        target_results: list[_TargetResult] = []

        async def _process_target(binding: ChannelBinding) -> None:
            channel = self._channels.get(binding.channel_id)
            if channel is None:
                logger.warning(
                    "Channel %s not found in registry, skipping. Available: %s",
                    binding.channel_id,
                    list(self._channels.keys()),
                )
                return

            tr = _TargetResult(channel_id=binding.channel_id)

            try:
                # Transcode content if needed
                transcoded_event = await self._maybe_transcode(event, source_binding, binding)
                if transcoded_event is None:
                    tr.error = "transcoding_failed"
                    logger.warning(
                        "Transcoding failed for channel %s — skipping delivery",
                        binding.channel_id,
                        extra={
                            "room_id": event.room_id,
                            "channel_id": binding.channel_id,
                            "event_id": event.id,
                        },
                    )
                    target_results.append(tr)
                    return

                # Enforce max_length on text content
                if binding.capabilities.max_length is not None:
                    transcoded_event = self._enforce_max_length(
                        transcoded_event, binding.capabilities.max_length
                    )

                # Orchestration: skip intelligence channels for internal events
                if binding.category == ChannelCategory.INTELLIGENCE:
                    if (transcoded_event.metadata or {}).get("_orchestration_internal"):
                        target_results.append(tr)
                        return
                    routed_to = (transcoded_event.metadata or {}).get("_routed_to")
                    if routed_to is not None:
                        always_process = (transcoded_event.metadata or {}).get(
                            "_always_process", []
                        )
                        if (
                            binding.channel_id != routed_to
                            and binding.channel_id not in always_process
                        ):
                            target_results.append(tr)
                            return

                # Step 1: on_event — all channels react
                output = await channel.on_event(transcoded_event, binding, context)
                tr.output = output

                # Streaming response: capture handle, skip reentry logic
                if output.response_stream is not None:
                    tr.streaming_response = StreamingResponse(
                        stream=output.response_stream,
                        source_channel_id=binding.channel_id,
                        source_channel_type=binding.channel_type,
                        trigger_event=transcoded_event,
                    )
                    tr.observations.extend(output.observations)
                    target_results.append(tr)
                    return

                # Step 2: deliver — only transport channels push to external
                if binding.category == ChannelCategory.TRANSPORT:
                    # Skip delivery for channels that already received streaming content
                    if exclude_delivery and binding.channel_id in exclude_delivery:
                        tr.observations.extend(output.observations)
                        target_results.append(tr)
                        return

                    breaker = self._get_breaker(binding.channel_id)

                    if not breaker.allow_request():
                        tr.error = "circuit_breaker_open"
                        logger.warning(
                            "Circuit breaker open for %s — skipping delivery",
                            binding.channel_id,
                            extra={
                                "room_id": transcoded_event.room_id,
                                "channel_id": binding.channel_id,
                            },
                        )
                    else:
                        # Rate limit
                        if binding.rate_limit is not None:
                            await self._rate_limiter.wait(binding.channel_id, binding.rate_limit)

                        try:
                            if binding.retry_policy is not None:
                                delivery_output = await retry_with_backoff(
                                    channel.deliver,
                                    binding.retry_policy,
                                    transcoded_event,
                                    binding,
                                    context,
                                )
                            else:
                                delivery_output = await channel.deliver(
                                    transcoded_event, binding, context
                                )
                            tr.delivery_output = delivery_output
                            breaker.record_success()
                        except Exception as exc:
                            breaker.record_failure()
                            tr.error = str(exc)
                            logger.exception(
                                "Delivery to %s failed",
                                binding.channel_id,
                                extra={
                                    "room_id": event.room_id,
                                    "channel_id": binding.channel_id,
                                    "event_id": event.id,
                                },
                            )

                # Always collect side effects (tasks, observations, metadata)
                # regardless of mute status — RFC: "muting silences the voice,
                # not the brain"
                tr.observations.extend(output.observations)

                # Muted channels: suppress response_events (the "voice")
                if binding.muted:
                    if output.responded:
                        logger.debug(
                            "Channel %s is muted — suppressing %d response events, "
                            "keeping %d tasks, %d observations",
                            binding.channel_id,
                            len(output.response_events),
                            len(output.tasks),
                            len(output.observations),
                        )
                    target_results.append(tr)
                    return

                # Collect reentry events with chain depth enforcement
                if output.responded:
                    for resp in output.response_events:
                        if resp.chain_depth < self._max_chain_depth:
                            tr.reentry_events.append(resp)
                        else:
                            blocked = resp.model_copy(
                                update={
                                    "status": EventStatus.BLOCKED,
                                    "blocked_by": "event_chain_depth_limit",
                                }
                            )
                            tr.blocked_events.append(blocked)
                            tr.observations.append(
                                Observation(
                                    id=f"obs_{blocked.id}",
                                    room_id=event.room_id,
                                    channel_id=binding.channel_id,
                                    content=(
                                        f"Event chain depth {resp.chain_depth}"
                                        f" exceeded limit {self._max_chain_depth}"
                                    ),
                                    category="event_chain_depth_exceeded",
                                    metadata={
                                        "chain_depth": resp.chain_depth,
                                        "max_chain_depth": self._max_chain_depth,
                                        "source_channel": binding.channel_id,
                                    },
                                )
                            )
                            logger.warning(
                                "Chain depth %d exceeded limit %d for channel %s — event blocked",
                                resp.chain_depth,
                                self._max_chain_depth,
                                binding.channel_id,
                                extra={
                                    "room_id": event.room_id,
                                    "channel_id": binding.channel_id,
                                    "chain_depth": resp.chain_depth,
                                },
                            )

            except Exception as exc:
                tr.error = str(exc)
                logger.exception(
                    "Processing target %s failed",
                    binding.channel_id,
                    extra={
                        "room_id": event.room_id,
                        "channel_id": binding.channel_id,
                        "event_id": event.id,
                    },
                )

            target_results.append(tr)

        await asyncio.gather(*[_process_target(t) for t in targets], return_exceptions=True)

        # Merge per-target results into BroadcastResult (single-threaded)
        for tr in target_results:
            if tr.output is not None:
                result.outputs[tr.channel_id] = tr.output
                result.tasks.extend(tr.output.tasks)
                result.metadata_updates.update(tr.output.metadata_updates)
            if tr.delivery_output is not None:
                result.delivery_outputs[tr.channel_id] = tr.delivery_output
            if tr.streaming_response is not None:
                result.streaming_responses.append(tr.streaming_response)
            if tr.error is not None:
                result.errors[tr.channel_id] = tr.error
            result.reentry_events.extend(tr.reentry_events)
            result.blocked_events.extend(tr.blocked_events)
            result.observations.extend(tr.observations)

        reset_span(broadcast_token)
        telemetry.end_span(
            span_id,
            attributes={
                "target_count": len(targets),
                "delivered_count": len(result.delivery_outputs),
                "failed_count": len(result.errors),
            },
        )

        return result

    def _filter_targets(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        all_bindings: list[ChannelBinding],
    ) -> list[ChannelBinding]:
        """Filter bindings to find valid delivery targets.

        Muted channels ARE included — they can still receive events via on_event()
        and produce side effects (tasks, observations). Their response_events are
        suppressed in broadcast().
        """
        targets: list[ChannelBinding] = []

        for binding in all_bindings:
            # Skip source channel
            if binding.channel_id == source_binding.channel_id:
                continue

            # Check access - must be able to read
            if binding.access in (Access.WRITE_ONLY, Access.NONE):
                continue

            # Check direction - must accept inbound delivery
            if binding.direction == ChannelDirection.OUTBOUND:
                continue

            # NOTE: muted channels are NOT skipped here — they receive events
            # but their response_events are suppressed in broadcast()

            # Check visibility
            if not self._check_visibility(event, source_binding, binding):
                continue

            targets.append(binding)

        return targets

    def _check_visibility(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> bool:
        """Check if source is visible to target based on visibility rules."""
        vis = source_binding.visibility

        if vis == "all":
            return True

        if vis == "none":
            return False

        # "transport" - only visible to transport channels
        if vis == "transport":
            return target_binding.category == ChannelCategory.TRANSPORT

        # "intelligence" - only visible to intelligence channels
        if vis == "intelligence":
            return target_binding.category == ChannelCategory.INTELLIGENCE

        # Comma-separated list of channel IDs
        if "," in vis:
            allowed = {cid.strip() for cid in vis.split(",") if cid.strip()}
            return target_binding.channel_id in allowed

        # Single channel ID
        return target_binding.channel_id == vis

    @staticmethod
    def _content_media_type(content: Any) -> ChannelMediaType | None:
        """Map event content to its primary media type."""
        if isinstance(content, TextContent):
            return ChannelMediaType.TEXT
        if isinstance(content, RichContent):
            return ChannelMediaType.RICH
        if isinstance(content, MediaContent):
            return ChannelMediaType.MEDIA
        if isinstance(content, AudioContent):
            return ChannelMediaType.AUDIO
        if isinstance(content, VideoContent):
            return ChannelMediaType.VIDEO
        if isinstance(content, LocationContent):
            return ChannelMediaType.LOCATION
        if isinstance(content, TemplateContent):
            return ChannelMediaType.TEMPLATE
        # Composite, Edit, Delete, System — no single media type
        return None

    async def _maybe_transcode(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> RoomEvent | None:
        """Transcode event content if the target doesn't support it.

        Returns ``None`` if the content cannot be transcoded for the target.
        """
        target_types = set(target_binding.capabilities.media_types)

        # Check if the specific content type is already supported
        content_media = self._content_media_type(event.content)
        if content_media is not None and content_media in target_types:
            return event

        # Edit/Delete: check capability flags directly
        if isinstance(event.content, EditContent) and target_binding.capabilities.supports_edit:
            return event
        if (
            isinstance(event.content, DeleteContent)
            and target_binding.capabilities.supports_delete
        ):
            return event

        transcoded_content = await self._transcoder.transcode(
            event.content, source_binding, target_binding
        )
        if transcoded_content is None:
            return None
        if transcoded_content is event.content:
            return event

        return event.model_copy(update={"content": transcoded_content})

    @staticmethod
    def _enforce_max_length(event: RoomEvent, max_length: int) -> RoomEvent:
        """Truncate text content if it exceeds the channel's max_length."""
        max_length = max(3, max_length)
        content = event.content
        if isinstance(content, TextContent) and len(content.body) > max_length:
            truncated = content.body[: max_length - 3] + "..."
            return event.model_copy(
                update={"content": TextContent(body=truncated, language=content.language)}
            )
        return event

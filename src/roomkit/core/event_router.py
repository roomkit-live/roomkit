"""Event routing with permission enforcement and transcoding."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels.base import Channel
from roomkit.core._failure_log import log_failure
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.exceptions import NoRecipientError
from roomkit.core.lanes import DeliveryPlan
from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.core.retry import retry_with_backoff
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.core.visibility import effective_visibility, visibility_allows
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    Access,
    AgentResponsePolicy,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    EventStatus,
    EventType,
    Visibility,
)
from roomkit.models.event import (
    AudioContent,
    DeleteContent,
    EditContent,
    EventSource,
    LocationContent,
    MediaContent,
    RichContent,
    RoomEvent,
    TemplateContent,
    TextContent,
    VideoContent,
    is_interruption_marker,
    is_tool_call_record,
)
from roomkit.models.response_metadata import TurnEntries, turn_summary
from roomkit.models.task import Observation, Task
from roomkit.providers.utils import _aclose_stream

logger = logging.getLogger("roomkit.event_router")


def _solicits(
    event: RoomEvent,
    channel_id: str,
    *,
    source_is_agent: bool = False,
    policy: AgentResponsePolicy = AgentResponsePolicy.AGENT_CHAIN,
) -> bool:
    """Is this intelligence channel asked to act on *event*? (RFC §19.3)

    An explicit address decides, ahead of any routing decision (§19.4 step
    0): a router cannot override what the sender asked for. With no address,
    the room's policy governs an agent's own output — under
    ``ADDRESSED_ONLY`` it solicits nobody, which is what keeps a room of
    independent agents from answering each other down to the chain-depth
    limit. Otherwise the router's stamp decides, and with neither, everyone
    acts.

    Solicitation only — the caller has already resolved *visibility*, which
    is a separate question this must not re-answer.
    """
    if is_interruption_marker(event):
        # It says a turn was cut, nothing to act on (RFC §19.3 rule 5)
        return False
    metadata = event.metadata or {}
    always_process = metadata.get("_always_process", [])

    addressed = event.addressed_to
    if addressed is not None:
        # The address decides — except the supervisor, which RFC §19.4 step 4
        # adds on top of the addressed channels when a router stamps it.
        return channel_id in addressed or channel_id in always_process

    if source_is_agent and policy is AgentResponsePolicy.ADDRESSED_ONLY:
        # ADDRESSED_ONLY governs who is solicited to *act*; a supervisor is not
        # being asked to act, it is watching (RFC §19.4 step 4). Excluding it
        # here left it blind to exactly the turns it exists to oversee — every
        # unaddressed agent-to-agent exchange.
        return channel_id in always_process

    routed_to = metadata.get("_routed_to")
    if routed_to is None:
        return True
    return channel_id == routed_to or channel_id in always_process


# The ``blocked_by`` of a response past ``max_chain_depth`` (RFC §8.3).
CHAIN_DEPTH_LIMIT = "event_chain_depth_limit"


def unanswered(trigger: RoomEvent, channel_id: str, channel_type: Any) -> RoomEvent:
    """The record that stands in for an answer a guard kept *channel_id* from giving.

    Empty text from the agent, at the depth, scope and thread the answer to
    *trigger* would have had; the guard that stopped it marks it BLOCKED
    (RFC §8.3).
    """
    return RoomEvent(
        room_id=trigger.room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=TextContent(body=""),
        chain_depth=trigger.chain_depth + 1,
        visibility=trigger.response_visibility or Visibility.ALL,
        parent_event_id=trigger.parent_event_id,
        responds_to=trigger.id,
    )


def _answering(responses: list[RoomEvent], trigger: RoomEvent) -> list[RoomEvent]:
    """*responses*, each naming *trigger* as the event it answers when it named none."""
    return [
        resp
        if resp.responds_to is not None
        else resp.model_copy(update={"responds_to": trigger.id})
        for resp in responses
    ]


def chain_depth_exceeded(blocked: RoomEvent, max_chain_depth: int) -> Observation:
    """Log a record blocked past the chain-depth limit and observe it.

    One per record, its id tied to the record.
    """
    channel_id = blocked.source.channel_id
    logger.warning(
        "Chain depth %d exceeded limit %d for channel %s — event blocked",
        blocked.chain_depth,
        max_chain_depth,
        channel_id,
        extra={
            "room_id": blocked.room_id,
            "channel_id": channel_id,
            "chain_depth": blocked.chain_depth,
        },
    )
    return Observation(
        id=f"obs_{blocked.id}",
        room_id=blocked.room_id,
        channel_id=channel_id,
        content=f"Event chain depth {blocked.chain_depth} exceeded limit {max_chain_depth}",
        category="event_chain_depth_exceeded",
        metadata={
            "chain_depth": blocked.chain_depth,
            "max_chain_depth": max_chain_depth,
            "source_channel": channel_id,
        },
    )


@dataclass
class StreamingResponse:
    """A streaming response from an intelligence channel."""

    stream: Any  # AsyncIterator[str]
    source_channel_id: str
    source_channel_type: Any  # ChannelType
    trigger_event: RoomEvent
    # The turn's live record (``ChannelOutput.response_metadata``), read by the
    # persistence of each segment as it stands then — never copied here.
    response_metadata: Mapping[str, Any] = field(default_factory=dict)
    # How the turn's loop ended (``loop_end_reason``, ``ai_usage``), once its
    # reader read the ``LoopEndMarker``: this stream's own, kept off the
    # caller's merged record (RFC §6.4).
    turn_record: dict[str, Any] | None = None
    # An answer to an answer, started by a reentry pass or a streamed
    # segment's delivery rather than by the caller's own event. Set when the
    # stream joins its cascade (``DeliveryCascade.add_streams``).
    chained: bool = False


def stream_record(sr: StreamingResponse) -> dict[str, Any]:
    """A stream's turn record: its response metadata (where an ACP agent
    writes its outcome) with how its loop ended, once read (RFC §6.4)."""
    return {**sr.response_metadata, **(sr.turn_record or {})}


def stream_turn_entry(sr: StreamingResponse) -> dict[str, Any] | None:
    """A stream's entry under the caller's ``turns`` once read: its end and
    usage (RFC §6.4)."""
    return turn_summary(stream_record(sr))


def reply_turn_entry(output: ChannelOutput) -> dict[str, Any] | None:
    """A buffered reply's entry under the caller's ``turns``: its end and
    usage, read off its record or else its last message (RFC §6.4)."""
    messages = [
        event.metadata or {}
        for event in reversed(output.response_events)
        if event.type == EventType.MESSAGE
    ]
    return turn_summary(output.response_metadata, *messages)


def responder_turn_entries(result: BroadcastResult) -> TurnEntries:
    """Each responder's entry under the caller's ``turns``, by its channel
    id: a buffered reply's, or a stream's once read (RFC §6.4)."""
    entries = {
        cid: reply_turn_entry(out)
        for cid, out in result.outputs.items()
        if out.response_stream is None
    }
    entries.update(
        (sr.source_channel_id, stream_turn_entry(sr)) for sr in result.streaming_responses
    )
    return {cid: entry for cid, entry in entries.items() if entry is not None}


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
    # Same failures as ``errors`` but the live exception object per channel, so a
    # caller (e.g. the inbound pipeline surfacing a non-streaming generation
    # failure on ``InboundResult.error``) can classify it with the cause chain
    # intact — ``errors`` only carries ``str(exc)``.
    errors_exc: dict[str, Exception] = field(default_factory=dict)


@dataclass
class _TargetResult:
    """Per-target result collected during concurrent broadcast."""

    channel_id: str
    output: ChannelOutput | None = None
    delivery_output: ChannelOutput | None = None
    streaming_response: StreamingResponse | None = None
    error: str | None = None
    error_exc: Exception | None = None
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
        greeting_gate_fn: Callable[[str, float], Awaitable[None]] | None = None,
    ) -> None:
        self._channels = channels
        self._transcoder = transcoder or DefaultContentTranscoder()
        self._max_chain_depth = max_chain_depth
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._telemetry = telemetry
        self._greeting_gate_fn = greeting_gate_fn
        # Set by RoomKit after construction — the router owns the breakers,
        # the framework owns the event surface (RFC §13.1 / §8.2).
        self._framework_emitter: Any = None
        # Set by RoomKit: whether a discussion holds the room, so no agent is
        # asked at broadcast (RFC §10.2, §19.7.5).
        self._defers: Callable[[str], bool] | None = None
        self._breaker_tasks: set[asyncio.Task[Any]] = set()

    def _get_breaker(self, channel_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a channel."""
        if channel_id not in self._circuit_breakers:
            self._circuit_breakers[channel_id] = CircuitBreaker(
                on_state_change=lambda state, cid=channel_id: self._on_breaker_state(cid, state)
            )
        return self._circuit_breakers[channel_id]

    def _on_breaker_state(self, channel_id: str, state: str) -> None:
        """Publish a breaker transition as a framework event (RFC §13.1 MUST).

        The breaker trips inside a delivery attempt, which is already running
        on the loop; the emission is scheduled rather than awaited so a slow
        framework-event handler cannot stall the delivery that tripped it.
        """
        if self._framework_emitter is None:
            return
        event_type = "circuit_breaker_opened" if state == "open" else "circuit_breaker_closed"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self._framework_emitter(
                event_type,
                channel_id=channel_id,
                data={"channel_id": channel_id, "state": state},
            )
        )
        task.add_done_callback(self._breaker_tasks.discard)
        self._breaker_tasks.add(task)

    def get_channel(self, channel_id: str) -> Channel | None:
        """Look up a channel by ID."""
        return self._channels.get(channel_id)

    def plan(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        context: RoomContext,
        *,
        exclude_delivery: set[str] | None = None,
    ) -> DeliveryPlan:
        """Resolve an event's delivery set (RFC §10.1 step 12).

        Pure — no I/O, no awaits — so it runs under the room lock: the target
        set is consistent with the committed timeline. Execution
        (:meth:`execute_plan`) does not require the lock.

        A source that cannot write (RFC §7.5 — guards reentry events and
        direct broadcast callers; inbound events from non-writable sources
        are blocked before broadcast) yields an empty delivery set.
        """
        from roomkit.telemetry.context import get_current_span

        if not source_binding.can_write:
            logger.debug(
                "Source %s cannot write (access=%s, muted=%s) — not broadcasting",
                source_binding.channel_id,
                source_binding.access,
                source_binding.muted,
                extra={"room_id": event.room_id, "channel_id": source_binding.channel_id},
            )
            targets: list[ChannelBinding] = []
        else:
            # Stamp visibility from source binding
            if event.visibility == Visibility.ALL and source_binding.visibility != Visibility.ALL:
                event = event.model_copy(update={"visibility": source_binding.visibility})
            # Target bindings include muted channels — they can still read.
            targets = self._filter_targets(event, source_binding, context.bindings)
        # The caller's span, carried across the lane boundary so the
        # broadcast span stays a child of the inbound/send_event span.
        parent_span_id = get_current_span()
        parent_span_ctx = (
            self._telemetry.get_span_context(parent_span_id)
            if self._telemetry is not None and parent_span_id is not None
            else None
        )
        return DeliveryPlan(
            event=event,
            source_binding=source_binding,
            context=context,
            targets=targets,
            exclude_delivery=exclude_delivery,
            parent_span_id=parent_span_id,
            parent_span_ctx=parent_span_ctx,
            defers_solicitation=self._defers is not None and self._defers(event.room_id),
        )

    async def execute_plan(self, plan: DeliveryPlan) -> BroadcastResult:
        """Execute a plan's delivery set (RFC §10.1 step 14).

        RFC §3.8: For each target channel:
        - on_event(): all channels react (intelligence generates, observers analyze)
        - deliver(): only transport channels push to external recipients

        Runs without the room lock; per-room ordering is the caller's
        contract (the delivery lane, or an inline caller running to
        completion).
        """
        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span, reset_span, set_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        event = plan.event
        source_binding = plan.source_binding
        context = plan.context
        exclude_delivery = plan.exclude_delivery

        telemetry = self._telemetry or NoopTelemetryProvider()
        session_id = (event.metadata or {}).get("voice_session_id")
        # Restore the planner's span context first: a lane executor runs on a
        # fresh contextvars context, and the broadcast span must stay a child
        # of the span that planned it (inline callers see their own span).
        parent_token = None
        if plan.parent_span_id is not None:
            parent_token = set_current_span(
                plan.parent_span_id, telemetry_ctx=plan.parent_span_ctx
            )
        span_id = telemetry.start_span(
            SpanKind.BROADCAST,
            "framework.broadcast",
            parent_id=get_current_span(),
            room_id=event.room_id,
            session_id=session_id,
            attributes={Attr.CHANNEL_ID: source_binding.channel_id if source_binding else ""},
        )
        # Propagate backend-specific context for robust parent linking
        broadcast_token = set_current_span(
            span_id, telemetry_ctx=telemetry.get_span_context(span_id)
        )

        result = BroadcastResult()
        targets = plan.targets

        # An injected plan (bare on_event/deliver, no transcode, no reentry)
        # is executed by the framework host, never here; a plan with no
        # source binding has nothing to transcode from either way.
        if not targets or source_binding is None:
            reset_span(broadcast_token)
            if parent_token is not None:
                reset_span(parent_token)
            telemetry.end_span(span_id, attributes={"target_count": 0})
            return result

        # Collect per-target results to avoid concurrent mutation
        target_results: list[_TargetResult] = []

        async def _process_target(binding: ChannelBinding) -> None:
            if binding.category == ChannelCategory.INTELLIGENCE:
                unasked = self._unasked_result(event, source_binding, binding, context, plan)
                if unasked is not None:
                    target_results.append(unasked)
                    return

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

                # Greeting gate: wait for greeting to be stored before AI processes
                if (
                    binding.category == ChannelCategory.INTELLIGENCE
                    and self._greeting_gate_fn is not None
                ):
                    await self._greeting_gate_fn(transcoded_event.room_id, 30.0)

                # on_event — all channels react.
                output = await channel.on_event(transcoded_event, binding, context)
                tr.output = output
                if output.error is not None:
                    # Delivered and failed at once: the output goes on, and
                    # the error surfaces as a raised one would.
                    tr.error, tr.error_exc = str(output.error), output.error

                # Streaming response: capture handle, skip reentry logic
                if output.response_stream is not None:
                    # Muting silences the voice — including the streaming voice.
                    # A muted channel's brain still ran (memory ingest + context
                    # build happened in on_event), but its streamed reply must
                    # not be delivered. Close the un-consumed generator so no
                    # provider round-trip is made — the reply is never generated
                    # (RFC §7.5 rule 2 permits it). A read-only source's stream
                    # is read instead, its rows stored BLOCKED as they commit.
                    if binding.muted:
                        await _aclose_stream(output.response_stream)
                        logger.debug(
                            "Channel %s is muted — suppressing streaming response",
                            binding.channel_id,
                        )
                        tr.observations.extend(output.observations)
                        target_results.append(tr)
                        return
                    tr.streaming_response = StreamingResponse(
                        stream=output.response_stream,
                        source_channel_id=binding.channel_id,
                        source_channel_type=binding.channel_type,
                        trigger_event=transcoded_event,
                        response_metadata=output.response_metadata,
                    )
                    tr.observations.extend(output.observations)
                    target_results.append(tr)
                    return

                # deliver — only transport channels push to external.
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
                            if not isinstance(exc, NoRecipientError):
                                # A binding with no recipient says nothing
                                # about the provider; it must not trip the
                                # breaker every room on the channel shares.
                                breaker.record_failure()
                            tr.error = str(exc)
                            tr.error_exc = exc
                            log_failure(
                                logger,
                                exc,
                                f"Delivery to {binding.channel_id}",
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

                # Collect reentry events with chain depth enforcement. A muted
                # channel's response re-enters too: its commit pass runs its
                # BEFORE_BROADCAST hooks, then stores it BLOCKED with
                # ``source_muted`` (RFC §7.5 rules 2 and 3), so the timeline
                # records what the muted brain wanted to say and what its
                # hooks decided, and nothing is broadcast.
                if output.responded:
                    # A buffered answer names the event it answers (RFC §8.5),
                    # unless its channel named one itself; written back so every
                    # reader of the output (a delegated turn's) sees it.
                    output.response_events = _answering(output.response_events, event)
                    limit = plan.max_chain_depth or self._max_chain_depth
                    for resp in output.response_events:
                        if resp.chain_depth < limit:
                            tr.reentry_events.append(resp)
                        else:
                            blocked = resp.model_copy(
                                update={
                                    "status": EventStatus.BLOCKED,
                                    "blocked_by": CHAIN_DEPTH_LIMIT,
                                }
                            )
                            tr.blocked_events.append(blocked)
                            tr.observations.append(chain_depth_exceeded(blocked, limit))

            except Exception as exc:
                tr.error = str(exc)
                tr.error_exc = exc
                log_failure(
                    logger,
                    exc,
                    f"Processing target {binding.channel_id}",
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
            if tr.error_exc is not None:
                result.errors_exc[tr.channel_id] = tr.error_exc
            result.reentry_events.extend(tr.reentry_events)
            result.blocked_events.extend(tr.blocked_events)
            result.observations.extend(tr.observations)

        reset_span(broadcast_token)
        if parent_token is not None:
            reset_span(parent_token)
        telemetry.end_span(
            span_id,
            attributes={
                "target_count": len(targets),
                "delivered_count": len(result.delivery_outputs),
                "failed_count": len(result.errors),
            },
        )

        return result

    def _unasked_result(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        binding: ChannelBinding,
        context: RoomContext,
        plan: DeliveryPlan | None = None,
    ) -> _TargetResult | None:
        """What an intelligence target leaves when it is not asked to act, or ``None``.

        Decided before anything is done for the target (RFC §19.3, §19.4 step
        0): a channel that is not asked costs nothing, no registry lookup, no
        transcode, and no warning about a channel a restart has not
        re-registered but that this event never needed. Reads the
        untranscoded event: transcoding rewrites content, never the address or
        the routing metadata read here. Transport delivery is untouched:
        addressing narrows who is asked, never who may see.

        Past the chain-depth limit a solicited channel is not asked either
        (RFC §8.3): no model call, no tool, streamed or buffered alike.

        In a room a discussion holds, no agent is asked at broadcast; a turn
        the discussion gives asks its one agent, against the discussion's own
        depth limit (RFC §19.7.5).
        """
        if plan is not None and plan.turn_for is not None:
            if binding.channel_id != plan.turn_for:
                return _TargetResult(channel_id=binding.channel_id)
            limit = plan.max_chain_depth or self._max_chain_depth
            if event.chain_depth + 1 >= limit:
                return self._depth_limit_record(event, binding, limit)
            return None
        if plan is not None and plan.defers_solicitation:
            return _TargetResult(channel_id=binding.channel_id)
        internal = bool((event.metadata or {}).get("_orchestration_internal"))
        asked = not internal and _solicits(
            event,
            binding.channel_id,
            source_is_agent=source_binding.category == ChannelCategory.INTELLIGENCE,
            policy=context.room.agent_response_policy,
        )
        if not asked:
            return _TargetResult(channel_id=binding.channel_id)
        if event.chain_depth + 1 >= self._max_chain_depth:
            return self._depth_limit_record(event, binding, self._max_chain_depth)
        return None

    def _depth_limit_record(
        self, event: RoomEvent, binding: ChannelBinding, limit: int
    ) -> _TargetResult:
        """The BLOCKED record of an agent not asked past the depth limit (RFC §8.3).

        One per agent, in place of the response it was not asked for: its
        source is that agent, its text empty, its depth the one the response
        would have had. A tool-call row leaves none, since no agent answers
        one.
        """
        result = _TargetResult(channel_id=binding.channel_id)
        if is_tool_call_record(event):
            return result
        record = unanswered(event, binding.channel_id, binding.channel_type).model_copy(
            update={"status": EventStatus.BLOCKED, "blocked_by": CHAIN_DEPTH_LIMIT}
        )
        result.blocked_events.append(record)
        result.observations.append(chain_depth_exceeded(record, limit))
        return result

    async def broadcast(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        context: RoomContext,
        *,
        exclude_delivery: set[str] | None = None,
    ) -> BroadcastResult:
        """Plan and execute in one call — the inline path.

        Running to completion where the caller stands trivially satisfies
        per-room order (RFC §10.2). The inbound pipeline instead calls
        :meth:`plan` under the room lock and hands execution to the room's
        delivery lane.

        Args:
            exclude_delivery: Channel IDs to skip delivery for (already
                received content via streaming).
        """
        return await self.execute_plan(
            self.plan(event, source_binding, context, exclude_delivery=exclude_delivery)
        )

    def _filter_targets(
        self,
        event: RoomEvent,
        source_binding: ChannelBinding,
        all_bindings: list[ChannelBinding],
    ) -> list[ChannelBinding]:
        """Filter bindings to find valid delivery targets.

        Muted channels ARE included — they can still receive events via on_event()
        and produce side effects (tasks, observations). Their response events
        are stored BLOCKED by their own commit pass (RFC §7.5 rule 2).
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

            # NOTE: muted channels are NOT skipped here — they receive events,
            # and their response events are stored BLOCKED when they re-enter

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
        """Check if source is visible to target based on visibility rules.

        Uses the event's visibility field which already incorporates the source
        binding's visibility (merged in broadcast() before this is called).
        This allows callers of send_event() to override visibility per-event.

        Delegates the resolution to :func:`effective_visibility` so delivery and
        the history rebuilt for a channel (RFC §7.5 rule 8) answer the same
        question the same way. The stamp makes the two indistinguishable here:
        by the time this runs, a non-default binding scope is already on the
        event.
        """
        return visibility_allows(effective_visibility(event, source_binding), target_binding)

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

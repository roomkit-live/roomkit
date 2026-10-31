"""Event routing with permission enforcement and transcoding."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels.base import Channel
from roomkit.core._changes import for_channels
from roomkit.core._failure_log import log_failure
from roomkit.core._instructions import response_metadata
from roomkit.core._reentrance import answers_past_limit, within_turn
from roomkit.core._room_metadata import without_framework_keys
from roomkit.core._room_turns import RoomTurns
from roomkit.core._turn_runners import TurnRunner
from roomkit.core._voice_delivery import delivering_injected
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.exceptions import KitClosedError, NoRecipientError
from roomkit.core.lanes import DeliveryCascade, DeliveryPlan
from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.core.retry import retry_with_backoff
from roomkit.core.task_utils import cancels_current_task
from roomkit.core.transcoder import DefaultContentTranscoder, takes_file, takes_files_of
from roomkit.core.visibility import answers_reach, committed_visibility, visibility_allows
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    Access,
    AgentResponsePolicy,
    ChannelCategory,
    ChannelMediaType,
    EventDirection,
    EventStatus,
    EventType,
)
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
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
    refuse_too_deep,
    with_pipeline_id,
)
from roomkit.models.response_metadata import TurnEntries, turn_summary
from roomkit.models.task import Observation, Task
from roomkit.providers.utils import _aclose_stream
from roomkit.turns.owner import TurnClient, drop_unread, registry_turn_client

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

# The error of an intelligence target the event solicits whose channel is
# not registered (RFC §10.2 step 3, §8.2): its turn is not taken, and the
# delivery reports it, not retryable.
CHANNEL_NOT_REGISTERED = "channel_not_registered"


def unanswered(trigger: RoomEvent, channel_id: str, channel_type: Any) -> RoomEvent:
    """The record that stands in for an answer a guard kept *channel_id* from giving.

    Empty text from the agent, at the depth, scope and thread the answer to
    *trigger* would have had; the guard that stopped it marks it BLOCKED
    (RFC §8.3).
    """
    return RoomEvent(
        room_id=trigger.room_id,
        # Travelling outbound, as the answer it stands in for would (RFC §8.3).
        source=EventSource(
            channel_id=channel_id, channel_type=channel_type, direction=EventDirection.OUTBOUND
        ),
        content=TextContent(body=""),
        chain_depth=trigger.chain_depth + 1,
        visibility=answers_reach(trigger),
        parent_event_id=trigger.parent_event_id,
        responds_to=trigger.id,
    )


CUT_MARKER = "..."
"""What ends a cut text (RFC §6.6): three ASCII dots, since U+2026 is outside
the GSM 7-bit alphabet and would turn an SMS into UCS-2."""


def _cut_text(text: TextContent, max_length: int) -> TextContent:
    """*text* cut to *max_length* Unicode code points, never inside one: it
    ends with :data:`CUT_MARKER` within the limit, except below four code
    points, where the cut has no marker."""
    if len(text.body) <= max_length:
        return text
    if max_length > len(CUT_MARKER):
        body = text.body[: max_length - len(CUT_MARKER)] + CUT_MARKER
    else:
        body = text.body[:max_length]
    return text.model_copy(update={"body": body})


def _cut_content(content: EventContent, max_length: int) -> EventContent:
    """*content* with its text cut to *max_length*: a text, or each text part
    of a composite (RFC §6.6). The same object when nothing is cut."""
    if isinstance(content, TextContent):
        return _cut_text(content, max_length)
    if isinstance(content, CompositeContent):
        parts = [_cut_content(part, max_length) for part in content.parts]
        if all(new is old for new, old in zip(parts, content.parts, strict=True)):
            return content
        return content.model_copy(update={"parts": parts})
    return content


def _answering(
    responses: list[RoomEvent], trigger: RoomEvent, responder: ChannelBinding
) -> list[RoomEvent]:
    """*responses* as the pipeline commits them (RFC §8.5): each one its
    responder's, travelling outbound, in the trigger's room at the next
    depth, answering *trigger* unless it named the event it answers, under an
    id the pipeline gives (§8.1), with no idempotency key (§13.4), and
    recording the instruction *trigger* is, if it is one (§10.1.1 step 6).
    One typed INSTRUCTION is dropped: it does not re-enter."""
    return [
        _as_response(resp, trigger, responder)
        for resp in responses
        if not _dropped_as_instruction(resp, trigger, responder)
    ]


def _dropped_as_instruction(
    resp: RoomEvent, trigger: RoomEvent, responder: ChannelBinding
) -> bool:
    """Whether *resp* is dropped for being typed INSTRUCTION, which a response
    may not be (RFC §8.5), with a warning: only the application issues
    instructions (§10.1.1), and one agent's answer would otherwise direct
    another in the application's name. An agent that wants another to act
    addresses its answer (§19.3.1)."""
    if resp.type != EventType.INSTRUCTION:
        return False
    logger.warning(
        "Channel %s returned a response typed INSTRUCTION: dropped, since only the "
        "application issues instructions; its other responses re-enter",
        responder.channel_id,
        extra={"room_id": trigger.room_id, "channel_id": responder.channel_id},
    )
    return True


def _as_response(resp: RoomEvent, trigger: RoomEvent, responder: ChannelBinding) -> RoomEvent:
    """One response of *responder* to *trigger*, as :func:`_answering` writes it."""
    # Under its own source a response keeps what its channel wrote, never a
    # verification: that is a received sender's, and a response is no message
    # received (RFC §8.5, §11.7).
    source = resp.source.model_copy(
        update={
            "channel_id": responder.channel_id,
            "channel_type": responder.channel_type,
            "direction": EventDirection.OUTBOUND,
            "verification": None,
        }
    )
    if resp.source.channel_id != responder.channel_id:
        logger.warning(
            "Channel %s returned a response under another source, %s: it is its own",
            responder.channel_id,
            resp.source.channel_id,
            extra={"room_id": trigger.room_id, "channel_id": responder.channel_id},
        )
        # Nothing of the claimed source is kept: not its participant, nor
        # its provider's ids, nor its verification.
        source = EventSource(
            channel_id=responder.channel_id,
            channel_type=responder.channel_type,
            direction=EventDirection.OUTBOUND,
        )
    return with_pipeline_id(
        resp.model_copy(
            update={
                "source": source,
                "room_id": trigger.room_id,
                "chain_depth": trigger.chain_depth + 1,
                "responds_to": resp.responds_to or trigger.id,
                "idempotency_key": None,
                "metadata": response_metadata(resp.metadata, trigger),
            }
        )
    )


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
    # The turn's record (``ChannelOutput.response_metadata``), read by the
    # persistence of each segment as it stands then, never copied here; for a
    # turn behind the turn backend, the owner's copy its frames update.
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
    # Same failures as ``errors`` but the exception object per channel, so a
    # caller (e.g. the inbound pipeline surfacing a non-streaming generation
    # failure on ``InboundResult.error``) can classify it with the cause chain
    # intact — ``errors`` only carries ``str(exc)``. An intelligence turn's is
    # the one the owner rebuilt from the turn's error record (RFC §10.6.8).
    errors_exc: dict[str, BaseException] = field(default_factory=dict)


def _merge_updates(
    result: BroadcastResult, output: ChannelOutput, channel_id: str, room_id: str
) -> None:
    """Merge *output*'s room metadata updates into *result*'s, key by key,
    without a key only the framework writes (RFC §10.1 step 15)."""
    if output.metadata_updates:
        result.metadata_updates.update(
            without_framework_keys(
                output.metadata_updates, room_id=room_id, writer=f"Channel {channel_id}'s output"
            )
        )


@dataclass
class _TargetResult:
    """Per-target result collected during concurrent broadcast."""

    channel_id: str
    output: ChannelOutput | None = None
    delivery_output: ChannelOutput | None = None
    streaming_response: StreamingResponse | None = None
    error: str | None = None
    error_exc: BaseException | None = None
    reentry_events: list[RoomEvent] = field(default_factory=list)
    blocked_events: list[RoomEvent] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


def _merged(event: RoomEvent, target_results: list[_TargetResult]) -> BroadcastResult:
    """The per-target results of *event*'s delivery set as one result, in
    the order the targets answered: on_event()'s room metadata updates, then
    deliver()'s (RFC §10.1 step 15)."""
    result = BroadcastResult()
    for tr in target_results:
        if tr.output is not None:
            result.outputs[tr.channel_id] = tr.output
            result.tasks.extend(tr.output.tasks)
            _merge_updates(result, tr.output, tr.channel_id, event.room_id)
        if tr.delivery_output is not None:
            result.delivery_outputs[tr.channel_id] = tr.delivery_output
            _merge_updates(result, tr.delivery_output, tr.channel_id, event.room_id)
        if tr.streaming_response is not None:
            result.streaming_responses.append(tr.streaming_response)
        if tr.error is not None:
            result.errors[tr.channel_id] = tr.error
        if tr.error_exc is not None:
            result.errors_exc[tr.channel_id] = tr.error_exc
        result.reentry_events.extend(tr.reentry_events)
        result.blocked_events.extend(tr.blocked_events)
        result.observations.extend(tr.observations)
    return result


def _cut_sends(
    plan: DeliveryPlan, target_results: list[_TargetResult], channels: Mapping[str, Channel]
) -> list[_TargetResult]:
    """The sends of *plan*'s set still under way when the kit's close cut it:
    each a failure, ``kit_closed``, retryable (RFC §12.10.4)."""
    answered = {tr.channel_id for tr in target_results}
    excluded = plan.exclude_delivery or set()
    return [
        _TargetResult(
            channel_id=binding.channel_id,
            error=KitClosedError.code,
            error_exc=KitClosedError("The kit closed while this delivery was being sent"),
        )
        for binding in plan.targets
        if binding.category == ChannelCategory.TRANSPORT
        and binding.sends
        and binding.channel_id in channels
        and binding.channel_id not in answered
        and binding.channel_id not in excluded
    ]


def _leaves_no_result(exc: BaseException, binding: ChannelBinding) -> bool:
    """Whether *exc* ends *binding*'s target without a result: a cancellation
    aimed at the target's own task (its caller gave up, the kit's close cut
    it), or one out of an agent's reaction, its turn's cut (an agent's own
    ``CancelledError`` reaches here as its turn's failure). A transport's
    own ``CancelledError`` is its failure, as any other exception it raises
    (RFC §10.2 step 3)."""
    if not isinstance(exc, asyncio.CancelledError):
        return False
    return cancels_current_task(exc) or binding.category == ChannelCategory.INTELLIGENCE


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
        turns: TurnClient | None = None,
        runner_of: Callable[[str | None, str], TurnRunner | None] | None = None,
        context_of: Callable[..., Awaitable[RoomContext]] | None = None,
    ) -> None:
        self._channels = channels
        self._transcoder = transcoder or DefaultContentTranscoder()
        self._max_chain_depth = max_chain_depth
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._telemetry = telemetry
        self._greeting_gate_fn = greeting_gate_fn
        # Every intelligence turn runs through a turn backend (RFC §10.6); a
        # router built without the kit's runs its own over its channels.
        self._turns = turns or registry_turn_client(channels)
        # A strategy's runner takes an agent's turns on the owner (RFC §10.6.1).
        self._runner_of = runner_of
        # The lines of the channels that take a room's turns one at a time
        # (RFC §6.4), which read the room as it is when they start.
        self._room_turns = RoomTurns(self.react, context_of, telemetry)
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
            # A committed event's visibility is already its own intersected
            # with its source binding's (RFC §7.3); intersecting again with
            # the binding the plan reads changes nothing for it, and holds
            # the rule for an event a caller broadcasts without a commit.
            stamped = committed_visibility(event.visibility, source_binding, context.bindings)
            if stamped != event.visibility:
                event = event.model_copy(update={"visibility": stamped})
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

    async def execute_plan(
        self, plan: DeliveryPlan, *, cascade: DeliveryCascade | None = None
    ) -> BroadcastResult:
        """Execute a plan's delivery set (RFC §10.1 step 14), its intelligence
        targets' turns as *cascade*'s turn work when it has one.

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
        # What every target is handed of the room's history: a deleted event
        # with its content withheld (RFC §10.3); the plan's own context, which
        # the hooks receive after delivery, stays whole.
        context = for_channels(plan.context)
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

        # A plan with no source binding has nothing to transcode from.
        if not targets or source_binding is None:
            reset_span(broadcast_token)
            if parent_token is not None:
                reset_span(parent_token)
            telemetry.end_span(span_id, attributes={"target_count": 0})
            return result

        # Collect per-target results to avoid concurrent mutation
        target_results: list[_TargetResult] = []
        # What the set has done so far, read when the kit's close cuts it:
        # each target delivered keeps its result, a send under way is failed,
        # even one that ignores its cut (RFC §12.10.4).
        plan.partial_result = lambda: _merged(
            event, target_results + _cut_sends(plan, target_results, self._channels)
        )

        async def _process_target(binding: ChannelBinding) -> None:
            if binding.category == ChannelCategory.INTELLIGENCE:
                unasked = self._unasked_result(event, source_binding, binding, context, plan)
                if unasked is not None:
                    target_results.append(unasked)
                    return

            channel = self._channels.get(binding.channel_id)
            if channel is None and binding.category == ChannelCategory.TRANSPORT:
                # A transport reaches a client from the process the client
                # connected to: another process, or none while nobody is
                # connected. Its binding outlives both, so a transport absent
                # here is the normal state, not a fault.
                logger.debug(
                    "Transport channel %s not registered in this process, skipping",
                    binding.channel_id,
                )
                return
            if channel is None:
                # An agent the event solicits (an unsolicited one returned
                # above): its turn is owed and never taken, which the
                # event's delivery reports (RFC §10.2 step 3).
                logger.warning(
                    "Intelligence channel %s is not registered; its turn is not taken",
                    binding.channel_id,
                    extra={
                        "room_id": event.room_id,
                        "channel_id": binding.channel_id,
                        "event_id": event.id,
                    },
                )
                target_results.append(
                    _TargetResult(channel_id=binding.channel_id, error=CHANNEL_NOT_REGISTERED)
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

                # on_event — all channels react; an intelligence channel's
                # call is a turn, run through the turn backend (RFC §10.6).
                output = await self.react_for(cascade, channel, transcoded_event, binding, context)
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
                        # As its hooks left it, not as transcoded for this
                        # target: its rows record that instruction (RFC
                        # §10.1.1 step 6).
                        trigger_event=event,
                        response_metadata=output.response_metadata,
                    )
                    tr.observations.extend(output.observations)
                    target_results.append(tr)
                    return

                # deliver — only a transport that sends pushes to external;
                # an INBOUND-only one is told through on_event() alone (RFC
                # §10.2 step 3d).
                if binding.category == ChannelCategory.TRANSPORT and binding.sends:
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
                        except (Exception, asyncio.CancelledError) as exc:
                            if _leaves_no_result(exc, binding):
                                raise
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
                    output.response_events = _answering(output.response_events, event, binding)
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

            except (Exception, asyncio.CancelledError) as exc:
                if _leaves_no_result(exc, binding):
                    raise
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

        try:
            # An injected event asks no agent (RFC §9.4): the targets' tasks
            # inherit the scope, which a host of a model reads.
            with delivering_injected() if plan.injected else nullcontext():
                await asyncio.gather(
                    *[_process_target(t) for t in targets], return_exceptions=True
                )
        except asyncio.CancelledError:
            # The caller gave up: the streams the targets already returned
            # will not be read, and are closed rather than left open.
            for tr in target_results:
                if tr.streaming_response is not None:
                    drop_unread(tr.streaming_response.stream)
            raise

        # Merge per-target results into BroadcastResult (single-threaded), in
        # the order the targets answered: on_event()'s room metadata updates,
        # then deliver()'s (RFC §10.1 step 15).
        result = _merged(event, target_results)

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

    async def react_for(
        self,
        cascade: DeliveryCascade | None,
        channel: Channel,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """*channel*'s reaction to an event *cascade* delivers: an intelligence
        channel's turn runs as the cascade's turn work, and answers nothing
        once the cascade is cancelled, before it starts or while it runs
        (RFC §10.1). A channel that takes a room's turns one at a time has
        its turn wait its place in the stream it answers with (RFC §6.4)."""
        reaction = (
            self._room_turns.ask(channel, event, binding, context)
            if self._serializes(channel, binding, context)
            else self.react(channel, event, binding, context)
        )
        if cascade is None or binding.category != ChannelCategory.INTELLIGENCE:
            return await reaction
        try:
            return await cascade.run(reaction, turn_work=True)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if cascade.cancelled is None or (current is not None and current.cancelling()):
                raise
            return ChannelOutput.empty()

    def _serializes(self, channel: Channel, binding: ChannelBinding, context: RoomContext) -> bool:
        """Whether *channel*'s turn waits its place in the room: an
        intelligence channel that declares it, unless a strategy takes its
        turns, which orders them itself (RFC §6.4, §10.6.1). A muted one
        reacts at once, as any muted channel does, its answer closed unread
        (RFC §7.2)."""
        if binding.category != ChannelCategory.INTELLIGENCE or not channel.serializes_room_turns:
            return False
        if binding.muted:
            return False
        runner_of = self._runner_of
        return runner_of is None or runner_of(context.room.id, channel.channel_id) is None

    async def react(
        self,
        channel: Channel,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """*channel*'s ``on_event``, an intelligence channel's as a turn
        through the turn backend (RFC §10.6.1).

        Raises:
            ValueError: a response it returned holds more than five composite
                levels (RFC §5.3): the reaction failed, nothing of it kept.
        """
        output = await self._reaction(channel, event, binding, context)
        for response in output.response_events:
            refuse_too_deep(response.content)
        return output

    async def _reaction(
        self,
        channel: Channel,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        if binding.category != ChannelCategory.INTELLIGENCE:
            return await channel.on_event(event, binding, context)
        room_id = context.room.id if context.room else event.room_id
        runner = self._runner_of(room_id, binding.channel_id) if self._runner_of else None
        # What the turn sends into a room is at least as deep as its response
        # (RFC §8.3, §10.6): an agent injecting at each turn meets the limit.
        with within_turn(event.chain_depth + 1):
            if runner is not None:
                # The strategy takes the turn here, on the owner; each
                # generation it asks of the agent is a turn of its own.
                return await runner(event, binding, context)
            return await self._turns.run(event, binding, context)

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
            if answers_past_limit(event.chain_depth, limit):
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
        if answers_past_limit(event.chain_depth, self._max_chain_depth):
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

        The event's visibility field is the whole answer: its own intersected
        with its source binding's at its commit (RFC §7.3), which the history
        rebuilt for a channel reads too (§7.5 rule 8).
        """
        return visibility_allows(event.visibility, target_binding)

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

        # Check if the specific content type is already supported; a file
        # also needs a type and a size the target takes (RFC §5.8).
        content_media = self._content_media_type(event.content)
        if content_media is not None and content_media in target_types:
            content = event.content
            if not isinstance(content, MediaContent) or takes_file(
                target_binding.capabilities, content
            ):
                return event

        # Edit/Delete: check capability flags directly; a native edit's new
        # content also needs files the target takes (RFC §5.8).
        if (
            isinstance(event.content, EditContent)
            and target_binding.capabilities.supports_edit
            and takes_files_of(target_binding.capabilities, event.content.new_content)
        ):
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
        """Cut the event's text to the channel's ``max_length`` (RFC §6.6):
        a text, or each text part of a composite the target keeps whole."""
        cut = _cut_content(event.content, max_length)
        return event if cut is event.content else event.model_copy(update={"content": cut})

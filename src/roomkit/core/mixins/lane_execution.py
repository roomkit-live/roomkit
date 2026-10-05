"""LaneExecutionMixin — the framework side of the delivery lanes.

The engine (:mod:`roomkit.core.lanes`) owns ordering; this mixin owns what a
plan *means*: committing an event together with its plan (the third commit
gate next to ``_persist_committed`` / ``_commit_indexed``), executing a
plan's delivery set, firing the per-event aftermath, and turning response
events into fresh commit passes (RFC §10.1 step 14 — reentry passes take the
room lock anew; they are never drained inside the trigger's lock tenure).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.event_router import CHAIN_DEPTH_LIMIT
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.core.mixins.helpers import (
    _RECENT_EVENTS_LIMIT,
    HelpersMixin,
    _refuses_writes,
)
from roomkit.core.mixins.inbound_locked import _Blocked, _Ready
from roomkit.models.delivery import DeliveryError, DeliveryResult
from roomkit.models.enums import ChannelCategory, EventStatus, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.response_metadata import (
    ResponseMetadata,
    add_turn_entry,
    merge_caller_record,
    merge_channel_record,
    turn_summary,
)
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.context import get_current_span, restored_span

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.channels.base import Channel
    from roomkit.core.event_router import BroadcastResult, EventRouter
    from roomkit.core.hooks import HookEngine, SyncPipelineResult
    from roomkit.core.lanes import DeliveryCascade, DeliveryPlan, RoomLaneRegistry
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.hook import InjectedEvent
    from roomkit.store.base import ConversationStore
    from roomkit.telemetry.base import TelemetryProvider

logger = logging.getLogger("roomkit.framework")


def _delivery_results(result: BroadcastResult) -> dict[str, DeliveryResult]:
    """Per-channel outcomes of one delivery set (RFC §5.13, §10.1 step 18).

    ``errors_exc`` carries the live exception, so the report says what the
    retry loop would decide rather than re-deriving it from a message string.
    """
    results: dict[str, DeliveryResult] = {}
    for channel_id, output in result.delivery_outputs.items():
        provider_result = output.provider_result
        results[channel_id] = DeliveryResult(
            channel_id=channel_id,
            status="sent",
            provider_message_id=(
                provider_result.provider_message_id if provider_result is not None else None
            ),
            provider_result=provider_result,
        )
    for channel_id, message in result.errors.items():
        exc = result.errors_exc.get(channel_id)
        provider_result = getattr(exc, "provider_result", None)
        results[channel_id] = DeliveryResult(
            channel_id=channel_id,
            status="failed",
            provider_message_id=(
                provider_result.provider_message_id if provider_result is not None else None
            ),
            error=DeliveryError(
                code=str(getattr(exc, "code", None) or type(exc).__name__)
                if exc is not None
                else "DeliveryFailed",
                message=message,
                retryable=bool(getattr(exc, "retryable", True)),
            ),
            provider_result=provider_result,
        )
    return results


def scoped(response: RoomEvent, scope: str | None) -> RoomEvent:
    """*response* within the answer scope its trigger asked for, if any.

    The router's visibility check reads ``visibility``, so the caller's
    response scope rides the response there; it rides as
    ``response_visibility`` too, so whatever answers the response, buffered
    or streamed, is scoped alike down the chain.
    """
    if not scope:
        return response
    return response.model_copy(update={"visibility": scope, "response_visibility": scope})


@dataclass(slots=True, frozen=True)
class DeliverySource:
    """Planning inputs shared by a run of events from one sender.

    Resolving them per event costs a room lock and a context read every
    time. A stream's segments share one sender and one delivery set, so the
    run resolves them once and hands the result to each commit — which is
    what the single batch broadcast they replaced did, against bindings of
    exactly the same freshness.
    """

    binding: ChannelBinding
    context: RoomContext

    @classmethod
    def of(cls, channel_id: str, context: RoomContext) -> DeliverySource | str:
        """The planning inputs of *channel_id*'s run in *context*: its
        binding there, or its id alone when the context holds none."""
        binding = next((b for b in context.bindings if b.channel_id == channel_id), None)
        return cls(binding=binding, context=context) if binding is not None else channel_id


def _delivery_source(context: RoomContext, channel_id: str) -> DeliverySource | None:
    """Who sends and against what state, read off *context*.

    ``None`` when the sender has no binding: there is nothing to broadcast
    from, and the event commits as a plain cursor entry.
    """
    binding = context.get_binding(channel_id)
    return DeliverySource(binding=binding, context=context) if binding is not None else None


def _log_refused(room_id: str, event: RoomEvent) -> None:
    """Log that *event* was refused by the room's status gate (RFC §5.1)."""
    logger.debug("Room %s refuses writes; %s not committed", room_id, event.type.value)


@runtime_checkable
class LaneExecutionHost(Protocol):
    """Contract: capabilities a host class must provide for LaneExecutionMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store.
        _channels: Channel registry.
        _hook_engine: Hook engine for AFTER_BROADCAST / mutation / ON_ERROR.
        _lock_manager: Room lock, taken fresh by each reentry pass.
        _lanes: The delivery-lane registry.
        _max_chain_depth: Chain depth ceiling (RFC §8.3).
        _telemetry: Telemetry / tracing provider.

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter``.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager
    _lanes: RoomLaneRegistry
    _max_chain_depth: int
    _telemetry: TelemetryProvider

    def _get_router(self) -> EventRouter: ...


class LaneExecutionMixin(HelpersMixin):
    """Commit-with-plan gate and lane-executor callbacks.

    Host contract: :class:`LaneExecutionHost`.
    """

    _lock_manager: RoomLockManager
    _max_chain_depth: int
    _telemetry: TelemetryProvider

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # RoomKit._get_router
    _handle_block: Any  # InboundLockedMixin
    _gate_commit: Any  # InboundLockedMixin
    _store_blocked: Any  # InboundLockedMixin
    _announce_blocked: Any  # InboundLockedMixin
    _process_streaming_responses: Any  # InboundStreamingMixin

    # -- The planned commit gate --

    async def _commit_to_lane(
        self,
        room_id: str,
        event: RoomEvent,
        cascade: DeliveryCascade,
        plan_factory: Callable[[RoomEvent], DeliveryPlan | None] | None,
        *,
        policy_aware: bool = True,
    ) -> RoomEvent | None:
        """Commit an event and enqueue its delivery plan in one gate.

        ``plan_factory`` receives the committed event (authoritative index)
        and returns its plan — or ``None`` when there is nothing to deliver,
        which reduces the commit to a cursor entry. An event the persistence
        policy excludes is delivered but not stored: its plan joins the lane
        index-less, anchored behind the room's ``latest_index``, and moves no
        cursor (RFC §14.3 — an unstored event consumes no index).

        Runs under the room lock (planning reads binding state consistent
        with the committed timeline, RFC §10.1 step 12).
        """
        # An instruction takes the same index-less path whatever the policy:
        # it is delivered to the agent it directs and never stored (RFC
        # §10.1.1).
        if event.type == EventType.INSTRUCTION or (
            policy_aware
            and self._persistence_policy is not None
            and not self._persistence_policy.should_persist(event.type)
        ):
            if plan_factory is not None:
                plan = plan_factory(event)
                if plan is not None:
                    room = await self._store.get_room(room_id)
                    anchor = -1 if room is None else room.latest_index
                    cascade.retain()
                    self._enqueue_exec(room_id, plan, cascade, index=None, after_index=anchor)
            return None
        committed = await self._store.commit_event(room_id, event)
        plan = plan_factory(committed) if plan_factory is not None else None
        if plan is None:
            await self._note_committed_index(room_id, committed.index)
            return committed
        cascade.retain()
        self._enqueue_exec(room_id, plan, cascade, index=committed.index)
        return committed

    async def _commit_and_deliver(
        self,
        room_id: str,
        event: RoomEvent,
        source: str | DeliverySource,
        *,
        exclude_delivery: set[str] | None = None,
        policy_aware: bool = True,
        cascade: DeliveryCascade | None = None,
        hook_result: SyncPipelineResult | None = None,
        gate_status: bool = True,
    ) -> RoomEvent | None:
        """Commit an event and hand its delivery to the room's lane.

        The one entry point for a caller that owns an event end to end (a
        greeting, a streamed segment) rather than committing it and then
        broadcasting it where it stands. Inline
        execution satisfies per-room order only while nothing else can
        deliver for the room (RFC §10.2), and committing publishes the
        index on the delivery cursor, which is precisely what releases the
        lane to execute the *next* one. Going through the lane keeps the
        single ordering authority: the cursor advances after this event's
        delivery set has run, never before it.

        ``source`` is the sending channel's id, which costs a room lock and
        a context read to resolve — right for a one-off event, wrong for a
        run of them. A caller emitting several (a stream's segments) resolves
        a :class:`DeliverySource` once and passes that instead: the commit
        then takes no lock and reads nothing, and the store still assigns the
        index atomically (RFC §8.1), which is what this path always relied on.

        The event is an agent's output like any other (RFC §19.3.1): what
        the other agents answer to it re-enters, and a stream one of them
        starts is read (§8.3), never discarded.

        Pass ``cascade`` to enqueue without waiting — for a caller emitting
        a run of events that must not block on each one's delivery; that
        caller owns the single wait at the end, and the reading of the
        streams the run started.

        ``hook_result`` carries effects from an already allowed sync hook
        pipeline. Tasks and observations join the plan's post-delivery work;
        injected events are committed after their triggering event.

        The room's status gate (RFC §5.1) reads the room before the commit.
        A run that gates its rows itself (a stream's :class:`LaneSink`, which
        reads the status once and again after a close) passes
        ``gate_status=False``. With a resolved ``source`` the commit then
        takes no lock and reads nothing, but for the anchor of a row the
        persistence policy excludes. A ``str`` source is resolved per event,
        from one read of the room's context under its lock, which the status
        gate and the sender's binding read too (RFC §10.1 steps 6 and 12).

        Returns the committed event, or ``None`` when the persistence
        policy excluded it (delivered, unstored — RFC §14.3).
        """
        from roomkit.core.lanes import DeliveryCascade

        own_cascade = cascade is None
        if cascade is None:
            cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)

        if isinstance(source, str):
            async with self._lock_manager.locked(room_id):
                context = await self._existing_room_context(room_id)
                if context is None or (gate_status and _refuses_writes(context.room)):
                    _log_refused(room_id, event)
                    return None
                resolved = _delivery_source(context, source)
                committed = await self._commit_to_lane(
                    room_id,
                    event,
                    cascade,
                    self._plan_factory(resolved, exclude_delivery, hook_result),
                    policy_aware=policy_aware,
                )
        else:
            # The status gate holds at every point the timeline grows (RFC §5.1).
            if gate_status and await self._room_refuses_writes(room_id):
                _log_refused(room_id, event)
                return None
            resolved = source
            committed = await self._commit_to_lane(
                room_id,
                event,
                cascade,
                self._plan_factory(resolved, exclude_delivery, hook_result),
                policy_aware=policy_aware,
            )

        if hook_result is not None:
            context = (
                resolved.context if resolved is not None else await self._build_context(room_id)
            )
            if resolved is None:
                # A detached source has no delivery plan to collect its effects.
                await self._persist_side_effects(
                    room_id,
                    hook_result.tasks,
                    hook_result.observations,
                    committed or event,
                    context,
                )
            if hook_result.injected_events:
                await self._lane_injected_events(
                    hook_result.injected_events, room_id, context, cascade
                )

        # Off the lock: waiting under it would deadlock the lane against its
        # own caller, and ``wait()`` short-circuits rather than hang.
        if own_cascade:
            await self._finish_cascade(cascade, room_id)
        return committed

    async def _existing_room_context(self, room_id: str) -> RoomContext | None:
        """The room's context, or ``None`` when the room is gone. Call under
        the room lock, so the context is what the lock protects."""
        try:
            return await self._build_context(room_id)
        except RoomNotFoundError:
            return None

    def _plan_factory(
        self,
        source: DeliverySource | None,
        exclude_delivery: set[str] | None,
        hook_result: SyncPipelineResult | None = None,
    ) -> Callable[[RoomEvent], DeliveryPlan] | None:
        """The plan builder ``_commit_to_lane`` calls on the committed event.

        ``None`` in, ``None`` out — the commit reduces to a cursor entry.
        The event's ``response_visibility`` scopes what re-enters from its
        delivery set, as the root plan's does (a streamed segment carries
        its trigger's).
        """
        if source is None:
            return None
        router = self._get_router()

        def factory(committed: RoomEvent) -> DeliveryPlan:
            plan = router.plan(
                committed,
                source.binding,
                source.context.model_copy(
                    update={
                        "recent_events": [
                            *source.context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                            committed,
                        ]
                    }
                ),
                exclude_delivery=exclude_delivery,
            )
            plan.response_visibility = committed.response_visibility
            if hook_result is not None:
                plan.hook_tasks = list(hook_result.tasks)
                plan.hook_observations = list(hook_result.observations)
            return plan

        return factory

    def _enqueue_exec(
        self,
        room_id: str,
        plan: DeliveryPlan,
        cascade: DeliveryCascade,
        *,
        index: int | None,
        after_index: int = -1,
    ) -> None:
        from roomkit.core.lanes import ExecEntry

        self._lanes.enqueue(
            room_id,
            ExecEntry(plan=plan, cascade=cascade, index=index, after_index=after_index),
        )

    async def _lane_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: Any,
        cascade: DeliveryCascade,
    ) -> None:
        """Commit hook-injected events and lane their delivery (RFC §9.5).

        The commit happens here, under the room lock and policy-exempt (an
        injected event is a real DELIVERED timeline event); the channel I/O
        rides the lane like everything else. Targets are resolved now, so
        the delivery set is consistent with the state the hook saw.
        """
        from roomkit.core.lanes import DeliveryPlan

        for injected in injected_events:
            event = injected.event.model_copy(update={"status": EventStatus.DELIVERED})
            target_ids = injected.target_channel_ids

            # Bindings are resolved under the lock so the delivery set is
            # consistent with the state the injecting hook saw; the factory
            # itself stays synchronous.
            targets: list[Any] = []
            if target_ids is not None:
                for target_id in target_ids:
                    binding = await self._store.get_binding(room_id, target_id)
                    if binding is not None:
                        targets.append(binding)

            if not targets:
                # No target specified (stored only) or none resolvable.
                await self._commit_to_lane(room_id, event, cascade, None, policy_aware=False)
                continue

            def factory(committed: RoomEvent, _targets: list[Any] = targets) -> DeliveryPlan:
                return DeliveryPlan(
                    event=committed,
                    source_binding=None,
                    context=context,
                    targets=_targets,
                    injected=True,
                    fire_after_broadcast=False,
                )

            await self._commit_to_lane(room_id, event, cascade, factory, policy_aware=False)

    def _consume_streams_when_cascade_completes(
        self, cascade: DeliveryCascade, room_id: str
    ) -> asyncio.Task[None]:
        """Arrange stream consumption for a detached caller.

        A ``send_event`` issued from inside a sync hook (under the room
        lock) or from a tool handler (inside the lane) cannot wait on its
        cascade — but a streaming provider's reply is only generated when
        its stream is consumed. This schedules the consumption on a clean
        background task (fresh context: an inherited ``_held_rooms`` would
        fake lock reentrancy), tracked like every fire-and-forget hook task.

        Returns the consumer task: its completion is when the caller's whole
        turn is over — cascade AND streamed responses — which is what a
        deferred ``process_inbound``'s :class:`DeliveryHandle` waits on. A
        consumption failure is recorded on the cascade so that handle can
        surface it; the fire-and-forget callers never look, as before.

        Trace continuity is explicit, never inherited (the same rule as
        ``DeliveryPlan.parent_span_id``): the caller's span is captured here
        and restored inside the task, so the streamed segments keep the
        parent the waiting path gives them. A ``framework.detached`` span,
        child of the caller's span and opened at the detachment instant,
        measures the tail — what a deferred call's caller did not wait for.
        It is not made current: it measures without re-parenting, so a
        streamed reply and a non-streaming one sit at the same depth.
        """
        telemetry = self._telemetry
        parent_id = get_current_span()
        parent_ctx = telemetry.get_span_context(parent_id) if parent_id is not None else None
        tail_span = telemetry.start_span(
            SpanKind.INBOUND_PIPELINE,
            "framework.detached",
            parent_id=parent_id,
            room_id=room_id,
        )

        async def _consume() -> None:
            with restored_span(parent_id, telemetry_ctx=parent_ctx):
                await cascade.wait_detached()
                if not cascade.streams or cascade.cancelled is not None:
                    return
                # An escape here would otherwise die un-retrieved on this
                # task while the waiting path propagates the same failure
                # to its caller — record it so a DeliveryHandle surfaces it.
                try:
                    stream_error, record = await self._process_streaming_responses(
                        cascade, room_id, response_events=cascade.response_events
                    )
                except Exception as exc:
                    logger.exception("Detached stream consumption failed for room %s", room_id)
                    cascade.record_error(exc)
                else:
                    merge_caller_record(cascade.response_metadata, record)
                    if stream_error is not None:
                        cascade.record_error(stream_error)

        def _end_tail(done: asyncio.Task[None]) -> None:
            # On the task's completion rather than inside it: close() cancels
            # the consumer before the telemetry provider closes, and a task
            # cancelled before its first step never runs a single line of
            # its coroutine — a span ended from within would stay open.
            if done.cancelled():
                status, message = "error", "cancelled"
            elif (exc := done.exception()) is not None:
                # Nothing in _consume escapes today; if something ever does,
                # retrieving it here silences asyncio's own warning, so the
                # failure must stay loud and reach the handle's waiter.
                logger.error("Detached tail failed for room %s", room_id, exc_info=exc)
                if isinstance(exc, Exception):
                    cascade.record_error(exc)
                status, message = "error", str(exc)
            elif cascade.error is not None:
                status, message = "error", str(cascade.error)
            elif cascade.cancelled is not None:
                status, message = "error", cascade.cancelled
            else:
                status, message = "ok", None
            telemetry.end_span(
                tail_span,
                status=status,
                error_message=message,
                attributes={"streams": len(cascade.streams)},
            )

        task = asyncio.get_running_loop().create_task(
            _consume(),
            name=f"roomkit-detached-streams-{room_id}",
            context=contextvars.Context(),
        )
        cascade.track(task)
        task.add_done_callback(_end_tail)
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)
        return task

    async def _finish_cascade(
        self,
        cascade: DeliveryCascade,
        room_id: str,
        *,
        caller_logs: bool = False,
        streamed_too: bool = False,
    ) -> tuple[Exception | None, ResponseMetadata]:
        """Wait for a caller's delivery set, then read every stream it started.

        A caller that cannot wait, from inside the room's lane or under its
        lock (a reentrant call from a hook or a tool handler), hands the
        reading to a background task instead: a streaming reply is only
        generated when its stream is read. The rows the streams write join
        ``cascade.response_events``, beside the answers that re-enter while
        they are read. The reading runs as the cascade's owned work, so a
        cancelled caller does not leave it half-done.

        ``caller_logs`` says the caller receives a stream's failure (on
        ``InboundResult.error``, or raised) and logs it: the framework's own
        line for a stream with no streaming target then drops to DEBUG. A
        caller that does not wait never receives it, so a background read logs
        at the failure's own level. A stream a streaming target rendered keeps
        its line (RMK-403) unless ``streamed_too``: a caller that logs every
        failure of its turn itself, a delegation whose task logs it.
        """
        completed = await cascade.wait()
        if cascade.cancelled is not None:
            await cascade.wait_drained()
        if not completed:
            self._consume_streams_when_cascade_completes(cascade, room_id)
        elif cascade.streams and cascade.cancelled is None:
            cascade.caller_logs = caller_logs
            cascade.caller_logs_streamed = caller_logs and streamed_too
            return await cascade.run(
                self._process_streaming_responses(
                    cascade, room_id, response_events=cascade.response_events
                )
            )
        return None, ResponseMetadata()

    # -- Lane executor callbacks (LaneHost) --

    async def _execute_plan(self, plan: DeliveryPlan) -> BroadcastResult:
        """Execute one plan's delivery set (called by the lane executor)."""
        if plan.injected:
            return await self._execute_injected_plan(plan)
        return await self._get_router().execute_plan(plan)

    async def _execute_injected_plan(self, plan: DeliveryPlan) -> BroadcastResult:
        """Deliver an injected event to its named channels.

        Deliberately bare — direct ``on_event`` and, for transports,
        ``deliver``; no transcoding, no response collection. An injected
        event must not be able to trigger an AI reply.
        """
        from roomkit.core.event_router import BroadcastResult

        for binding in plan.targets:
            channel = self._channels.get(binding.channel_id)
            if channel is None:
                continue
            try:
                await channel.on_event(plan.event, binding, plan.context)
                if binding.category == ChannelCategory.TRANSPORT:
                    await channel.deliver(plan.event, binding, plan.context)
            except Exception:
                logger.exception(
                    "Failed to deliver injected event to %s",
                    binding.channel_id,
                    extra={"room_id": plan.event.room_id, "channel_id": binding.channel_id},
                )
        return BroadcastResult()

    async def _record_failed_deliveries(
        self, event: RoomEvent, results: dict[str, DeliveryResult]
    ) -> None:
        """Persist the outcomes onto the event, but only when one of them failed.

        A delivery set that all succeeded needs no record: its absence *is* the
        record, and paying an UPDATE per event to write "everything worked"
        costs the whole message volume to answer a question nobody asks. A set
        with a failure in it is the one an operator comes back to hours later —
        "which channels did this never reach?" — and the live
        ``delivery_failed`` framework event has long since gone.

        The whole map is written, successes included, so the answer is complete
        rather than a list of casualties with no denominator.
        """
        if not any(r.status == "failed" for r in results.values()):
            return
        try:
            await self._store.update_event(
                event.model_copy(
                    update={
                        "delivery_results": {
                            channel_id: r.model_dump(mode="json")
                            for channel_id, r in results.items()
                        }
                    }
                )
            )
        except Exception:
            logger.warning(
                "Could not record delivery results for event %s",
                event.id,
                exc_info=True,
                extra={"room_id": event.room_id, "event_id": event.id},
            )

    async def _post_plan_effects(
        self, plan: DeliveryPlan, result: BroadcastResult, cascade: DeliveryCascade
    ) -> None:
        """Per-event aftermath, off the claim and off the room lock.

        Fires the RFC §10.3 mutation trigger first (observers see the
        mutation before the edit/delete event's own AFTER_BROADCAST), then —
        on the root pass — delivery reporting and the intelligence ON_ERROR
        funnel, then blocked-event commits, side effects and AFTER_BROADCAST
        (RFC §10.1 step 16: after the event's delivery set completes).
        """
        event = plan.event
        context = plan.context
        room_id = event.room_id

        if plan.mutation_hook is not None:
            trigger, target = plan.mutation_hook
            await self._hook_engine.run_async_hooks(room_id, trigger, target, context)

        if plan.injected:
            return

        if plan.emit_processed:
            # Root pass only: delivery tracking, partial-failure
            # reporting and the caller-facing error all describe the
            # trigger's own delivery set, never a reentry's.
            if result.errors:
                total = len(result.delivery_outputs) + len(result.errors)
                logger.warning(
                    "Partial broadcast failure: %d/%d channels failed",
                    len(result.errors),
                    total,
                    extra={
                        "room_id": room_id,
                        "event_id": event.id,
                        "failed_channels": list(result.errors.keys()),
                    },
                )
                await self._emit_framework_event(
                    "broadcast_partial_failure",
                    room_id=room_id,
                    event_id=event.id,
                    data={
                        "failed": len(result.errors),
                        "total": total,
                        "errors": result.errors,
                    },
                )
            for ch_id in result.delivery_outputs:
                await self._emit_framework_event(
                    "delivery_succeeded", room_id=room_id, event_id=event.id, channel_id=ch_id
                )
            for ch_id, error_msg in result.errors.items():
                await self._emit_framework_event(
                    "delivery_failed",
                    room_id=room_id,
                    event_id=event.id,
                    channel_id=ch_id,
                    data={"error": error_msg},
                )
            cascade.delivery_results = _delivery_results(result)
            intelligence = {
                target.channel_id
                for target in plan.targets
                if target.category == ChannelCategory.INTELLIGENCE
            }
            reached = result.outputs.keys() | result.errors.keys()
            cascade.unavailable_targets = [
                target
                for target in event.addressed_to or []
                if target not in intelligence or target not in reached
            ]
            await self._record_failed_deliveries(event, cascade.delivery_results)
            await self._settle_buffered_replies(
                cascade, event, context, result, root=plan.emit_processed
            )

        # A stream any pass started is read by the caller (RFC §8.3); one a
        # reentry pass or a streamed segment's delivery started answers an
        # answer, and is chained.
        cascade.add_streams(result.streaming_responses, chained=not plan.emit_processed)

        await self._commit_blocked_events(room_id, result)

        if plan.fire_after_broadcast:
            await self._persist_side_effects(
                room_id,
                plan.hook_tasks + result.tasks,
                plan.hook_observations + result.observations,
                event,
                context,
            )
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.AFTER_BROADCAST, event, context
            )

        if plan.emit_processed:
            await self._emit_framework_event("event_processed", room_id=room_id, event_id=event.id)

    async def _settle_buffered_replies(
        self,
        cascade: DeliveryCascade,
        event: RoomEvent,
        context: RoomContext,
        result: BroadcastResult,
        *,
        root: bool,
    ) -> None:
        """What a delivery set's buffered replies leave their caller (RFC
        §10.1 step 18), on every path that waits for one (the lane, a
        regeneration): each failure to ON_ERROR, the first as the cascade's
        error, before any stream's; each reply's end under ``turns``.

        A non-streaming channel has finished generation before the delivery
        set returns, so its record is final now. Streaming records stay live
        until their generators are consumed and are merged there instead:
        copying them here would freeze late tool writes.
        """
        await self._report_intelligence_errors(event, context, result)
        first_error = self._first_intelligence_error(result, context)
        if first_error is not None:
            cascade.record_error(first_error)
        for channel_id, output in result.outputs.items():
            if output.response_stream is None:
                record_buffered_reply(cascade, channel_id, output, root=root)

    async def _report_intelligence_errors(
        self, event: RoomEvent, context: RoomContext, result: BroadcastResult
    ) -> None:
        """Surface intelligence-channel failures to ON_ERROR so hosts can
        render an error card (transport delivery failures are not turn-level
        agent errors). Fired off the room lock."""
        for binding in context.bindings:
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            error_msg = result.errors.get(binding.channel_id)
            if not error_msg:
                continue
            exc = result.errors_exc.get(binding.channel_id)
            await self._fire_error_hook(
                event.room_id,
                context,
                EventSource(
                    channel_id=binding.channel_id,
                    channel_type=binding.channel_type,
                ),
                error=error_msg,
                # As the streaming path names it: the exception's type.
                error_type=type(exc).__name__ if exc is not None else "unknown",
                error_category="generation",
                chain_depth=event.chain_depth + 1,
                visibility=event.response_visibility or "all",
                parent_event_id=event.parent_event_id,
            )

    async def _commit_blocked_events(self, room_id: str, result: BroadcastResult) -> None:
        """Commit the records a delivery set blocked (RFC §8.1, §8.3, §14.3).

        Blocked events are still indexed: an agent not asked past the depth
        limit and a muted source's response (§7.5 rule 2) both land here,
        whichever path broadcast the trigger. A room that refuses writes takes
        neither (§5.1).
        """
        if result.blocked_events and not await self._room_refuses_writes(room_id):
            for blocked in result.blocked_events:
                await self._commit_blocked_response(room_id, blocked)

    async def _commit_blocked_response(self, room_id: str, blocked: RoomEvent) -> None:
        """Commit a record the router blocked, and announce why.

        Shared by every blocked record a delivery set leaves: an agent not
        asked past the depth limit, a muted source's response (RFC §8.3,
        §7.5), so each is indexed and announced alike.
        """
        await self._commit_indexed(room_id, blocked)
        if blocked.blocked_by == CHAIN_DEPTH_LIMIT:
            await self._emit_framework_event(
                "chain_depth_exceeded",
                room_id=room_id,
                event_id=blocked.id,
                channel_id=blocked.source.channel_id,
                data={
                    "chain_depth": blocked.chain_depth,
                    "max_chain_depth": self._max_chain_depth,
                },
            )
        else:
            await self._announce_blocked(
                room_id, blocked, reason=blocked.blocked_by, blocked_by=blocked.blocked_by
            )

    async def _reentry_commit_pass(
        self,
        room_id: str,
        plan: DeliveryPlan,
        result: BroadcastResult,
        cascade: DeliveryCascade,
    ) -> None:
        """Turn an executed plan's response events into fresh commit passes.

        Each response takes the room lock for ITS OWN commit (RFC §10.1 step
        14): BEFORE_BROADCAST sync hooks, atomic commit, broadcast planning —
        and its plan joins the same lane behind the trigger. The child's
        cascade unit is retained inside ``_commit_to_lane`` before the
        parent's release, so the caller's wait covers the whole chain. A
        concurrent inbound MAY commit between a trigger and its response —
        the RFC's explicit relaxation (index monotonicity and parent
        linkage, never adjacency).
        """
        if plan.injected or not result.reentry_events:
            return
        await self._commit_responses(
            room_id, result.reentry_events, plan.response_visibility, cascade
        )

    async def _commit_responses(
        self,
        room_id: str,
        responses: list[RoomEvent],
        response_visibility: str | None,
        cascade: DeliveryCascade,
    ) -> None:
        """Give each buffered response its own commit pass, within the
        cascade's reentry budget (RFC §10.1 step 14, §8.3).

        The one way a buffered response reaches the timeline, whichever pass
        produced it: a trigger's delivery set or a regeneration.
        """
        for response in responses:
            reentry = scoped(response, response_visibility)
            if not cascade.consume_reentry_budget():
                await self._store_past_reentry_cap(room_id, reentry)
                continue
            await self._run_reentry_pass(room_id, reentry, response_visibility, cascade)

    async def _store_past_reentry_cap(self, room_id: str, response: RoomEvent) -> None:
        """Store a response past the cascade's reentry budget as its BLOCKED record.

        Shared by a buffered answer and by a chained stream closed unread,
        so the cap leaves the same trace whichever way the answer came.
        """
        logger.warning(
            "Reentry chain hit its cap, storing response as BLOCKED",
            extra={"room_id": room_id},
        )
        async with self._lock_manager.locked(room_id):
            # Same status gate as every other growth point (RFC §5.1):
            # a closed room does not take the audit record either.
            if not await self._room_refuses_writes(room_id):
                await self._commit_indexed(
                    room_id,
                    response.model_copy(
                        update={"status": EventStatus.BLOCKED, "blocked_by": "reentry_loop_cap"}
                    ),
                )

    async def _reentry_context(self, room_id: str, reentry: RoomEvent) -> RoomContext | None:
        """The room's context for a reentry pass, or ``None`` when the room is
        gone, the refusal announced with a null status (RFC §8.2)."""
        context = await self._existing_room_context(room_id)
        if context is None:
            await self._refuse_closed_room(
                room_id, status=None, operation="reentry", event=reentry
            )
        return context

    async def _refuse_reentry(
        self, room_id: str, reentry: RoomEvent, context: RoomContext
    ) -> bool:
        """Whether the room refuses this reentry pass, the refusal announced.

        RFC §10.1 step 6 / §5.1: a reentry re-enters the locked section, so it
        meets the same status gate as any other write. The room may have been
        closed while the trigger's delivery set was executing, and an answer
        landing after ``close_room()`` records nothing, not even a BLOCKED
        row; the blocked result :meth:`_refuse_closed_room` returns is for a
        caller with someone to answer, and a pass has none. A room gone
        meanwhile was refused already, when its context could not be read
        (:meth:`_reentry_context`).
        """
        if not _refuses_writes(context.room):
            return False
        await self._refuse_closed_room(
            room_id, status=context.room.status, operation="reentry", event=reentry
        )
        return True

    async def _run_reentry_pass(
        self,
        room_id: str,
        reentry: RoomEvent,
        response_visibility: str | None,
        cascade: DeliveryCascade,
    ) -> None:
        """One response event's own commit pass, under a fresh room lock."""
        async with self._lock_manager.locked(room_id):
            # One read of what the lock protects (RFC §10.1 steps 6 and 12):
            # the status gate, the source's right to write and the delivery
            # plan all read this context, fresh under the lock, since
            # concurrent commits may have landed after the trigger's plan.
            context = await self._reentry_context(room_id, reentry)
            if context is None:
                return
            if await self._refuse_reentry(room_id, reentry, context):
                return

            # Provisional index for the hook, mirroring the main inbound
            # path; the authoritative index is (re)assigned at commit.
            reentry = reentry.model_copy(update={"index": context.room.event_count})
            reentry_ctx = context.model_copy(
                update={
                    "recent_events": [
                        *context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                        reentry,
                    ]
                }
            )

            # BEFORE_BROADCAST sync hooks, then the source's right to write
            # (RFC §10.1 steps 9 and 11): orchestration routing stamps
            # _routed_to here, and a muted or read-only agent's answer is
            # stored BLOCKED with what its hooks decided kept (§7.5 rule 3).
            decision = await self._gate_commit(room_id, reentry, reentry_ctx)
            if isinstance(decision, _Blocked):
                await self._store_blocked(room_id, decision, cascade)
                return
            await self._commit_reentry(room_id, decision, response_visibility, cascade)

    def _reentry_plan_factory(
        self, ready: _Ready, response_visibility: str | None
    ) -> Callable[[RoomEvent], DeliveryPlan]:
        """The plan builder of a response its gates accepted, called on the
        committed event: its delivery set, scoped as its trigger asked, with
        its hooks' tasks and observations for after delivery."""
        router = self._get_router()

        def factory(committed: RoomEvent) -> DeliveryPlan:
            child = router.plan(committed, ready.source_binding, ready.context)
            child.response_visibility = response_visibility
            child.hook_tasks = list(ready.sync_result.tasks)
            child.hook_observations = list(ready.sync_result.observations)
            return child

        return factory

    async def _commit_reentry(
        self,
        room_id: str,
        ready: _Ready,
        response_visibility: str | None,
        cascade: DeliveryCascade,
    ) -> None:
        """Commit a response its gates accepted and lane its delivery (RFC
        §10.1 step 12), under the lock its pass holds."""
        reentry, binding = ready.event, ready.source_binding
        sync_result, context = ready.sync_result, ready.context

        # Commit the response BEFORE delivering any events its hook
        # injected: the response causes the injection, so it takes the
        # lower index (mirrors the main path).
        if binding is None:
            # A detached source has no delivery plan. Keep its timeline
            # response, but only after the same filtering and side effects
            # as a response whose source is still attached.
            stored = await self._persist_committed(
                room_id, reentry.model_copy(update={"status": EventStatus.DELIVERED})
            )
            await self._persist_side_effects(
                room_id,
                sync_result.tasks,
                sync_result.observations,
                stored or reentry,
                context,
            )
        else:
            stored = await self._commit_to_lane(
                room_id,
                reentry.model_copy(update={"status": EventStatus.DELIVERED}),
                cascade,
                self._reentry_plan_factory(ready, response_visibility),
            )
        if stored is not None:
            cascade.response_events.append(stored)
        if sync_result.injected_events:
            await self._lane_injected_events(
                sync_result.injected_events, room_id, context, cascade
            )


def record_buffered_reply(
    cascade: DeliveryCascade, channel_id: str, output: ChannelOutput, *, root: bool
) -> None:
    """Merge a buffered reply's record into the caller's, final once the
    channel returned. A reply to the caller's own event (*root*), not to an
    answer, puts its end under its channel in ``turns``, read off its record
    or else its last message (RFC §6.4)."""
    merge_channel_record(cascade.response_metadata, output.response_metadata)
    if not root:
        return
    messages = [
        event.metadata or {}
        for event in reversed(output.response_events)
        if event.type == EventType.MESSAGE
    ]
    entry = turn_summary(output.response_metadata, *messages)
    add_turn_entry(cascade.response_metadata, channel_id, entry)

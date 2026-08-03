"""LaneExecutionMixin — the framework side of the delivery lanes.

The engine (:mod:`roomkit.core.lanes`) owns ordering; this mixin owns what a
plan *means*: committing an event together with its plan (the third commit
gate next to ``_persist_committed`` / ``_commit_indexed``), executing a
plan's delivery set, firing the per-event aftermath, and turning response
events into fresh commit passes (RFC §10.1 step 14 — reentry passes take the
room lock anew; they are never drained inside the trigger's lock tenure).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT, HelpersMixin
from roomkit.models.enums import ChannelCategory, EventStatus, HookTrigger
from roomkit.models.event import EventSource, RoomEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from roomkit.channels.base import Channel
    from roomkit.core.event_router import BroadcastResult, EventRouter
    from roomkit.core.hooks import HookEngine
    from roomkit.core.lanes import DeliveryCascade, DeliveryPlan, RoomLaneRegistry
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.hook import InjectedEvent
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


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

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter``.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager
    _lanes: RoomLaneRegistry
    _max_chain_depth: int

    def _get_router(self) -> EventRouter: ...


class LaneExecutionMixin(HelpersMixin):
    """Commit-with-plan gate and lane-executor callbacks.

    Host contract: :class:`LaneExecutionHost`.
    """

    _lock_manager: RoomLockManager
    _max_chain_depth: int

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # RoomKit._get_router
    _handle_block: Any  # InboundLockedMixin
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
        if (
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
    ) -> None:
        """Arrange stream consumption for a detached caller.

        A ``send_event`` issued from inside a sync hook (under the room
        lock) or from a tool handler (inside the lane) cannot wait on its
        cascade — but a streaming provider's reply is only generated when
        its stream is consumed. This schedules the consumption on a clean
        background task (fresh context: an inherited ``_held_rooms`` would
        fake lock reentrancy), tracked like every fire-and-forget hook task.
        """
        import asyncio
        import contextvars

        async def _consume() -> None:
            await cascade.wait_detached()
            if cascade.streams and cascade.cancelled is None:
                await self._process_streaming_responses(cascade.streams, room_id)

        task = asyncio.get_running_loop().create_task(
            _consume(),
            name=f"roomkit-detached-streams-{room_id}",
            context=contextvars.Context(),
        )
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)

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
            # Surface intelligence-channel failures to ON_ERROR so hosts can
            # render an error card (transport delivery failures above are not
            # turn-level agent errors). Fired here, off the room lock.
            for binding in context.bindings:
                if binding.category != ChannelCategory.INTELLIGENCE:
                    continue
                error_msg = result.errors.get(binding.channel_id)
                if not error_msg:
                    continue
                await self._fire_error_hook(
                    room_id,
                    context,
                    EventSource(
                        channel_id=binding.channel_id,
                        channel_type=binding.channel_type,
                    ),
                    error=error_msg,
                    error_type="unknown",
                    error_category="generation",
                    chain_depth=event.chain_depth + 1,
                    visibility=event.response_visibility or "all",
                    parent_event_id=event.parent_event_id,
                )
            first_error = self._first_intelligence_error(result, context)
            if first_error is not None:
                cascade.record_error(first_error)
            cascade.add_streams(result.streaming_responses)

        # Commit blocked events from chain depth enforcement atomically (RFC
        # §8.1 / §8.3 / §14.3 — blocked events are still indexed).
        for blocked in result.blocked_events:
            await self._commit_indexed(room_id, blocked)
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

        # Stamp response_visibility from the root trigger onto reentry
        # events' *visibility* field — the router's visibility check reads
        # ``visibility``, so the caller's response scope rides the event.
        reentries = result.reentry_events
        if plan.response_visibility:
            reentries = [
                r.model_copy(update={"visibility": plan.response_visibility}) for r in reentries
            ]

        for reentry in reentries:
            if not cascade.consume_reentry_budget():
                logger.warning(
                    "Reentry chain hit its cap, storing response as BLOCKED",
                    extra={"room_id": room_id},
                )
                async with self._lock_manager.locked(room_id):
                    await self._commit_indexed(
                        room_id,
                        reentry.model_copy(
                            update={
                                "status": EventStatus.BLOCKED,
                                "blocked_by": "reentry_loop_cap",
                            }
                        ),
                    )
                continue
            await self._run_reentry_pass(room_id, reentry, plan, cascade)

    async def _run_reentry_pass(
        self,
        room_id: str,
        reentry: RoomEvent,
        parent_plan: DeliveryPlan,
        cascade: DeliveryCascade,
    ) -> None:
        """One response event's own commit pass, under a fresh room lock."""
        router = self._get_router()
        async with self._lock_manager.locked(room_id):
            reentry_binding = await self._store.get_binding(room_id, reentry.source.channel_id)
            if reentry_binding is None:
                # No channel to broadcast to, but the response is still part
                # of the timeline: commit it DELIVERED so it is indexed and
                # counted like any other event (RFC §10.1).
                await self._persist_committed(
                    room_id, reentry.model_copy(update={"status": EventStatus.DELIVERED})
                )
                return

            # Fresh context: concurrent commits may have landed since the
            # trigger's plan was made, and this pass must see them.
            context = await self._build_context(room_id)

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

            # BEFORE_BROADCAST sync hooks on reentry events so orchestration
            # routing can stamp _routed_to metadata and prevent AI-to-AI loops.
            reentry_sync = await self._hook_engine.run_sync_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, reentry, reentry_ctx
            )
            if not reentry_sync.allowed:
                # RFC §9.5: commit BLOCKED, emit event_blocked, deliver
                # injected side effects; hook side effects are still persisted.
                blocked_event = await self._handle_block(
                    room_id=room_id,
                    event=reentry,
                    reason=reentry_sync.reason,
                    blocked_by=reentry_sync.blocked_by,
                    injected_events=reentry_sync.injected_events,
                    context=reentry_ctx,
                    cascade=cascade,
                )
                await self._persist_side_effects(
                    room_id,
                    reentry_sync.tasks,
                    reentry_sync.observations,
                    blocked_event,
                    reentry_ctx,
                )
                return
            reentry = reentry_sync.event or reentry

            def factory(
                committed: RoomEvent,
                _binding: Any = reentry_binding,
                _ctx: Any = reentry_ctx,
                _sync: Any = reentry_sync,
            ) -> DeliveryPlan:
                child = router.plan(committed, _binding, _ctx)
                child.response_visibility = parent_plan.response_visibility
                child.hook_tasks = list(_sync.tasks)
                child.hook_observations = list(_sync.observations)
                return child

            # Commit the response BEFORE delivering any events its hook
            # injected: the response causes the injection, so it takes the
            # lower index (mirrors the main path).
            await self._commit_to_lane(
                room_id,
                reentry.model_copy(update={"status": EventStatus.DELIVERED}),
                cascade,
                factory,
            )
            if reentry_sync.injected_events:
                await self._lane_injected_events(
                    reentry_sync.injected_events, room_id, reentry_ctx, cascade
                )

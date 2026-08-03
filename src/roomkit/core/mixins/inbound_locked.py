"""InboundLockedMixin — the locked half of inbound processing (RFC §10.1).

Everything here runs under the room lock and ends at the commit point: the
status gate, idempotency, index assignment, hooks, permission checks, the
atomic commit and the broadcast *planning* (§10.1 step 12). Execution of the
delivery set, the reentry passes and the async hooks live off the lock, in
the room's delivery lane (:mod:`roomkit.core.lanes`,
:class:`~roomkit.core.mixins.lane_execution.LaneExecutionMixin`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT, HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    DeleteType,
    EventStatus,
    EventType,
    HookTrigger,
    ParticipantRole,
    RoomStatus,
)
from roomkit.models.event import DeleteContent, EditContent, RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityResult
from roomkit.models.task import Observation, Task

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter
    from roomkit.core.hooks import HookEngine
    from roomkit.core.lanes import DeliveryCascade
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# RFC §10.3 fixes the ``edit_source`` vocabulary at "sender" | "system". The
# field stays a free-form ``str`` for compatibility, so anything outside the
# vocabulary is treated as unprivileged rather than trusted.
_EDIT_SOURCE_SYSTEM = "system"

# RFC §5.1 — statuses that refuse new events. CLOSED and ARCHIVED refuse
# identically; they differ in intent, not in what they accept.
_REFUSING_STATUSES = frozenset({RoomStatus.CLOSED, RoomStatus.ARCHIVED})


class _Proceed:
    """Marker returned by ``_run_precommit`` once the event has committed,
    with its broadcast planned and enqueued on the room's delivery lane
    (RFC §10.1 step 12).

    Carries the state the caller still needs after the commit — kept off the
    event so the phases stay decoupled. ``mutation_hook`` is the (trigger,
    updated target) pair for an edit/delete (RFC §10.3); it rides the plan
    and fires from the lane executor.
    """

    __slots__ = ("context", "event", "mutation_hook", "source_binding", "sync_result")

    def __init__(
        self,
        event: RoomEvent,
        source_binding: Any,
        sync_result: Any,
        context: RoomContext,
        mutation_hook: tuple[HookTrigger, RoomEvent] | None = None,
    ) -> None:
        self.event = event
        self.source_binding = source_binding
        self.sync_result = sync_result
        self.context = context
        self.mutation_hook = mutation_hook


@runtime_checkable
class InboundLockHost(Protocol):
    """Contract: capabilities a host class must provide for InboundLockedMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for events, bindings, participants.
        _channels: Channel registry for injected-event delivery.
        _hook_engine: Hook engine for BEFORE_BROADCAST / AFTER_BROADCAST.
        _max_chain_depth: Maximum reentry chain depth (RFC §10).

    Methods provided by the host class (RoomKit):
        _get_router: Lazily create / return the ``EventRouter`` for broadcast.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int

    def _get_router(self) -> EventRouter: ...


class InboundLockedMixin(HelpersMixin):
    """Locked inbound processing: gates, commit, broadcast planning.

    Host contract: :class:`InboundLockHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int
    _process_timeout: float

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _commit_to_lane: Any  # see LaneExecutionMixin
    _lane_injected_events: Any  # see LaneExecutionMixin

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _resolve_thread_root(self, room_id: str, parent_event_id: str) -> str | None:
        """Resolve an in-app thread parent to its thread root.

        Flat two-level threading: a reply always points at the thread ROOT.
        If the referenced event is itself a reply, its root is returned so
        replying-to-a-reply stays in a single thread. A parent that does not
        exist or belongs to another room drops to ``None`` (top level) with a
        warning — a stale reference must not lose the sender's message.
        """
        parent = await self._store.get_event(parent_event_id)
        if parent is None or parent.room_id != room_id:
            logger.warning(
                "Thread parent %s not found in room %s; posting at top level",
                parent_event_id,
                room_id,
                extra={"room_id": room_id, "parent_event_id": parent_event_id},
            )
            return None
        return parent.parent_event_id or parent_event_id

    def _is_system_source(self, event: RoomEvent) -> bool:
        """Whether an event genuinely originates from a system channel.

        ``EventSource.channel_type`` is set by the framework from the
        registered channel, not by the payload, so a remote party cannot
        claim it.
        """
        return event.source.channel_type == ChannelType.SYSTEM

    async def _has_admin_authority(self, room_id: str, participant_id: str) -> bool:
        """Whether a participant holds administrative authority in the room.

        RFC §10.3 requires administrative authority to be *verified* rather
        than asserted by the inbound payload; the room roster is the only
        thing here the sender does not control. Moderation that legitimately
        outranks the roster belongs on the host-side API
        (``update_event`` / ``delete_event``), which documents that the host
        owns authorization on that path.
        """
        participant = await self._store.get_participant(room_id, participant_id)
        return participant is not None and participant.role == ParticipantRole.OWNER

    async def _authorize_edit_delete(
        self, room_id: str, event: RoomEvent, sender_id: str, target_author_id: str
    ) -> str | None:
        """Authorize an inbound EDIT/DELETE, returning a block reason or ``None``.

        RFC §10.3: authorship covers ``sender`` edits and SENDER deletes,
        ADMIN deletes require verified administrative authority, and SYSTEM
        must come from a system channel. Any other ``edit_source`` is
        unprivileged and falls back to the author check — an unrecognized
        value must not buy a caller past authorization, which is exactly what
        an attacker would send.
        """
        content = event.content
        privileged: bool | None = None
        if isinstance(content, DeleteContent):
            if content.delete_type == DeleteType.ADMIN:
                privileged = await self._has_admin_authority(room_id, sender_id)
            elif content.delete_type == DeleteType.SYSTEM:
                privileged = self._is_system_source(event)
        elif isinstance(content, EditContent) and content.edit_source == _EDIT_SOURCE_SYSTEM:
            privileged = self._is_system_source(event)

        if privileged is True:
            return None
        if privileged is False:
            logger.warning(
                "Edit/Delete rejected: sender %s lacks the claimed authority in room %s",
                sender_id,
                room_id,
                extra={"room_id": room_id},
            )
            return "not_authorized"

        # Unprivileged path — the sender must be the original author.
        if sender_id != target_author_id:
            logger.warning(
                "Edit/Delete rejected: sender %s is not author %s",
                sender_id,
                target_author_id,
                extra={"room_id": room_id},
            )
            return "not_original_author"
        return None

    async def _commit_event(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Commit an event to the timeline (RFC §10.1 step 12 / §14.3).

        Persists the event and bumps the room counters (event_count,
        latest_index, timers.last_activity_at) as ONE atomic store transaction
        (:meth:`ConversationStore.commit_event`), so an observer never sees a
        stored event the room counters do not reflect, and the authoritative
        index is assigned inside that transaction (§8.1) — safe even without a
        cross-process room lock. Returns the committed event (its index may
        differ from the provisional pre-hook value if the store serialized a
        concurrent writer).

        When the persistence policy excludes the event, nothing is stored: the
        event is delivered but not persisted, so it consumes no index and MUST
        NOT advance the room counters (latest_index must never point at an
        unstored event, §14.3). The input event is returned unchanged.
        """
        committed = await self._persist_committed(room_id, event)
        return event if committed is None else committed

    async def _process_locked(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext | None,
        cascade: DeliveryCascade,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
    ) -> InboundResult:
        """Process an event under the room lock (RFC §10.1).

        Split at the commit point (RFC §10.1): the pre-commit critical
        section (:meth:`_run_precommit`) is bounded by ``process_timeout``
        (§13.6) and aborts before any durable write. The commit itself plans
        the broadcast and enqueues it on the room's delivery lane (§10.2) —
        execution, the RFC §10.3 mutation trigger, AFTER_BROADCAST and
        ON_ERROR hooks all run off the lock, tracked by *cascade*; the
        caller awaits the cascade once the lock is released.

        *context* is the context the caller built BEFORE taking the lock — the
        inbound pipeline builds one for the channel and the identity resolver
        (RFC §10.1 steps 3-5). It is carried into the locked rebuild rather
        than thrown away. A caller that already holds the lock has no such
        context and passes ``None``.
        """
        try:
            outcome = await asyncio.wait_for(
                self._run_precommit(
                    event,
                    room_id,
                    context,
                    cascade,
                    resolved_identity=resolved_identity,
                    pending_id_result=pending_id_result,
                ),
                timeout=self._process_timeout,
            )
        except TimeoutError:
            logger.error(
                "Inbound pre-commit timed out after %.1fs",
                self._process_timeout,
                extra={"room_id": room_id, "event_id": event.id},
            )
            await self._emit_framework_event(
                "process_timeout",
                room_id=room_id,
                event_id=event.id,
                data={"timeout": self._process_timeout},
            )
            return InboundResult(blocked=True, reason="process_timeout")
        if isinstance(outcome, InboundResult):
            return outcome
        # Injected events from allow/modify hooks: committed and laned after
        # the trigger (outside the pre-commit timeout — the trigger is
        # committed and must never be contradicted by a late cancellation).
        if outcome.sync_result.injected_events:
            await self._lane_injected_events(
                outcome.sync_result.injected_events, room_id, outcome.context, cascade
            )
        return InboundResult(event=outcome.event)

    async def _run_precommit(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext | None,
        cascade: DeliveryCascade,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
    ) -> InboundResult | _Proceed:
        """Pre-commit critical section (RFC §10.1).

        Returns an :class:`InboundResult` for any block/duplicate case, or a
        :class:`_Proceed` once the event has been committed — with its
        broadcast planned and enqueued on the room's delivery lane (§10.2).
        Performs no durable write of the inbound event before the commit
        point, so a ``process_timeout`` here aborts cleanly with nothing
        persisted (§13.6).
        """
        # Re-read under the lock: the status gate must not act on an answer
        # taken before it (§10.1 step 6), and planning reads bindings under the
        # lock (step 12). The pre-lock context is carried in rather than
        # discarded — its history is the expensive half of a context, and it is
        # reused only when the room's counter proves nothing committed since.
        context = await self._build_context(room_id, carrying=context)

        # RFC §5.1 / §10.1 step 6 — a room whose status refuses new events
        # refuses them here, at the one point every entry converges on: inbound
        # messages, send_event(), hook-injected events and the framework's own
        # re-injection all pass through this. Under the lock, because
        # close_room() takes the same one and an earlier answer can be stale by
        # the time the event would commit.
        #
        # Nothing is written, not even a BLOCKED record: appending an audit
        # event to a closed room is the thing the status forbids (§5.1).
        if context.room.status in _REFUSING_STATUSES:
            logger.info(
                "Event refused: room %s is %s",
                room_id,
                context.room.status,
                extra={"room_id": room_id, "event_id": event.id},
            )
            await self._emit_framework_event(
                "room_refused_event",
                room_id=room_id,
                event_id=event.id,
                data={"status": str(context.room.status), "event_type": str(event.type)},
            )
            return InboundResult(blocked=True, reason="room_closed")

        # Persist deferred participant creation inside the lock (Fix #1)
        if resolved_identity is not None:
            await self._ensure_identified_participant(room_id, event, resolved_identity)
        elif pending_id_result is not None:
            await self._create_pending_participant(room_id, event, pending_id_result)

        # Idempotency check (inside lock to prevent TOCTOU race)
        if event.idempotency_key and await self._store.check_idempotency(
            room_id, event.idempotency_key
        ):
            logger.info(
                "Duplicate event %s",
                event.idempotency_key,
                extra={"room_id": room_id, "idempotency_key": event.idempotency_key},
            )
            return InboundResult(blocked=True, reason="duplicate")

        # Provisional index for the hooks. The authoritative one is (re)assigned
        # inside the commit (§8.1), so this reads the counter the context
        # already carries under the lock rather than paying a COUNT(*) over the
        # room's whole timeline on every single inbound.
        event = event.model_copy(update={"index": context.room.event_count})

        # Normalize the in-app thread parent to the thread ROOT (flat two-level
        # model). This is the single choke point for every entry point — direct
        # send_event and inbound both traverse here (RFC §10.5) — so the
        # invariant "parent_event_id points at a thread root" holds regardless
        # of how the caller referenced the parent.
        if event.parent_event_id is not None:
            root_id = await self._resolve_thread_root(room_id, event.parent_event_id)
            if root_id != event.parent_event_id:
                event = event.model_copy(update={"parent_event_id": root_id})

        # Edit/Delete validation (RFC §10.3). The target mutation is deferred
        # until after BEFORE_BROADCAST hooks allow the event (applied below via
        # ``_apply_edit_delete_state``), so a moderation hook that blocks an
        # edit/delete cannot leave the target already mutated.
        edit_delete_target: RoomEvent | None = None
        if event.type in (EventType.EDIT, EventType.DELETE) and isinstance(
            event.content, (EditContent, DeleteContent)
        ):
            target_id = event.content.target_event_id
            target_event = await self._store.get_event(target_id)

            if target_event is None or target_event.room_id != room_id:
                logger.warning(
                    "Edit/Delete target %s not found in room %s",
                    target_id,
                    room_id,
                    extra={"room_id": room_id, "target_event_id": target_id},
                )
                return InboundResult(blocked=True, reason="target_event_not_found")

            # Identity required: anonymous users must not edit/delete others' messages
            sender_id = event.source.participant_id
            target_author_id = target_event.source.participant_id
            if sender_id is None or target_author_id is None:
                return InboundResult(blocked=True, reason="identity_required_for_edit")

            # Authorization check (RFC §10.3)
            block_reason = await self._authorize_edit_delete(
                room_id, event, sender_id, target_author_id
            )
            if block_reason is not None:
                return InboundResult(blocked=True, reason=block_reason)

            edit_delete_target = target_event

        # Run sync hooks (before_broadcast)
        sync_result = await self._hook_engine.run_sync_hooks(
            room_id, HookTrigger.BEFORE_BROADCAST, event, context
        )

        # Emit framework events for any hook errors
        for hook_err in sync_result.hook_errors:
            await self._emit_framework_event(
                "hook_error",
                room_id=room_id,
                event_id=event.id,
                data=hook_err,
            )

        if not sync_result.allowed:
            return await self._blocked_result(
                room_id,
                event,
                context,
                cascade,
                reason=sync_result.reason,
                blocked_by=sync_result.blocked_by,
                injected_events=sync_result.injected_events,
                tasks=sync_result.tasks,
                observations=sync_result.observations,
            )

        # Use potentially modified event. HookResult.event carries whatever
        # payload its trigger passed, so only a RoomEvent may replace one.
        if isinstance(sync_result.event, RoomEvent):
            event = sync_result.event

        # RFC §7.5 — a source that cannot write (READ_ONLY/NONE or muted) must
        # not inject a DELIVERED event into the timeline. Persist it BLOCKED for
        # audit, still collecting hook side effects (RFC §7.5 rule 3 — side
        # effects are ALWAYS collected), and stop before broadcast. The source
        # binding is fetched once here and reused for broadcast below.
        source_binding = await self._store.get_binding(room_id, event.source.channel_id)
        if source_binding is not None and not source_binding.can_write:
            reason = "source_muted" if source_binding.muted else "source_read_only"
            return await self._blocked_result(
                room_id,
                event,
                context,
                cascade,
                reason=reason,
                blocked_by=reason,
                injected_events=sync_result.injected_events,
                tasks=sync_result.tasks,
                observations=sync_result.observations,
            )

        # Apply edit/delete target mutation now that the event is authorized and
        # hook-allowed (RFC §10.3 — mutation must not precede the block decision).
        mutation_hook: tuple[HookTrigger, RoomEvent] | None = None
        if edit_delete_target is not None:
            mutation_hook = await self._apply_edit_delete_state(event, edit_delete_target)

        # Commit point (RFC §10.1 step 12): persist DELIVERED and bump the
        # room counters atomically, PLAN the broadcast against binding state
        # consistent with this commit, and enqueue it on the room's delivery
        # lane. Past this line the event is authoritative; its delivery set
        # executes off the lock (§10.2), tracked by *cascade*.
        router = self._get_router()

        def plan_factory(committed: RoomEvent) -> Any:
            from roomkit.core.lanes import DeliveryPlan

            if source_binding is None:
                if mutation_hook is None:
                    return None  # nothing to broadcast, nothing to fire
                # No source binding: no broadcast, but the RFC §10.3 mutation
                # trigger must still fire once the commit is visible — an
                # empty plan carries it. A binding-less trigger fires no
                # AFTER_BROADCAST and persists no hook side effects.
                return DeliveryPlan(
                    event=committed,
                    source_binding=None,
                    context=context,
                    targets=[],
                    mutation_hook=mutation_hook,
                    fire_after_broadcast=False,
                )
            # Refresh context locally by appending the committed event, so
            # the delivery set (and the AI's on_event) sees the trigger.
            broadcast_ctx = context.model_copy(
                update={
                    "recent_events": [
                        *context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                        committed,
                    ]
                }
            )
            plan = router.plan(committed, source_binding, broadcast_ctx)
            plan.mutation_hook = mutation_hook
            plan.response_visibility = committed.response_visibility
            plan.hook_tasks = list(sync_result.tasks)
            plan.hook_observations = list(sync_result.observations)
            plan.emit_processed = True
            return plan

        event = event.model_copy(update={"status": EventStatus.DELIVERED})
        committed = await self._commit_to_lane(room_id, event, cascade, plan_factory)
        # A persistence-policy-excluded event is delivered but not persisted;
        # it keeps its provisional identity (RFC §14.3 — no index consumed).
        event = event if committed is None else committed
        return _Proceed(event, source_binding, sync_result, context, mutation_hook)

    async def _handle_block(
        self,
        *,
        room_id: str,
        event: RoomEvent,
        reason: str | None,
        blocked_by: str | None,
        injected_events: list[InjectedEvent],
        context: RoomContext,
        cascade: DeliveryCascade,
    ) -> RoomEvent:
        """RFC §9.5 block handling: commit the event as BLOCKED, emit the
        framework event, and lane injected side effects. Shared by every
        block path (main-inbound hook block, reentry hook block, source
        write-permission block) so they cannot drift.

        The BLOCKED event is committed atomically like any other event — index,
        status, and room counters in one store transaction (§14.3) — because a
        blocked event is still part of the timeline and consumes an index (§8.3).

        Returns the committed BLOCKED event so the caller can include it in its
        return value.
        """
        blocked_event = event.model_copy(
            update={
                "status": EventStatus.BLOCKED,
                "blocked_by": blocked_by or reason,
            }
        )
        blocked_event = await self._commit_indexed(room_id, blocked_event)
        await self._emit_framework_event(
            "event_blocked",
            room_id=room_id,
            event_id=blocked_event.id,
            data={
                "reason": reason,
                "blocked_by": blocked_by,
            },
        )
        await self._lane_injected_events(injected_events, room_id, context, cascade)
        return blocked_event

    async def _blocked_result(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
        cascade: DeliveryCascade,
        *,
        reason: str | None,
        blocked_by: str | None,
        injected_events: list[InjectedEvent],
        tasks: list[Task],
        observations: list[Observation],
    ) -> InboundResult:
        """Persist a BLOCKED event, persist its hook side effects (RFC §7.5
        rule 3 — side effects are always collected), and return the blocked
        :class:`InboundResult`. Shared by the hook-block and source
        write-permission paths so they cannot drift.
        """
        blocked_event = await self._handle_block(
            room_id=room_id,
            event=event,
            reason=reason,
            blocked_by=blocked_by,
            injected_events=injected_events,
            context=context,
            cascade=cascade,
        )
        await self._persist_side_effects(room_id, tasks, observations, blocked_event, context)
        return InboundResult(event=blocked_event, blocked=True, reason=reason)

    async def _run_deferred_async_hooks(
        self, room_id: str, pending: list[tuple[HookTrigger, RoomEvent, RoomContext]]
    ) -> None:
        """Run async-hook firings collected under the room lock, once it is
        released (RFC §10.1). Still used by the regenerate path, which
        broadcasts inline under the lock.
        """
        for trigger, ev, ctx in pending:
            await self._hook_engine.run_async_hooks(room_id, trigger, ev, ctx)

    async def _apply_edit_delete_state(
        self, event: RoomEvent, target_event: RoomEvent
    ) -> tuple[HookTrigger, RoomEvent] | None:
        """Apply RFC §10.3 state updates to an edit/delete target event.

        Invoked only after the edit/delete event has passed authorization
        *and* been allowed by BEFORE_BROADCAST hooks, so a blocked
        edit/delete never mutates the target. Uses the final (post-modify)
        event content so a ``modify`` hook on the edit is honored.

        Returns the (mutation trigger, updated target) pair the caller must
        fire once the room lock is released — ON_EVENT_UPDATED for an edit,
        ON_EVENT_DELETED for a (soft) delete — or ``None`` when the content
        was not an edit/delete payload.
        """
        content = event.content
        if isinstance(content, EditContent):
            updated = target_event.model_copy(
                update={
                    "content": content.new_content,
                    "metadata": {**target_event.metadata, "edited": True},
                }
            )
            await self._store.update_event(updated)
            return (HookTrigger.ON_EVENT_UPDATED, updated)
        if isinstance(content, DeleteContent):
            updated = target_event.model_copy(
                update={"metadata": {**target_event.metadata, "deleted": True}}
            )
            await self._store.update_event(updated)
            return (HookTrigger.ON_EVENT_DELETED, updated)
        return None

    async def _deliver_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: RoomContext,
    ) -> None:
        """Store and deliver injected events to their target channels, inline.

        Only the identity pipeline still uses this (it runs before the room
        lock and has no cascade); the locked pipeline lanes its injected
        events via ``_lane_injected_events``. Inline execution trivially
        satisfies per-room order (RFC §10.2).
        """
        for injected in injected_events:
            # Commit the injected event atomically as DELIVERED (index + room
            # counters, RFC §8.1 / §14.3) — it is a real, delivered timeline
            # event, not a PENDING draft.
            stored = await self._commit_indexed(
                room_id, injected.event.model_copy(update={"status": EventStatus.DELIVERED})
            )

            # Deliver to target channels
            target_ids = injected.target_channel_ids
            if target_ids is None:
                # No target specified — skip delivery (stored only)
                continue

            for target_id in target_ids:
                channel = self._channels.get(target_id)
                binding = await self._store.get_binding(room_id, target_id)
                if channel is not None and binding is not None:
                    try:
                        await channel.on_event(stored, binding, context)
                        if binding.category == ChannelCategory.TRANSPORT:
                            await channel.deliver(stored, binding, context)
                    except Exception:
                        logger.exception(
                            "Failed to deliver injected event to %s",
                            target_id,
                            extra={"room_id": room_id, "channel_id": target_id},
                        )

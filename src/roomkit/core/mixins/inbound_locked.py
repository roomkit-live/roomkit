"""InboundLockedMixin — locked inbound processing, broadcast, and reentry."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import _RECENT_EVENTS_LIMIT, HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    DeleteType,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import DeleteContent, EditContent, EventSource, RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityResult
from roomkit.models.task import Observation, Task

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter
    from roomkit.core.hooks import HookEngine
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# Deferred ON_ERROR invocations collected under the room lock and fired after it
# is released (RFC §10.1): (context, error source, _fire_error_hook kwargs).
_ErrorHookSink = list[tuple[RoomContext, EventSource, dict[str, Any]]]

# Deferred async-hook firings collected under the room lock and fired after it
# is released (RFC §10.1): (trigger, event, context). Carries AFTER_BROADCAST
# plus the RFC §10.3 mutation triggers (ON_EVENT_UPDATED / ON_EVENT_DELETED).
_AsyncHookSink = list[tuple[HookTrigger, RoomEvent, RoomContext]]


class _Proceed:
    """Marker returned by ``_run_precommit`` once the event has committed and the
    caller should run the post-commit broadcast phase (RFC §10.1).

    Carries the state the broadcast phase needs — kept off the event so the
    pre-commit and post-commit phases stay decoupled. ``mutation_hook`` is the
    (trigger, updated target) pair to fire when the event edited/deleted a
    target (RFC §10.3), or ``None``.
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
    """Locked inbound processing, broadcast, and reentry.

    Host contract: :class:`InboundLockHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _hook_engine: HookEngine
    _max_chain_depth: int
    _process_timeout: float

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
        context: RoomContext,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
        pending_streams_out: list[Any] | None = None,
        pending_after_broadcast_out: _AsyncHookSink | None = None,
        pending_error_hooks_out: _ErrorHookSink | None = None,
    ) -> InboundResult:
        """Process an event under the room lock (RFC §10.1).

        Split at the commit point (RFC §10.1): the pre-commit critical
        section (:meth:`_run_precommit`) is bounded by ``process_timeout``
        (§13.6) and aborts before any durable write, while the post-commit
        broadcast (:meth:`_process_broadcast`) runs unbounded — so a committed
        event is never cancelled and the returned result never contradicts the
        stored timeline.

        AFTER_BROADCAST (and RFC §10.3 mutation-trigger) async hooks are
        collected into ``pending_after_broadcast_out`` (when provided) so the
        caller can run them once the room lock is released (RFC §10.1 — async
        hooks run after the lock, not under it). Without a sink they run
        inline. ON_ERROR hooks
        are likewise collected into ``pending_error_hooks_out`` — a failing
        provider must not hold the room lock while its error hooks run.
        """
        try:
            outcome = await asyncio.wait_for(
                self._run_precommit(
                    event,
                    room_id,
                    context,
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
        # RFC §10.3 — the target of an edit/delete mutated at the commit point;
        # fire its mutation trigger first so observers see mutation before the
        # edit/delete event's own AFTER_BROADCAST, matching execution order.
        if outcome.mutation_hook is not None:
            trigger, target = outcome.mutation_hook
            await self._dispatch_async_hooks(
                room_id, trigger, target, outcome.context, pending_after_broadcast_out
            )
        return await self._process_broadcast(
            outcome.event,
            room_id,
            outcome.source_binding,
            outcome.sync_result,
            outcome.context,
            pending_streams_out=pending_streams_out,
            pending_after_broadcast_out=pending_after_broadcast_out,
            pending_error_hooks_out=pending_error_hooks_out,
        )

    async def _run_precommit(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
    ) -> InboundResult | _Proceed:
        """Pre-commit critical section (RFC §10.1).

        Returns an :class:`InboundResult` for any block/duplicate case, or a
        :class:`_Proceed` once the event has been committed and the caller
        should run the post-commit broadcast. Performs no durable write of the
        inbound event before the commit point, so a ``process_timeout`` here
        aborts cleanly with nothing persisted (§13.6).
        """
        # Rebuild context under lock to prevent stale reads
        context = await self._build_context(room_id)

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

        # Assign index
        count = await self._store.get_event_count(room_id)
        event = event.model_copy(update={"index": count})

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
            if event.source.participant_id is None or target_event.source.participant_id is None:
                return InboundResult(blocked=True, reason="identity_required_for_edit")

            # Authorization check
            if isinstance(event.content, EditContent):
                if (
                    event.content.edit_source in (None, "sender")
                    and event.source.participant_id != target_event.source.participant_id
                ):
                    logger.warning(
                        "Edit rejected: sender %s is not author %s",
                        event.source.participant_id,
                        target_event.source.participant_id,
                        extra={"room_id": room_id},
                    )
                    return InboundResult(blocked=True, reason="not_original_author")
            elif isinstance(event.content, DeleteContent) and (
                event.content.delete_type == DeleteType.SENDER
                and event.source.participant_id != target_event.source.participant_id
            ):
                logger.warning(
                    "Delete rejected: sender %s is not author %s",
                    event.source.participant_id,
                    target_event.source.participant_id,
                    extra={"room_id": room_id},
                )
                return InboundResult(blocked=True, reason="not_original_author")

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
                reason=sync_result.reason,
                blocked_by=sync_result.blocked_by,
                injected_events=sync_result.injected_events,
                tasks=sync_result.tasks,
                observations=sync_result.observations,
            )

        # Use potentially modified event
        event = sync_result.event or event

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

        # Commit point (RFC §10.1): persist DELIVERED and bump the room
        # counters atomically, before broadcast, so the timeline and counters
        # never diverge (§14.3) even if the post-commit phase is slow or times
        # out. Past this line the event is authoritative. ``_commit_event``
        # returns the committed event carrying the store-authoritative index.
        event = event.model_copy(update={"status": EventStatus.DELIVERED})
        event = await self._commit_event(room_id, event)
        return _Proceed(event, source_binding, sync_result, context, mutation_hook)

    async def _process_broadcast(
        self,
        event: RoomEvent,
        room_id: str,
        source_binding: Any,
        sync_result: Any,
        context: RoomContext,
        *,
        pending_streams_out: list[Any] | None = None,
        pending_after_broadcast_out: _AsyncHookSink | None = None,
        pending_error_hooks_out: _ErrorHookSink | None = None,
    ) -> InboundResult:
        """Post-commit phase (RFC §10.1): deliver injected events,
        broadcast, and drain reentries.

        Runs WITHOUT ``process_timeout`` — the event committed in
        :meth:`_run_precommit` is authoritative and must never be cancelled or
        re-marked by a timeout (§13.6). Delivery slowness is bounded per channel
        and reported via ``delivery_failed`` / ``broadcast_partial_failure``.
        """
        # Deliver any injected events from allow/modify hooks
        if sync_result.injected_events:
            await self._deliver_injected_events(sync_result.injected_events, room_id, context)

        # No source binding → nothing to broadcast to.
        if source_binding is None:
            return InboundResult(event=event)

        # Refresh context locally by appending the new event (avoids 4 store queries)
        context = context.model_copy(
            update={
                "recent_events": [*context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :], event]
            }
        )

        # Broadcast to other channels
        router = self._get_router()
        broadcast_result = await router.broadcast(event, source_binding, context)

        # H8: Warn on partial broadcast failure
        if broadcast_result.errors:
            total = len(broadcast_result.delivery_outputs) + len(broadcast_result.errors)
            logger.warning(
                "Partial broadcast failure: %d/%d channels failed",
                len(broadcast_result.errors),
                total,
                extra={
                    "room_id": room_id,
                    "event_id": event.id,
                    "failed_channels": list(broadcast_result.errors.keys()),
                },
            )
            await self._emit_framework_event(
                "broadcast_partial_failure",
                room_id=room_id,
                event_id=event.id,
                data={
                    "failed": len(broadcast_result.errors),
                    "total": total,
                    "errors": broadcast_result.errors,
                },
            )

        # Emit delivery tracking framework events
        for ch_id in broadcast_result.delivery_outputs:
            await self._emit_framework_event(
                "delivery_succeeded",
                room_id=room_id,
                event_id=event.id,
                channel_id=ch_id,
            )
        for ch_id, error_msg in broadcast_result.errors.items():
            await self._emit_framework_event(
                "delivery_failed",
                room_id=room_id,
                event_id=event.id,
                channel_id=ch_id,
                data={"error": error_msg},
            )

        # Surface intelligence-channel failures to ON_ERROR so hosts can render an
        # error card. A failure raised in the AI channel's on_event — eager context
        # build, tool/skill resolution, or a non-streaming provider error — lands
        # here as a broadcast error; the streaming consumption path fires ON_ERROR
        # on its own (see inbound_streaming). Transport delivery failures above are
        # not turn-level agent errors, so they are deliberately excluded.
        #
        # ON_ERROR is DEFERRED past the room lock (RFC §10.1 — like
        # AFTER_BROADCAST): a slow ON_ERROR hook (up to the hook timeout) must
        # not hold the lock and stall every following message for the room.
        for binding in context.bindings:
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            error_msg = broadcast_result.errors.get(binding.channel_id)
            if not error_msg:
                continue
            await self._dispatch_error_hook(
                room_id,
                context,
                EventSource(
                    channel_id=binding.channel_id,
                    channel_type=binding.channel_type,
                ),
                pending_error_hooks_out,
                error=error_msg,
                error_type="unknown",
                error_category="generation",
                chain_depth=event.chain_depth + 1,
                visibility=event.response_visibility or "all",
                parent_event_id=event.parent_event_id,
            )

        # Commit blocked events from chain depth enforcement atomically (RFC
        # §8.1 / §8.3 / §14.3 — blocked events are still indexed, and the commit
        # keeps the room counters in step with the timeline).
        for blocked in broadcast_result.blocked_events:
            await self._store.commit_event(room_id, blocked)
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

        # Pass streaming responses to caller (handled outside room lock
        # to avoid blocking concurrent process_inbound calls during TTS)
        if pending_streams_out is not None:
            pending_streams_out.extend(broadcast_result.streaming_responses)

        # Store reentry events and re-broadcast them (drain loop)
        # Stamp response_visibility from the trigger event onto reentry events'
        # *visibility* field.  The event router's _check_visibility() reads
        # `visibility`, not `response_visibility`, so propagating the scope
        # here avoids a separate codepath in the router.
        _reentry_vis = event.response_visibility
        if _reentry_vis:
            pending_reentries = deque(
                rev.model_copy(update={"visibility": _reentry_vis})
                for rev in broadcast_result.reentry_events
            )
        else:
            pending_reentries = deque(broadcast_result.reentry_events)
        max_reentries = self._max_chain_depth * 10
        reentry_count = 0
        reentry_tasks: list[Any] = []
        reentry_observations: list[Any] = []
        while pending_reentries:
            if reentry_count >= max_reentries:
                logger.warning(
                    "Reentry drain loop hit cap (%d iterations), storing %d remaining as BLOCKED",
                    max_reentries,
                    len(pending_reentries),
                    extra={"room_id": room_id},
                )
                for remaining in pending_reentries:
                    blocked_remaining = remaining.model_copy(
                        update={
                            "status": EventStatus.BLOCKED,
                            "blocked_by": "reentry_loop_cap",
                        }
                    )
                    await self._store.commit_event(room_id, blocked_remaining)
                break
            reentry_count += 1
            reentry = pending_reentries.popleft()

            # Tool call events are safe to broadcast — the AI channel's
            # self-loop guard skips events from its own channel_id.
            reentry_binding = await self._store.get_binding(room_id, reentry.source.channel_id)
            if reentry_binding is None:
                # No channel to broadcast to, but the response is still part of
                # the timeline: commit it atomically as DELIVERED (RFC §10.1
                # step 13) so it is indexed and counted like any other event.
                await self._commit_event(
                    room_id, reentry.model_copy(update={"status": EventStatus.DELIVERED})
                )
                continue

            # Provisional index for the hook, mirroring the main inbound path;
            # the authoritative index is (re)assigned atomically at commit.
            reentry = reentry.model_copy(
                update={"index": await self._store.get_event_count(room_id)}
            )
            # Append reentry event to context locally instead of full rebuild
            reentry_ctx = context.model_copy(
                update={
                    "recent_events": [
                        *context.recent_events[-(_RECENT_EVENTS_LIMIT - 1) :],
                        reentry,
                    ]
                }
            )

            # Run BEFORE_BROADCAST sync hooks on reentry events so that
            # orchestration routing (ConversationRouter) can stamp
            # _routed_to metadata and prevent AI-to-AI loops.
            reentry_sync = await self._hook_engine.run_sync_hooks(
                room_id, HookTrigger.BEFORE_BROADCAST, reentry, reentry_ctx
            )
            # Collect side effects even if the hook blocks this event
            reentry_tasks.extend(reentry_sync.tasks)
            reentry_observations.extend(reentry_sync.observations)
            if not reentry_sync.allowed:
                # RFC §9.5: commit BLOCKED (fresh insert — no pre-persist),
                # emit event_blocked, deliver InjectedEvents.
                await self._handle_block(
                    room_id=room_id,
                    event=reentry,
                    reason=reentry_sync.reason,
                    blocked_by=reentry_sync.blocked_by,
                    injected_events=reentry_sync.injected_events,
                    context=reentry_ctx,
                )
                continue
            reentry = reentry_sync.event or reentry
            # Commit point (RFC §10.1 step 13): assign the authoritative index,
            # store DELIVERED, and bump the room counters as ONE transaction —
            # the same atomic commit as the trigger event (step 12). Commit the
            # response BEFORE delivering any events its hook injected: the
            # response causes the injection, so it must take the lower index
            # (mirrors the main path, where the event commits before broadcast).
            reentry = await self._commit_event(
                room_id, reentry.model_copy(update={"status": EventStatus.DELIVERED})
            )
            # RFC §9.5: deliver InjectedEvents produced by an allow/modify hook
            # on this reentry event (same shape as the main inbound path).
            if reentry_sync.injected_events:
                await self._deliver_injected_events(
                    reentry_sync.injected_events, room_id, reentry_ctx
                )

            reentry_result = await router.broadcast(
                reentry,
                reentry_binding,
                reentry_ctx,
            )
            # Collect tasks/observations from reentry broadcast
            reentry_tasks.extend(reentry_result.tasks)
            reentry_observations.extend(reentry_result.observations)
            # Commit reentry's own blocked events atomically
            for blocked in reentry_result.blocked_events:
                await self._store.commit_event(room_id, blocked)
            # Queue nested reentry events for further broadcasting
            pending_reentries.extend(reentry_result.reentry_events)
            # AFTER_BROADCAST hooks for reentry events (e.g., AI responses)
            await self._dispatch_async_hooks(
                room_id,
                HookTrigger.AFTER_BROADCAST,
                reentry,
                reentry_ctx,
                pending_after_broadcast_out,
            )

        # Persist side effects from hooks and broadcast (including reentry)
        all_tasks = sync_result.tasks + broadcast_result.tasks + reentry_tasks
        all_observations = (
            sync_result.observations + broadcast_result.observations + reentry_observations
        )
        await self._persist_side_effects(
            room_id,
            all_tasks,
            all_observations,
            event,
            context,
        )

        # AFTER_BROADCAST async hooks (deferred to run outside the room lock)
        await self._dispatch_async_hooks(
            room_id, HookTrigger.AFTER_BROADCAST, event, context, pending_after_broadcast_out
        )

        # No separate room-state write here (RFC §10.1 step 15): every event —
        # the trigger, each reentry, and every blocked/injected event — bumped
        # the room counters atomically at its own commit, so the timeline and
        # the counters can never diverge (§14.3).
        await self._emit_framework_event("event_processed", room_id=room_id, event_id=event.id)

        return InboundResult(event=event)

    async def _handle_block(
        self,
        *,
        room_id: str,
        event: RoomEvent,
        reason: str | None,
        blocked_by: str | None,
        injected_events: list[InjectedEvent],
        context: RoomContext,
    ) -> RoomEvent:
        """RFC §9.5 block handling: commit the event as BLOCKED, emit the
        framework event, and deliver injected side effects. Shared by every
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
        blocked_event = await self._store.commit_event(room_id, blocked_event)
        await self._emit_framework_event(
            "event_blocked",
            room_id=room_id,
            event_id=blocked_event.id,
            data={
                "reason": reason,
                "blocked_by": blocked_by,
            },
        )
        await self._deliver_injected_events(injected_events, room_id, context)
        return blocked_event

    async def _blocked_result(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
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
        )
        await self._persist_side_effects(room_id, tasks, observations, blocked_event, context)
        return InboundResult(event=blocked_event, blocked=True, reason=reason)

    async def _dispatch_async_hooks(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: RoomEvent,
        context: RoomContext,
        deferred: _AsyncHookSink | None,
    ) -> None:
        """Run *trigger* async hooks, or defer them to *deferred* so the
        caller runs them once the room lock is released (RFC §10.1 — async
        hooks run after the lock, not under it; a slow hook must not hold the
        lock and block concurrent inbound processing for the room).
        """
        if deferred is not None:
            deferred.append((trigger, event, context))
        else:
            await self._hook_engine.run_async_hooks(room_id, trigger, event, context)

    async def _run_deferred_async_hooks(self, room_id: str, pending: _AsyncHookSink) -> None:
        """Run the async-hook firings collected during locked processing,
        called by inbound/direct-injection callers once the room lock is
        released (RFC §10.1). Order matches execution: mutation triggers and
        reentry events first, then the trigger event.
        """
        for trigger, ev, ctx in pending:
            await self._hook_engine.run_async_hooks(room_id, trigger, ev, ctx)

    async def _dispatch_error_hook(
        self,
        room_id: str,
        context: RoomContext,
        source: EventSource,
        deferred: _ErrorHookSink | None,
        **kwargs: Any,
    ) -> None:
        """Fire ON_ERROR now, or defer it to *deferred* so the caller runs it
        once the room lock is released (RFC §10.1 — like AFTER_BROADCAST). A
        provider failure must not hold the room lock while its ON_ERROR hooks
        run (up to the hook timeout) and stall the room's next messages.
        """
        if deferred is not None:
            deferred.append((context, source, kwargs))
        else:
            await self._fire_error_hook(room_id, context, source, **kwargs)

    async def _run_deferred_error_hooks(
        self,
        room_id: str,
        pending: _ErrorHookSink,
    ) -> None:
        """Run the ON_ERROR hooks collected during locked processing, called by
        inbound/direct-injection callers once the room lock is released.
        """
        for context, source, kwargs in pending:
            await self._fire_error_hook(room_id, context, source, **kwargs)

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
        """Store and deliver injected events to their target channels."""
        for injected in injected_events:
            # Commit the injected event atomically as DELIVERED (index + room
            # counters, RFC §8.1 / §14.3) — it is a real, delivered timeline
            # event, not a PENDING draft.
            stored = await self._store.commit_event(
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

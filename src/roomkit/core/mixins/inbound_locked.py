"""InboundLockedMixin — locked inbound processing, broadcast, and reentry."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime
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


class _Proceed:
    """Marker returned by ``_run_precommit`` once the event has committed and the
    caller should run the post-commit broadcast phase (RFC §10.1).

    Carries the state the broadcast phase needs — kept off the event so the
    pre-commit and post-commit phases stay decoupled.
    """

    __slots__ = ("context", "event", "source_binding", "sync_result")

    def __init__(
        self, event: RoomEvent, source_binding: Any, sync_result: Any, context: RoomContext
    ) -> None:
        self.event = event
        self.source_binding = source_binding
        self.sync_result = sync_result
        self.context = context


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

    async def _bump_room_counters(self, room_id: str, latest_index: int) -> None:
        """Refresh room counters to match stored events (RFC §14.3).

        Updates ``latest_index``, ``event_count`` and ``last_activity_at`` from
        the store's authoritative event count.
        """
        room = await self._store.get_room(room_id)
        if room is None:
            return
        updates: dict[str, object] = {
            "latest_index": latest_index,
            "event_count": await self._store.get_event_count(room_id),
        }
        if room.timers:
            updates["timers"] = room.timers.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )
        room = room.model_copy(update=updates)
        await self._store.update_room(room)

    async def _commit_event(self, room_id: str, event: RoomEvent) -> None:
        """Commit an event to the timeline (RFC §10.1).

        Persists the event as DELIVERED and bumps the room counters as one
        logical unit, so an observer never sees a DELIVERED event that the room
        counters do not reflect (§14.3).
        """
        await self._persist_event(event)
        await self._bump_room_counters(room_id, event.index)

    async def _process_locked(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
        pending_streams_out: list[Any] | None = None,
        pending_after_broadcast_out: list[tuple[RoomEvent, RoomContext]] | None = None,
    ) -> InboundResult:
        """Process an event under the room lock (RFC §10.1).

        Split at the commit point (RFC §10.1): the pre-commit critical
        section (:meth:`_run_precommit`) is bounded by ``process_timeout``
        (§13.6) and aborts before any durable write, while the post-commit
        broadcast (:meth:`_process_broadcast`) runs unbounded — so a committed
        event is never cancelled and the returned result never contradicts the
        stored timeline.

        AFTER_BROADCAST async hooks are collected into
        ``pending_after_broadcast_out`` (when provided) so the caller can run
        them once the room lock is released (RFC §10.1 — async hooks run after
        the lock, not under it). Without a sink they run inline.
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
        return await self._process_broadcast(
            outcome.event,
            room_id,
            outcome.source_binding,
            outcome.sync_result,
            outcome.context,
            pending_streams_out=pending_streams_out,
            pending_after_broadcast_out=pending_after_broadcast_out,
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
        if edit_delete_target is not None:
            await self._apply_edit_delete_state(event, edit_delete_target)

        # Commit point (RFC §10.1): persist DELIVERED and bump the room
        # counters atomically, before broadcast, so the timeline and counters
        # never diverge (§14.3) even if the post-commit phase is slow or times
        # out. Past this line the event is authoritative.
        event = event.model_copy(update={"status": EventStatus.DELIVERED})
        await self._commit_event(room_id, event)
        return _Proceed(event, source_binding, sync_result, context)

    async def _process_broadcast(
        self,
        event: RoomEvent,
        room_id: str,
        source_binding: Any,
        sync_result: Any,
        context: RoomContext,
        *,
        pending_streams_out: list[Any] | None = None,
        pending_after_broadcast_out: list[tuple[RoomEvent, RoomContext]] | None = None,
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
        for binding in context.bindings:
            if binding.category != ChannelCategory.INTELLIGENCE:
                continue
            error_msg = broadcast_result.errors.get(binding.channel_id)
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

        # Store blocked events from chain depth enforcement with an atomic,
        # monotonic index (RFC §8.1 / §8.3 — blocked events are still indexed).
        for blocked in broadcast_result.blocked_events:
            await self._store.add_event_auto_index(room_id, blocked)
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
                    await self._store.add_event_auto_index(room_id, blocked_remaining)
                break
            reentry_count += 1
            reentry = pending_reentries.popleft()
            stored = await self._persist_event_auto_index(room_id, reentry)
            if stored is not None:
                reentry = stored

            # Tool call events are safe to broadcast — the AI channel's
            # self-loop guard skips events from its own channel_id.
            reentry_binding = await self._store.get_binding(room_id, reentry.source.channel_id)
            if reentry_binding:
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
                    # RFC §9.5 block handling: update to BLOCKED, emit
                    # event_blocked, deliver InjectedEvents.
                    # ``update_existing=True`` because reentry events are
                    # pre-persisted in PENDING above (atomic indexing
                    # requires the row to exist before the hook fires).
                    await self._handle_block(
                        room_id=room_id,
                        event=reentry,
                        reason=reentry_sync.reason,
                        blocked_by=reentry_sync.blocked_by,
                        injected_events=reentry_sync.injected_events,
                        context=reentry_ctx,
                        update_existing=True,
                    )
                    continue
                reentry = reentry_sync.event or reentry
                # RFC §9.5: deliver InjectedEvents produced by an
                # allow/modify hook on this reentry event (same shape as
                # the main inbound path in ``_process_locked``).
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
                # Store reentry's blocked events with an atomic, monotonic index
                for blocked in reentry_result.blocked_events:
                    await self._store.add_event_auto_index(room_id, blocked)
                # Queue nested reentry events for further broadcasting
                pending_reentries.extend(reentry_result.reentry_events)
                # AFTER_BROADCAST hooks for reentry events (e.g., AI responses)
                await self._dispatch_after_broadcast(
                    room_id, reentry, reentry_ctx, pending_after_broadcast_out
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
        await self._dispatch_after_broadcast(room_id, event, context, pending_after_broadcast_out)

        # Post-commit counter refresh (RFC §10.1) — captures reentry
        # events added during broadcast; the source event was already reflected
        # at the commit point.
        await self._bump_room_counters(room_id, event.index)

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
        update_existing: bool,
    ) -> RoomEvent:
        """RFC §9.5 block handling: persist BLOCKED, emit framework event,
        deliver injected side effects. Shared by the hook-block paths (main
        inbound + reentry) and the source write-permission block so they
        cannot drift.

        ``update_existing=True`` when the event is already in the store
        (reentry path — events are pre-persisted in PENDING before the
        hook fires so they get an atomic index). The existing row is
        updated in place to avoid double-insert.

        ``update_existing=False`` when the event has not yet been stored
        (main inbound / permission path — the terminal status is decided
        before first storage).

        Returns the stored BLOCKED event so the caller can include it
        in its return value.
        """
        blocked_event = event.model_copy(
            update={
                "status": EventStatus.BLOCKED,
                "blocked_by": blocked_by or reason,
            }
        )
        if update_existing:
            await self._store.update_event(blocked_event)
        else:
            await self._store.add_event(blocked_event)

        await self._emit_framework_event(
            "event_blocked",
            room_id=room_id,
            event_id=event.id,
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
            update_existing=False,
        )
        await self._persist_side_effects(room_id, tasks, observations, blocked_event, context)
        return InboundResult(event=blocked_event, blocked=True, reason=reason)

    async def _dispatch_after_broadcast(
        self,
        room_id: str,
        event: RoomEvent,
        context: RoomContext,
        deferred: list[tuple[RoomEvent, RoomContext]] | None,
    ) -> None:
        """Run AFTER_BROADCAST async hooks, or defer them to *deferred* so the
        caller runs them once the room lock is released (RFC §10.1 — async
        hooks run after the lock, not under it; a slow hook must not hold the
        lock and block concurrent inbound processing for the room).
        """
        if deferred is not None:
            deferred.append((event, context))
        else:
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.AFTER_BROADCAST, event, context
            )

    async def _run_deferred_after_broadcast(
        self, room_id: str, pending: list[tuple[RoomEvent, RoomContext]]
    ) -> None:
        """Run the AFTER_BROADCAST hooks collected during locked processing,
        called by inbound/direct-injection callers once the room lock is
        released (RFC §10.1). Order matches execution: reentry events first,
        then the trigger event.
        """
        for ev, ctx in pending:
            await self._hook_engine.run_async_hooks(room_id, HookTrigger.AFTER_BROADCAST, ev, ctx)

    async def _apply_edit_delete_state(self, event: RoomEvent, target_event: RoomEvent) -> None:
        """Apply RFC §10.3 state updates to an edit/delete target event.

        Invoked only after the edit/delete event has passed authorization
        *and* been allowed by BEFORE_BROADCAST hooks, so a blocked
        edit/delete never mutates the target. Uses the final (post-modify)
        event content so a ``modify`` hook on the edit is honored.
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
        elif isinstance(content, DeleteContent):
            updated = target_event.model_copy(
                update={"metadata": {**target_event.metadata, "deleted": True}}
            )
            await self._store.update_event(updated)

    async def _deliver_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: RoomContext,
    ) -> None:
        """Store and deliver injected events to their target channels."""
        for injected in injected_events:
            # Store the injected event with an atomic, monotonic index (RFC §8.1)
            stored = await self._store.add_event_auto_index(room_id, injected.event)

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

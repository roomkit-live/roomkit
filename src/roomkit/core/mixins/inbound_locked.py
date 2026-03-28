"""InboundLockedMixin — locked inbound processing, broadcast, and reentry."""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    DeleteType,
    EventStatus,
    EventType,
    HookTrigger,
)
from roomkit.models.event import DeleteContent, EditContent, RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter
    from roomkit.core.hooks import HookEngine
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


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

    # Stub for cross-mixin call — implemented by RoomKit._get_router().
    def _get_router(self) -> EventRouter: ...

    async def _process_locked(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
        pending_streams_out: list[Any] | None = None,
    ) -> InboundResult:
        """Process an event under the room lock."""
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

        # Edit/Delete validation and state updates (RFC §10.3)
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

            # Apply state updates to the target event
            if isinstance(event.content, EditContent):
                updated_target = target_event.model_copy(
                    update={
                        "content": event.content.new_content,
                        "metadata": {**target_event.metadata, "edited": True},
                    }
                )
                await self._store.update_event(updated_target)
            elif isinstance(event.content, DeleteContent):
                updated_target = target_event.model_copy(
                    update={
                        "metadata": {**target_event.metadata, "deleted": True},
                    }
                )
                await self._store.update_event(updated_target)

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
            # RFC §4.2: Store original event as BLOCKED with audit trail
            blocked_event = event.model_copy(
                update={
                    "status": EventStatus.BLOCKED,
                    "blocked_by": sync_result.blocked_by or sync_result.reason,
                }
            )
            await self._store.add_event(blocked_event)

            await self._emit_framework_event(
                "event_blocked",
                room_id=room_id,
                event_id=event.id,
                data={
                    "reason": sync_result.reason,
                    "blocked_by": sync_result.blocked_by,
                },
            )

            # RFC §4.2: Deliver injected events to their target channels
            await self._deliver_injected_events(sync_result.injected_events, room_id, context)

            # Persist side effects from hooks even on blocked path
            await self._persist_side_effects(
                room_id,
                sync_result.tasks,
                sync_result.observations,
                blocked_event,
                context,
            )

            return InboundResult(event=blocked_event, blocked=True, reason=sync_result.reason)

        # Use potentially modified event
        event = sync_result.event or event

        # Store event as DELIVERED
        event = event.model_copy(update={"status": EventStatus.DELIVERED})
        await self._store.add_event(event)

        # Deliver any injected events from allow/modify hooks
        if sync_result.injected_events:
            await self._deliver_injected_events(sync_result.injected_events, room_id, context)

        # Get source binding for broadcast
        source_binding = await self._store.get_binding(room_id, event.source.channel_id)
        if source_binding is None:
            return InboundResult(event=event)

        # Refresh context locally by appending the new event (avoids 4 store queries)
        context = context.model_copy(
            update={"recent_events": [*context.recent_events[-49:], event]}
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

        # Store blocked events from chain depth enforcement
        for blocked in broadcast_result.blocked_events:
            await self._store.add_event(blocked)
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
            reentry = await self._store.add_event_auto_index(room_id, reentry)
            reentry_binding = await self._store.get_binding(room_id, reentry.source.channel_id)
            if reentry_binding:
                # Append reentry event to context locally instead of full rebuild
                reentry_ctx = context.model_copy(
                    update={"recent_events": [*context.recent_events[-49:], reentry]}
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
                    # Hook blocked this reentry event — skip broadcast
                    continue
                reentry = reentry_sync.event or reentry

                reentry_result = await router.broadcast(
                    reentry,
                    reentry_binding,
                    reentry_ctx,
                )
                # Collect tasks/observations from reentry broadcast
                reentry_tasks.extend(reentry_result.tasks)
                reentry_observations.extend(reentry_result.observations)
                # Store reentry's blocked events
                for blocked in reentry_result.blocked_events:
                    await self._store.add_event(blocked)
                # Queue nested reentry events for further broadcasting
                pending_reentries.extend(reentry_result.reentry_events)
                # Run AFTER_BROADCAST hooks for reentry events (e.g., AI responses)
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.AFTER_BROADCAST, reentry, reentry_ctx
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

        # Run async hooks (after_broadcast)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.AFTER_BROADCAST, event, context
        )

        # Update room state per RFC §3.5 step 15
        room = await self._store.get_room(room_id)
        if room is not None:
            updates: dict[str, object] = {
                "latest_index": event.index,
                "event_count": await self._store.get_event_count(room_id),
            }
            if room.timers:
                updates["timers"] = room.timers.model_copy(
                    update={"last_activity_at": datetime.now(UTC)}
                )
            room = room.model_copy(update=updates)
            await self._store.update_room(room)

        await self._emit_framework_event("event_processed", room_id=room_id, event_id=event.id)

        return InboundResult(event=event)

    async def _deliver_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        context: RoomContext,
    ) -> None:
        """Store and deliver injected events to their target channels."""
        for injected in injected_events:
            # Store the injected event
            await self._store.add_event(injected.event)

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
                        await channel.on_event(injected.event, binding, context)
                        if binding.category == ChannelCategory.TRANSPORT:
                            await channel.deliver(injected.event, binding, context)
                    except Exception:
                        logger.exception(
                            "Failed to deliver injected event to %s",
                            target_id,
                            extra={"room_id": room_id, "channel_id": target_id},
                        )

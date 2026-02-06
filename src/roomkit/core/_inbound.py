"""InboundMixin — inbound message processing pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from roomkit.core._helpers import HelpersMixin
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage, InboundResult
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    DeleteType,
    EventStatus,
    EventType,
    HookTrigger,
    IdentificationStatus,
)
from roomkit.models.event import DeleteContent, EditContent, RoomEvent
from roomkit.models.hook import InjectedEvent
from roomkit.models.identity import Identity, IdentityResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.inbound_router import InboundRoomRouter
    from roomkit.core.locks import RoomLockManager
    from roomkit.identity.base import IdentityResolver
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


class InboundMixin(HelpersMixin):
    """Inbound message processing pipeline."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
    _identity_resolver: IdentityResolver | None
    _identity_channel_types: set[ChannelType] | None
    _identity_timeout: float
    _process_timeout: float
    _inbound_router: InboundRoomRouter
    _max_chain_depth: int

    async def process_inbound(
        self, message: InboundMessage, *, room_id: str | None = None
    ) -> InboundResult:
        """Process an inbound message through the full pipeline.

        Args:
            message: The inbound message to process.
            room_id: Explicit room to route to, bypassing the inbound router.
                Useful for shared channels attached to multiple rooms.
        """
        from roomkit.core.framework import ChannelNotRegisteredError

        channel = self._channels.get(message.channel_id)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel {message.channel_id} not registered")

        # Route to room (or auto-create)
        if room_id is None:
            room_id = await self._inbound_router.route(
                channel_id=message.channel_id,
                channel_type=channel.channel_type,
                participant_id=message.sender_id,
            )
        if room_id is None:
            # Auto-create room and attach channel
            room = await self.create_room()  # type: ignore[attr-defined]
            room_id = room.id
            await self.attach_channel(room_id, message.channel_id)  # type: ignore[attr-defined]

        context = await self._build_context(room_id)

        # Let channel process inbound
        event = await channel.handle_inbound(message, context)

        # Identity resolution pipeline (RFC §7)
        # Skip if channel type not in identity_channel_types filter (when set)
        resolver = self._identity_resolver
        should_resolve = resolver is not None and (
            self._identity_channel_types is None
            or channel.channel_type in self._identity_channel_types
        )
        if should_resolve and resolver is not None:
            try:
                id_result = await asyncio.wait_for(
                    resolver.resolve(message, context),
                    timeout=self._identity_timeout,
                )
            except TimeoutError:
                logger.warning(
                    "Identity resolution timed out after %.1fs",
                    self._identity_timeout,
                    extra={"room_id": room_id, "channel_id": message.channel_id},
                )
                await self._emit_framework_event(
                    "identity_timeout",
                    room_id=room_id,
                    channel_id=message.channel_id,
                    data={"timeout": self._identity_timeout},
                )
                id_result = IdentityResult(status=IdentificationStatus.UNKNOWN)

            # Backfill address and channel_type from the message if not set by resolver
            # This ensures identity hooks always have access to sender info
            updates = {}
            if id_result.address is None:
                updates["address"] = message.sender_id
            if id_result.channel_type is None:
                updates["channel_type"] = str(channel.channel_type)
            if updates:
                id_result = id_result.model_copy(update=updates)

            # Deferred participant creation — capture intent only; persist
            # inside _process_locked to avoid race on event index (Fix #1).
            resolved_identity: Identity | None = None
            pending_id_result: IdentityResult | None = None

            if id_result.status == IdentificationStatus.IDENTIFIED and id_result.identity:
                # Known identity — stamp participant_id; persist later
                event = event.model_copy(
                    update={
                        "source": event.source.model_copy(
                            update={"participant_id": id_result.identity.id}
                        )
                    }
                )
                resolved_identity = id_result.identity

            elif id_result.status in (
                IdentificationStatus.AMBIGUOUS,
                IdentificationStatus.PENDING,
            ):
                # Multiple candidates or pending — run identity-specific hooks
                hook_result = await self._run_identity_hooks(
                    room_id, HookTrigger.ON_IDENTITY_AMBIGUOUS, event, context, id_result
                )
                # Also fire regular async hooks for observation/logging
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.ON_IDENTITY_AMBIGUOUS, event, context
                )
                if (
                    hook_result
                    and hook_result.status == IdentificationStatus.IDENTIFIED
                    and hook_result.identity
                ):
                    event = event.model_copy(
                        update={
                            "source": event.source.model_copy(
                                update={"participant_id": hook_result.identity.id}
                            )
                        }
                    )
                    resolved_identity = hook_result.identity
                elif hook_result and hook_result.status == IdentificationStatus.CHALLENGE_SENT:
                    if hook_result.inject:
                        await self._deliver_injected_events([hook_result.inject], room_id, context)
                    return InboundResult(blocked=True, reason="identity_challenge_sent")
                elif hook_result and hook_result.status == IdentificationStatus.REJECTED:
                    return InboundResult(
                        blocked=True,
                        reason=hook_result.reason or "identity_rejected",
                    )
                else:
                    # No hook resolved it — mark for pending creation
                    pending_id_result = id_result

            elif id_result.status in (
                IdentificationStatus.UNKNOWN,
                IdentificationStatus.REJECTED,
            ):
                # No match or rejected — run identity-specific hooks
                hook_result = await self._run_identity_hooks(
                    room_id, HookTrigger.ON_IDENTITY_UNKNOWN, event, context, id_result
                )
                # Also fire regular async hooks for observation/logging
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.ON_IDENTITY_UNKNOWN, event, context
                )
                if hook_result and hook_result.status == IdentificationStatus.REJECTED:
                    return InboundResult(
                        blocked=True,
                        reason=hook_result.reason or "unknown_sender",
                    )
                elif (
                    hook_result
                    and hook_result.status == IdentificationStatus.IDENTIFIED
                    and hook_result.identity
                ):
                    event = event.model_copy(
                        update={
                            "source": event.source.model_copy(
                                update={"participant_id": hook_result.identity.id}
                            )
                        }
                    )
                    resolved_identity = hook_result.identity

        if not should_resolve or resolver is None:
            resolved_identity = None
            pending_id_result = None

        # Process under room lock
        async with self._lock_manager.locked(room_id):
            try:
                return await asyncio.wait_for(
                    self._process_locked(
                        event, room_id, context,
                        resolved_identity=resolved_identity,
                        pending_id_result=pending_id_result,
                    ),
                    timeout=self._process_timeout,
                )
            except TimeoutError:
                logger.error(
                    "Process locked timed out after %.1fs",
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

    async def _process_locked(
        self,
        event: RoomEvent,
        room_id: str,
        context: RoomContext,
        *,
        resolved_identity: Identity | None = None,
        pending_id_result: IdentityResult | None = None,
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
                return InboundResult(
                    blocked=True, reason="target_event_not_found"
                )

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
        context = context.model_copy(update={
            "recent_events": [*context.recent_events[-49:], event]
        })

        # Broadcast to other channels
        router = self._get_router()  # type: ignore[attr-defined]
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

        # Store reentry events and re-broadcast them (drain loop)
        pending_reentries = deque(broadcast_result.reentry_events)
        max_reentries = self._max_chain_depth * 10
        reentry_count = 0
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
                    await self._store.add_event(blocked_remaining)
                break
            reentry_count += 1
            reentry = pending_reentries.popleft()
            await self._store.add_event(reentry)
            reentry_binding = await self._store.get_binding(room_id, reentry.source.channel_id)
            if reentry_binding:
                # Append reentry event to context locally instead of full rebuild
                reentry_ctx = context.model_copy(update={
                    "recent_events": [*context.recent_events[-49:], reentry]
                })
                reentry_result = await router.broadcast(reentry, reentry_binding, reentry_ctx)
                # Store reentry's blocked events
                for blocked in reentry_result.blocked_events:
                    await self._store.add_event(blocked)
                # Queue nested reentry events for further broadcasting
                pending_reentries.extend(reentry_result.reentry_events)
                # Run AFTER_BROADCAST hooks for reentry events (e.g., AI responses)
                await self._hook_engine.run_async_hooks(
                    room_id, HookTrigger.AFTER_BROADCAST, reentry, reentry_ctx
                )

        # Persist side effects from hooks and broadcast
        all_tasks = sync_result.tasks + broadcast_result.tasks
        all_observations = sync_result.observations + broadcast_result.observations
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
                "event_count": room.event_count + 1,
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

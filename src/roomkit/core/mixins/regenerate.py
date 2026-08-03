"""RegenerateMixin — re-run the intelligence channel on the last inbound message."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import ChannelCategory, EventStatus
from roomkit.models.event import EventSource, RoomEvent

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class RegenerateHost(Protocol):
    """Contract: capabilities a host class must provide for RegenerateMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _lock_manager: Per-room lock for serialised mutation.
        _process_timeout: Timeout in seconds for locked processing.

    Cross-mixin methods (provided by other mixins in the MRO):
        _get_router: From :class:`InboundLockedMixin`.
        _commit_and_deliver: From :class:`LaneExecutionMixin`.
        _process_streaming_responses: From :class:`InboundStreamingMixin`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _process_timeout: float


class RegenerateMixin(HelpersMixin):
    """Adds ``regenerate_response()`` to RoomKit.

    Host contract: :class:`RegenerateHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _process_timeout: float

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # see RegenerateHost
    _commit_and_deliver: Any  # see RegenerateHost
    _process_streaming_responses: Any  # see RegenerateHost

    async def regenerate_response(self, room_id: str) -> InboundResult | None:
        """Re-run the room's intelligence channel on the last inbound message.

        Produces a fresh response to the most recent transport (human) message
        *without* ingesting a new inbound event — the triggering message keeps
        its identity, index, and timestamp. The existing broadcast + streaming
        pipeline is reused, so the new response is persisted, streamed, and runs
        its AFTER_BROADCAST hooks exactly like a first-time turn. The trigger
        message's own hooks are not re-run.

        Replacement semantics are the caller's concern: any responses already
        present after the last inbound message should be removed *before* calling
        this (the method only generates — it does not delete the prior answer).

        Returns the :class:`InboundResult` for the regenerated turn, or ``None``
        when there is no inbound message to regenerate (no transport message, or
        its source binding can no longer write).

        The re-broadcast is scoped to ``visibility="intelligence"`` so only the
        agent reacts — transports never receive the user message again (no
        duplicate bubble, no echo to other participants). Targets the single
        intelligence-channel path; orchestrated rooms (routing installed as
        BEFORE_BROADCAST hooks) are not re-routed here.
        """
        pending_streams: list[Any] = []
        regenerated: list[RoomEvent] = []
        trigger: RoomEvent | None = None
        broadcast_error: Exception | None = None
        error_source: EventSource | None = None

        async with self._lock_manager.locked(room_id):
            context = await self._build_context(room_id)

            transport_ids = {
                b.channel_id for b in context.bindings if b.category == ChannelCategory.TRANSPORT
            }
            trigger = next(
                (
                    e
                    for e in reversed(context.recent_events)
                    if e.source.channel_id in transport_ids
                ),
                None,
            )
            if trigger is None:
                return None

            source_binding = await self._store.get_binding(room_id, trigger.source.channel_id)
            if source_binding is None or not source_binding.can_write:
                return None

            # Scope the re-broadcast to intelligence channels: only the agent
            # regenerates, no transport re-delivery of the user's message.
            intel_trigger = trigger.model_copy(update={"visibility": "intelligence"})

            router = self._get_router()
            broadcast_result = await asyncio.wait_for(
                router.broadcast(intel_trigger, source_binding, context),
                timeout=self._process_timeout,
            )

            pending_streams.extend(broadcast_result.streaming_responses)
            # A non-streaming intelligence failure surfaces as a broadcast error.
            # Capture it (reported on InboundResult.error) and its source so an
            # ON_ERROR card fires after the lock — parity with _process_broadcast.
            # The streaming path fires its own ON_ERROR via
            # _process_streaming_responses below, so the two never double up.
            for b in context.bindings:
                if b.category != ChannelCategory.INTELLIGENCE:
                    continue
                exc = broadcast_result.errors_exc.get(b.channel_id)
                if exc is not None:
                    broadcast_error = exc
                    error_source = EventSource(
                        channel_id=b.channel_id, channel_type=b.channel_type
                    )
                    break

            # Non-streaming providers return the response as reentry events.
            # They commit and deliver after the lock (below): the delivery
            # cursor must not reach a regenerated answer before the room's
            # lane has actually delivered it (RFC §10.2).
            regenerated = [
                r.model_copy(update={"status": EventStatus.DELIVERED})
                for r in broadcast_result.reentry_events
            ]

        # Outside the room lock (RFC §10.1): the regenerated answers reach
        # transports through the room's delivery lane — which also fires their
        # AFTER_BROADCAST hooks once each delivery set completes (step 16) —
        # then streaming delivery (which can take seconds).
        for reentry in regenerated:
            await self._commit_and_deliver(room_id, reentry, reentry.source.channel_id)
        # A non-streaming regeneration failure fires ON_ERROR here (the streaming
        # path fires its own inside _process_streaming_responses), so the host
        # renders an error card for a failed regenerate on either path.
        if broadcast_error is not None and error_source is not None and trigger is not None:
            await self._fire_error_hook(
                room_id,
                context,
                error_source,
                error=str(broadcast_error),
                error_type=type(broadcast_error).__name__,
                error_category="generation",
                chain_depth=trigger.chain_depth + 1,
                visibility=trigger.response_visibility or "all",
                parent_event_id=trigger.parent_event_id,
            )
        stream_error: Exception | None = None
        if pending_streams:
            stream_error = await self._process_streaming_responses(pending_streams, room_id)

        return InboundResult(event=trigger, error=stream_error or broadcast_error)

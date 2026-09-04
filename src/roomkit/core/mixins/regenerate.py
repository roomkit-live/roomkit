"""RegenerateMixin — re-run the intelligence channel on the last inbound message."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import RoomClosedError
from roomkit.core.mixins.helpers import _REFUSING_STATUSES, HelpersMixin
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import ChannelCategory, EventStatus
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.response_metadata import ResponseMetadata

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.store.base import ConversationStore


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
    """Adds ``regenerate_response()`` and ``regenerate_target()`` to RoomKit.

    Host contract: :class:`RegenerateHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _process_timeout: float

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # see RegenerateHost
    _commit_and_deliver: Any  # see RegenerateHost
    _process_streaming_responses: Any  # see RegenerateHost

    async def regenerate_target(self, room_id: str) -> RoomEvent | None:
        """The event :meth:`regenerate_response` would re-run the agent on.

        The primitive's own choice, from the primitive's own read: the newest
        message a transport binding wrote and the room accepted (a message a
        hook blocked is never replayed), in the history window the room's
        channels derive, whose source binding can still write. A host that
        must act on the trigger *before* regenerating — delete the answer the
        new one replaces, refuse a trigger it recognises as a runner's prompt
        rather than a person's question — asks here instead of re-implementing
        the selection off a window of its own.

        Returns ``None`` when a regenerate would do nothing: no transport
        message in the window, or its source binding can no longer write.

        Raises :class:`RoomClosedError` when the room's status refuses new
        events (RFC §5.1): :meth:`regenerate_response` would refuse, and an
        accessor that returns an event has no way to hand back that refusal —
        the same reasoning as :meth:`send_event`. Raises
        :class:`RoomNotFoundError` for an unknown room.

        A read, taken outside the room lock, so the answer can be stale by
        the time a regenerate acts on it — the caveat of any answer taken
        before the lock (RFC §10.1 step 6): the regenerating call re-selects
        under the lock, and a message that lands in between becomes the
        trigger there.
        """
        context, found = await self._regenerate_target(room_id)
        if context.room.status in _REFUSING_STATUSES:
            raise RoomClosedError(f"Room {room_id} does not accept new events")
        return found[0] if found is not None else None

    async def _regenerate_target(
        self, room_id: str
    ) -> tuple[RoomContext, tuple[RoomEvent, ChannelBinding] | None]:
        """The event a regenerate re-runs the agent on, with the binding that
        wrote it, and the context it was found in.

        One selection for both readers — :meth:`regenerate_response` under the
        lock and :meth:`regenerate_target` outside it — so a host asking for
        the trigger sees the primitive's own choice, same window and same
        predicate, rather than a copy that drifts: the newest message of the
        room's recent history written by a TRANSPORT binding and accepted by
        the room, provided that binding can still write (a muted or read-only
        source has no turn to regenerate). The window is the one the room's
        channels derive, floored because this caller scans the tail itself
        (``reads_history``).

        Returns ``(context, None)`` when nothing qualifies. The context comes
        back because the status gate reads it, and because building it is the
        expensive half of the call.
        """
        context = await self._build_context(room_id, reads_history=True)
        transports = {
            b.channel_id: b for b in context.bindings if b.category == ChannelCategory.TRANSPORT
        }
        # A BLOCKED message is stored, never broadcast (RFC §10.1 step 10):
        # a hook refused it, or its source could not write. The room never
        # answered it, so a regenerate does not answer it either.
        trigger = next(
            (
                e
                for e in reversed(context.recent_events)
                if e.source.channel_id in transports and e.status != EventStatus.BLOCKED
            ),
            None,
        )
        if trigger is None:
            return context, None
        source_binding = transports[trigger.source.channel_id]
        if not source_binding.can_write:
            return context, None
        return context, (trigger, source_binding)

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
        :meth:`regenerate_target` names the message this call would re-run on,
        so the caller can key that removal on it.

        Returns the :class:`InboundResult` for the regenerated turn, or ``None``
        when there is no inbound message to regenerate (no transport message, or
        its source binding can no longer write). A room whose status refuses
        new events (RFC §5.1) is refused *before* the agent runs, with
        ``InboundResult(blocked=True, reason="room_closed")`` and a
        ``room_refused_event`` framework event — exactly as
        :meth:`process_inbound` refuses — rather than after a generation whose
        answer nothing could commit.

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
            context, found = await self._regenerate_target(room_id)

            # RFC §5.1 — the regenerated answer would be refused at commit, so
            # refuse here, before the agent runs: a closed room must not cost
            # a generation (tools, tokens) for an answer nothing can persist.
            # Under the lock for the same reason as the inbound gate (§10.1
            # step 6): close_room() takes it, and a status read before it can
            # be stale by the time the turn would commit. Nothing is written.
            if context.room.status in _REFUSING_STATUSES:
                return await self._refuse_closed_room(
                    room_id,
                    status=context.room.status,
                    operation="regenerate",
                    event=found[0] if found is not None else None,
                )

            if found is None:
                return None
            trigger, source_binding = found

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
        record = ResponseMetadata()
        for output in broadcast_result.outputs.values():
            if output.response_stream is None:
                record.update(output.response_metadata)
        stream_error: Exception | None = None
        if pending_streams:
            stream_error, stream_record = await self._process_streaming_responses(
                pending_streams, room_id
            )
            record.update(stream_record)

        return InboundResult(
            event=trigger,
            error=stream_error or broadcast_error,
            response_metadata=record,
        )

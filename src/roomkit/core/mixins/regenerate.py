"""RegenerateMixin — re-run the intelligence channel on the last inbound message."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import RoomClosedError
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins.helpers import _REFUSING_STATUSES, HelpersMixin
from roomkit.core.mixins.lane_execution import record_buffered_reply
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import ChannelCategory, EventStatus
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import merge_caller_record

if TYPE_CHECKING:
    from roomkit.core.event_router import BroadcastResult
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class RegenerateHost(Protocol):
    """Contract: capabilities a host class must provide for RegenerateMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _lock_manager: Per-room lock for serialised mutation.
        _process_timeout: Timeout in seconds for locked processing.
        _max_chain_depth: Chain depth ceiling, which sizes the reentry budget.

    Cross-mixin methods (provided by other mixins in the MRO):
        _get_router: From :class:`InboundLockedMixin`.
        _commit_responses: From :class:`LaneExecutionMixin`.
        _commit_blocked_events: From :class:`LaneExecutionMixin`.
        _finish_cascade: From :class:`LaneExecutionMixin`.
        _report_intelligence_errors: From :class:`LaneExecutionMixin`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _process_timeout: float
    _max_chain_depth: int


class RegenerateMixin(HelpersMixin):
    """Adds ``regenerate_response()`` and ``regenerate_target()`` to RoomKit.

    Host contract: :class:`RegenerateHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _process_timeout: float
    _max_chain_depth: int

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # see RegenerateHost
    _commit_responses: Any  # see RegenerateHost
    _commit_blocked_events: Any  # see RegenerateHost
    _finish_cascade: Any  # see RegenerateHost
    _report_intelligence_errors: Any  # see RegenerateHost

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
        trigger there. A host that must not answer that message twice hands
        this event's id back as ``regenerate_response(trigger_id=...)``,
        which refuses a selection that moved instead of regenerating it.
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
        # answered it, so a regenerate does not answer it either. A per-event
        # check on a reversed scan; the list-building form of the same
        # predicate is ``received_events``.
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

    async def regenerate_response(
        self, room_id: str, *, trigger_id: str | None = None
    ) -> InboundResult | None:
        """Re-run the room's intelligence channel on the last inbound message.

        Produces a fresh response to the most recent transport (human) message
        *without* ingesting a new inbound event — the triggering message keeps
        its identity, index, and timestamp. The existing broadcast + streaming
        pipeline is reused, so the new response re-enters like a first-time
        turn's (RFC §10.1 step 14): its BEFORE_BROADCAST hooks run, its
        source's right to write is checked, it counts against the reentry
        budget, it is persisted, streamed, and runs its AFTER_BROADCAST hooks,
        and it comes back in ``response_events``. The trigger message's own
        hooks are not re-run.

        Replacement semantics are the caller's concern: any responses already
        present after the last inbound message should be removed *before* calling
        this (the method only generates — it does not delete the prior answer).
        :meth:`regenerate_target` names the message this call would re-run on,
        so the caller can key that removal on it.

        ``trigger_id`` makes the call a compare-and-regenerate. The target is
        read outside the room lock, so a message can land between that read
        and this call: the pipeline has answered it already, and a regenerate
        that re-selects under the lock would answer it a second time, with
        the earlier answer gone. Naming the trigger closes that window: when
        the selection under the lock is no longer that event — another
        message moved in, or nothing qualifies any more — the call returns
        ``InboundResult(blocked=True, reason="trigger_moved")`` without
        running the agent and without writing anything. ``None`` (the
        default) regenerates whatever the selection is. A closed room is
        refused first: ``room_closed`` wins over ``trigger_moved``. The
        compare is on the trigger, not on the answers that follow it: two
        concurrent calls naming the same trigger both pass it, and an answer
        landing between them does not move a transport selection.

        Returns the :class:`InboundResult` for the regenerated turn, or ``None``
        when there is no inbound message to regenerate (no transport message, or
        its source binding can no longer write) — with ``trigger_id`` set, that
        same state is a moved trigger and comes back blocked instead of
        ``None``. A room whose status refuses
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

            # The compare half of a compare-and-regenerate: the host prepared
            # for one trigger, and the selection under the lock is the truth.
            if trigger_id is not None and (found is None or found[0].id != trigger_id):
                logger.info(
                    "Regenerate refused: trigger %s is no longer the selection of room %s",
                    trigger_id,
                    room_id,
                    extra={"room_id": room_id, "event_id": trigger_id},
                )
                return InboundResult(blocked=True, reason="trigger_moved")

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
            # A non-streaming intelligence failure surfaces as a broadcast error:
            # the first one is the caller's (InboundResult.error), as on the
            # inbound path; each fires its ON_ERROR after the lock (below).
            broadcast_error = self._first_intelligence_error(broadcast_result, context)

            # Non-streaming providers return the response as reentry events.
            # Each takes its own commit pass after the lock (below), as any
            # response does (RFC §10.1 step 14): the delivery cursor must not
            # reach a regenerated answer before the room's lane has actually
            # delivered it (RFC §10.2).
            regenerated = list(broadcast_result.reentry_events)

        return await self._finish_regeneration(
            room_id,
            context,
            trigger,
            broadcast_result,
            regenerated,
            pending_streams,
            broadcast_error=broadcast_error,
        )

    async def _finish_regeneration(
        self,
        room_id: str,
        context: RoomContext,
        trigger: RoomEvent,
        broadcast_result: BroadcastResult,
        regenerated: list[RoomEvent],
        pending_streams: list[Any],
        *,
        broadcast_error: Exception | None,
    ) -> InboundResult:
        """Deliver what a regeneration produced, off the room lock, and report it."""
        # Outside the room lock (RFC §10.1): the regenerated answers reach
        # transports through the room's delivery lane — which also fires their
        # AFTER_BROADCAST hooks once each delivery set completes (step 16) —
        # then streaming delivery (which can take seconds). One cascade for
        # the whole regeneration: what the other agents answer to the new
        # answer is read with it (RFC §8.3), the regenerated stream first.
        cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        cascade.add_streams(pending_streams)
        # What the re-broadcast blocked is stored and announced, and its side
        # effects kept, as the inbound path does (RFC §8.3).
        await self._commit_blocked_events(room_id, broadcast_result)
        await self._persist_side_effects(
            room_id, broadcast_result.tasks, broadcast_result.observations, trigger, context
        )
        await self._commit_responses(room_id, regenerated, trigger.response_visibility, cascade)
        # Each non-streaming failure fires ON_ERROR here, as on the inbound path
        # (the streaming path fires its own while its stream is read), so the
        # host renders an error card per failed agent on either path.
        await self._report_intelligence_errors(trigger, context, broadcast_result)
        # The buffered failure is the cascade's first, as on the inbound path
        # (RFC §10.1 step 18): a stream's failure is the caller's only after it.
        if broadcast_error is not None:
            cascade.record_error(broadcast_error)
        # Each buffered reply's record read as the inbound path reads it: its
        # end under ``turns``, a ``turns`` key its hooks wrote left out (RFC §6.4).
        for channel_id, output in broadcast_result.outputs.items():
            if output.response_stream is None:
                record_buffered_reply(cascade, channel_id, output, root=True)
        stream_error, stream_meta = await self._finish_cascade(cascade, room_id, caller_logs=True)

        result = InboundResult(event=trigger)
        result.report_cascade(cascade)
        if stream_error is not None and result.error is None:
            result.error = stream_error
        merge_caller_record(result.response_metadata, stream_meta)
        return result

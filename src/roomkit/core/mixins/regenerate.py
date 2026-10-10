"""RegenerateMixin — re-run the intelligence channel on the last inbound message."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.event_router import CHAIN_DEPTH_LIMIT
from roomkit.core.exceptions import RoomClosedError
from roomkit.core.lanes import DeliveryCascade, DeliveryPlan
from roomkit.core.mixins.helpers import _REFUSING_STATUSES, HelpersMixin
from roomkit.models.delivery import InboundResult
from roomkit.models.enums import ChannelCategory, EventStatus
from roomkit.models.event import RoomEvent
from roomkit.models.response_metadata import merge_caller_record

if TYPE_CHECKING:
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
        _process_timeout: Bound on the wait for the room lock and the choice
            of the event to replay (RFC §13.6).
        _max_chain_depth: Chain depth ceiling, which sizes the reentry budget.

    Cross-mixin methods (provided by other mixins in the MRO):
        _get_router: From :class:`InboundLockedMixin`.
        _enqueue_exec: From :class:`LaneExecutionMixin`.
        _finish_cascade: From :class:`LaneExecutionMixin`.
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
    _discussions: dict[str, Any]  # rooms a discussion holds (RFC §19.7.5)

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # see RegenerateHost
    _enqueue_exec: Any  # see RegenerateHost
    _finish_cascade: Any  # see RegenerateHost

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
        its identity, index, and timestamp. The re-broadcast runs in the
        room's delivery lane, off the room lock and unbounded, as an inbound
        event's broadcast does (RFC §13.5, §13.6): a strategy that works in
        ``on_event`` (a Loop, a Supervisor's delegation) goes to its end, and
        the room takes messages meanwhile. The new response re-enters like a
        first-time turn's (RFC §10.1 step 14): its BEFORE_BROADCAST hooks run,
        its source's right to write is checked, it counts against the reentry
        budget, it is persisted, streamed, and runs its AFTER_BROADCAST hooks,
        and it comes back in ``response_events``. The trigger message's own
        hooks are not re-run, nor its delivery report or ``event_processed``.

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
        ``None``. ``process_timeout`` bounds only the wait for the room lock
        and the choice of the trigger: past it, the call returns
        ``InboundResult(blocked=True, reason="process_timeout")`` without
        running the agent. The call waits behind the deliveries already in
        the room's lane; called from inside that lane or under the room lock,
        it returns at once and the regeneration follows in lane order. A
        caller cancelled cuts the regenerated turn. A room whose status
        refuses new events (RFC §5.1), at the call or while the regeneration
        waits in the lane, is refused *before* the agent runs, with
        ``InboundResult(blocked=True, reason="room_closed")`` and a
        ``room_refused_event`` framework event — exactly as
        :meth:`process_inbound` refuses — rather than after a generation whose
        answer nothing could commit.

        The re-broadcast is scoped to ``visibility="intelligence"`` so only the
        agent reacts — transports never receive the user message again (no
        duplicate bubble, no echo to other participants). Targets the single
        intelligence-channel path; orchestrated rooms (routing installed as
        BEFORE_BROADCAST hooks) are not re-routed here.

        In a room a discussion holds (RFC §19.7.5 rule 7), each agent that
        answered the trigger is queued at the front for a turn of its own,
        and the call returns once they are queued, with no answer in its
        result: the answers come in those turns.
        """
        discussion = self._discussions.get(room_id)
        if discussion is not None:
            return await self._regenerate_in_discussion(discussion, room_id, trigger_id)
        cascade = DeliveryCascade(room_id, reentry_budget=self._max_chain_depth * 10)
        try:
            planned = await self._plan_regeneration(room_id, trigger_id, cascade)
            if not isinstance(planned, tuple):
                return planned
            trigger, plan = planned
            return await self._finish_regeneration(room_id, trigger, plan, cascade)
        except BaseException:
            # The caller owns the cascade: a caller cancelled stops its
            # delivery tail, off the lock, as on the inbound path.
            await cascade.abandon("caller_cancelled")
            raise

    async def _plan_regeneration(
        self, room_id: str, trigger_id: str | None, cascade: DeliveryCascade
    ) -> tuple[RoomEvent, DeliveryPlan] | InboundResult | None:
        """Choose the event to replay under the room lock and put its
        re-broadcast in the room's lane: the trigger and its plan, or what
        the call returns instead (a refusal, ``None`` when nothing qualifies). The wait for
        the lock and the choice are bounded by ``process_timeout``, the
        broadcast is not (RFC §13.6)."""
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout(self._process_timeout):
                    await stack.enter_async_context(self._lock_manager.locked(room_id))
                    context, found = await self._regenerate_target(room_id)
            except TimeoutError:
                return await self._refuse_on_timeout(room_id, operation="regenerate")
            refusal = await self._refuse_regeneration(room_id, context, found, trigger_id)
            if refusal is not None or found is None:
                return refusal
            trigger, source_binding = found
            plan = self._enqueue_regeneration(room_id, trigger, source_binding, context, cascade)
            return trigger, plan

    async def _regenerate_in_discussion(
        self, discussion: Any, room_id: str, trigger_id: str | None
    ) -> InboundResult | None:
        """Queue a regenerated answer to the trigger for every agent of the
        discussion that answered it (RFC §19.7.5 rule 7)."""
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout(self._process_timeout):
                    await stack.enter_async_context(self._lock_manager.locked(room_id))
                    context, found = await self._regenerate_target(room_id)
            except TimeoutError:
                return await self._refuse_on_timeout(room_id, operation="regenerate")
            refusal = await self._refuse_regeneration(room_id, context, found, trigger_id)
            if refusal is not None or found is None:
                return refusal
        trigger, _ = found
        answered = [
            e.source.channel_id
            for e in context.recent_events
            if e.responds_to == trigger.id and e.blocked_by != CHAIN_DEPTH_LIMIT
        ]
        await discussion.queue_regeneration(trigger, list(dict.fromkeys(answered)))
        return InboundResult(event=trigger)

    async def _refuse_regeneration(
        self,
        room_id: str,
        context: RoomContext,
        found: tuple[RoomEvent, ChannelBinding] | None,
        trigger_id: str | None,
    ) -> InboundResult | None:
        """What refuses the regeneration under the lock, before the agent
        runs: a room that accepts no new event, or a trigger that moved."""
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
        return None

    def _enqueue_regeneration(
        self,
        room_id: str,
        trigger: RoomEvent,
        source_binding: ChannelBinding,
        context: RoomContext,
        cascade: DeliveryCascade,
    ) -> DeliveryPlan:
        """Put *trigger*'s re-broadcast in the room's lane, behind every
        event committed so far, as a rerun (``DeliveryPlan.rerun``); its plan."""
        # Scope the re-broadcast to intelligence channels: only the agent
        # regenerates, no transport re-delivery of the user's message.
        intel_trigger = trigger.model_copy(update={"visibility": "intelligence"})
        plan = self._get_router().plan(intel_trigger, source_binding, context)
        plan.rerun = True
        plan.response_visibility = trigger.response_visibility
        cascade.retain()
        self._enqueue_exec(
            room_id, plan, cascade, index=None, after_index=context.room.latest_index
        )
        return plan

    async def _finish_regeneration(
        self, room_id: str, trigger: RoomEvent, plan: DeliveryPlan, cascade: DeliveryCascade
    ) -> InboundResult:
        """The regeneration's result once its cascade ends: the trigger, the
        new answers and what they ended with (RFC §10.1 step 18), or the
        refusal of a room that closed while it waited in the lane."""
        # The lane delivers the new answers — their commit passes, their
        # AFTER_BROADCAST hooks, what the other agents answer to them (RFC
        # §8.3) — then the streams the regeneration started are read here,
        # outside the lock (TTS delivery can take seconds).
        stream_error, stream_meta = await self._finish_cascade(cascade, room_id, caller_logs=True)
        if plan.refusal is not None:
            return plan.refusal
        result = InboundResult(event=trigger)
        result.report_cascade(cascade)
        if stream_error is not None and result.error is None:
            result.error = stream_error
        merge_caller_record(result.response_metadata, stream_meta)
        return result

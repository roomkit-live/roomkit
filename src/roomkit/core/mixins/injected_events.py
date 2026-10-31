"""InjectedEventsMixin — the events a hook injects (RFC §9.4).

Each one is committed as a pass of its cause's cascade (§8.3, §10.1 step 14):
it takes the room lock for its own commit, runs the ``BEFORE_BROADCAST``
chain with every hook under the lock, and counts against the cascade's
reentry budget. It keeps its cause's chain depth, names its targets as its
visibility, carries no idempotency key, and takes no source check: it is the
hook's, not a channel's. It is delivered to its transport targets only,
under the rules of any delivery, never to its own source, and asks no agent:
a transport hosting a model takes it into the model's context silently. Its
delivery set, empty when it is stored only, fires ``AFTER_BROADCAST`` like
any other.
An injected INSTRUCTION is dropped with a warning before any pass: only the
application issues instructions.

A locked section queues the events its hooks injected
(:class:`~roomkit.core._queued_injections.QueuedInjections`) and their
passes run once it has released the room lock (§9.4, §10.1 steps 10 to 14),
each in a section of its own: another event may commit between a cause and
what its hooks injected. The events an injected pass's own hooks inject run
the same way, right after it, before the next of its cause's.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from roomkit.core._changes import with_target
from roomkit.core._instructions import without_fingerprint
from roomkit.core._observed import Observed, observed_once_over
from roomkit.core._queued_injections import QueuedInjections
from roomkit.core._reentrance import Entry, Placement, section
from roomkit.core.lanes import DeliveryCascade, DeliveryPlan
from roomkit.core.locks import unlocked_context
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.mixins.inbound_locked import (
    _Blocked,
    _Ready,
    _Refused,
    committed_in,
    target_update,
)
from roomkit.core.mixins.lane_execution import provisional
from roomkit.core.task_utils import await_unless_cut, start_named
from roomkit.core.visibility import intersect_visibility, narrowed_to_target, visibility_allows
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import RoomEvent, with_pipeline_id

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.hook import InjectedEvent

logger = logging.getLogger("roomkit.framework")


def _as_injected(event: RoomEvent, cause: RoomEvent) -> RoomEvent:
    """*event* as its pass commits it (RFC §9.4): in its cause's room, at
    its cause's chain depth, with no idempotency key (§13.4) and no
    instruction fingerprint, the pipeline's on a response alone (§5.2),
    under an id the pipeline gives (§8.1)."""
    return with_pipeline_id(
        event.model_copy(
            update={
                "room_id": cause.room_id,
                "chain_depth": cause.chain_depth,
                "idempotency_key": None,
                "metadata": without_fingerprint(event.metadata),
            }
        )
    )


def _without_instructions(
    injected_events: list[InjectedEvent], cause: RoomEvent
) -> list[InjectedEvent]:
    """*injected_events* without the instructions among them (RFC §9.4,
    §10.1.1): only the application issues an instruction, which a hook sends
    with ``send_event()``. Each one dropped is logged, before any pass: no
    hook runs on it, and nothing of it is stored or delivered."""
    kept: list[InjectedEvent] = []
    for injected in injected_events:
        if injected.event.type != EventType.INSTRUCTION:
            kept.append(injected)
            continue
        logger.warning(
            "A hook injected an INSTRUCTION while deciding on event %s: dropped, since only "
            "the application issues instructions (send_event); the rest of its result stands",
            cause.id,
            extra={"room_id": cause.room_id, "event_id": cause.id},
        )
    return kept


def _targets_scope(injected: InjectedEvent, context: RoomContext) -> str:
    """The visibility an injected event is committed with: the targets the
    room binds, whatever visibility it came with, and ``none`` when it is
    stored only (RFC §9.4)."""
    bound = {binding.channel_id for binding in context.bindings}
    named = sorted(bound.intersection(injected.target_channel_ids or ()))
    return ",".join(named) if named else Visibility.NONE


class InjectedEventsMixin(HelpersMixin):
    """The passes of the events hooks inject, the identity challenge's included.

    Host contract: the lane execution and the locked pipeline.
    """

    _lock_manager: RoomLockManager
    _max_chain_depth: int

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _commit_to_lane: Any  # LaneExecutionMixin
    _finish_cascade: Any  # LaneExecutionMixin
    _open_pass: Any  # LaneExecutionMixin
    _store_past_reentry_cap: Any  # LaneExecutionMixin
    _check_change: Any  # InboundLockedMixin
    _gate_commit: Any  # InboundLockedMixin
    _store_blocked: Any  # InboundLockedMixin

    async def _lane_injected_events(
        self,
        injected_events: list[InjectedEvent],
        room_id: str,
        cascade: DeliveryCascade,
        *,
        cause: RoomEvent,
    ) -> None:
        """Commit each event *cause*'s hooks injected as a pass of
        *cascade*, in order (RFC §9.4), whichever way *cause* ended; an
        instruction among them is dropped before any pass."""
        for injected in _without_instructions(injected_events, cause):
            await self._run_injected_pass(room_id, injected, cascade, cause=cause)

    async def _inject_off_lock(
        self, injected_events: list[InjectedEvent], room_id: str, *, cause: RoomEvent
    ) -> None:
        """Commit and deliver events injected outside any cascade (an
        identity challenge, before the room lock), waiting for their
        delivery."""
        cascade = self._cascades.new(room_id, reentry_budget=self._max_chain_depth * 10)
        await self._lane_injected_events(injected_events, room_id, cascade, cause=cause)
        await self._finish_cascade(cascade, room_id)

    def _hand_off_queued(
        self, queued: QueuedInjections, room_id: str, cascade: DeliveryCascade
    ) -> asyncio.Task[None] | None:
        """Hand the passes of what a decided event's hooks injected to a task
        the kit holds, so that they no longer depend on the caller: they
        commit whichever way the event ends (RFC §9.4), its caller cancelled
        or failing after the decision included. ``None`` when nothing was
        queued.

        Started under the event's lock, in a context that holds none, the
        task asks for the lock as any later entry and waits for the event's
        section, and any section ahead of it, to release it: that wait has
        no bound, as an adopted entry's has none. The kit's close cancels it
        and waits for its end, with its other background work.
        """
        if not queued.entries:
            return None
        task = start_named(
            self._run_queued(queued, room_id, cascade),
            name=f"roomkit-injected-events-{room_id}",
            context=unlocked_context(),
        )
        task.add_done_callback(_log_failed_passes)
        task.add_done_callback(self._pending_hook_tasks.discard)
        self._pending_hook_tasks.add(task)
        return task

    @staticmethod
    def _queue_of(entry: Entry) -> QueuedInjections:
        """Where an entry's locked section queues what its hooks inject (RFC
        §9.4): nested in a pass's section (a locked hook, code it runs, that
        injects directly), that section's queue, whose passes run once the
        section has released the lock; else a queue of its own."""
        return entry.queued if entry.queued is not None else QueuedInjections()

    def _hand_off_decided(
        self, entry: Entry, queued: QueuedInjections, room_id: str, cascade: DeliveryCascade
    ) -> asyncio.Task[None] | None:
        """Hand off the passes a decided entry's section queued
        (:meth:`_hand_off_queued`), called under its lock; the task to wait
        for once the lock is released, if any.

        None for an entry whose passes joined the queue of the pass section
        it is nested in, which runs them; and for one nested under a lock
        held outside any pass (a host's own section): its passes wait for
        that lock's release, which its caller holds, so the caller cannot
        wait for them.
        """
        if entry.queued is not None:
            return None
        handed = self._hand_off_queued(queued, room_id, cascade)
        return None if entry.placement is Placement.NESTED else handed

    @staticmethod
    async def _finish_queued(handed: asyncio.Task[None] | None) -> None:
        """Wait, once the caller's section released the lock, for the
        passes it handed off (:meth:`_hand_off_decided`), shielded from the
        caller's cancellation, so they commit before the call returns. The
        kit's close cutting them ends the wait quietly: the caller was not
        cancelled."""
        if handed is not None:
            await await_unless_cut(asyncio.shield(handed))

    async def _run_queued(
        self, queued: QueuedInjections, room_id: str, cascade: DeliveryCascade
    ) -> None:
        """Run the passes of the events a locked section queued, once it has
        released the room lock (RFC §9.4, §10.1 steps 10 to 14), in the
        order they were queued."""
        for injected_events, cause in queued.take():
            await self._lane_injected_events(injected_events, room_id, cascade, cause=cause)

    async def _run_injected_pass(
        self,
        room_id: str,
        injected: InjectedEvent,
        cascade: DeliveryCascade,
        *,
        cause: RoomEvent,
    ) -> None:
        """One injected event's own commit pass, in a locked section of its
        own; then the passes of what its hooks injected, once it released
        the lock."""
        queued = QueuedInjections()
        async with self._lock_manager.locked(room_id):
            with section(room_id, cascade.budget, queued):
                async with observed_once_over(
                    self._hook_engine, room_id, closing=lambda: self._closed
                ) as observed:
                    await self._injected_section(
                        room_id, injected, cascade, queued, observed, cause=cause
                    )
        await self._run_queued(queued, room_id, cascade)

    async def _injected_section(
        self,
        room_id: str,
        injected: InjectedEvent,
        cascade: DeliveryCascade,
        queued: QueuedInjections,
        observed: Observed,
        *,
        cause: RoomEvent,
    ) -> None:
        """An injected event's locked section: its gates and its commit,
        the events its hooks inject queued on *queued*, what its chain owes
        its observers on *observed*."""
        event = _as_injected(injected.event, cause)
        room_context = await self._open_pass(room_id, event)
        if room_context is None:
            return
        # Its thread parent, the hook's, rooted (RFC §10.1 step 8), as its
        # own hooks read it, in their context too.
        event = await self._with_thread_root(room_id, event)
        event = event.model_copy(update={"visibility": _targets_scope(injected, room_context)})
        event, context = provisional(event, room_context)
        if not cascade.consume_reentry_budget():
            # Past the budget it is not a pass: its hooks never run (§8.3).
            await self._store_past_reentry_cap(room_id, event, channels=False)
            return
        # An edit or a delete it carries is the hook's, a SYSTEM change
        # only (RFC §10.3 rule 6); a refused one is logged, nothing stored.
        check = await self._check_change(room_id, event, by_hook=True)
        if check.refusal is not None:
            return
        decision = await self._gate_commit(
            room_id,
            event,
            context,
            source_check=False,
            change=check,
            operation="reentry",
            observed=observed,
        )
        if isinstance(decision, _Refused):
            return
        if isinstance(decision, _Blocked):
            await self._store_blocked(room_id, decision, queued)
            return
        await self._commit_injected(room_id, decision, injected, event.visibility, cascade, queued)

    async def _commit_injected(
        self,
        room_id: str,
        ready: _Ready,
        injected: InjectedEvent,
        scope: str,
        cascade: DeliveryCascade,
        queued: QueuedInjections,
    ) -> None:
        """Commit an injected event its hooks let through and lane its
        delivery, under the lock its pass holds; then its hooks' side
        effects, the events they injected queued on *queued*."""
        categories = {binding.channel_id: binding.category for binding in ready.context.bindings}
        event = ready.event.model_copy(
            update={
                "status": EventStatus.DELIVERED,
                # A hook may narrow the scope of its targets, never widen it.
                "visibility": intersect_visibility(ready.event.visibility, scope, categories.get),
            }
        )
        # An edit's or a delete's target, written in the event's own commit
        # (RFC §10.3, §14.3): the change reaches those of its targets that
        # saw the target, with the target as changed.
        target, mutation_hook = target_update(event, ready.edit_delete_target)
        event = narrowed_to_target(event, target, ready.context.bindings)
        context = with_target(ready.context, target)
        targets = await self._injected_targets(room_id, event, injected)

        def factory(committed: RoomEvent) -> DeliveryPlan:
            planned = committed_in(context, committed)
            return _injected_plan(committed, ready, planned, targets, mutation_hook)

        stored = await self._commit_to_lane(
            room_id, event, cascade, factory, policy_aware=False, target=target
        )
        committed = stored or event
        sync_result = ready.sync_result
        await self._persist_side_effects(
            room_id, sync_result.tasks, sync_result.observations, committed, context
        )
        queued.add(sync_result.injected_events, committed)

    async def _injected_targets(
        self, room_id: str, event: RoomEvent, injected: InjectedEvent
    ) -> list[ChannelBinding]:
        """The transports *event* is delivered to: its targets as the room
        binds them now (a hook of its pass may have changed a binding),
        under the rules of any delivery, never its own source (RFC §9.4,
        §7.5 rule 5). An agent among them is not called: it reads the event
        in its history."""
        targets: list[ChannelBinding] = []
        for target_id in dict.fromkeys(injected.target_channel_ids or ()):
            if target_id == event.source.channel_id:
                continue
            binding = await self._store.get_binding(room_id, target_id)
            if (
                binding is not None
                and binding.category == ChannelCategory.TRANSPORT
                and binding.access not in (Access.WRITE_ONLY, Access.NONE)
                and visibility_allows(event.visibility, binding)
            ):
                targets.append(binding)
        return targets


def _log_failed_passes(task: asyncio.Task[None]) -> None:
    """Log the injected passes a handed-off task failed to commit: its
    caller, cancelled meanwhile, may never read the failure."""
    if task.cancelled() or (error := task.exception()) is None:
        return
    logger.error("A hook's injected events failed to commit", exc_info=error)


def _injected_plan(
    committed: RoomEvent,
    ready: _Ready,
    context: RoomContext,
    targets: list[ChannelBinding],
    mutation_hook: tuple[HookTrigger, RoomEvent] | None,
) -> DeliveryPlan:
    """The delivery plan of an injected event, under *context*: its
    transport targets, none when it is stored only, no reentry, the mutation
    trigger of an edit or a delete (RFC §10.3), and its ``AFTER_BROADCAST``
    once its delivery set has run, like any other (§9.4)."""
    # An unbound source (a hook's own, as "system") transcodes as a channel
    # of its type with default capabilities.
    source_binding = ready.source_binding or ChannelBinding(
        channel_id=committed.source.channel_id,
        room_id=committed.room_id,
        channel_type=committed.source.channel_type,
    )
    return DeliveryPlan(
        event=committed,
        source_binding=source_binding,
        context=context,
        targets=targets,
        injected=True,
        mutation_hook=mutation_hook,
    )

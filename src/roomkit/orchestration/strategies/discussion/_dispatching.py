"""Who takes a person's unaddressed message, decided by the lease holder (RFC §19.7.5 rule 18).

Routing leaves such a message waiting in the stored queue, with the
candidates it reaches and the place in the queue it came in at
(``DiscussionRoom._leave_for_dispatch``). The turn driver, holding the
lease, hands the first message waiting to a :class:`Dispatcher` before it
gives another turn: the policy decides off the room lock, within the
discussion's bound, and the agents it picks are put at the front of the
queue at the message's place, as rule 8 puts ``everyone``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from roomkit.channels._speaker import turn_labels
from roomkit.models.enums import EventStatus, EventType, HookTrigger
from roomkit.models.event import is_tool_call_record

from ._people import sees
from ._queue import Pending
from .dispatch import DispatchCandidate, DispatchDecision, DispatchDecisionEvent, DispatchTurn
from .models import SpeakQueueChange

if TYPE_CHECKING:
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent

    from ._room import DiscussionRoom

logger = logging.getLogger("roomkit.orchestration.discussion")

FALLBACK = "fallback"
"""The reason of a decision the policy did not take: every candidate asked."""


class Dispatcher:
    """Decides, in the process holding the lease, who takes a message waiting."""

    def __init__(self, room: DiscussionRoom) -> None:
        self._room = room

    async def decide(self, pending: Pending) -> None:
        """Decide *pending* and queue the agents picked. A message gone from
        the room, or whose candidates all listen now, asks no agent; a
        decision that fails in any way asks every candidate. The decision is
        reported once applied, never when another process decided first."""
        room = self._room
        listening = set(room.state.listening)
        candidates = [a for a in pending.candidates if a in room.agents and a not in listening]
        report: DispatchDecisionEvent | None = None
        try:
            report = await self._decision(pending, candidates)
            agents = report.decision.agents if report is not None else ()
        except Exception:
            logger.exception(
                "The dispatch decision on %s in room %s failed; asking every candidate",
                pending.event_id,
                room.room_id,
            )
            agents = tuple(candidates)
        if await self._apply(pending, agents) and report is not None:
            hooks = room.kit.hook_engine
            if hooks.has_hooks(HookTrigger.ON_DISPATCH_DECISION):
                applied = report
                room.announce(lambda: room.kit._fire_dispatch_decision(applied))

    async def _decision(
        self, pending: Pending, candidates: list[str]
    ) -> DispatchDecisionEvent | None:
        """The decision on *pending* as it will be reported, or None when there
        is nothing to decide."""
        room = self._room
        if not candidates:
            return None
        event = await room.kit.store.get_event(pending.event_id)
        if event is None or event.room_id != room.room_id:
            return None
        context = await room.kit._build_context(room.room_id, reads_history=True)
        decision, duration_ms = await self._bounded(self._turn(event, context, candidates))
        return DispatchDecisionEvent(room.room_id, event, tuple(candidates), decision, duration_ms)

    async def _bounded(self, turn: DispatchTurn) -> tuple[DispatchDecision, int]:
        """The policy's decision within the discussion's bound, cut down to
        the candidates, and how long it took: a policy that fails, does not
        decide in time or decides something unreadable, or none in this
        process, asks every candidate."""
        strategy = self._room.strategy
        policy = strategy.dispatch if strategy is not None else None
        candidates = [c.channel_id for c in turn.candidates]
        fallback = DispatchDecision(tuple(candidates), FALLBACK)
        if strategy is None or policy is None:
            logger.warning("No dispatch policy in this process for room %s", turn.room_id)
            return fallback, 0
        started = time.monotonic()
        try:
            decision = await asyncio.wait_for(policy.decide(turn), strategy.dispatch_timeout)
            if not isinstance(decision, DispatchDecision):
                raise TypeError(f"decide() returned {type(decision).__name__}")
            return _within(decision, candidates), round((time.monotonic() - started) * 1000)
        except TimeoutError:
            logger.warning(
                "Dispatch policy took over %.1f s on %s; asking every candidate",
                strategy.dispatch_timeout,
                turn.event.id,
            )
            return fallback, round(strategy.dispatch_timeout * 1000)
        except Exception:
            logger.warning(
                "Dispatch policy failed on %s; asking every candidate",
                turn.event.id,
                exc_info=True,
            )
            return fallback, round((time.monotonic() - started) * 1000)

    def _turn(self, event: RoomEvent, context: RoomContext, candidates: list[str]) -> DispatchTurn:
        """What the policy judges: the room's messages before *event* a
        candidate may read (a policy may hand them to a classifier outside),
        who said each, and the candidates' identity."""
        room = self._room
        recent = tuple(
            e
            for e in context.recent_events
            if e.index < event.index
            and e.type == EventType.MESSAGE
            and e.status != EventStatus.BLOCKED
            and not is_tool_call_record(e)
            and any(sees(c, e, context) for c in candidates)
        )
        labels = turn_labels((*recent, event), context)
        speakers: dict[str, str] = {}
        for e in (*recent, event):
            author = e.source.channel_id
            label = f"@{author}" if author in room.agents else labels.get(e.id)
            if label:
                speakers[e.id] = label
        return DispatchTurn(
            room_id=room.room_id,
            event=event,
            recent=recent,
            speakers=speakers,
            candidates=tuple(self._candidate(a) for a in candidates),
        )

    def _candidate(self, agent_id: str) -> DispatchCandidate:
        agent = self._room.agents.get(agent_id)
        return DispatchCandidate(
            channel_id=agent_id,
            name=getattr(agent, "name", None),
            role=getattr(agent, "role", None),
            description=getattr(agent, "description", None),
        )

    async def _apply(self, pending: Pending, agents: Sequence[str]) -> bool:
        """Put *agents* at the front at the message's place; whether the
        decision was applied: not when another process decided the message
        meanwhile, the discussion ended, or this process stopped (the message
        then waits for the next holder)."""
        room = self._room
        async with room.editing() as state:
            if room.closed or not any(p.event_id == pending.event_id for p in state.dispatching):
                return False
            state.dispatching = [p for p in state.dispatching if p.event_id != pending.event_id]
            if state.over:
                return False
            state.queue_picked(pending, agents)
        if agents:
            room.fire(SpeakQueueChange.QUEUED, agents, pending.event_id)
        return True


def _within(decision: DispatchDecision, candidates: list[str]) -> DispatchDecision:
    """*decision* with only candidates, each once: a policy is no way around
    visibility or the listening state."""
    picked = tuple(dict.fromkeys(a for a in decision.agents if a in candidates))
    left_out = [a for a in decision.agents if a not in candidates]
    if left_out:
        logger.warning("Dispatch decision named agents that are not candidates: %s", left_out)
    return dataclasses.replace(decision, agents=picked)

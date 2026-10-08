"""Speaking turns on the AI channel: the speak policy consulted before a turn runs
(RFC §6.4)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from roomkit.channels._ai_cuts import cut_records, cut_reply
from roomkit.channels._speaker import author_name, participant_name
from roomkit.core.visibility import visible_events
from roomkit.models.enums import EventType, ParticipantRole, ParticipantStatus
from roomkit.models.event import is_tool_call_record
from roomkit.speaking.base import SpeakDecision, SpeakDecisionEvent, SpeakPolicy, SpeakTurn

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import SpeakDecisionHook
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.speaking.thought import Thought

logger = logging.getLogger("roomkit.channels.ai")

OFFER_NOTE = (
    "Nobody asked you on this turn. Offer, in one short sentence, what you could "
    "add, without giving it."
)
"""The block an ``offer`` decision adds to the turn's notes."""

_FALLBACK = "fallback"

_NOT_PEOPLE = frozenset({ParticipantRole.AGENT, ParticipantRole.BOT})


class AISpeakingMixin:
    """Consults the channel's speak policy, once per event it would answer."""

    channel_id: str
    _speak_policy: SpeakPolicy | None
    _speak_timeout: float
    _speak_decision_hook: SpeakDecisionHook | None

    def _store_speak_policy(self, policy: SpeakPolicy | None, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("speak_timeout must be positive")
        self._speak_policy = policy
        self._speak_timeout = timeout
        self._speak_decision_hook = None

    async def _speak_decision(
        self,
        event: RoomEvent,
        context: RoomContext,
        thought: Thought | None = None,
        *,
        asked_again: bool = False,
    ) -> SpeakDecision | None:
        """The policy's decision on *event*, reported to ``ON_SPEAK_DECISION``
        with how long it took, or None when nothing is decided: no policy, an
        instruction (the application asked for that turn), the channel's own
        event or a tool record. *asked_again*: the policy decides again once the
        agent thought (RFC §6.4)."""
        policy = self._speak_policy
        if policy is None or not self._submitted_to_policy(event):
            return None
        turn = _speak_turn(event, context, self.channel_id, thought)
        started = time.monotonic()
        decision = await self._bounded_decision(policy, turn)
        duration_ms = round((time.monotonic() - started) * 1000)
        room_id = context.room.id if context.room else event.room_id
        await self._report_speak_decision(
            SpeakDecisionEvent(room_id, self.channel_id, event, decision, duration_ms, asked_again)
        )
        return decision

    async def _bounded_decision(self, policy: SpeakPolicy, turn: SpeakTurn) -> SpeakDecision:
        """The policy's decision on *turn* within the channel's bound: a policy
        that fails or does not decide in time does not silence the agent, which
        speaks with the reason ``fallback``."""
        try:
            return await asyncio.wait_for(policy.decide(turn), self._speak_timeout)
        except TimeoutError:
            logger.warning(
                "Speak policy took over %.1f s on %s; speaking",
                self._speak_timeout,
                turn.event.id,
            )
        except Exception:
            logger.warning("Speak policy failed on %s; speaking", turn.event.id, exc_info=True)
        return SpeakDecision("speak", reason=_FALLBACK)

    def _submitted_to_policy(self, event: RoomEvent) -> bool:
        return (
            event.type != EventType.INSTRUCTION
            and event.source.channel_id != self.channel_id
            and not is_tool_call_record(event)
        )

    async def _report_speak_decision(self, report: SpeakDecisionEvent) -> None:
        hook = self._speak_decision_hook
        if hook is None:
            return
        try:
            await hook(report)
        except Exception:
            logger.warning("ON_SPEAK_DECISION failed on %s", report.event.id, exc_info=True)


def speak_notes(decision: SpeakDecision | None) -> tuple[str, ...]:
    """The blocks a decision adds to the turn's notes: its own, and for an offer
    the channel's ask to offer rather than answer."""
    if decision is None:
        return ()
    if decision.mode == "offer":
        return (*decision.notes, OFFER_NOTE)
    return decision.notes


def _speak_turn(
    event: RoomEvent, context: RoomContext, channel_id: str, thought: Thought | None = None
) -> SpeakTurn:
    """What the policy judges: the room's messages before *event* that the channel
    may know (RFC §7.5 rule 8: a policy may send them to a classifier outside),
    who said each, and the people taking part besides the agent."""
    recent = tuple(
        e
        for e in visible_events(context, channel_id)
        if e.id != event.id and e.type == EventType.MESSAGE and not is_tool_call_record(e)
    )
    speakers = {
        e.id: name for e in (*recent, event) if (name := author_name(e, context)) is not None
    }
    return SpeakTurn(
        event=event,
        recent=recent,
        people=_people(context, (*recent, event), speakers, channel_id),
        channel_id=channel_id,
        speakers=speakers,
        thought=thought,
        cut=cut_reply(recent, cut_records(context, channel_id), channel_id),
    )


def _people(
    context: RoomContext,
    events: tuple[RoomEvent, ...],
    speakers: dict[str, str],
    channel_id: str,
) -> tuple[str, ...]:
    """The room's active participants that are neither agents nor bots nor on the
    agent's channel, or the distinct speakers of *events* when more: one
    microphone is one participant but may carry several diarized voices."""
    participants = tuple(
        participant_name(p)
        for p in context.participants
        if p.status == ParticipantStatus.ACTIVE
        and p.role not in _NOT_PEOPLE
        and p.channel_id != channel_id
    )
    voices = tuple(
        dict.fromkeys(
            speakers[e.id]
            for e in events
            if e.source.channel_id != channel_id and e.id in speakers
        )
    )
    return voices if len(voices) > len(participants) else participants

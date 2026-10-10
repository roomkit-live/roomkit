"""A dispatch policy on a classifier: one yes/no question per candidate (RFC §19.7.5 rule 18).

The policy asks a :class:`~roomkit.classifiers.base.Classifier`, in one call,
whether each candidate should take the message, over the team (each agent's
identity), the recent conversation named by speaker and the message. The
candidates whose probability reaches the threshold take it, likeliest first.
The question was measured with Jev (TypeSafe's System One model) on sixteen
messages to a team of four, then live: 15/16 right first picks, no agent
woken for a thanks, 17 turns where asking everyone gave 64.
"""

from __future__ import annotations

from typing import Any

from roomkit.classifiers.base import Classifier, YesNoQuestion
from roomkit.models.event import RoomEvent, TextContent

from .base import DispatchCandidate, DispatchDecision, DispatchPolicy, DispatchTurn

YES = "the message asks for, or plainly falls within, this agent's part"
NO = "it is another agent's part, or nobody needs to act (thanks, small talk)"


def question(candidate: DispatchCandidate) -> YesNoQuestion:
    """Whether *candidate* should take the message, as the policy asks it."""
    handle = f"@{candidate.channel_id}"
    who = ": ".join(p for p in (candidate.name or candidate.role, candidate.description) if p)
    named = f"{handle} ({who})" if who else handle
    return YesNoQuestion(
        instructions=(
            f"Should {named} take the latest message, that is act on it or answer it, "
            f"given what {handle} can do and what the others can do?"
        ),
        yes=YES,
        no=NO,
    )


class ClassifierDispatchPolicy(DispatchPolicy):
    """Picks the candidates a classifier judges the message is for.

    Args:
        classifier: Answers the questions, all candidates in one call
            (``JevClassifier``, ``LLMClassifier``...). Its failure is the
            discussion's fallback: every candidate takes the message.
        threshold: The probability of yes a candidate needs. None reaching
            it: no agent takes the message (thanks, small talk).
        max_agents: How many candidates take it at most, likeliest first.
        recent: How many earlier messages the classifier reads.
    """

    def __init__(
        self,
        classifier: Classifier,
        *,
        threshold: float = 0.5,
        max_agents: int = 2,
        recent: int = 8,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        if recent < 0:
            raise ValueError("recent must not be negative")
        self._classifier = classifier
        self._threshold = threshold
        self._max_agents = max_agents
        self._recent = recent

    async def decide(self, turn: DispatchTurn) -> DispatchDecision:
        if not turn.candidates:
            return DispatchDecision(reason="no candidate")
        names = {f"agent_{i}": c.channel_id for i, c in enumerate(turn.candidates)}
        questions = {name: question(c) for name, c in zip(names, turn.candidates, strict=True)}
        answers = await self._classifier.classify(self.state(turn), questions)
        judgments = {names[name]: round(answers.yes(name), 3) for name in questions}
        chosen = sorted(
            (agent for agent, p in judgments.items() if p >= self._threshold),
            key=lambda agent: judgments[agent],
            reverse=True,
        )[: self._max_agents]
        reason = "above threshold" if chosen else "nobody above threshold"
        return DispatchDecision(tuple(chosen), reason, judgments)

    def state(self, turn: DispatchTurn) -> dict[str, Any]:
        """What the classifier reads: the team, the conversation, the message."""
        recent = [e for e in turn.recent if _text(e)][-self._recent :] if self._recent else []
        return {
            "team": [_member(c) for c in turn.candidates],
            "conversation": [_line(e, turn) for e in recent],
            "message": _line(turn.event, turn),
        }


def _member(candidate: DispatchCandidate) -> dict[str, str]:
    member = {
        "agent": f"@{candidate.channel_id}",
        "name": candidate.name,
        "role": candidate.role,
        "can": candidate.description,
    }
    return {key: value for key, value in member.items() if value}


def _line(event: RoomEvent, turn: DispatchTurn) -> dict[str, str]:
    return {"speaker": turn.speakers.get(event.id, ""), "text": _text(event)}


def _text(event: RoomEvent) -> str:
    return event.content.body if isinstance(event.content, TextContent) else ""

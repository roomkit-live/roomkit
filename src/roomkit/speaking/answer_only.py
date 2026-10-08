"""A policy around another that answers only some people (RFC §6.4).

An agent may listen to everyone in a room and answer only some of them: a
television on in the living room, a meeting where it assists one person.
:class:`AnswerOnly` wraps any :class:`~roomkit.speaking.base.SpeakPolicy`: a
turn from anyone else is left silent without asking it, and the people it
judges with are only those the agent answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from roomkit._text import person_name
from roomkit.speaking.base import SpeakDecision, SpeakPolicy, SpeakTurn

LISTENED_TO = "only listened to"
"""The reason of a decision on a speaker the agent does not answer."""


def _key(name: str) -> str:
    """*name* as compared: kept to a name's characters, on one line, as the
    room keeps a speaker's (so a configured name matches the one the room
    gives), then ignoring case."""
    return person_name(name).casefold()


class AnswerOnly(SpeakPolicy):
    """Answers only *people*; everyone else is heard, never answered.

    A turn whose speaker is not one of *people*, or whom the room does not
    name, is decided ``silent`` with the reason :data:`LISTENED_TO`, without
    asking *policy*: it is stored and the agent's thinker thinks about it, as
    any turn left silent. A turn from one of *people* is *policy*'s to decide,
    with only *people* in ``SpeakTurn.people``, the speaker among them: a voice
    only listened to does not turn a conversation with one person into a group
    one.

    A speaker is matched by the name the room gives them (``SpeakTurn.speakers``:
    the name the sender's transport stamped, else the participant's display
    name), ignoring case and spacing. It chooses whom the agent answers; it is
    not an access control, since a participant may take any display name.

    Args:
        policy: Decides on the turns of the people answered.
        people: The names of the people the agent answers.

    Raises:
        TypeError: *people* is one string rather than names.
        ValueError: *people* names no one.
    """

    def __init__(self, policy: SpeakPolicy, people: Iterable[str]) -> None:
        if isinstance(people, str):
            raise TypeError("AnswerOnly takes the names of the people answered, not one string")
        self._policy = policy
        self._people = frozenset(key for name in people if (key := _key(name)))
        if not self._people:
            raise ValueError("AnswerOnly needs at least one person to answer")

    def answers(self, name: str | None) -> bool:
        """Whether *name* is one of the people the agent answers."""
        return name is not None and _key(name) in self._people

    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        speaker = turn.speakers.get(turn.event.id)
        if speaker is None or not self.answers(speaker):
            return SpeakDecision("silent", reason=LISTENED_TO)
        return await self._policy.decide(replace(turn, people=self._judged_with(turn, speaker)))

    def _judged_with(self, turn: SpeakTurn, speaker: str) -> tuple[str, ...]:
        """The people the wrapped policy judges with: those of the turn it
        answers, and the speaker when none of them names them (a participant
        record named otherwise than the voice its transport stamps)."""
        people = tuple(name for name in turn.people if self.answers(name))
        if any(_key(name) == _key(speaker) for name in people):
            return people
        return (*people, speaker)

    async def close(self) -> None:
        """Close the policy it wraps."""
        await self._policy.close()

"""Speaking turns: whether an agent speaks now, offers to, or stays silent (RFC §6.4).

An agent in a conversation with several people, or listening to one who thinks
aloud, does not answer every turn. An :class:`~roomkit.channels.ai.AIChannel`
given a :class:`SpeakPolicy` asks it, once per event it would answer, before the
turn runs; a ``silent`` decision runs no turn at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, get_args

from roomkit.models.event import RoomEvent

SpeakMode = Literal["speak", "offer", "silent"]
"""``speak`` answers; ``offer`` says in one short sentence what the agent could add,
without giving it; ``silent`` runs no turn."""

_MODES = frozenset(get_args(SpeakMode))


@dataclass(frozen=True)
class SpeakTurn:
    """What a policy judges: the event, and the conversation and people around it."""

    event: RoomEvent
    """The event the turn would answer (its trigger, RFC §8.5)."""

    recent: tuple[RoomEvent, ...] = ()
    """The conversation before it, oldest first, the agent's own answers included:
    the room's messages, without tool records."""

    people: tuple[str, ...] = ()
    """Who takes part besides the agent, by name: one name is a conversation
    between the agent and one person. The room's active participants that are
    neither agents nor bots, or the distinct speakers of the recent events when
    more (one microphone may carry several diarized voices)."""

    channel_id: str = ""
    """The agent's channel: its own answers in ``recent`` come from it."""

    speakers: Mapping[str, str] = field(default_factory=dict)
    """Who said ``event`` and each of ``recent``, by event id, where the room
    names them: the name the sender's transport stamped on the event, else the
    participant's display name, as the AI context names speakers."""

    def by_agent(self, event: RoomEvent) -> bool:
        """Whether *event* is one of the agent's own answers."""
        return bool(self.channel_id) and event.source.channel_id == self.channel_id


@dataclass(frozen=True)
class SpeakDecision:
    """Whether the agent speaks on a turn, and what it reads if it does."""

    mode: SpeakMode
    reason: str = ""
    """Why, in a few words, for logs and ``ON_SPEAK_DECISION``."""

    judgments: dict[str, float] = field(default_factory=dict)
    """What the policy weighed, by name: what makes a decision measurable."""

    notes: tuple[str, ...] = ()
    """Blocks the turn's notes carry when the agent speaks or offers (RFC §6.4)."""

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"speak mode {self.mode!r} is not one of {sorted(_MODES)}")


@dataclass(frozen=True)
class SpeakDecisionEvent:
    """Carried to ``ON_SPEAK_DECISION`` hooks: one channel's decision on one event."""

    room_id: str
    channel_id: str
    event: RoomEvent
    """The event decided on."""

    decision: SpeakDecision


class SpeakPolicy(ABC):
    """Decides, on every event an AI channel would answer, whether its agent speaks.

    A policy that raises, or does not decide within the channel's bound, does
    not silence the agent: the channel speaks and reports the reason
    ``fallback``. An instruction, a task's hand-back among them, never reaches
    a policy: the application asked for that turn.
    """

    @abstractmethod
    async def decide(self, turn: SpeakTurn) -> SpeakDecision:
        """The decision on *turn*."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (a client, a model)."""

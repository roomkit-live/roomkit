"""Who takes a person's unaddressed message in a discussion (RFC §19.7.5 rule 18).

A person's message that names no agent goes, by rule 8, to the agents that
asked that person, else to ``everyone``, one turn each: in a room of
specialists most of those turns say nothing, the agent that should answer
may come third, and an agent that asked a question takes the next message
even when it answers nothing of it. A :class:`DispatchPolicy` decides
instead, once for the room, which agents take it, in which order, or that
none does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from roomkit.models.event import RoomEvent


@dataclass(frozen=True)
class DispatchCandidate:
    """An agent a policy may pick, with the identity the host gave it."""

    channel_id: str
    name: str | None = None
    role: str | None = None
    description: str | None = None
    """What the agent does, as a turn's notes tell the others (rule 14)."""


@dataclass(frozen=True)
class DispatchTurn:
    """What a policy judges: the message, the conversation before it, the candidates."""

    room_id: str
    event: RoomEvent
    """The person's message."""

    recent: tuple[RoomEvent, ...] = ()
    """The conversation before it, oldest first: the room's messages a
    candidate may read, without tool records."""

    speakers: Mapping[str, str] = field(default_factory=dict)
    """Who said ``event`` and each of ``recent``, by event id: a person by the
    label the transcript gives them (RFC §6.4), an agent as ``@channel_id``."""

    candidates: tuple[DispatchCandidate, ...] = ()
    """The agents the policy may pick: those that asked the message's author,
    then ``everyone`` in its order; only those the message reaches, less
    those that only listen."""

    asked: tuple[str, ...] = ()
    """The candidates that asked the message's author a question and wait for
    the answer (RFC §19.7.5 rule 10): the message may answer them, or not."""


@dataclass(frozen=True)
class DispatchDecision:
    """Which agents take a message, in the order they answer, and why."""

    agents: tuple[str, ...] = ()
    """Their channel ids; empty, no agent takes the message."""

    reason: str = ""
    """Why, in a few words, for logs and ``ON_DISPATCH_DECISION``."""

    judgments: dict[str, float] = field(default_factory=dict)
    """What the policy weighed, by name: what makes a decision measurable."""


@dataclass(frozen=True)
class DispatchDecisionEvent:
    """Carried to ``ON_DISPATCH_DECISION`` hooks: one decision on one message."""

    room_id: str
    event: RoomEvent
    """The person's message decided on."""

    candidates: tuple[str, ...]
    decision: DispatchDecision
    """As applied: its agents are candidates, each once."""

    duration_ms: int = 0
    """How long the policy took: the discussion's bound when it did not decide
    in time."""


class DispatchPolicy(ABC):
    """Decides which agents of a discussion take a person's unaddressed message.

    A policy that raises, or does not decide within the discussion's
    ``dispatch_timeout``, does not silence the room: the message asks what it
    asks with no policy (the agents that asked its author, else every
    candidate), and the decision reported carries the reason ``fallback``. A
    message that names an agent never reaches a policy: a name is the
    person's word.
    """

    @abstractmethod
    async def decide(self, turn: DispatchTurn) -> DispatchDecision:
        """The decision on *turn*: agents among its candidates, in order."""

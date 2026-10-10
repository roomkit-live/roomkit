"""What a host reads of a discussion: its speak queue, and each change of it (RFC §19.7.5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique


@dataclass(frozen=True)
class SpeakQueue:
    """A room's speak queue as the host reads it (RFC §19.7.5)."""

    speaking: str | None
    """The agent whose turn runs now, or None between turns."""

    queue: tuple[str, ...]
    """The agents owed a turn, in the order they get it."""

    listening: frozenset[str]
    """The agents that only listen (``listen_only``)."""

    asked: tuple[tuple[str, str], ...]
    """(agent, person): an agent that asked a person and waits for the answer."""

    waiting: bool
    """Whether the discussion waits for a person's message."""

    over: bool
    """Whether the discussion is over: it gives no further turn."""


@unique
class SpeakQueueChange(StrEnum):
    """What changed in a room's speak queue, as ``ON_SPEAK_QUEUE`` reports it."""

    QUEUED = "queued"
    TURN_GIVEN = "turn_given"
    TURN_ENDED = "turn_ended"
    INSTRUCTION_DROPPED = "instruction_dropped"
    LISTENING = "listening"
    TALKING_AGAIN = "talking_again"
    WAITING = "waiting"
    OVER = "over"


@dataclass(frozen=True)
class SpeakQueueEvent:
    """Carried to ``ON_SPEAK_QUEUE`` hooks: one change of a room's speak queue."""

    room_id: str
    queue: SpeakQueue
    """The queue as it is once changed."""

    change: SpeakQueueChange
    channel_ids: tuple[str, ...] = ()
    """The agents the change is about (queued, given a turn, listening...)."""

    event_id: str | None = None
    """The event behind the change, when one is (the message that queued, the
    event a turn answers)."""

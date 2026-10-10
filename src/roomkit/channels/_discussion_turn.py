"""What an AI channel reads of a turn a discussion gives (RFC §19.7.5).

A discussion gives a turn as a rerun of the event the turn answers, the event
carrying a mark in its metadata. The channel adds the turn's notes (who is in
the room, which event the turn answers and who asked, how the room works) to
the turn's input and reads the answered event at its place in the history,
before what was said after it, instead of as the conversation's last message.
The channel side imports nothing of the strategy.

The notes join the turn as the runtime's own block, so they never travel in
the event: event metadata can come from outside (a transport, a host passing a
webhook's fields through). The mark carries only a random turn id, issued by
:func:`issue_mark` and retired once the turn ends, and the notes are kept here
under that id. A mark this process did not issue, already retired, or of any
other shape is no mark.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.event import RoomEvent

DISCUSSION_TURN = "_discussion_turn"
"""The metadata key a discussion's turn trigger carries."""

# The notes of each turn this process issued and has not retired yet.
_issued: dict[str, tuple[str, ...]] = {}


def issue_mark(notes: list[str]) -> dict[str, Any]:
    """A mark for one turn a discussion gives, its *notes* kept until the turn
    ends (:func:`retire_mark`)."""
    turn = uuid4().hex
    _issued[turn] = tuple(note for note in notes if note)
    return {"turn": turn}


def retire_mark(mark: dict[str, Any]) -> None:
    """The turn *mark* was issued for has ended: the mark reads as no mark."""
    turn = mark.get("turn")
    if isinstance(turn, str):
        _issued.pop(turn, None)


def _turn_of(event: RoomEvent) -> str | None:
    mark = (event.metadata or {}).get(DISCUSSION_TURN)
    turn = mark.get("turn") if isinstance(mark, dict) else None
    return turn if isinstance(turn, str) and turn in _issued else None


def discussion_turn(event: RoomEvent) -> dict[str, Any] | None:
    """The discussion's mark on *event*, or None when no discussion gave the
    turn: no mark, or one this process did not issue or already retired."""
    turn = _turn_of(event)
    return {"turn": turn} if turn is not None else None


def discussion_notes(event: RoomEvent) -> list[str]:
    """The notes a discussion's turn adds to the turn's input."""
    turn = _turn_of(event)
    return list(_issued[turn]) if turn is not None else []

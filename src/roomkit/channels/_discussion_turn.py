"""What an AI channel reads of a turn a discussion gives (RFC §19.7.5).

A discussion gives a turn as a rerun of the event the turn answers, the event
carrying a mark in its metadata: the turn's notes (who is in the room, which
event the turn answers and who asked, how the room works). The channel adds
the notes to the turn's input and reads the answered event at its place in
the history, before what was said after it, instead of as the conversation's
last message. The channel side imports nothing of the strategy.

The notes join the turn as the runtime's own block, so a mark must be one the
discussion issued: event metadata can come from outside (a transport, a host
passing a webhook's fields through), and a mark anyone could write would put
their words in the runtime's voice. A mark carries a random turn id, issued by
:func:`issue_mark` and retired once the turn ends; one this process did not
issue, or already retired, is ignored.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from roomkit.models.event import RoomEvent

DISCUSSION_TURN = "_discussion_turn"
"""The metadata key a discussion's turn trigger carries."""

# Turn ids this process issued and has not retired yet.
_issued: set[str] = set()


def issue_mark(notes: list[str]) -> dict[str, Any]:
    """A mark for one turn a discussion gives, carrying *notes*; retire it when
    the turn ends (:func:`retire_mark`)."""
    turn = uuid4().hex
    _issued.add(turn)
    return {"turn": turn, "notes": list(notes)}


def retire_mark(mark: dict[str, Any]) -> None:
    """The turn *mark* was issued for has ended: the mark reads as no mark."""
    _issued.discard(str(mark.get("turn", "")))


def discussion_turn(event: RoomEvent) -> dict[str, Any] | None:
    """The discussion's mark on *event*, or None when no discussion gave the
    turn: no mark, or one this process did not issue or already retired."""
    mark = (event.metadata or {}).get(DISCUSSION_TURN)
    if not isinstance(mark, dict) or mark.get("turn") not in _issued:
        return None
    return mark


def discussion_notes(event: RoomEvent) -> list[str]:
    """The notes a discussion's turn adds to the turn's input."""
    mark = discussion_turn(event)
    if mark is None:
        return []
    return [note for note in mark.get("notes", ()) if isinstance(note, str) and note]

"""What an AI channel reads of a turn a discussion gives (RFC §19.7.5).

A discussion gives a turn as a rerun of the event the turn answers, the event
carrying a mark in its metadata: the turn's notes (who is in the room, which
event the turn answers and who asked, how the room works). The channel adds
the notes to the turn's input and reads the answered event at its place in
the history, before what was said after it, instead of as the conversation's
last message. The channel side imports nothing of the strategy.
"""

from __future__ import annotations

from typing import Any

from roomkit.models.event import RoomEvent

DISCUSSION_TURN = "_discussion_turn"
"""The metadata key a discussion's turn trigger carries."""


def discussion_turn(event: RoomEvent) -> dict[str, Any] | None:
    """The discussion's mark on *event*, or None when no discussion gave the turn."""
    mark = (event.metadata or {}).get(DISCUSSION_TURN)
    return mark if isinstance(mark, dict) else None


def discussion_notes(event: RoomEvent) -> list[str]:
    """The notes a discussion's turn adds to the turn's input."""
    mark = discussion_turn(event)
    if mark is None:
        return []
    return [note for note in mark.get("notes", ()) if isinstance(note, str) and note]

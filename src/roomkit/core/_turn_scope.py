"""The turns the current code runs under (RFC §10.6.5).

The turn backend sets it around a turn's work and the calls the owner serves
for it, and a channel reads it to know the turn its loop serves.
"""

from __future__ import annotations

import contextvars

from roomkit.core.task_utils import callers_read

_TURNS_RUN_UNDER: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "roomkit_turns_run_under", default=()
)
"""The turns the current code runs under, outermost first: a close called
from one of them must not cut it (the turns' :func:`held_runs`)."""


def turns_run_under() -> tuple[str, ...]:
    """The ids of the turns the current code runs under and, inside the
    kit's shutdown, those every ``close()`` it answers was called under
    (RMK-1019)."""
    return tuple(dict.fromkeys(t for turns in callers_read(_TURNS_RUN_UNDER) for t in turns))


def enter_turn(turn_id: str) -> None:
    """The current code runs under *turn_id* too: a close it calls spares
    that turn (the turns' :func:`held_runs`)."""
    turns = _TURNS_RUN_UNDER.get()
    if turn_id not in turns:
        _TURNS_RUN_UNDER.set((*turns, turn_id))


def current_turn_id() -> str | None:
    """The id of the turn the current code runs in (the innermost, when a
    turn runs inside another): inside a turn's work on its executor, its
    tools and hooks included, and inside a call its room owner serves for
    it. What ``RoomKit.steer(room_id, channel_id, directive, turn_id=...)``
    addresses within its room. ``None`` outside a turn."""
    turns = _TURNS_RUN_UNDER.get()
    return turns[-1] if turns else None


_STREAM_CUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "roomkit_turn_stream_cut", default=False
)


def set_turn_stream_cut(cut: bool) -> None:
    """Whether the turn's work now running closes its stream as a cut's (the
    turn was cut, its executor closes, a frame could not cross), not as its
    reader's: the owner then takes nothing more of what the stream handed out
    (RFC §10.6.4). Set by the turn backend for each piece of a turn's work."""
    _STREAM_CUT.set(cut)


def turn_stream_cut() -> bool:
    """Whether the stream the current code closes was cut rather than
    stopped by its reader. A reader that stops took what it was handed; a
    cut turn's owner closes as cancelled each call whose end had not reached
    it (RFC §10.6.4). ``False`` outside a turn: a reader that closes a
    stream it reads itself stops it."""
    return _STREAM_CUT.get()

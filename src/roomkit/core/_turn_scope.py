"""The turns the current code runs under (RFC §10.6.5).

The turn backend sets it around a turn's work and the calls the owner serves
for it, and a channel reads it to know the turn its loop serves.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from roomkit.core.task_utils import callers_read, callers_run, joined_as

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


_TURN_CALL: contextvars.ContextVar[asyncio.Task[Any] | None] = contextvars.ContextVar(
    "roomkit_turn_call", default=None
)
"""The task that runs the call of the innermost turn the current code is in:
the turn's work, or a call its loop runs on a task of its own (a tool's). A
task that call's code starts carries it unchanged: it is no call of the turn."""


def mark_turn_call(task: asyncio.Task[Any]) -> None:
    """*task* runs a call of the turn its code runs under: the turn's work,
    or a tool call its loop starts (RFC §12.10.4 step 3). Set in the task's
    own context, before its first step."""
    task.get_context().run(_TURN_CALL.set, task)


def turns_a_close_spares() -> tuple[str, ...]:
    """The turns a ``close()`` called from the current code lets run on and,
    inside the kit's shutdown, those of every ``close()`` it answers: the
    turns it runs under as their call, the call that awaits ``close()``
    itself. A task a turn's code started on its own carries the turn's marks,
    but a close it awaits is cut with that turn (RFC §12.10.4 step 3)."""
    spared = callers_run(_turns_spared_here)
    return tuple(dict.fromkeys(turn for turns in spared for turn in turns))


def _turns_spared_here() -> tuple[str, ...]:
    turns = _TURNS_RUN_UNDER.get()
    call = _TURN_CALL.get()
    if call is None or call is joined_as():
        return turns
    return turns[:-1]


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

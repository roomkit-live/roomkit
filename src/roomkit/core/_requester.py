"""Who asked for the work a turn delegates (RFC §6.4, §19.7).

When several people speak in the room, a delegated task names out of its block
who asked for it, by the label the conversation gives them. In a tool the
delegating agent called, the task is that agent's wording and the asker the
author of the turn it answers (:func:`~roomkit.tools.current_tool_requester`);
where a strategy hands a participant's own words to its workers, the strategy
names their author (:func:`asked_by`). A one-to-one conversation names no one.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from roomkit.channels._speaker import several_speakers, turn_labels
from roomkit.core.visibility import visible_events
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from roomkit.tools.context import _current_loop_ctx, current_tool_requester

_ASKED_BY: ContextVar[str | None] = ContextVar("roomkit_asked_by", default=None)
_NOT_TURNS = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_END})


def asking_label(event: RoomEvent, context: RoomContext, channel_id: str) -> str | None:
    """The label of *event*'s author when the turns *channel_id* may read and
    *event* hold several speakers; ``None`` in a one-to-one conversation."""
    window = [
        e
        for e in visible_events(context, channel_id)
        if e.id != event.id and e.source.channel_id != channel_id and _a_turn(e)
    ]
    labels = turn_labels([*window, event], context)
    counted = (labels.get(e.id) for e in (*window, event) if _a_turn(e))
    return labels.get(event.id) if several_speakers(counted) else None


def _a_turn(event: RoomEvent) -> bool:
    """Whether *event* is a turn the conversation counts: it has text, and is
    neither a tool call's record nor the application's instruction."""
    if event.type in _NOT_TURNS or event.type == EventType.INSTRUCTION:
        return False
    return bool(extract_event_text(event).strip())


@contextmanager
def asked_by(label: str | None) -> Iterator[None]:
    """Run a delegation that hands *label*'s own words to its workers
    (``None`` in a one-to-one conversation)."""
    token = _ASKED_BY.set(label)
    try:
        yield
    finally:
        _ASKED_BY.reset(token)


def requested_line() -> str:
    """The line a delegated task opens with, out of its block: who asked for
    it, ``Alice asked:`` for their own words, ``Requested by Alice (2), in the
    delegating agent's words:`` for a task an agent wrote; ``""`` when no one
    is named. Inside a tool loop the turn's author is the asker, whatever a
    delegation around it named."""
    if _current_loop_ctx.get() is not None:
        label = current_tool_requester()
        return f"Requested by {label}, in the delegating agent's words:" if label else ""
    label = _ASKED_BY.get()
    return f"{label} asked:" if label else ""


def task_heading(default: str) -> str:
    """The heading of a block holding the user's task: who asked for it when
    someone is named, else *default* (``User request:``)."""
    return requested_line() or default


def with_requester(task: str) -> str:
    """*task*, a model's own input, after the line that names who asked."""
    line = requested_line()
    return f"{line}\n{task}" if line else task

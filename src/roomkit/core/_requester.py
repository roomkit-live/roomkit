"""Who asked for the task a strategy hands a model (RFC §6.4, §19.7).

When several people speak in the room, the user's task a strategy copies into
a model's input as a ``<task>`` block is headed by who asked for it, by the
label the conversation gives them: ``Alice asked:`` before a participant's own
words, named by the strategy that hands them on (:func:`asking_label`);
``Requested by Alice (2), in the delegating agent's words:`` before a task an
agent wrote in a tool call (:func:`~roomkit.tools.current_tool_requester`).
Only the block's heading names the asker: the runtime's own prompts and the
input a worker acts on as its own carry none. A one-to-one conversation names
no one.
"""

from __future__ import annotations

from roomkit.channels._speaker import several_speakers, turn_labels
from roomkit.core.visibility import visible_events
from roomkit.memory.token_estimator import extract_event_text
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from roomkit.tools.context import current_tool_requester

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


def task_heading(default: str, *, asked_by: str | None = None) -> str:
    """The heading of a ``<task>`` block holding the user's task: *asked_by*'s
    label for a participant's own words, else who asked for the task the
    delegating agent wrote in this tool call, else *default*
    (``User request:``) when no one is named."""
    if asked_by:
        return f"{asked_by} asked:"
    label = current_tool_requester()
    return f"Requested by {label}, in the delegating agent's words:" if label else default

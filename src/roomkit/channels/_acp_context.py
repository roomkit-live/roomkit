"""What an ACP session missed while it was not being asked (RFC §19.3.2).

An unsolicited channel is skipped entirely — not asked to answer, and not told
the event happened. For a channel whose context is rebuilt from the room's
timeline every turn that costs nothing. An ACP session holds its history inside
the agent's own process, so what it was not told is gone for good: the room
looks like a shared conversation and is a bundle of private threads.

The catch-up is read from the timeline at the moment the agent *is* solicited,
which is the only form that also works when the session was born after the
conversation — sessions open on the first prompt, so an agent addressed for the
first time in a busy room has an empty one.
"""

from __future__ import annotations

from roomkit.channels._speaker import speaker_label
from roomkit.channels.base import Channel
from roomkit.core.visibility import visible_events
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RichContent, RoomEvent, TextContent

_SKIPPED_TYPES = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_END})
"""Another agent's tool calls are its business, not room conversation."""


def event_text(event: RoomEvent) -> str:
    """The text an ACP agent should read for *event*.

    Rich content is offered as its plain-text rendering: the prompt is a
    string, and a session that received the markup would answer about it.
    """
    content = event.content
    if isinstance(content, TextContent):
        return content.body
    if isinstance(content, RichContent):
        return content.plain_text or content.body
    return Channel.extract_text(event)


def room_context_block(
    context: RoomContext,
    channel_id: str,
    *,
    after_index: int,
    trigger: RoomEvent,
    limit: int,
) -> str:
    """The room's conversation since *after_index*, as one prompt section.

    Returns ``""`` when there is nothing the agent missed — the common case
    once it is in a back-and-forth, and the reason an ordinary exchange pays
    nothing for this.

    What it deliberately leaves out:

    - **Anything visibility withheld.** ``visible_events`` answers this (RFC
      §7.5 rule 8): catching up is not a second door into the room, and the
      agent reads exactly what it would have been delivered had it been asked.
    - **The triggering event.** It follows the block as the actual request.
    - **The agent's own past events.** Its session already holds what it said,
      and a block headed "messages you did not receive" is the wrong place to
      quote it back to itself.

    ``limit`` bounds what is shown, and the header says so when it bites —
    §19.3.2 requires the reader be told its history is partial, because an
    agent that knows it was truncated can ask for the rest while one that
    believes it holds the whole room cannot. The count is taken over the tail
    the framework loaded (``recent_events``), so a gap longer than that tail
    is reported as the part of it we can see.
    """
    if limit <= 0:
        return ""

    missed = [
        event
        for event in visible_events(context, channel_id)
        if event.index > after_index
        and event.id != trigger.id
        and event.source.channel_id != channel_id
        and event.type not in _SKIPPED_TYPES
        and event_text(event).strip()
    ]
    if not missed:
        return ""

    shown = missed[-limit:]
    lines = [
        f"[{position}] {speaker_label(event, context)}: {event_text(event).strip()}"
        for position, event in enumerate(shown, start=1)
    ]
    return "\n".join([_header(len(shown), len(missed)), *lines, "[End of room context]"])


def _header(shown: int, total: int) -> str:
    """Name the block and, when it is cut, say so and by how much."""
    if shown < total:
        return (
            f"[Room context — the {shown} most recent of {total} messages you did not "
            f"receive; the earlier ones are not shown. Context only; the request follows.]"
        )
    plural = "message" if shown == 1 else "messages"
    return (
        f"[Room context — {shown} {plural} you did not receive. "
        f"Context only; the request follows.]"
    )

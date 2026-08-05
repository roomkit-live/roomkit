"""What a solicited ACP turn is asked to read: the prompt and its sections.

Three sections, in this order: what the host contributes, what the session
missed, the request itself.

The middle one is RFC §19.3.2. An unsolicited channel is skipped entirely —
not asked to answer, and not told the event happened. For a channel whose
context is rebuilt from the room's timeline every turn that costs nothing. An
ACP session holds its history inside the agent's own process, so what it was
not told is gone for good: the room looks like a shared conversation and is a
bundle of private threads.

The catch-up is read from the timeline at the moment the agent *is* solicited,
which is the only form that also works when the session was born after the
conversation — sessions open on the first prompt, so an agent addressed for the
first time in a busy room has an empty one.

The first section is the host's. Only the host holds member memories, a
document corpus, or an organisation's rules, and an ACP agent cannot go and
fetch them. It leads the prompt because what the agent missed of the
conversation belongs nearer the request than background does.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from roomkit.channels._speaker import speaker_label
from roomkit.channels.base import Channel
from roomkit.core.visibility import visible_events
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RichContent, RoomEvent, TextContent

logger = logging.getLogger("roomkit.channels.acp")

_SKIPPED_TYPES = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_END})
"""Another agent's tool calls are its business, not room conversation."""

ACPContextContributor = Callable[[RoomContext, RoomEvent], Awaitable[Sequence[str]]]
"""What a host adds to one turn's prompt: blocks, for this request, right now."""


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


async def contributed_blocks(
    contributor: ACPContextContributor | None,
    context: RoomContext,
    trigger: RoomEvent,
    *,
    channel_id: str,
) -> list[str]:
    """What the host adds to this turn — never at the cost of the turn.

    Fail-open: a contributor that fails is logged and the turn goes without
    its blocks, which is how the other host-supplied callbacks in this channel
    behave. Losing the answer because the background context could not be
    assembled would be the worse trade. Reading what came back is inside the
    same guard as the call: a contributor that returns ``None``, or something
    that is not a block, is as broken as one that raised and must cost no
    more. ``BaseException`` is deliberately not caught — a cancelled turn must
    stay cancelled.

    A lone ``str`` counts as one block. ``str`` satisfies ``Sequence[str]``,
    so a type checker passes it through, and iterating it would spell the
    prompt out one character per section.
    """
    if contributor is None:
        return []
    try:
        blocks = await contributor(context, trigger)
        if isinstance(blocks, str):
            blocks = [blocks]
        return [stripped for block in blocks if (stripped := block.strip())]
    except Exception:
        logger.exception("ACP context contributor failed (%s); prompting without it", channel_id)
        return []


def compose_prompt(blocks: Sequence[str], catch_up: str, request: str) -> str:
    """Join the turn's sections: host context, then catch-up, then request.

    With neither blocks nor catch-up the request is the whole prompt, which is
    what an ordinary back-and-forth sends.
    """
    sections = [*blocks, catch_up, request]
    return "\n\n".join(section for section in sections if section)

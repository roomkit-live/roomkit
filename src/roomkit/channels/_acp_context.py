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

from roomkit._text import quoted
from roomkit.channels._speaker import speaker_label
from roomkit.channels.base import Channel
from roomkit.core.visibility import visible_events
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RichContent, RoomEvent, TextContent

logger = logging.getLogger("roomkit.channels.acp")

_SKIPPED_TYPES = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_END})
"""Another agent's tool calls are its business, not room conversation."""

ENTRY_LIMIT = 4000
"""Characters of one message the room context quotes."""

ROOM_CONTEXT_OPENING = "[Room context"
"""How the room context block opens; what follows says how much it holds."""

ROOM_CONTEXT_CLOSING = " Context only; the request follows.]"
"""Ends the room context block's first line."""

ROOM_CONTEXT_END = "[End of room context]"
"""The room context block's last line."""

ACPContextContributor = Callable[[RoomContext, RoomEvent], Awaitable[Sequence[str]]]
"""What a host adds to one turn's prompt: blocks, for this request, right now."""


def acp_event_text(event: RoomEvent) -> str:
    """The text an ACP agent reads for *event*: its prompt, and each line of
    the room context it is given (RFC §6.4, ACP agent channel, item 4).

    Rich content is offered as its plain-text rendering: the prompt is a
    string, and a session that received the markup would answer about it.
    That is where it differs from ``extract_event_text``, which reads a rich
    event's markup body. A host building an ACP prompt of its own reads an
    event as the channel does through this.
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

    - **Anything visibility withheld, and anything the room refused.**
      ``visible_events`` answers both (RFC §7.5 rule 8): catching up is not a
      second door into the room, and the agent reads exactly what it would
      have been delivered had it been asked — a BLOCKED event was delivered to
      nobody.
    - **The triggering event.** It follows the block as the actual request.
    - **The agent's own past events.** Its session already holds what it said,
      and a block headed "messages you did not receive" is the wrong place to
      quote it back to itself. The exception is a reply from a standalone
      turn (RFC §10.1.1 step 7): another session produced it, so this one
      never held it, and it is shown as the agent's own words.

    ``limit`` bounds what is shown, and the header says so when it bites —
    §19.3.2 requires the reader be told its history is partial, because an
    agent that knows it was truncated can ask for the rest while one that
    believes it holds the whole room cannot. The count is taken over the tail
    the framework loaded (``recent_events``), which may stop short of the
    cursor: on a room with no hook the framework loads exactly the window the
    channels declare (RMK-103). The indices say so without loading more — a
    tail whose oldest event sits past ``after_index + 1`` left events of the
    room unread, and the header reports that gap as an upper bound (it counts
    room events, some of which the agent would not have been shown).
    """
    if limit <= 0:
        return ""

    missed = [
        event
        for event in visible_events(context, channel_id)
        if event.index > after_index
        and event.id != trigger.id
        and (event.source.channel_id != channel_id or _from_standalone_turn(event))
        and event.type not in _SKIPPED_TYPES
        and acp_event_text(event).strip()
    ]
    oldest = min((event.index for event in context.recent_events), default=after_index + 1)
    unloaded = max(0, oldest - after_index - 1)
    if not missed:
        # Nothing loaded is new to the agent (its own turn's tool calls and
        # replies can fill the tail), yet the gap may reach past the tail:
        # that is still partial, and silence would say otherwise.
        if not unloaded:
            return ""
        return (
            f"{ROOM_CONTEXT_OPENING} — none of the loaded messages are new to you; "
            f"{_not_loaded(unloaded)}.{ROOM_CONTEXT_CLOSING}"
        )

    shown = missed[-limit:]
    # Each message quoted on its line: it cannot end the block nor start a
    # line of its own (RFC §6.4).
    lines = [
        f"[{position}] {_label(event, context, channel_id)}: "
        f"{quoted(acp_event_text(event), ENTRY_LIMIT)}"
        for position, event in enumerate(shown, start=1)
    ]
    header = _header(len(shown), len(missed), unloaded)
    return "\n".join([header, *lines, ROOM_CONTEXT_END])


def _from_standalone_turn(event: RoomEvent) -> bool:
    """Whether *event* is a reply the channel produced in a standalone turn's session."""
    acp_meta = event.metadata.get("acp")
    return isinstance(acp_meta, dict) and acp_meta.get("standalone") is True


def _label(event: RoomEvent, context: RoomContext, channel_id: str) -> str:
    if event.source.channel_id == channel_id:
        return "you (in a separate session)"
    return speaker_label(event, context)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _not_loaded(unloaded: int) -> str:
    verb = "was" if unloaded == 1 else "were"
    return f"up to {_count(unloaded, 'earlier room event')} {verb} not loaded"


def _header(shown: int, total: int, unloaded: int) -> str:
    """Name the block and, when it is cut, say so and by how much.

    Two cuts can apply: ``room_history`` trims what the loaded tail holds
    (*total* counted), and the tail itself may not reach back to the cursor
    (*unloaded*, an upper bound in room events).
    """
    if unloaded:
        lead = (
            f"the {shown} most recent of {total} loaded messages"
            if shown < total
            else f"the {_count(shown, 'most recent message')}"
        )
        return (
            f"{ROOM_CONTEXT_OPENING} — {lead} you did not receive; "
            f"{_not_loaded(unloaded)}.{ROOM_CONTEXT_CLOSING}"
        )
    if shown < total:
        return (
            f"{ROOM_CONTEXT_OPENING} — the {shown} most recent of {total} messages you did not "
            f"receive; the earlier ones are not shown.{ROOM_CONTEXT_CLOSING}"
        )
    return (
        f"{ROOM_CONTEXT_OPENING} — {_count(shown, 'message')} you did not receive."
        f"{ROOM_CONTEXT_CLOSING}"
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

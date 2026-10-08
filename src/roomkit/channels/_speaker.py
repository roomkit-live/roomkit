"""Who is speaking, as a transcript should name them.

Shared by the surfaces that write a room's conversation out for someone else
to read — the console transcript, and the room context an ACP session is
handed. Both answer the same question, and a room where the console says
"Marie · sms" while an agent is told "sms" names one person two ways.
"""

from __future__ import annotations

from collections.abc import Callable

from roomkit._text import identifier, person_name, quoted
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant

SPEAKER_KEY = "speaker"
"""The ``AIMessage.metadata`` key naming whose words a user message carries when
the context prefixes them with that name (``"Name: text"``): a transcript
reads the name there, never from the text, where anyone can write ``Name:``."""


def speaker_label(
    event: RoomEvent,
    context: RoomContext,
    agent_label: Callable[[str], str] | None = None,
) -> str:
    """Name the author of *event* the way a reader should see it.

    A person gets their own name and the channel they speak through —
    ``"Marie · sms"`` — because in a room holding several humans, the channel
    id names none of them: two colleagues texting in would otherwise share one
    handle. Anything without a participant (an agent, a system event) keeps the
    channel-derived label.

    ``agent_label`` renames those channel-derived labels for presentation; it
    defaults to the channel id itself, which is what a room addressed by
    ``@channel-id`` should show.

    ``source.participant_id`` holds a ``Participant.id`` when the channel names
    its own sender, and an ``Identity.id`` when the identity pipeline resolved
    one (RFC §11) — two namespaces in one field, so both are tried.
    """
    label = agent_label if agent_label is not None else _channel_id_label
    participant_id = event.source.participant_id
    if not participant_id:
        return label(event.source.channel_id)

    person = next(
        (p for p in context.participants if p.id == participant_id),
        None,
    ) or next(
        (p for p in context.participants if p.identity_id == participant_id),
        None,
    )
    if person is None:
        return label(event.source.channel_id)
    return f"{participant_name(person)} · {event.source.channel_id}"


def participant_name(participant: Participant) -> str:
    """*participant*'s name as a model reads it: written by whoever registers the
    person, so kept to a name's characters, or to an identifier's when it has
    no name (RFC §6.4)."""
    return person_name(participant.display_name) or identifier(participant.id, "participant")


def channel_label(channel_id: str) -> str:
    """The label of a turn whose author has no name, in a transcript that names
    people by their bare names: its channel as the room addresses it
    (``@sms1``), kept to an identifier's characters. No name takes that form
    (a person's name drops ``@``), so a person named ``ai2`` or ``sms1`` does
    not read as an agent or as a nameless channel (RFC §6.4)."""
    return f"@{identifier(channel_id, 'channel')}"


def _channel_id_label(channel_id: str) -> str:
    """The channel id, unchanged — the default way to name a non-person."""
    return channel_id


def said_by(text: str, speaker: object, limit: int, *, label: str | None = None) -> str:
    """*text* quoted within *limit*, after the name the context gave its speaker
    (``SPEAKER_KEY``) when it gave one, shown as *label* when given: the name
    out of the quote, so a person who writes ``Name:`` is not read as someone
    else (RFC §6.4)."""
    prefix = f"{speaker}: "
    if isinstance(speaker, str) and text.startswith(prefix):
        return f"{label or speaker}: {quoted(text[len(prefix) :], limit)}"
    return quoted(text, limit)

"""Who is speaking, as a transcript should name them.

A model reads a participant's turn under one label wherever it reads it (the
conversation, the room context handed to an ACP agent, a line broadcast into a
realtime session): :func:`turn_labels`, from :func:`author_name`. The console
names a person for a human reader, with their channel (:func:`speaker_label`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from roomkit._lookalike import skeletons
from roomkit._text import identifier, person_name, quoted
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
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
    person = _participant(event, context)
    if person is None:
        return label(event.source.channel_id)
    return f"{participant_name(person)} · {event.source.channel_id}"


def _participant(event: RoomEvent, context: RoomContext) -> Participant | None:
    """The room's participant behind *event*: ``source.participant_id`` holds a
    ``Participant.id`` when the channel names its own sender, and an
    ``Identity.id`` when the identity pipeline resolved one (RFC §11), so both
    are tried."""
    participant_id = event.source.participant_id
    if not participant_id:
        return None
    return next(
        (p for p in context.participants if p.id == participant_id),
        None,
    ) or next((p for p in context.participants if p.identity_id == participant_id), None)


def author_name(event: RoomEvent, context: RoomContext) -> str | None:
    """Display name of whoever is behind *event*, kept to a name's characters
    (RFC §6.4), or ``None``.

    ``metadata["sender_name"]`` is the stamp transports and hosts write at
    ingress (the Teams and WhatsApp providers do, and so does a host's
    session ingress); the room's participant record is the fallback for
    transports that register named participants without stamping events.
    Either is written by whoever sends: given unquoted before their words, a
    name must not open a line or a frame of its own.
    """
    name = event.metadata.get("sender_name")
    if isinstance(name, str) and (kept := person_name(name)):
        return kept
    person = _participant(event, context)
    return (person_name(person.display_name) or None) if person is not None else None


def turn_labels(events: Iterable[RoomEvent], context: RoomContext) -> dict[str, str | None]:
    """The label each of *events* opens with where a model reads it (RFC §6.4):
    its author's name, or its channel (:func:`channel_label`) when they have
    none; ``None`` for the runtime's system events, no participant's turn.

    Names are told apart across sources (a channel and its sender): when a
    later source's name reads like an earlier one's (in case or in Unicode's
    confusables, ``Alice`` and ``Аlice``), it carries its rank, ``Аlice (2)``,
    a form no name takes, so a sender who takes another's name does not read
    as that person. The first source seen keeps the name."""
    labels: dict[str, str | None] = {}
    ranks: list[tuple[frozenset[str], tuple[str, str | None]]] = []
    for event in events:
        labels[event.id] = _label(event, context, ranks)
    return labels


def _label(
    event: RoomEvent,
    context: RoomContext,
    ranks: list[tuple[frozenset[str], tuple[str, str | None]]],
) -> str | None:
    """*event*'s label, *ranks* holding the sources whose names were seen, in
    order, each with the forms its name reads as."""
    if event.source.channel_type == ChannelType.SYSTEM:
        return None
    name = author_name(event, context)
    if name is None:
        return channel_label(event.source.channel_id)
    source = (event.source.channel_id, event.source.participant_id)
    keys = skeletons(name)
    alike = [seen for read, seen in ranks if read & keys]
    if source not in alike:
        ranks.append((keys, source))
        alike.append(source)
    rank = alike.index(source) + 1
    return name if rank == 1 else f"{name} ({rank})"


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

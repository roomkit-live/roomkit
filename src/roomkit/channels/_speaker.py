"""Who is speaking, as a transcript should name them.

A model reads a participant's turn under one label wherever it reads it (the
conversation, the room context handed to an ACP agent, a line broadcast into a
realtime session): :func:`turn_labels`, from :func:`author_name`. The console
names a person for a human reader, with their channel (:func:`speaker_label`).
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Iterable
from typing import Any

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
    """The room's participant behind *event*."""
    return _people(context).get(event.source.participant_id or "")


def _people(context: RoomContext) -> dict[str, Participant]:
    """The room's participants by id and by identity: ``source.participant_id``
    holds a ``Participant.id`` when the channel names its own sender, and an
    ``Identity.id`` when the identity pipeline resolved one (RFC §11); an id
    wins over an identity."""
    people = {p.identity_id: p for p in context.participants if p.identity_id}
    return people | {p.id: p for p in context.participants}


def author_name(event: RoomEvent, context: RoomContext) -> str | None:
    """Display name of whoever is behind *event*, kept to a name's characters
    (RFC §6.4), or ``None`` (:func:`_author`)."""
    return _author(event, _people(context))


def _author(event: RoomEvent, people: dict[str, Participant]) -> str | None:
    """``metadata["sender_name"]`` first: the stamp a transport or host writes
    at ingress, and the voice a channel's diarization names on a shared
    microphone (RFC §12.2.3); the name the room's participant record holds
    otherwise. Either is kept to a name's characters: given unquoted before
    their words, a name must not open a line or a frame of its own. A sender
    who takes a registered person's name is another source, and carries a
    rank (:func:`turn_labels`)."""
    name = event.metadata.get("sender_name")
    if isinstance(name, str) and (kept := person_name(name)):
        return kept
    person = people.get(event.source.participant_id or "")
    return (person_name(person.display_name) or None) if person is not None else None


AUTHOR_RANK = "author_rank"
"""The event metadata key holding its author's rank among the room's sources
whose names read alike, fixed when the event is committed (RFC §10.1 step
12)."""

AUTHOR_REGISTER = "author_register"
"""The room metadata key holding the sources whose names the room has seen,
in order, each with what its name reads as (RFC §6.4). Both are kept as
digests salted with the room's id: the register compares, it never needs to
show a sender's id or name, and the room's metadata may reach clients."""

Register = list[dict[str, Any]]


def turn_labels(events: Iterable[RoomEvent], context: RoomContext) -> dict[str, str | None]:
    """The label each of *events* opens with where a model reads it (RFC §6.4):
    its author's name, or its channel (:func:`channel_label`) when they have
    none; ``None`` for the runtime's system events, no participant's turn.

    Names are told apart across sources (a participant, whichever channel
    reached them, or a sender on its channel): when a source's name reads
    like an earlier one's (in case or in Unicode's confusables, ``Alice`` and
    ``Аlice``), it carries its rank, ``Аlice (2)``, a form no name takes, so
    a sender who takes another's name does not read as that person. The
    room's named participants come first, in the order they joined, whoever
    spoke first. The rank an event carries (:data:`AUTHOR_RANK`, fixed for
    the room when it was committed) is the one read. A name that reads as a
    label the agent's own turns carry (``You``) carries ``(a participant)``,
    so no one reads as the agent."""
    people = _people(context)
    window: Register = []
    _seed(window, context, "")
    return {event.id: _label(event, people, window) for event in events}


def _label(event: RoomEvent, people: dict[str, Participant], window: Register) -> str | None:
    if event.source.channel_type == ChannelType.SYSTEM:
        return None
    name = _author(event, people)
    if name is None:
        return channel_label(event.source.channel_id)
    keys = skeletons(name)
    if keys & _agent_labels():
        return f"{name} (a participant)"
    rank = _rank_in(window, _digest("", _source(event, people)), _digests("", keys))
    stamped = event.metadata.get(AUTHOR_RANK)
    rank = stamped if isinstance(stamped, int) and stamped > 0 else rank
    return name if rank == 1 else f"{name} ({rank})"


def author_rank(
    event: RoomEvent, context: RoomContext, register: Register, *, enters: bool = True
) -> int | None:
    """*event*'s author rank in the room (:data:`AUTHOR_RANK`), *register*
    (the room's :data:`AUTHOR_REGISTER`) extended with the room's named
    participants, and with the event's source when it is new and *enters*;
    ``None`` for a system event or an author with no name."""
    if event.source.channel_type == ChannelType.SYSTEM:
        return None
    salt = context.room.id
    _seed(register, context, salt)
    people = _people(context)
    name = _author(event, people)
    if name is None:
        return None
    source = _digest(salt, _source(event, people))
    return _rank_in(register, source, _digests(salt, skeletons(name)), enters=enters)


def _seed(register: Register, context: RoomContext, salt: str) -> None:
    """The room's named participants enter *register* first, in the order they
    joined: a name the application registered holds its rank against a
    sender who takes it, whoever spoke first."""
    for person in sorted(context.participants, key=lambda p: p.joined_at):
        if name := person_name(person.display_name):
            source = _digest(salt, ["participant", person.id])
            _rank_in(register, source, _digests(salt, skeletons(name)))


def _digest(salt: str, parts: Iterable[str]) -> str:
    return hashlib.sha256("\x1f".join([salt, *parts]).encode()).hexdigest()[:24]


def _digests(salt: str, keys: Iterable[str]) -> set[str]:
    return {_digest(salt, [key]) for key in keys}


def register_of(value: object) -> Register:
    """A room's :data:`AUTHOR_REGISTER` as stored, its well-formed entries
    only: the room's metadata is the application's to write too."""
    if not isinstance(value, list):
        return []
    return [
        {"source": entry["source"], "names": list(entry["names"])}
        for entry in value
        if isinstance(entry, dict)
        and isinstance(entry.get("source"), str)
        and isinstance(entry.get("names"), list)
        and all(isinstance(name, str) for name in entry["names"])
    ]


def _rank_in(register: Register, source: str, keys: set[str], *, enters: bool = True) -> int:
    """*source*'s rank among the sources of *register* whose names read like
    *keys*, *register* extended when the source is new to it and *enters*."""
    alike: list[str] = []
    for entry in register:
        if keys.intersection(entry["names"]) and entry["source"] not in alike:
            alike.append(entry["source"])
    if source not in alike:
        if enters:
            register.append({"source": source, "names": sorted(keys)})
        alike.append(source)
    return alike.index(source) + 1


@functools.cache
def _agent_labels() -> frozenset[str]:
    """What the labels a model reads as its own turns read as: ``You`` (the
    thinker) and ``you (in a separate session)`` (an ACP agent's room
    context). A name reading as one carries ``(a participant)``."""
    return skeletons("You") | skeletons("you (in a separate session)")


def _source(event: RoomEvent, people: dict[str, Participant]) -> list[str]:
    """Who *event* comes from: the room's participant, whichever channel and
    id reached them (an identity is one person); else its sender on its
    channel, since a sender id from one transport says nothing of another's;
    else its channel."""
    person = people.get(event.source.participant_id or "")
    if person is not None:
        return ["participant", person.id]
    if event.source.participant_id:
        return ["sender", event.source.channel_id, event.source.participant_id]
    return ["channel", event.source.channel_id]


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

"""Who is speaking, as a transcript should name them.

A model reads a participant's turn under one label wherever it reads it (the
conversation, the room context handed to an ACP agent, a line broadcast into a
realtime session): :func:`turn_labels`, from :func:`author_name`. The console
names a person for a human reader, with their channel (:func:`speaker_label`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable

from roomkit._lookalike import skeletons
from roomkit._text import identifier, person_name, quoted
from roomkit.core._authors import (
    AUTHOR_REGISTER,
    AuthorRegister,
    People,
    digest,
    name_keys,
    recorded_author,
    source_of,
)
from roomkit.core._authors import author_name as _author_name
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
    return People(context.participants).of(event)


def author_name(event: RoomEvent, context: RoomContext) -> str | None:
    """Display name of whoever is behind *event*, kept to a name's characters
    (RFC §6.4), or ``None``."""
    return _author_name(event, People(context.participants))


def turn_labels(events: Iterable[RoomEvent], context: RoomContext) -> dict[str, str | None]:
    """The label each of *events* opens with where a model reads it (RFC §6.4):
    its author's name, or its channel (:func:`channel_label`) when they have
    none; ``None`` for the runtime's system events, no participant's turn.

    Names are told apart across sources (a participant, whichever channel
    reached them, or a sender on its channel): when a source's name reads
    like another's (in case or in Unicode's confusables, ``Alice`` and
    ``Аlice``), it carries its rank, ``Аlice (2)``, a form no name takes, so
    a sender who takes another's name does not read as that person. The
    record of its author a turn was committed with (its name and rank,
    :data:`~roomkit.core._authors.AUTHOR`) is the one read, while the turn's
    source is the one recorded; a turn with none is ranked against the room's
    register. A name that reads as a label the agent's own turns carry
    (``You``) carries ``(a participant)``, so no one reads as the agent."""
    unrecorded = _Unrecorded(context)
    return {event.id: _label(event, unrecorded) for event in events}


def _label(event: RoomEvent, unrecorded: _Unrecorded) -> str | None:
    if event.source.channel_type == ChannelType.SYSTEM:
        return None
    people, salt = unrecorded.people, unrecorded.salt
    source = digest(salt, source_of(event, people))
    record = recorded_author(event)
    name = person_name(record[0]) if record is not None and record[2] == source else ""
    if name and record is not None:
        rank = record[1]
    else:
        name = _author_name(event, people)
        if name is None:
            return channel_label(event.source.channel_id)
        rank = unrecorded.rank(source, name_keys(salt, name))
    if skeletons(name) & _agent_labels():
        return f"{name} (a participant)"
    return name if rank == 1 else f"{name} ({rank})"


class _Unrecorded:
    """Ranks for turns that carry no record of their author (from before the
    register, or whose metadata or source was replaced since): a copy of the
    room's register, extended with the room's named participants and with
    each new source in the order the turns are read. Copied on first use,
    which a window of recorded turns never makes."""

    def __init__(self, context: RoomContext) -> None:
        self.people = People(context.participants)
        self.salt = context.room.id
        self._stored = context.room.metadata.get(AUTHOR_REGISTER)
        self._register: AuthorRegister | None = None

    def rank(self, source: str, names: set[str]) -> int:
        if self._register is None:
            stored = AuthorRegister.stored(self._stored)
            self._register = stored.copy() if stored is not None else AuthorRegister()
            self._register.seed(self.people, self.salt)
        return self._register.rank(source, names)


@functools.cache
def _agent_labels() -> frozenset[str]:
    """What the labels a model reads as its own turns read as: ``You`` (the
    thinker) and ``you (in a separate session)`` (an ACP agent's room
    context). A name reading as one carries ``(a participant)``."""
    return skeletons("You") | skeletons("you (in a separate session)")


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

"""Who wrote a turn, and the rank that tells apart a room's sources whose names
read alike (RFC §6.4, §10.1 step 12).

A turn's author is the name a transport, a host or a diarized voice stamps on
it (``metadata["sender_name"]``), else the room's participant record. Two
sources whose names read alike (``Alice``, ``ALICE``, ``Аlice``) are told
apart by a rank the room fixes when a turn is committed: the room keeps a
register (:data:`AUTHOR_REGISTER`) of the names it has seen, grouped by what
they read as, and of each source's rank in each group; the turn carries its
author's name, rank and source (:data:`AUTHOR`), so a reader names it as it
was named then. The register holds digests salted with the room's id: it
compares, it never needs to show a sender's id or name, and the room's
metadata may reach clients.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Iterable
from typing import Any

from roomkit._lookalike import skeletons
from roomkit._text import person_name
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventStatus
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant
from roomkit.store.base import ConversationStore

AUTHOR = "author"
"""The event metadata key holding the record of its author, fixed when the
event is committed: ``{"name": "Alice", "rank": 2, "source": "<digest>"}``."""

AUTHOR_REGISTER = "author_register"
"""The room metadata key holding the room's register of authors."""

_VERSION = 3
_PAGE = 500
_PAIR = re.compile(r"([0-9a-f]{16}):([1-9][0-9]{0,8})")


class People:
    """The room's participants as a turn's ``source.participant_id`` names
    them: a ``Participant.id`` names its participant on a channel they are
    reached through (``channel_id``, ``connected_via``), since an id one
    transport gives says nothing of another's; an ``Identity.id`` the
    identity pipeline resolved names its participant on any (RFC §5.5, §11)."""

    def __init__(self, participants: Iterable[Participant]) -> None:
        self.participants = list(participants)
        self._by_id = {p.id: p for p in self.participants}
        self._by_identity = {p.identity_id: p for p in self.participants if p.identity_id}

    def of(self, event: RoomEvent) -> Participant | None:
        """The participant behind *event*, or ``None``."""
        pid = event.source.participant_id
        if not pid:
            return None
        person = self._by_id.get(pid)
        # connected_via holds the primary channel too.
        if person is not None and event.source.channel_id in person.connected_via:
            return person
        return self._by_identity.get(pid)


def author_name(event: RoomEvent, people: People) -> str | None:
    """``metadata["sender_name"]`` first: the stamp a transport or host writes
    at ingress, and the voice a channel's diarization names on a shared
    microphone (RFC §12.2.3); the name the room's participant record holds
    otherwise. Either is kept to a name's characters: given unquoted before
    their words, a name must not open a line or a frame of its own."""
    name = event.metadata.get("sender_name")
    if isinstance(name, str) and (kept := person_name(name)):
        return kept
    person = people.of(event)
    return (person_name(person.display_name) or None) if person is not None else None


def source_of(event: RoomEvent, people: People) -> list[str]:
    """Who *event* comes from: the room's participant, whichever channel
    reached them; else its sender on its channel; else its channel."""
    person = people.of(event)
    if person is not None:
        return ["participant", person.id]
    if event.source.participant_id:
        return ["sender", event.source.channel_id, event.source.participant_id]
    return ["channel", event.source.channel_id]


def digest(salt: str, parts: Iterable[str]) -> str:
    """*parts* hashed with *salt* (the room's id)."""
    return hashlib.sha256("\x1f".join([salt, *parts]).encode()).hexdigest()[:16]


def name_keys(salt: str, name: str) -> set[str]:
    """What *name* reads as (:func:`~roomkit._lookalike.skeletons`), hashed."""
    return {digest(salt, [key]) for key in skeletons(name)}


def recorded_author(event: RoomEvent) -> tuple[str, int, str] | None:
    """The name, rank and source *event* was committed with (:data:`AUTHOR`),
    or ``None`` when it carries no well-formed record."""
    record = event.metadata.get(AUTHOR)
    if not isinstance(record, dict):
        return None
    name, rank, source = record.get("name"), record.get("rank"), record.get("source")
    if isinstance(name, str) and _is_rank(rank) and isinstance(source, str):
        return name, rank, source
    return None


class AuthorRegister:
    """A room's register of authors: for each thing a name reads as (a
    skeleton, hashed), the sources whose names read so and the rank of each.

    Two sources whose names read alike share a skeleton, so their ranks are
    told apart there: a source new to the names it uses takes one more than
    the highest rank the sources whose names read like them hold (``Lan``,
    then ``Ian (2)``, then ``ian (3)``), and a registered participant the
    lowest rank none of them holds. A source keeps its rank while no source
    whose name reads like the one it uses holds it, so a name that reads like
    two others ranks after both and renumbers neither.

    Kept as JSON in the room's metadata, which every read of the room copies
    or parses, so its values are strings: ``names`` maps each hashed
    skeleton to its ``"source:rank"`` pairs. The application may write the
    metadata too: what is malformed in it is read as absent."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data or {"version": _VERSION, "names": {}}
        self.changed = data is None

    @classmethod
    def stored(cls, value: object) -> AuthorRegister | None:
        """The register *value* (a room's :data:`AUTHOR_REGISTER`) holds, or
        ``None`` when it holds none of this version."""
        if not isinstance(value, dict) or type(value.get("version")) is not int:
            return None
        if value["version"] == _VERSION and isinstance(value.get("names"), dict):
            return cls(value)
        return None

    def copy(self) -> AuthorRegister:
        """A register a reader may extend without touching this one: its
        values are strings, so a copy of the map suffices."""
        return AuthorRegister({**self.data, "names": dict(self.data["names"])})

    def rank(self, source: str, names: set[str]) -> int:
        """*source*'s rank under *names*: the rank it holds when no other
        source under them holds it, else one more than the highest held;
        filed under each of *names*."""
        mine, others = self._ranks(source, names)
        kept = sorted(mine - others)
        rank = kept[0] if kept else max(mine | others, default=0) + 1
        self._file(source, names, rank)
        return rank

    def seat(self, source: str, names: set[str]) -> int:
        """*source*'s rank under *names* as a registered participant's: the
        rank it holds when no other source under them holds it, else the
        lowest none of them holds."""
        mine, others = self._ranks(source, names)
        kept = sorted(mine - others)
        rank = kept[0] if kept else next(n for n in itertools.count(1) if n not in others)
        self._file(source, names, rank)
        return rank

    def claim(self, source: str, names: set[str], rank: int) -> None:
        """Record *source* at *rank* under *names*, as a turn's record says it
        was; a rank another source under them holds stays theirs, and
        *source* takes a new one at its next turn."""
        _mine, others = self._ranks(source, names)
        if rank not in others:
            self._file(source, names, rank)

    def seed(self, people: People, salt: str) -> None:
        """The room's named participants take their seats, in the order they
        joined: a name the application registered holds the lowest rank free
        against a sender who takes it, whoever spoke first."""
        for person in sorted(people.participants, key=lambda p: p.joined_at):
            if name := person_name(person.display_name):
                self.seat(digest(salt, ["participant", person.id]), name_keys(salt, name))

    def _ranks(self, source: str, names: set[str]) -> tuple[set[int], set[int]]:
        """The ranks *source* holds under *names*, and those the other sources
        under them hold."""
        mine: set[int] = set()
        others: set[int] = set()
        for name in names:
            for holder, rank in self._entries(name).items():
                (mine if holder == source else others).add(rank)
        return mine, others

    def _entries(self, name: str) -> dict[str, int]:
        """The sources filed under *name* and their ranks, its malformed pairs
        left out."""
        value = self.data["names"].get(name)
        if not isinstance(value, str):
            return {}
        pairs = (_PAIR.fullmatch(pair) for pair in value.split())
        return {found[1]: int(found[2]) for found in pairs if found}

    def _file(self, source: str, names: set[str], rank: int) -> None:
        for name in sorted(names):
            entries = self._entries(name)
            if entries.get(source) != rank:
                entries[source] = rank
                self.data["names"][name] = " ".join(f"{s}:{r}" for s, r in entries.items())
                self.changed = True


def _is_rank(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def author_record(
    event: RoomEvent, context: RoomContext, register: AuthorRegister
) -> dict[str, Any] | None:
    """The record *event* is committed with (:data:`AUTHOR`): its author's
    name, their rank in *register* (extended with the room's named
    participants, and with the source or its name when new), its source;
    ``None`` for a system event or an author with no name."""
    if event.source.channel_type == ChannelType.SYSTEM:
        return None
    people = People(context.participants)
    name = author_name(event, people)
    if name is None:
        return None
    salt = context.room.id
    register.seed(people, salt)
    source = digest(salt, source_of(event, people))
    return {"name": name, "rank": register.rank(source, name_keys(salt, name)), "source": source}


class _Rebuild:
    """A register for a room that holds none (a room from before it), from
    its timeline, page by page: the ranks its turns' records hold, then its
    named participants' seats, then the authors of its turns that hold no
    record, in index order."""

    def __init__(self, context: RoomContext) -> None:
        self.register = AuthorRegister()
        self._people = People(context.participants)
        self._salt = context.room.id
        self._unrecorded: list[tuple[str, set[str]]] = []

    def add(self, events: Iterable[RoomEvent]) -> None:
        for event in events:
            if event.status == EventStatus.BLOCKED:
                continue
            if event.source.channel_type == ChannelType.SYSTEM:
                continue
            record = recorded_author(event)
            if record is not None and (name := person_name(record[0])):
                self.register.claim(record[2], name_keys(self._salt, name), record[1])
            elif (name := author_name(event, self._people)) is not None:
                source = digest(self._salt, source_of(event, self._people))
                self._unrecorded.append((source, name_keys(self._salt, name)))

    def done(self) -> AuthorRegister:
        self.register.seed(self._people, self._salt)
        for source, names in self._unrecorded:
            self.register.rank(source, names)
        return self.register


async def with_author(store: ConversationStore, room_id: str, event: RoomEvent) -> RoomEvent:
    """*event* carrying the record of its author (:data:`AUTHOR`), the room's
    register extended first when the source or its name is new, or rebuilt
    from the timeline when the room holds none (RFC §10.1 step 12).

    Run under the room lock, before the commit: a stored turn's record never
    changes, and a store shared across processes needs the distributed lock
    of RFC §13.5. The register is written first, so a commit that fails after
    it leaves a rank unused, never one held twice. A record the event came
    with is dropped, the runtime's alone; a BLOCKED record takes none. A
    restricted turn joins the register too: ranked without joining it, a
    later source could take its rank in the view of a reader who sees both."""
    if AUTHOR in event.metadata:
        metadata = {k: v for k, v in event.metadata.items() if k != AUTHOR}
        event = event.model_copy(update={"metadata": metadata})
    named = event.source.participant_id or event.metadata.get("sender_name")
    if not named or event.status == EventStatus.BLOCKED:
        return event
    room = await store.get_room(room_id)
    if room is None:
        return event
    context = RoomContext(room=room, participants=await store.list_participants(room_id))
    register = AuthorRegister.stored(room.metadata.get(AUTHOR_REGISTER))
    if register is None:
        register = await _rebuilt(store, room_id, context)
    record = author_record(event, context, register)
    if register.changed:
        await store.patch_room_metadata(room_id, {AUTHOR_REGISTER: register.data})
    if record is None:
        return event
    return event.model_copy(update={"metadata": {**event.metadata, AUTHOR: record}})


async def _rebuilt(store: ConversationStore, room_id: str, context: RoomContext) -> AuthorRegister:
    """The register *room_id*'s timeline rebuilds (:class:`_Rebuild`)."""
    rebuild = _Rebuild(context)
    offset = 0
    while page := await store.list_events(room_id, offset=offset, limit=_PAGE):
        rebuild.add(page)
        offset += len(page)
    return rebuild.done()

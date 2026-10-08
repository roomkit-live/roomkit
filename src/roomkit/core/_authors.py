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

_VERSION = 2
_PAGE = 500


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
    """A room's register of authors: the names the room has seen, grouped by
    what they read as, directly or through another name (``Lan``, ``Ian`` and
    ``ian`` form one group), and each source's rank in each group.

    A source new to a group takes the group's highest rank plus one, so two
    sources whose names read alike never share a rank; a source keeps its
    rank in a group. When a name joins two groups, a source of the younger
    whose rank a source of the older holds takes a new one.

    Kept as JSON in the room's metadata, which every read of the room copies
    or parses, so its values are strings and numbers only: ``names`` files
    each name under a group, ``groups`` holds each group's highest rank, and
    ``ranks`` each source's ``"group:rank"`` pairs. The application may write
    the metadata too: what is malformed in it is read as absent."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data or {
            "version": _VERSION,
            "next": 1,
            "names": {},
            "groups": {},
            "ranks": {},
        }
        self.changed = data is None
        self._index: _Index | None = None

    @classmethod
    def stored(cls, value: object) -> AuthorRegister | None:
        """The register *value* (a room's :data:`AUTHOR_REGISTER`) holds, or
        ``None`` when it holds none of this version."""
        if not isinstance(value, dict) or value.get("version") != _VERSION:
            return None
        if not _is_rank(value.get("next")):
            return None
        if all(isinstance(value.get(key), dict) for key in ("names", "groups", "ranks")):
            return cls(value)
        return None

    def copy(self) -> AuthorRegister:
        """A register a reader may extend without touching this one: its
        values are strings and numbers, so a copy of each map suffices."""
        maps = {key: dict(self.data[key]) for key in ("names", "groups", "ranks")}
        return AuthorRegister({**self.data, **maps})

    def rank(self, source: str, names: set[str]) -> int:
        """*source*'s rank in the group of *names*, the register extended when
        the source or one of the names is new to it."""
        gid = self._join(names)
        held = self._ranks(source).get(gid)
        if held is not None:
            return held
        return self._enter(source, gid, self.data["groups"][gid] + 1)

    def claim(self, source: str, names: set[str], rank: int) -> None:
        """Record *source* at *rank* in the group of *names*, as a turn's
        record says it was; a rank another source of the group holds stays
        theirs, and *source* takes a new one at its next turn."""
        gid = self._join(names)
        if gid in self._ranks(source) or rank in self._held(gid):
            return
        self._enter(source, gid, rank)

    def seed(self, people: People, salt: str) -> None:
        """The room's named participants enter first, in the order they
        joined: a name the application registered holds its rank against a
        sender who takes it, whoever spoke first."""
        for person in sorted(people.participants, key=lambda p: p.joined_at):
            if name := person_name(person.display_name):
                self.rank(digest(salt, ["participant", person.id]), name_keys(salt, name))

    def _join(self, names: set[str]) -> str:
        """The group *names* belong to, the oldest of theirs with the others
        merged into it, or a new one; each name filed under it."""
        filed = (self.data["names"].get(name) for name in names)
        found = sorted({gid for gid in filed if self._is_group(gid)}, key=int)
        if found:
            gid = found[0]
            for other in found[1:]:
                self._merge(gid, other)
        else:
            gid = str(self.data["next"])
            self.data["next"] += 1
            self.data["groups"][gid] = 0
            self.changed = True
        for name in sorted(names):
            if self.data["names"].get(name) != gid:
                self.data["names"][name] = gid
                if self._index is not None:
                    self._index.names.setdefault(gid, []).append(name)
                self.changed = True
        return gid

    def _merge(self, gid: str, other: str) -> None:
        index = self._indexed()
        held = self._held(gid)
        top = max(self.data["groups"][gid], self.data["groups"].pop(other))
        for name in index.names.pop(other, []):
            self.data["names"][name] = gid
            index.names.setdefault(gid, []).append(name)
        for source in index.members.pop(other, []):
            ranks = self._ranks(source)
            rank = ranks.pop(other, None)
            if gid not in ranks:
                if rank is None or rank in held:
                    top += 1
                    rank = top
                ranks[gid] = rank
                held.add(rank)
                index.members.setdefault(gid, []).append(source)
            self._write_ranks(source, ranks)
        self.data["groups"][gid] = top
        self.changed = True

    def _enter(self, source: str, gid: str, rank: int) -> int:
        ranks = self._ranks(source)
        ranks[gid] = rank
        self._write_ranks(source, ranks)
        self.data["groups"][gid] = max(self.data["groups"][gid], rank)
        if self._index is not None:
            self._index.members.setdefault(gid, []).append(source)
        return rank

    def _is_group(self, gid: object) -> bool:
        if not isinstance(gid, str) or not gid.isdigit():
            return False
        top = self.data["groups"].get(gid)
        return isinstance(top, int) and not isinstance(top, bool) and top >= 0

    def _held(self, gid: str) -> set[int]:
        sources = self._indexed().members.get(gid, [])
        return {rank for source in sources if (rank := self._ranks(source).get(gid))}

    def _ranks(self, source: str) -> dict[str, int]:
        """*source*'s rank in each group, its malformed pairs left out."""
        value = self.data["ranks"].get(source)
        if not isinstance(value, str):
            return {}
        ranks: dict[str, int] = {}
        for pair in value.split():
            gid, _, rank = pair.partition(":")
            if self._is_group(gid) and rank.isdigit() and int(rank) > 0:
                ranks[gid] = int(rank)
        return ranks

    def _write_ranks(self, source: str, ranks: dict[str, int]) -> None:
        self.data["ranks"][source] = " ".join(f"{gid}:{rank}" for gid, rank in ranks.items())
        self.changed = True

    def _indexed(self) -> _Index:
        """Each group's names and sources, read once from the register when a
        merge or a replayed record needs them, then kept up to date."""
        if self._index is None:
            index = _Index()
            for name, gid in self.data["names"].items():
                if self._is_group(gid):
                    index.names.setdefault(gid, []).append(name)
            for source in self.data["ranks"]:
                for gid in self._ranks(source):
                    index.members.setdefault(gid, []).append(source)
            self._index = index
        return self._index


class _Index:
    """A register's groups, read the other way: each one's names and sources."""

    def __init__(self) -> None:
        self.names: dict[str, list[str]] = {}
        self.members: dict[str, list[str]] = {}


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


def rebuilt(events: Iterable[RoomEvent], context: RoomContext) -> AuthorRegister:
    """A register for a room that holds none (a room from before it): the
    ranks its turns' records hold, then its named participants, then the
    authors of its turns that hold no record, in index order."""
    register = AuthorRegister()
    people = People(context.participants)
    salt = context.room.id
    unrecorded: list[tuple[str, set[str]]] = []
    for event in events:
        if event.status == EventStatus.BLOCKED or event.source.channel_type == ChannelType.SYSTEM:
            continue
        record = recorded_author(event)
        if record is not None and (name := person_name(record[0])):
            register.claim(record[2], name_keys(salt, name), record[1])
        elif (name := author_name(event, people)) is not None:
            unrecorded.append((digest(salt, source_of(event, people)), name_keys(salt, name)))
    register.seed(people, salt)
    for source, names in unrecorded:
        register.rank(source, names)
    return register


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
        register = rebuilt(await _timeline(store, room_id), context)
    record = author_record(event, context, register)
    if register.changed:
        await store.patch_room_metadata(room_id, {AUTHOR_REGISTER: register.data})
    if record is None:
        return event
    return event.model_copy(update={"metadata": {**event.metadata, AUTHOR: record}})


async def _timeline(store: ConversationStore, room_id: str) -> list[RoomEvent]:
    """The turns *room_id* received, in index order."""
    events: list[RoomEvent] = []
    while page := await store.list_events(room_id, offset=len(events), limit=_PAGE):
        events.extend(page)
    return events

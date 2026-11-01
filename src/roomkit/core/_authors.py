"""Who wrote a turn, and the rank that tells apart a room's sources whose names
read alike (RFC §6.4, §10.1 step 12).

A turn's author is the name a transport, a host or a diarized voice stamps on
it (``metadata["sender_name"]``), else the room's participant record. Two
sources whose names read alike (``Alice``, ``ALICE``, ``Аlice``) are told
apart by a rank the room fixes when a turn is committed: the room keeps a
register of the names it has seen, grouped by what they read as, and of each
source's rank in each group; the turn carries its author's name, rank and
source (:data:`AUTHOR`), so a reader names it as it was named then. The
register holds digests salted with the room's id: it compares, it never
needs to show a sender's id or name. The room's id is not secret and a phone
number or a name is a small space, so whoever reads the digests finds the
senders by trying them all: the store keeps the register apart from the
room's metadata, which may reach clients
(:data:`AUTHOR_REGISTER` of its private state,
:meth:`~roomkit.store.base.ConversationStore.get_private_state`), and a
reader takes it from the room's context (``RoomContext.author_register``).
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Iterable
from typing import Any

from roomkit._lookalike import skeletons
from roomkit._text import person_name
from roomkit.core._private_state import AUTHOR_REGISTER
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventStatus
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant, own_record_id
from roomkit.models.room import Room
from roomkit.store.base import ConversationStore

AUTHOR = "author"
"""The event metadata key holding the record of its author, fixed when the
event is committed: ``{"name": "Alice", "rank": 2, "source": "<digest>"}``."""

_VERSION = 3
_PAGE = 500
_PAIR = re.compile(r"([0-9a-f]{16}):([1-9][0-9]{0,8})")


class People:
    """The room's participants as a turn's ``source.participant_id`` names
    them: a ``Participant.id``, or an ``Identity.id``, names its participant
    only on a channel they are reached through (``connected_via``, which the
    identity pipeline extends with the channel it resolved them on), since an
    id one transport gives says nothing of another's (RFC §5.5). The one
    answer to who a sender is: their label, role, authority, identification
    and verification all read it.

    An identity is one person however many records name it (a member the host
    added under it, and the record the identity pipeline keeps for it): the
    record that joined first stands for them all, so their turns share one
    source and one rank whichever record and channel they came through."""

    def __init__(self, participants: Iterable[Participant]) -> None:
        records = sorted(participants, key=lambda p: (p.joined_at, p.id))
        self._by_id = {p.id: p for p in records}
        self._by_identity: dict[str, list[Participant]] = {}
        for record in records:
            if record.identity_id:
                self._by_identity.setdefault(record.identity_id, []).append(record)
        # One record per person, in the order they joined.
        self.persons = [p for p in records if self._standing_for(p) is p]

    def of(self, event: RoomEvent) -> Participant | None:
        """The participant behind *event*, or ``None``."""
        record = self.named(event)
        return self._standing_for(record) if record is not None else None

    def named(self, event: RoomEvent) -> Participant | None:
        """The record *event*'s ``participant_id`` names on its channel, or
        ``None``: the record of that id when its participant is reached
        through the channel, else one of the identity's records reached
        through it (the channel the identity pipeline resolved them on).
        An id that names nobody on the channel names nobody at all."""
        return self.named_on(event.source.participant_id, event.source.channel_id)

    def named_on(self, pid: str | None, channel: str) -> Participant | None:
        """The record *pid* names on *channel*, as :meth:`named` reads an
        event's sender: a ``post()``'s speaker is read before its event
        exists (RFC §10.5). Past the records of that id and of that identity,
        the sender's own record on the channel (:func:`own_record_id`, RFC
        §11.2), which their address names there."""
        if not pid:
            return None
        record = self._named_directly(pid, channel)
        if record is not None:
            return record
        own = self._by_id.get(own_record_id(channel, pid))
        return own if own is not None and reached(own, channel) else None

    def _named_directly(self, pid: str, channel: str) -> Participant | None:
        """The record of id *pid*, else of identity *pid*, reached through
        *channel*."""
        record = self._by_id.get(pid)
        if record is not None and reached(record, channel):
            return record
        return next((r for r in self._by_identity.get(pid, ()) if reached(r, channel)), None)

    def names_another(self, event: RoomEvent) -> bool:
        """Whether *event*'s id is a participant's the room reached through
        other channels only: it names nobody on the event's channel, and read
        back on its own it would name them."""
        return self.names_another_on(event.source.participant_id, event.source.channel_id)

    def names_another_on(self, pid: str | None, channel: str) -> bool:
        """Whether *pid* is a participant's the room reached through channels
        other than *channel* only, as :meth:`names_another` reads an event."""
        return (
            pid is not None and pid in self._by_id and self._named_directly(pid, channel) is None
        )

    def _standing_for(self, record: Participant) -> Participant:
        """The record standing for *record*'s person: its identity's first."""
        if record.identity_id is None:
            return record
        return next(iter(self._by_identity.get(record.identity_id, ())), record)


def reached(record: Participant, channel_id: str) -> bool:
    """Whether the room reached *record*'s participant through *channel_id*
    (RFC §5.5): ``connected_via`` holds the primary channel too."""
    return channel_id in record.connected_via


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


def with_committed_record(metadata: dict[str, Any], stored: RoomEvent) -> dict[str, Any]:
    """*metadata*, a direct update's of *stored*, with the record of the
    author *stored* was committed with (:data:`AUTHOR`), and none when it was
    committed with none: the record is the room's, fixed at the commit (RFC
    §10.1 step 12), so a direct update never writes one of its own."""
    kept = {key: value for key, value in metadata.items() if key != AUTHOR}
    if AUTHOR in stored.metadata:
        kept[AUTHOR] = stored.metadata[AUTHOR]
    return kept


class AuthorRegister:
    """A room's register of authors: for each thing a name reads as (a
    skeleton, hashed), the sources whose names read so and the rank of each.

    Two sources whose names read alike share a skeleton, so their ranks are
    told apart there: a source new to the names it uses takes one more than
    the highest rank the sources whose names read like them hold (``Lan``,
    then ``Ian (2)``, then ``ian (3)``), and a registered participant the
    lowest rank none of them holds. A source keeps its rank while no source
    whose name reads like the one it uses holds it, so a name that reads like
    two others ranks after both and renumbers neither; a rank once given is
    never given to another source, since a turn was recorded with it.

    Kept as JSON by the store, apart from the room's metadata, and copied or
    parsed for every context of the room, so its values are strings:
    ``names`` maps each hashed skeleton to its ``"source:rank"`` pairs. What
    is malformed in it (a room's metadata from before, which the application
    may write) is read as absent."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data or {"version": _VERSION, "names": {}}
        self.changed = data is None

    @classmethod
    def stored(cls, value: object) -> AuthorRegister | None:
        """The register *value* (a room's, as the store keeps it) holds, or
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

    def merge(self, other: AuthorRegister) -> None:
        """File here every rank *other* holds: a register only adds, so the
        union keeps every rank either gave, and neither gives one again to
        another source. Two sources a rank was given to apart keep it both,
        and each takes a new one at its next turn."""
        for name in list(other.data["names"]):
            for source, rank in other._entries(name):
                self._file(source, {name}, rank)

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
        for person in people.persons:
            if name := person_name(person.display_name):
                self.seat(digest(salt, ["participant", person.id]), name_keys(salt, name))

    def _ranks(self, source: str, names: set[str]) -> tuple[set[int], set[int]]:
        """The ranks *source* holds under *names*, and those the other sources
        under them hold."""
        mine: set[int] = set()
        others: set[int] = set()
        for name in names:
            for holder, rank in self._entries(name):
                (mine if holder == source else others).add(rank)
        return mine, others

    def _entries(self, name: str) -> list[tuple[str, int]]:
        """The ranks filed under *name*, each with its source, its malformed
        pairs left out."""
        value = self.data["names"].get(name)
        if not isinstance(value, str):
            return []
        pairs = (_PAIR.fullmatch(pair) for pair in value.split())
        return [(found[1], int(found[2])) for found in pairs if found]

    def _file(self, source: str, names: set[str], rank: int) -> None:
        """*rank* filed for *source* under each of *names*, beside any rank it
        held there before: a turn was recorded with that one, so no other
        source may take it."""
        for name in sorted(names):
            if (source, rank) not in self._entries(name):
                held = self.data["names"].get(name)
                pair = f"{source}:{rank}"
                self.data["names"][name] = f"{held} {pair}" if isinstance(held, str) else pair
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


def _keeps_apart(store: ConversationStore) -> bool:
    """Whether *store* keeps the register apart from the rooms' metadata:
    both private-state methods are its own, where the base class's keep it
    in the metadata."""
    kind = type(store)
    return (
        kind.get_private_state is not ConversationStore.get_private_state
        and kind.set_private_state is not ConversationStore.set_private_state
    )


async def _held_register(
    store: ConversationStore, room: Room
) -> tuple[AuthorRegister | None, bool]:
    """The register *room* holds, and whether its metadata holds one to move
    to the store: a room stored before the store kept it apart, or one a
    process from before wrote there since (a store shared while a deployment
    upgrades, or rolled back). The store's takes every rank the metadata's
    gave: a rank given twice would name two sources alike."""
    held = AuthorRegister.stored(await store.get_private_state(room.id, AUTHOR_REGISTER))
    if AUTHOR_REGISTER not in room.metadata or not _keeps_apart(store):
        return held, False
    # The room is the store's snapshot (RFC §14.4): its register is copied
    # before the turn extends it.
    legacy = AuthorRegister.stored(room.metadata[AUTHOR_REGISTER])
    legacy = legacy.copy() if legacy is not None else None
    if held is None:
        return legacy, True
    if legacy is not None:
        held.merge(legacy)
    return held, True


async def stored_register(store: ConversationStore, room: Room) -> dict[str, Any] | None:
    """The register *room* holds, as its readers take it: the store's, with
    what its metadata still holds until it moves at the room's next turn
    that names its sender (:func:`with_author`)."""
    register, _moved = await _held_register(store, room)
    return register.data if register is not None else None


async def with_author(
    store: ConversationStore,
    room_id: str,
    event: RoomEvent,
    snapshot: RoomContext | None = None,
) -> RoomEvent:
    """*event* carrying the record of its author (:data:`AUTHOR`), the room's
    register extended first when the source or its name is new, or rebuilt
    from the timeline when the room holds none (RFC §10.1 step 12).

    Run under the room lock, before the commit: a stored turn's record never
    changes, and a store shared across processes needs the distributed lock
    of RFC §13.5. The register is written first, so a commit that fails after
    it leaves a rank unused, never one held twice. A record the event came
    with is dropped, the runtime's alone; a BLOCKED record takes none. A
    restricted turn joins the register too: ranked without joining it, a
    later source could take its rank in the view of a reader who sees both.

    A register the room's metadata holds (:func:`_held_register`) is written
    to the store, merged into the one it holds, and the store removes it from
    the metadata: its digests are the same, so every rank holds.

    *snapshot* is the room's context read under the lock the commit runs
    within, after the hooks that ran there (RFC §10.1 steps 6 and 9a): the
    record is made from its room, participants and register, which are not
    read again."""
    if AUTHOR in event.metadata:
        metadata = {k: v for k, v in event.metadata.items() if k != AUTHOR}
        event = event.model_copy(update={"metadata": metadata})
    named = event.source.participant_id or event.metadata.get("sender_name")
    if not named or event.status == EventStatus.BLOCKED:
        return event
    read = await _room_and_register(store, room_id, snapshot)
    if read is None:
        return event
    context, register, moved = read
    if register is None:
        register = await _rebuilt(store, room_id, context)
    record = author_record(event, context, register)
    if register.changed or moved:
        await store.set_private_state(room_id, AUTHOR_REGISTER, register.data)
    if record is None:
        return event
    return event.model_copy(update={"metadata": {**event.metadata, AUTHOR: record}})


async def _room_and_register(
    store: ConversationStore, room_id: str, snapshot: RoomContext | None
) -> tuple[RoomContext, AuthorRegister | None, bool] | None:
    """The room's context, its register and whether its metadata holds one to
    move (:func:`_held_register`), from *snapshot* when the lock read it, else
    from the store; ``None`` for a room the store lacks. A snapshot's register
    is copied: the record extends it, and the context is its readers' too."""
    if snapshot is None:
        room = await store.get_room(room_id)
        if room is None:
            return None
        context = RoomContext(room=room, participants=await store.list_participants(room_id))
        register, moved = await _held_register(store, room)
        return context, register, moved
    held = AuthorRegister.stored(snapshot.author_register)
    moved = AUTHOR_REGISTER in snapshot.room.metadata and _keeps_apart(store)
    return snapshot, None if held is None else held.copy(), moved


async def _rebuilt(store: ConversationStore, room_id: str, context: RoomContext) -> AuthorRegister:
    """The register *room_id*'s timeline rebuilds (:class:`_Rebuild`)."""
    rebuild = _Rebuild(context)
    offset = 0
    while page := await store.list_events(room_id, offset=offset, limit=_PAGE):
        rebuild.add(page)
        offset += len(page)
    return rebuild.done()

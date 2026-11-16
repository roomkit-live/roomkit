"""What each side of the turn backend records per room and channel (RFC
§10.6.2, §10.6.7): the owner its previous turn, the backend where the room's
state is, the executor the turn that left it.

A record falls out once a side keeps :data:`ROOMS_KEPT` newer ones. A room
that was quiet that long starts over and never resumes a stale state: the
owner then names no previous turn, which an executor holding a record reads
as stale, and an executor whose record falls out drops the state it vouched
for (RFC §10.6.2).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

ROOMS_KEPT = 10_000
"""How many (channel, room) pairs a side keeps a record for."""

ROOM_STATE_KEY = "roomkit:channel_state:"
"""The prefix of the room-metadata key under which the owner keeps what the
room holds for a channel (RFC §6.4): an ACP session's config values."""


def room_state_key(channel_id: str) -> str:
    """The room-metadata key of what the room holds for *channel_id*."""
    return f"{ROOM_STATE_KEY}{channel_id}"


def stored_room_state(metadata: dict[str, object], channel_id: str) -> dict[str, object]:
    """What *metadata* holds for *channel_id*; empty when it holds nothing
    readable."""
    state = metadata.get(room_state_key(channel_id))
    return dict(state) if isinstance(state, dict) else {}


class RoomRecords[T]:
    """One record per (channel, room), the least recently written falling out
    past *kept*, *evicted* told of each (channel id, room id) that does."""

    def __init__(
        self,
        kept: int = ROOMS_KEPT,
        *,
        evicted: Callable[[str, str], None] | None = None,
    ) -> None:
        self._kept = kept
        self._evicted = evicted
        self._records: OrderedDict[tuple[str, str], T] = OrderedDict()
        # Room id → the channels it holds a record of.
        self._channels: dict[str, set[str]] = {}

    def get(self, channel_id: str, room_id: str) -> T | None:
        return self._records.get((channel_id, room_id))

    def put(self, channel_id: str, room_id: str, record: T) -> None:
        key = (channel_id, room_id)
        self._records[key] = record
        self._records.move_to_end(key)
        self._channels.setdefault(room_id, set()).add(channel_id)
        while len(self._records) > self._kept:
            (gone_channel, gone_room), _ = self._records.popitem(last=False)
            self._unindex(gone_channel, gone_room)
            if self._evicted is not None:
                self._evicted(gone_channel, gone_room)

    def pop(self, channel_id: str, room_id: str) -> T | None:
        record = self._records.pop((channel_id, room_id), None)
        self._unindex(channel_id, room_id)
        return record

    def pop_room(self, room_id: str) -> None:
        """Drop every channel's record of *room_id*, telling no one."""
        for channel_id in self._channels.pop(room_id, set()):
            del self._records[(channel_id, room_id)]

    def _unindex(self, channel_id: str, room_id: str) -> None:
        channels = self._channels.get(room_id)
        if channels is None:
            return
        channels.discard(channel_id)
        if not channels:
            del self._channels[room_id]

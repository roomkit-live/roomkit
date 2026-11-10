"""What an executor's wiring reads beside its own hooks (RFC §10.6.2): the
room's context for a hook, the rows a channel reads, and where a channel's
publications and framework events go outside a turn.

The kit's own executor reads the kit (``core/_kit_wiring.py``); an executor a
host builds reads :class:`TurnWiringEnv`, what its turn carries and what the host gave it.
"""

from __future__ import annotations

import contextvars
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from roomkit.core._changes import withheld
from roomkit.core.exceptions import RoomNotFoundError, TurnLinkError
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.store_filter import EventFilter
from roomkit.turns.host import current_turn_host

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.models.event import RoomEvent
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.store.base import ConversationStore
    from roomkit.turns._outlet import FrameworkEmitter
    from roomkit.turns.wire import TurnRequest


TOOL_MEMORY_CALLS = 30
"""Calls a channel's stored tool rows are read for to rebuild its tool memory
(the 12 tools re-revealed), after the infra rows are filtered out."""

HISTORY_ROWS_PAGE = 500
"""A channel's own rows a turn's history reads from the store at a time."""

LAST_CONTEXTS_KEPT = 256
"""Rooms whose last turn's context an executor keeps for a hook outside any turn."""

_TOOL_ROWS = frozenset({EventType.TOOL_CALL_START, EventType.TOOL_CALL_END})

_TURN_REQUEST: contextvars.ContextVar[TurnRequest | None] = contextvars.ContextVar(
    "roomkit_turn_request", default=None
)


def set_turn_request(request: TurnRequest) -> None:
    """The request of the turn the current context runs."""
    _TURN_REQUEST.set(request)


def _carried(room_id: str, channel_id: str) -> list[RoomEvent]:
    """The rows the current turn brought, when it is *channel_id*'s in *room_id*."""
    request = _TURN_REQUEST.get()
    if request is None or (request.room_id, request.channel_id) != (room_id, channel_id):
        return []
    return request.channel_rows


async def read_tool_rows(
    store: ConversationStore, room_id: str, channel_id: str, limit: int
) -> list[RoomEvent]:
    """*channel_id*'s latest *limit* stored tool rows in the room, oldest first."""
    return await store.get_timeline(
        room_id,
        event_filter=EventFilter(
            event_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END],
            source_channel_id=channel_id,
        ),
        limit=limit,
        newest_first=True,  # most recent N, returned ascending
    )


async def read_history_rows(
    store: ConversationStore, room_id: str, channel_id: str, *, after_index: int, limit: int
) -> list[RoomEvent]:
    """*channel_id*'s own rows a turn's history replays, its tool calls and
    the instructions it read, the first *limit* after *after_index*."""
    own = EventFilter(
        event_types=[
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
            EventType.INSTRUCTION_READ,
        ],
        source_channel_id=channel_id,
    )
    return await store.get_timeline(
        room_id, event_filter=own, limit=limit, after_index=after_index
    )


async def read_message_before(store: ConversationStore, room_id: str, index: int) -> int:
    """The index of the room's last message before *index*, -1 for none."""
    before = await store.list_events(
        room_id,
        limit=1,
        before_index=index,
        event_filter=EventFilter(event_types=[EventType.MESSAGE], include_blocked=True),
    )
    return before[-1].index if before else -1


async def read_channel_rows(
    store: ConversationStore, room_id: str, channel_id: str, first: int | None
) -> list[RoomEvent]:
    """What a turn of *channel_id* on an executor reads of the room, as its
    owner reads it for the turn: the rows, each once, oldest first, deleted
    ones withheld (RFC §7.5 rule 8). *first* is the index its history starts
    at, ``None`` for none."""
    latest = await read_tool_rows(store, room_id, channel_id, TOOL_MEMORY_CALLS * 2)
    rows = {row.id: row for row in latest}
    if first is not None:
        position = await read_message_before(store, room_id, first)
        while True:
            page = await read_history_rows(
                store, room_id, channel_id, after_index=position, limit=HISTORY_ROWS_PAGE
            )
            rows.update((row.id, row) for row in page)
            if len(page) < HISTORY_ROWS_PAGE:
                break
            position = page[-1].index
    return withheld(sorted(rows.values(), key=lambda row: row.index))


class WiringEnv(ABC):
    """What an executor's wiring gives the channels it wires."""

    @property
    def media_resolver(self) -> MediaResolver | None:
        """The host's media resolver (RFC §6.4)."""
        return None

    @property
    def attached_media(self) -> AttachedMedia | None:
        """The host's attached-media reader (RFC §6.4)."""
        return None

    @abstractmethod
    async def context(self, room_id: str, *, carrying: RoomContext | None = None) -> RoomContext:
        """The room's context for a hook, built when it fires (RFC §10.6.6);
        *carrying* is one already built for the same call, whose history may
        be reused. Raises when it cannot be built."""

    @abstractmethod
    async def tool_rows(self, room_id: str, channel_id: str, limit: int) -> list[RoomEvent]:
        """The channel's latest stored tool rows in the room, oldest first."""

    @abstractmethod
    async def history_rows(
        self, room_id: str, channel_id: str, *, after_index: int, limit: int
    ) -> list[RoomEvent]:
        """The channel's own rows a turn's history replays (its tool calls,
        the instructions it read), the first *limit* after *after_index*,
        oldest first."""

    @abstractmethod
    async def message_before(self, room_id: str, index: int) -> int:
        """The index of the room's last message before *index*, -1 for none."""

    def realtime(self) -> RealtimeBackend | None:
        """Where a channel's publication made outside a turn goes, ``None``
        for nowhere."""
        return None

    def emitter(self) -> FrameworkEmitter | None:
        """Where a framework event emitted outside a turn goes, ``None`` for
        nowhere."""
        return None

    def turn_started(self, request: TurnRequest) -> None:  # noqa: B027
        """A turn of the executor starts: *request* is what it was asked."""

    def room_left(self, room_id: str, channel_id: str) -> None:  # noqa: B027
        """*channel_id* of the executor was detached from *room_id*: what was
        kept of the room's conversation for it goes (RFC §10.6.2)."""

    async def room_tasks(self, room_id: str) -> list[dict[str, Any]]:
        """The room's background tasks, outside a turn's host (RFC §23.4)."""
        return []

    def room_vision(self, context: RoomContext) -> str | None:
        """What the room's video last showed, outside a turn's host."""
        return None


class TurnWiringEnv(WiringEnv):
    """The wiring environment of an executor a host builds: it has no store,
    so what a turn reads comes with the turn, or from its room's owner (RFC
    §10.6.3, §10.6.6), and what the host gave the executor.

    Args:
        media_resolver: The host's media resolver (RFC §6.4).
        attached_media: The host's attached-media reader (RFC §6.4).
    """

    def __init__(
        self,
        *,
        media_resolver: MediaResolver | None = None,
        attached_media: AttachedMedia | None = None,
    ) -> None:
        self._media_resolver = media_resolver
        self._attached_media = attached_media
        self._last_contexts: OrderedDict[str, RoomContext] = OrderedDict()
        # The channels whose turns ran here in each room kept: the room's last
        # context goes once the last of them is detached.
        self._ran_in: dict[str, set[str]] = {}

    def turn_started(self, request: TurnRequest) -> None:
        self._last_contexts[request.room_id] = request.context
        self._last_contexts.move_to_end(request.room_id)
        self._ran_in.setdefault(request.room_id, set()).add(request.channel_id)
        while len(self._last_contexts) > LAST_CONTEXTS_KEPT:
            dropped, _ = self._last_contexts.popitem(last=False)
            self._ran_in.pop(dropped, None)

    def room_left(self, room_id: str, channel_id: str) -> None:
        """Once no channel whose turns ran here is attached to the room, a
        room created again under the id reads nothing of the earlier
        conversation outside a turn: a hook there fails closed until the
        room's next turn here, as one in a room no turn ran in (RFC
        §10.6.6). Another channel still attached keeps reading it."""
        channels = self._ran_in.get(room_id)
        if channels is None:
            return
        channels.discard(channel_id)
        if not channels:
            del self._ran_in[room_id]
            self._last_contexts.pop(room_id, None)

    @property
    def media_resolver(self) -> MediaResolver | None:
        return self._media_resolver

    @property
    def attached_media(self) -> AttachedMedia | None:
        return self._attached_media

    async def context(self, room_id: str, *, carrying: RoomContext | None = None) -> RoomContext:
        """The room as its owner builds it now; once the turn is over, or
        being cut so the owner no longer answers it, as the turn started
        with it; outside any turn (an external agent's own activity), as the
        executor's last turn in the room had it. Any other failure is the
        hook's to fail closed on."""
        request = _TURN_REQUEST.get()
        if request is None or request.room_id != room_id:
            last = self._last_contexts.get(room_id)
            if last is None:
                raise RoomNotFoundError(f"No turn of room {room_id} ran here")
            return last
        host = current_turn_host()
        if host is None or not host.open:
            return request.context
        try:
            return RoomContext.model_validate(await host.room_context())
        except TurnLinkError:
            return request.context

    async def tool_rows(self, room_id: str, channel_id: str, limit: int) -> list[RoomEvent]:
        return [row for row in _carried(room_id, channel_id) if row.type in _TOOL_ROWS][-limit:]

    async def history_rows(
        self, room_id: str, channel_id: str, *, after_index: int, limit: int
    ) -> list[RoomEvent]:
        return [row for row in _carried(room_id, channel_id) if row.index > after_index][:limit]

    async def message_before(self, room_id: str, index: int) -> int:
        return -1  # what the turn brought is already bounded: a history replays
        # no round older than its first turn

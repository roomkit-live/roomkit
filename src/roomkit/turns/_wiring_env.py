"""What an executor's wiring reads beside its own hooks (RFC §10.6.2): the
room's context for a hook, the store a channel reads, and where a channel's
publications and framework events go outside a turn.

The kit's own executor reads the kit (``core/_kit_wiring.py``); an executor a
host builds reads :class:`StoreWiringEnv`, its store and what the host gave it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from roomkit.core._authors import stored_register
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.store_filter import EventFilter

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.models.event import RoomEvent
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.store.base import ConversationStore
    from roomkit.turns._outlet import FrameworkEmitter

HOOK_HISTORY = 50
"""Recent events a hook's context carries on an executor a host built, as the
kit's floor for hooks does."""


async def read_tool_rows(
    store: ConversationStore,
    room_id: str,
    channel_id: str,
    limit: int,
    *,
    after_index: int | None = None,
) -> list[RoomEvent]:
    """*channel_id*'s stored tool rows in the room, oldest first: its latest
    *limit*, or its first *limit* after *after_index*."""
    tools = EventFilter(
        event_types=[EventType.TOOL_CALL_START, EventType.TOOL_CALL_END],
        source_channel_id=channel_id,
    )
    if after_index is not None:
        return await store.get_timeline(
            room_id, event_filter=tools, limit=limit, after_index=after_index
        )
    # The most recent *limit*, returned ascending.
    return await store.get_timeline(room_id, event_filter=tools, limit=limit, newest_first=True)


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
    async def tool_rows(
        self, room_id: str, channel_id: str, limit: int, *, after_index: int | None = None
    ) -> list[RoomEvent]:
        """The channel's stored tool rows in the room, oldest first: its
        latest *limit*, or its first *limit* after *after_index*."""

    def realtime(self) -> RealtimeBackend | None:
        """Where a channel's publication made outside a turn goes, ``None``
        for nowhere."""
        return None

    def emitter(self) -> FrameworkEmitter | None:
        """Where a framework event emitted outside a turn goes, ``None`` for
        nowhere."""
        return None

    async def room_tasks(self, room_id: str) -> list[dict[str, Any]]:
        """The room's background tasks, outside a turn's host (RFC §23.4)."""
        return []

    def room_vision(self, context: RoomContext) -> str | None:
        """What the room's video last showed, outside a turn's host."""
        return None


class StoreWiringEnv(WiringEnv):
    """The wiring environment of an executor a host builds: the shared store,
    and what the host gave the executor (RFC §10.6.2, §10.6.9).

    Args:
        store: The shared store; ``None`` leaves hooks that need a room's
            context unable to build one (a ``BEFORE_TOOL_USE`` then denies).
        media_resolver: The host's media resolver (RFC §6.4).
        attached_media: The host's attached-media reader (RFC §6.4).
    """

    def __init__(
        self,
        store: ConversationStore | None,
        *,
        media_resolver: MediaResolver | None = None,
        attached_media: AttachedMedia | None = None,
    ) -> None:
        self._store = store
        self._media_resolver = media_resolver
        self._attached_media = attached_media

    @property
    def media_resolver(self) -> MediaResolver | None:
        return self._media_resolver

    @property
    def attached_media(self) -> AttachedMedia | None:
        return self._attached_media

    async def context(self, room_id: str, *, carrying: RoomContext | None = None) -> RoomContext:
        store = self._shared_store()
        async with store.connection():
            room, bindings, participants = await store.load_room_context(room_id)
            if room is None:
                raise RoomNotFoundError(f"Room {room_id} not found")
            recent = await store.get_conversation(room_id, limit=HOOK_HISTORY)
            register = await stored_register(store, room)
        return RoomContext(
            room=room,
            bindings=bindings,
            participants=participants,
            recent_events=recent,
            author_register=register,
        )

    async def tool_rows(
        self, room_id: str, channel_id: str, limit: int, *, after_index: int | None = None
    ) -> list[RoomEvent]:
        return await read_tool_rows(
            self._shared_store(), room_id, channel_id, limit, after_index=after_index
        )

    def _shared_store(self) -> ConversationStore:
        if self._store is None:
            raise RuntimeError("This executor was given no store (TurnExecutor(store=...))")
        return self._store

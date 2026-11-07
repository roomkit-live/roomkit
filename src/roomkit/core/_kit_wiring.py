"""The kit's side of its turns' wiring (RFC §10.6.2, §10.6.3).

:class:`KitWiringEnv` is what the kit's own executor wires the channels the
kit registers from: the kit's context builder, store, realtime backend and
framework events for what happens outside a turn. :class:`KitEvents` is where
the kit publishes what its turns published on their executors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.channels._video_hooks import room_vision_note
from roomkit.core.visibility import binding_reaches_room
from roomkit.turns._wiring_env import WiringEnv, read_tool_rows
from roomkit.turns.base import OwnerEvents

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.realtime.base import EphemeralEvent, RealtimeBackend
    from roomkit.turns._outlet import FrameworkEmitter


class KitWiringEnv(WiringEnv):
    """What the kit's own executor wires channels from, read off the kit when
    used."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit

    @property
    def media_resolver(self) -> MediaResolver | None:
        return self._kit._media_resolver

    @property
    def attached_media(self) -> AttachedMedia | None:
        return self._kit._attached_media

    async def context(self, room_id: str, *, carrying: RoomContext | None = None) -> RoomContext:
        return await self._kit._build_context(room_id, carrying=carrying)

    async def tool_rows(
        self, room_id: str, channel_id: str, limit: int, *, after_index: int | None = None
    ) -> list[RoomEvent]:
        with self._kit._resource_lease():
            return await read_tool_rows(
                self._kit._store, room_id, channel_id, limit, after_index=after_index
            )

    def realtime(self) -> RealtimeBackend | None:
        return self._kit._realtime

    def emitter(self) -> FrameworkEmitter | None:
        return self._kit._emit_framework_event

    async def room_tasks(self, room_id: str) -> list[dict[str, Any]]:
        return await self._kit._build_room_tasks_loader()(room_id)

    def room_vision(self, context: RoomContext) -> str | None:
        return room_vision_note(self._kit._channels, context)


class KitEvents(OwnerEvents):
    """Where the kit publishes what its turns published on their executors:
    its realtime backend, its own framework-event bus."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        # A turn publishes to its room's channel, through the room publish a
        # backend may implement apart.
        room_id = channel.removeprefix("room:")
        if room_id != channel:
            await self._kit._realtime.publish_to_room(room_id, event)
        else:
            await self._kit._realtime.publish(channel, event)

    async def reaches_room(self, room_id: str, channel_id: str) -> bool:
        binding = await self._kit._store.get_binding(room_id, channel_id)
        return binding is not None and binding_reaches_room(binding)

    async def emit(
        self,
        event_type: str,
        *,
        room_id: str | None,
        channel_id: str | None,
        event_id: str | None,
        data: dict[str, Any],
    ) -> None:
        await self._kit._emit_framework_event(
            event_type, room_id=room_id, channel_id=channel_id, event_id=event_id, data=data
        )

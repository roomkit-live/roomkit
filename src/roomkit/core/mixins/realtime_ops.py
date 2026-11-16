"""RealtimeOpsMixin — ephemeral event publishing and subscriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from roomkit.core._room_hook_lifetime import RoomHookLifetime
    from roomkit.core._room_subscriptions import RoomSubscriptions
    from roomkit.realtime.base import EphemeralCallback, RealtimeBackend


class RealtimeOpsMixin:
    """Ephemeral event publishing and room subscription operations.

    Each takes the organization the caller acts for (RFC §17.2): with one,
    the room is read first, and another organization's room raises
    :class:`~roomkit.core.exceptions.RoomNotFoundError` with nothing
    published or subscribed. Unset, the room of any organization is read,
    and a room id the store does not hold is published to and subscribed to
    as before.

    A publication carries the creation mark of its room
    (``EphemeralEvent.room_created_at``), and a subscription reads the
    events of its own room alone: once the store deleted the room, the room
    created again under its id, by this process or another, reaches none of
    the deleted room's subscriptions (RFC §8.4, §17.2).
    """

    _realtime: RealtimeBackend
    _room_hook_lifetime: RoomHookLifetime
    _room_subscriptions: RoomSubscriptions

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # RoomLifecycleMixin
    _room_mark: Any  # HooksApiMixin

    async def publish_typing(
        self,
        room_id: str,
        user_id: str,
        is_typing: bool = True,
        data: dict[str, Any] | None = None,
        *,
        organization_id: str | None = None,
    ) -> None:
        """Publish a typing indicator for a user in a room.

        Args:
            room_id: The room to publish the typing event in.
            user_id: The user who is typing.
            is_typing: True for typing_start, False for typing_stop.
            data: Optional additional data (e.g., {"name": "User Name"}).
            organization_id: The organization the caller acts for (RFC §17.2).
        """
        mark = await self._room_mark(room_id, organization_id)
        event = EphemeralEvent(
            room_id=room_id,
            room_created_at=mark,
            type=EphemeralEventType.TYPING_START if is_typing else EphemeralEventType.TYPING_STOP,
            user_id=user_id,
            data=data or {},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_tool_call(
        self,
        room_id: str,
        channel_id: str,
        tool_calls: list[dict[str, Any]],
        event_type: EphemeralEventType = EphemeralEventType.TOOL_CALL_START,
        *,
        duration_ms: int | None = None,
        organization_id: str | None = None,
    ) -> None:
        """Publish a tool call ephemeral event for a channel in a room.

        Args:
            room_id: The room to publish the event in.
            channel_id: The AI channel executing tool calls.
            tool_calls: List of tool call dicts (id, name, and
                arguments/result/arguments_chars, per event type).
            event_type: ``TOOL_CALL_DELTA``, ``TOOL_CALL_START`` or
                ``TOOL_CALL_END``.
            duration_ms: Optional execution duration (for END events).
            organization_id: The organization the caller acts for (RFC §17.2).
        """
        mark = await self._room_mark(room_id, organization_id)
        data: dict[str, Any] = {"tool_calls": tool_calls, "channel_id": channel_id}
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        event = EphemeralEvent(
            room_id=room_id,
            room_created_at=mark,
            type=event_type,
            user_id=channel_id,
            channel_id=channel_id,
            data=data,
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_presence(
        self, room_id: str, user_id: str, status: str, *, organization_id: str | None = None
    ) -> None:
        """Publish a presence update for a user in a room.

        Args:
            room_id: The room to publish the presence event in.
            user_id: The user whose presence changed.
            status: One of "online", "away", or "offline".
            organization_id: The organization the caller acts for (RFC §17.2).
        """
        mark = await self._room_mark(room_id, organization_id)
        type_map = {
            "online": EphemeralEventType.PRESENCE_ONLINE,
            "away": EphemeralEventType.PRESENCE_AWAY,
            "offline": EphemeralEventType.PRESENCE_OFFLINE,
        }
        event_type = type_map.get(status, EphemeralEventType.CUSTOM)
        event = EphemeralEvent(
            room_id=room_id,
            room_created_at=mark,
            type=event_type,
            user_id=user_id,
            data={"status": status} if event_type == EphemeralEventType.CUSTOM else {},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_read_receipt(
        self, room_id: str, user_id: str, event_id: str, *, organization_id: str | None = None
    ) -> None:
        """Publish a read receipt for a user in a room.

        Args:
            room_id: The room to publish the read receipt in.
            user_id: The user who read the message.
            event_id: The ID of the event that was read.
            organization_id: The organization the caller acts for (RFC §17.2).
        """
        mark = await self._room_mark(room_id, organization_id)
        event = EphemeralEvent(
            room_id=room_id,
            room_created_at=mark,
            type=EphemeralEventType.READ_RECEIPT,
            user_id=user_id,
            data={"event_id": event_id},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_reaction(
        self,
        room_id: str,
        user_id: str,
        target_event_id: str,
        emoji: str,
        action: str = "add",
        *,
        organization_id: str | None = None,
    ) -> None:
        """Publish a reaction to a message in a room.

        Args:
            room_id: The room containing the target event.
            user_id: The user reacting.
            target_event_id: The ID of the event being reacted to.
            emoji: The emoji reaction (empty string to remove).
            action: ``"add"`` or ``"remove"``.
            organization_id: The organization the caller acts for (RFC §17.2).
        """
        mark = await self._room_mark(room_id, organization_id)
        event = EphemeralEvent(
            room_id=room_id,
            room_created_at=mark,
            type=EphemeralEventType.REACTION,
            user_id=user_id,
            data={
                "target_event_id": target_event_id,
                "emoji": emoji,
                "action": action,
            },
        )
        await self._realtime.publish_to_room(room_id, event)

    async def subscribe_room(
        self, room_id: str, callback: EphemeralCallback, *, organization_id: str | None = None
    ) -> str:
        """Subscribe to ephemeral events for a room.

        Args:
            room_id: The room to subscribe to.
            callback: Async callback invoked for each ephemeral event.
            organization_id: The organization the caller acts for (RFC §17.2).

        Returns:
            A subscription ID that can be used to unsubscribe.

        A subscription ends with its room (RFC §8.4, §17.2): once the store
        deleted the room, :meth:`create_room` under its id ends it, and
        keeps those made while no room held the id, which are the new
        room's. Unscoped, the kit reads whether a room holds *room_id*.
        """
        with self._room_hook_lifetime.registering(room_id) as registration:
            registration.mark = await self._room_mark(room_id, organization_id)
            subscription_id = await self._realtime.subscribe_to_room(
                room_id,
                _of_room(callback, registration.mark, lambda: self._room_mark(room_id, None)),
            )
            self._room_subscriptions.made(room_id, subscription_id, held=registration.held)
        return subscription_id

    async def unsubscribe_room(self, subscription_id: str) -> bool:
        """Unsubscribe from ephemeral events.

        Args:
            subscription_id: The subscription ID returned by subscribe_room.

        Returns:
            True if the subscription existed and was removed.
        """
        self._room_subscriptions.ended(subscription_id)
        return await self._realtime.unsubscribe(subscription_id)


def _of_room(
    callback: EphemeralCallback,
    mark: datetime | None,
    held_mark: Callable[[], Awaitable[datetime | None]],
) -> EphemeralCallback:
    """*callback* for the events of the room of creation mark *mark* alone
    (RFC §8.4, §17.2): an event another room of the id published, created
    after that room was deleted, is not read. Made while no room held the id
    (*mark* ``None``), the subscription is the room's that holds the id when
    it reads its first event (*held_mark*, read once): an event of a room
    gone before it is not read. An event that names no room's mark,
    published on the backend directly, is read."""
    bound = mark

    async def read(event: EphemeralEvent) -> None:
        nonlocal bound
        published = event.room_created_at
        if published is not None:
            if bound is None:
                bound = await held_mark()
            if published != bound:
                return
        await callback(event)

    return read

"""RealtimeOpsMixin — ephemeral event publishing and subscriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from roomkit.realtime.base import EphemeralCallback, RealtimeBackend


class RealtimeOpsMixin:
    """Ephemeral event publishing and room subscription operations."""

    _realtime: RealtimeBackend

    async def publish_typing(
        self,
        room_id: str,
        user_id: str,
        is_typing: bool = True,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Publish a typing indicator for a user in a room.

        Args:
            room_id: The room to publish the typing event in.
            user_id: The user who is typing.
            is_typing: True for typing_start, False for typing_stop.
            data: Optional additional data (e.g., {"name": "User Name"}).
        """
        event = EphemeralEvent(
            room_id=room_id,
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
    ) -> None:
        """Publish a tool call ephemeral event for a channel in a room.

        Args:
            room_id: The room to publish the event in.
            channel_id: The AI channel executing tool calls.
            tool_calls: List of tool call dicts (id, name, arguments/result).
            event_type: ``TOOL_CALL_START`` or ``TOOL_CALL_END``.
            duration_ms: Optional execution duration (for END events).
        """
        data: dict[str, Any] = {"tool_calls": tool_calls, "channel_id": channel_id}
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        event = EphemeralEvent(
            room_id=room_id,
            type=event_type,
            user_id=channel_id,
            channel_id=channel_id,
            data=data,
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_presence(self, room_id: str, user_id: str, status: str) -> None:
        """Publish a presence update for a user in a room.

        Args:
            room_id: The room to publish the presence event in.
            user_id: The user whose presence changed.
            status: One of "online", "away", or "offline".
        """
        type_map = {
            "online": EphemeralEventType.PRESENCE_ONLINE,
            "away": EphemeralEventType.PRESENCE_AWAY,
            "offline": EphemeralEventType.PRESENCE_OFFLINE,
        }
        event_type = type_map.get(status, EphemeralEventType.CUSTOM)
        event = EphemeralEvent(
            room_id=room_id,
            type=event_type,
            user_id=user_id,
            data={"status": status} if event_type == EphemeralEventType.CUSTOM else {},
        )
        await self._realtime.publish_to_room(room_id, event)

    async def publish_read_receipt(self, room_id: str, user_id: str, event_id: str) -> None:
        """Publish a read receipt for a user in a room.

        Args:
            room_id: The room to publish the read receipt in.
            user_id: The user who read the message.
            event_id: The ID of the event that was read.
        """
        event = EphemeralEvent(
            room_id=room_id,
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
    ) -> None:
        """Publish a reaction to a message in a room.

        Args:
            room_id: The room containing the target event.
            user_id: The user reacting.
            target_event_id: The ID of the event being reacted to.
            emoji: The emoji reaction (empty string to remove).
            action: ``"add"`` or ``"remove"``.
        """
        event = EphemeralEvent(
            room_id=room_id,
            type=EphemeralEventType.REACTION,
            user_id=user_id,
            data={
                "target_event_id": target_event_id,
                "emoji": emoji,
                "action": action,
            },
        )
        await self._realtime.publish_to_room(room_id, event)

    async def subscribe_room(self, room_id: str, callback: EphemeralCallback) -> str:
        """Subscribe to ephemeral events for a room.

        Args:
            room_id: The room to subscribe to.
            callback: Async callback invoked for each ephemeral event.

        Returns:
            A subscription ID that can be used to unsubscribe.
        """
        return await self._realtime.subscribe_to_room(room_id, callback)

    async def unsubscribe_room(self, subscription_id: str) -> bool:
        """Unsubscribe from ephemeral events.

        Args:
            subscription_id: The subscription ID returned by subscribe_room.

        Returns:
            True if the subscription existed and was removed.
        """
        return await self._realtime.unsubscribe(subscription_id)

"""WebSocket channel implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent

logger = logging.getLogger("roomkit.channels.websocket")

SendFn = Callable[[str, RoomEvent], Coroutine[Any, Any, None]]


class WebSocketChannel(Channel):
    """WebSocket transport channel with connection registry."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self._connections: dict[str, SendFn] = {}

    @property
    def info(self) -> dict[str, Any]:
        return {"connection_count": len(self._connections)}

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[
                ChannelMediaType.TEXT,
                ChannelMediaType.RICH,
                ChannelMediaType.MEDIA,
                ChannelMediaType.AUDIO,
                ChannelMediaType.VIDEO,
                ChannelMediaType.LOCATION,
            ],
            supports_typing=True,
            supports_read_receipts=True,
            supports_reactions=True,
            supports_edit=True,
            supports_delete=True,
            supports_rich_text=True,
            supports_media=True,
            supports_buttons=True,
            supports_cards=True,
            supports_quick_replies=True,
        )

    def register_connection(self, connection_id: str, send_fn: SendFn) -> None:
        """Register a WebSocket connection."""
        self._connections[connection_id] = send_fn

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        self._connections.pop(connection_id, None)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        for conn_id, send_fn in list(self._connections.items()):
            try:
                await send_fn(conn_id, event)
            except Exception:
                logger.exception("WebSocket send failed for connection %s", conn_id)
                self._connections.pop(conn_id, None)
        return ChannelOutput.empty()

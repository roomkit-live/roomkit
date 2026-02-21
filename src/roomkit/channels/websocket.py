"""WebSocket channel implementation."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Literal

from pydantic import BaseModel

from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent

logger = logging.getLogger("roomkit.channels.websocket")

SendFn = Callable[[str, RoomEvent], Coroutine[Any, Any, None]]


# -- Streaming protocol models ------------------------------------------------


class StreamStart(BaseModel):
    """Sent when a streaming response begins."""

    type: Literal["stream_start"] = "stream_start"
    room_id: str
    stream_id: str
    source: EventSource


class StreamChunk(BaseModel):
    """Sent for each text delta during streaming."""

    type: Literal["stream_chunk"] = "stream_chunk"
    room_id: str
    stream_id: str
    delta: str
    text: str


class StreamEnd(BaseModel):
    """Sent when a streaming response completes."""

    type: Literal["stream_end"] = "stream_end"
    room_id: str
    stream_id: str
    event: RoomEvent


class StreamError(BaseModel):
    """Sent when a streaming response fails."""

    type: Literal["stream_error"] = "stream_error"
    room_id: str
    stream_id: str
    error: str


StreamMessage = StreamStart | StreamChunk | StreamEnd | StreamError

StreamSendFn = Callable[[str, StreamMessage], Coroutine[Any, Any, None]]


class WebSocketChannel(Channel):
    """WebSocket transport channel with connection registry."""

    channel_type = ChannelType.WEBSOCKET

    _MAX_CONSECUTIVE_ERRORS = 3

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self._connections: dict[str, SendFn] = {}
        self._stream_send_fns: dict[str, StreamSendFn] = {}
        self._error_counts: dict[str, int] = {}

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

    def register_connection(
        self,
        connection_id: str,
        send_fn: SendFn,
        *,
        stream_send_fn: StreamSendFn | None = None,
    ) -> None:
        """Register a WebSocket connection.

        Args:
            connection_id: Unique connection identifier.
            send_fn: Callback for delivering complete events.
            stream_send_fn: Optional callback for delivering streaming messages.
                When provided, this connection receives progressive text delivery
                via the ``stream_start``/``stream_chunk``/``stream_end`` protocol.
        """
        self._connections[connection_id] = send_fn
        if stream_send_fn is not None:
            self._stream_send_fns[connection_id] = stream_send_fn
        else:
            self._stream_send_fns.pop(connection_id, None)
        self._error_counts.pop(connection_id, None)

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        self._connections.pop(connection_id, None)
        self._stream_send_fns.pop(connection_id, None)
        self._error_counts.pop(connection_id, None)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether any connected client supports streaming text delivery."""
        return bool(self._stream_send_fns)

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                provider=self.provider_name,
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
                self._error_counts.pop(conn_id, None)
            except Exception:
                self._handle_send_error(conn_id)
        return ChannelOutput.empty()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Deliver a streaming text response to connected clients.

        Streaming-capable connections receive ``stream_start``, ``stream_chunk``,
        and ``stream_end`` messages progressively. Non-streaming connections
        receive the final complete event via the regular ``send_fn``.
        """
        stream_id = uuid.uuid4().hex
        room_id = event.room_id
        source = event.source

        logger.debug(
            "deliver_stream: channel=%s, stream=%s, streaming_conns=%d, total_conns=%d",
            self.channel_id,
            stream_id[:8],
            len(self._stream_send_fns),
            len(self._connections),
        )

        # Send stream_start to all streaming connections
        start_msg = StreamStart(room_id=room_id, stream_id=stream_id, source=source)
        for conn_id in list(self._stream_send_fns):
            await self._send_stream_message(conn_id, start_msg)

        # Stream chunks — build text incrementally to avoid O(N^2) joins
        accumulated: list[str] = []
        running_text = ""
        try:
            async for delta in text_stream:
                accumulated.append(delta)
                running_text += delta
                chunk_msg = StreamChunk(
                    room_id=room_id,
                    stream_id=stream_id,
                    delta=delta,
                    text=running_text,
                )
                for conn_id in list(self._stream_send_fns):
                    await self._send_stream_message(conn_id, chunk_msg)
        except Exception as exc:
            error_msg = StreamError(room_id=room_id, stream_id=stream_id, error=str(exc))
            for conn_id in list(self._stream_send_fns):
                await self._send_stream_message(conn_id, error_msg)
            raise

        # Build final event
        final_text = running_text
        final_event = event.model_copy(update={"content": TextContent(body=final_text)})

        # Send stream_end to streaming connections
        end_msg = StreamEnd(room_id=room_id, stream_id=stream_id, event=final_event)
        for conn_id in list(self._stream_send_fns):
            await self._send_stream_message(conn_id, end_msg)

        # Deliver final event to non-streaming connections
        for conn_id, send_fn in list(self._connections.items()):
            if conn_id in self._stream_send_fns:
                continue
            try:
                await send_fn(conn_id, final_event)
                self._error_counts.pop(conn_id, None)
            except Exception:
                self._handle_send_error(conn_id)

        return ChannelOutput.empty()

    async def _send_stream_message(self, conn_id: str, msg: StreamMessage) -> None:
        """Send a streaming protocol message, tracking errors."""
        stream_send_fn = self._stream_send_fns.get(conn_id)
        if stream_send_fn is None:
            logger.warning("_send_stream_message: no stream_send_fn for %s", conn_id)
            return
        try:
            await stream_send_fn(conn_id, msg)
            self._error_counts.pop(conn_id, None)
        except Exception:
            logger.exception(
                "_send_stream_message: failed for %s (msg type=%s)", conn_id, msg.type
            )
            self._handle_send_error(conn_id)

    def _handle_send_error(self, conn_id: str) -> None:
        """Increment error count and remove connection after threshold."""
        consecutive = self._error_counts.get(conn_id, 0) + 1
        self._error_counts[conn_id] = consecutive
        if consecutive >= self._MAX_CONSECUTIVE_ERRORS:
            logger.warning(
                "WebSocket connection %s removed after %d consecutive failures",
                conn_id,
                consecutive,
            )
            self._connections.pop(conn_id, None)
            self._stream_send_fns.pop(conn_id, None)
            self._error_counts.pop(conn_id, None)
        else:
            logger.warning(
                "WebSocket send failed for connection %s (attempt %d/%d)",
                conn_id,
                consecutive,
                self._MAX_CONSECUTIVE_ERRORS,
            )

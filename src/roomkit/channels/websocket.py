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
        # Which rooms each connection is subscribed to, and the reverse index
        # used on the delivery path. A connection receives a room's events only
        # if it appears here for that room — the channel never broadcasts to
        # every socket it happens to hold.
        self._connection_rooms: dict[str, set[str]] = {}
        self._room_connections: dict[str, set[str]] = {}

    @property
    def info(self) -> dict[str, Any]:
        return {
            "connection_count": len(self._connections),
            "room_count": len(self._room_connections),
        }

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
            supports_threading=True,
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
        room_id: str,
        stream_send_fn: StreamSendFn | None = None,
    ) -> None:
        """Register a WebSocket connection and subscribe it to a room.

        Args:
            connection_id: Unique connection identifier.
            send_fn: Callback for delivering complete events.
            room_id: The room this connection is for. Required: a channel
                instance can be attached to several rooms, and without this the
                channel has no way to tell which of its sockets belongs to the
                room it is delivering — it would have to send everything to
                everyone. Use :meth:`subscribe` to add further rooms to the
                same connection.
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
        self.subscribe(connection_id, room_id)

    def subscribe(self, connection_id: str, room_id: str) -> None:
        """Also deliver *room_id*'s events to an existing connection.

        For the client that holds several conversations open on one socket.
        """
        self._connection_rooms.setdefault(connection_id, set()).add(room_id)
        self._room_connections.setdefault(room_id, set()).add(connection_id)

    def unsubscribe(self, connection_id: str, room_id: str) -> None:
        """Stop delivering *room_id*'s events to a connection."""
        rooms = self._connection_rooms.get(connection_id)
        if rooms is not None:
            rooms.discard(room_id)
            if not rooms:
                self._connection_rooms.pop(connection_id, None)
        conns = self._room_connections.get(room_id)
        if conns is not None:
            conns.discard(connection_id)
            if not conns:
                self._room_connections.pop(room_id, None)

    def rooms_for(self, connection_id: str) -> set[str]:
        """Rooms a connection currently receives."""
        return set(self._connection_rooms.get(connection_id, ()))

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection and drop its subscriptions."""
        self._connections.pop(connection_id, None)
        self._stream_send_fns.pop(connection_id, None)
        self._error_counts.pop(connection_id, None)
        for room_id in self.rooms_for(connection_id):
            self.unsubscribe(connection_id, room_id)

    def _connections_in(self, room_id: str) -> list[tuple[str, SendFn]]:
        """Live connections subscribed to *room_id*, as (id, send_fn) pairs."""
        return [
            (conn_id, self._connections[conn_id])
            for conn_id in sorted(self._room_connections.get(room_id, ()))
            if conn_id in self._connections
        ]

    def _streaming_connections_in(self, room_id: str) -> list[str]:
        """Ids of connections in *room_id* that speak the streaming protocol."""
        return [
            conn_id
            for conn_id in sorted(self._room_connections.get(room_id, ()))
            if conn_id in self._stream_send_fns
        ]

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether any connected client supports streaming text delivery."""
        return bool(self._stream_send_fns)

    def supports_streaming_delivery_for(self, room_id: str) -> bool:
        """Whether any client *in this room* speaks the streaming protocol.

        The channel-wide property answers for every socket the channel holds,
        which is the wrong question once connections are scoped: a room whose
        clients are all non-streaming would otherwise take the streaming path
        and fall back at the end, having set up a stream nobody reads.
        """
        return bool(self._streaming_connections_in(room_id))

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
        # Only this room's connections. The binding names the room being
        # delivered; sending to every socket the channel holds would put one
        # room's messages into another's client.
        for conn_id, send_fn in self._connections_in(binding.room_id):
            try:
                await send_fn(conn_id, event)
                self._error_counts.pop(conn_id, None)
            except Exception:
                self._handle_send_error(conn_id)
        return ChannelOutput.empty()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Deliver a streaming response with interleaved events to clients.

        The stream yields ``str`` for text deltas and ``RoomEvent`` for
        persisted events (text segments, tool calls). Streaming-capable
        connections receive:

        - ``stream_start`` — streaming begins
        - ``stream_chunk`` — text delta (drives the live bubble)
        - ``event`` — persisted event (tool call, text segment)
        - ``stream_end`` — streaming complete

        Non-streaming connections receive all persisted events via
        the regular ``send_fn``.
        """
        stream_id = uuid.uuid4().hex
        room_id = event.room_id
        source = event.source

        # Every fan-out below is scoped to the room being delivered, not to
        # every socket the channel holds.
        streaming_conns = self._streaming_connections_in(binding.room_id)
        room_conns = self._connections_in(binding.room_id)

        logger.debug(
            "deliver_stream: channel=%s, room=%s, stream=%s, streaming_conns=%d, room_conns=%d",
            self.channel_id,
            binding.room_id,
            stream_id[:8],
            len(streaming_conns),
            len(room_conns),
        )

        # Send stream_start to this room's streaming connections
        start_msg = StreamStart(room_id=room_id, stream_id=stream_id, source=source)
        for conn_id in streaming_conns:
            await self._send_stream_message(conn_id, start_msg)

        # Stream: text deltas as stream_chunk, RoomEvents as event messages
        accumulated: list[str] = []
        running_text = ""
        segment_events: list[RoomEvent] = []
        try:
            async for delta in text_stream:
                if isinstance(delta, str):
                    accumulated.append(delta)
                    running_text += delta
                    chunk_msg = StreamChunk(
                        room_id=room_id,
                        stream_id=stream_id,
                        delta=delta,
                        text=running_text,
                    )
                    for conn_id in streaming_conns:
                        await self._send_stream_message(conn_id, chunk_msg)
                elif isinstance(delta, RoomEvent):
                    # Deliver persisted event inline during streaming
                    segment_events.append(delta)
                    for conn_id, send_fn in room_conns:
                        try:
                            await send_fn(conn_id, delta)
                        except Exception:
                            self._handle_send_error(conn_id)
        except Exception as exc:
            error_msg = StreamError(room_id=room_id, stream_id=stream_id, error=str(exc))
            for conn_id in streaming_conns:
                await self._send_stream_message(conn_id, error_msg)
            raise

        # Build final event — use last text segment if segments exist,
        # otherwise use the full accumulated text
        if segment_events:
            # Segments were delivered inline — stream_end carries no event
            # (the UI already has all events)
            segmented_meta = {**event.metadata, "_segmented": True}
            final_event = event.model_copy(
                update={"content": TextContent(body=""), "metadata": segmented_meta}
            )
        else:
            # Simple text response — stream_end carries the full text
            final_event = event.model_copy(update={"content": TextContent(body=running_text)})

        # Send stream_end to streaming connections
        end_msg = StreamEnd(room_id=room_id, stream_id=stream_id, event=final_event)
        for conn_id in streaming_conns:
            await self._send_stream_message(conn_id, end_msg)

        # Deliver final event to this room's non-streaming connections
        for conn_id, send_fn in room_conns:
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

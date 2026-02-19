"""WebSocket event source for RoomKit."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.sources.base import BaseSourceProvider, EmitCallback, SourceStatus

# Optional dependency - import for type checking and availability check
try:
    import websockets
    from websockets import ClientConnection

    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    ClientConnection = None  # type: ignore[assignment, misc]
    HAS_WEBSOCKETS = False

logger = logging.getLogger("roomkit.sources.websocket")

# Type alias for message parser
MessageParser = Callable[[str | bytes], InboundMessage | None]


def default_json_parser(channel_id: str) -> MessageParser:
    """Create a default JSON message parser.

    Expects messages in format:
    {
        "sender_id": "user123",
        "text": "Hello world",
        "external_id": "msg-456",  # optional
        "metadata": {}             # optional
    }

    Args:
        channel_id: Channel ID to use for parsed messages.

    Returns:
        A parser function that converts JSON to InboundMessage.
    """

    def parser(raw: str | bytes) -> InboundMessage | None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            data = json.loads(raw)

            # Skip non-message events (e.g., pings, acks)
            if not isinstance(data, dict):
                return None
            if "sender_id" not in data:
                return None

            return InboundMessage(
                channel_id=channel_id,
                sender_id=data["sender_id"],
                content=TextContent(body=data.get("text", "")),
                external_id=data.get("external_id"),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse message: %s", e)
            return None

    return parser


class WebSocketSource(BaseSourceProvider):
    """WebSocket client source for receiving messages.

    Connects to a WebSocket server and emits parsed messages into RoomKit.
    Handles reconnection automatically when the connection drops.

    Example:
        from roomkit import RoomKit
        from roomkit.sources.websocket import WebSocketSource

        # Simple usage with default JSON parser
        source = WebSocketSource(
            url="wss://chat.example.com/events",
            channel_id="websocket-chat",
        )
        await kit.attach_source("websocket-chat", source)

        # Custom parser for non-JSON messages
        def my_parser(raw: str) -> InboundMessage | None:
            parts = raw.split("|")
            if len(parts) < 2:
                return None
            return InboundMessage(
                channel_id="custom",
                sender_id=parts[0],
                content=TextContent(body=parts[1]),
            )

        source = WebSocketSource(
            url="wss://custom.example.com/stream",
            channel_id="custom",
            parser=my_parser,
        )
    """

    def __init__(
        self,
        url: str,
        channel_id: str,
        *,
        parser: MessageParser | None = None,
        headers: dict[str, str] | None = None,
        subprotocols: list[str] | None = None,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        close_timeout: float = 10.0,
        max_size: int = 2**20,  # 1 MB
        origin: str | None = None,
        reconnect: bool = False,
        max_reconnect_backoff: float = 30.0,
    ) -> None:
        """Initialize WebSocket source.

        Args:
            url: WebSocket URL to connect to (ws:// or wss://).
            channel_id: Channel ID for emitted messages.
            parser: Function to parse raw messages into InboundMessage.
                If None, uses default JSON parser.
            headers: Additional HTTP headers for the connection.
            subprotocols: WebSocket subprotocols to request.
            ping_interval: Interval between ping frames in seconds.
                Set to None to disable pings.
            ping_timeout: Timeout for pong response in seconds.
            close_timeout: Timeout for close handshake in seconds.
            max_size: Maximum message size in bytes.
            origin: Origin header value.
            reconnect: Automatically reconnect with exponential backoff on errors.
            max_reconnect_backoff: Maximum backoff between reconnect attempts in seconds.
        """
        super().__init__()
        self._url = url
        self._channel_id = channel_id
        self._parser = parser or default_json_parser(channel_id)
        self._headers = headers or {}
        self._subprotocols = subprotocols
        self._reconnect = reconnect
        self._max_reconnect_backoff = max_reconnect_backoff
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._close_timeout = close_timeout
        self._max_size = max_size
        self._origin = origin
        self._ws: ClientConnection | None = None

    @property
    def name(self) -> str:
        return f"websocket:{self._url}"

    async def start(self, emit: EmitCallback) -> None:
        """Connect and start receiving messages.

        This method handles reconnection automatically using the websockets
        library's built-in reconnection support.
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is required for WebSocketSource. "
                "Install it with: pip install roomkit[websocket]"
            )

        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)

        # Build connection kwargs — only include optional keys when explicitly set
        connect_kwargs: dict[str, Any] = {
            "uri": self._url,
            "ping_interval": self._ping_interval,
            "ping_timeout": self._ping_timeout,
            "close_timeout": self._close_timeout,
            "max_size": self._max_size,
        }
        # Only include optional keys when a value was provided
        if self._headers:
            connect_kwargs["additional_headers"] = self._headers
        if self._subprotocols is not None:
            connect_kwargs["subprotocols"] = self._subprotocols
        if self._origin is not None:
            connect_kwargs["origin"] = self._origin

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(**connect_kwargs) as ws:
                    self._ws = ws
                    self._set_status(SourceStatus.CONNECTED)
                    logger.info("Connected to %s", self._url)
                    backoff = 1.0  # Reset on successful connection

                    await self._receive_loop(ws, emit)
                    # Normal exit from receive loop (stop signaled)
                    if self._should_stop() or not self._reconnect:
                        return

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._reconnect or self._should_stop():
                    self._set_status(SourceStatus.ERROR, str(e))
                    raise
                self._set_status(SourceStatus.ERROR, str(e))
                logger.warning(
                    "WebSocket connection lost (%s), reconnecting in %.1fs",
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_reconnect_backoff)
                self._set_status(SourceStatus.CONNECTING)

    async def _receive_loop(
        self,
        ws: ClientConnection,
        emit: EmitCallback,
    ) -> None:
        """Main receive loop - reads messages and emits them."""
        while not self._should_stop():
            try:
                # Use wait_for to allow checking stop flag periodically
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)

                # Parse the message
                message = self._parser(raw)
                if message is not None:
                    result = await emit(message)
                    self._record_message()

                    if result.blocked:
                        logger.debug(
                            "Message blocked: %s",
                            result.reason,
                        )

            except TimeoutError:
                # Normal timeout - check stop flag and continue
                continue
            except Exception:
                # Connection error or other issue
                if self._should_stop():
                    break
                raise

    async def stop(self) -> None:
        """Stop receiving and close the connection."""
        await super().stop()

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        logger.info("WebSocket source stopped")

    async def send(self, message: str | bytes) -> None:
        """Send a message through the WebSocket connection.

        This allows bidirectional communication if needed.

        Args:
            message: Message to send (string or bytes).

        Raises:
            RuntimeError: If not connected.
        """
        if self._ws is None or self._status != SourceStatus.CONNECTED:
            raise RuntimeError("WebSocket not connected")

        await self._ws.send(message)

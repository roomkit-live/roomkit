"""Server-Sent Events (SSE) source for RoomKit."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.sources.base import BaseSourceProvider, EmitCallback, SourceStatus

# Optional dependency - import for availability check
try:
    import httpx
    from httpx_sse import aconnect_sse

    HAS_SSE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    aconnect_sse = None  # type: ignore[assignment]
    HAS_SSE = False

logger = logging.getLogger("roomkit.sources.sse")

# Type alias for event parser
SSEEventParser = Callable[[str, str, str | None], InboundMessage | None]


def default_json_parser(channel_id: str) -> SSEEventParser:
    """Create a default JSON event parser.

    Expects SSE data field to contain JSON:
    {
        "sender_id": "user123",
        "text": "Hello world",
        "external_id": "msg-456",  # optional
        "metadata": {}             # optional
    }

    Args:
        channel_id: Channel ID to use for parsed messages.

    Returns:
        A parser function that converts SSE events to InboundMessage.
    """

    def parser(event: str, data: str, event_id: str | None) -> InboundMessage | None:
        # Skip non-message events (e.g., heartbeats, pings)
        if event not in ("message", "msg", "chat", ""):
            logger.debug("Skipping event type: %s", event)
            return None

        try:
            payload = json.loads(data)

            if not isinstance(payload, dict):
                return None
            if "sender_id" not in payload:
                return None

            return InboundMessage(
                channel_id=channel_id,
                sender_id=payload["sender_id"],
                content=TextContent(body=payload.get("text", "")),
                external_id=payload.get("external_id") or event_id,
                metadata=payload.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse SSE data: %s", e)
            return None

    return parser


class SSESource(BaseSourceProvider):
    """Server-Sent Events (SSE) source for receiving messages.

    Connects to an SSE endpoint and emits parsed events into RoomKit.
    Handles reconnection automatically when the connection drops.

    Example:
        from roomkit import RoomKit
        from roomkit.sources import SSESource

        # Simple usage with default JSON parser
        source = SSESource(
            url="https://api.example.com/events",
            channel_id="sse-events",
        )
        await kit.attach_source("sse-events", source)

        # With authentication
        source = SSESource(
            url="https://api.example.com/events",
            channel_id="sse-events",
            headers={"Authorization": "Bearer token123"},
        )

        # Custom parser for non-JSON events
        def my_parser(event: str, data: str, event_id: str | None) -> InboundMessage | None:
            if event != "chat":
                return None
            return InboundMessage(
                channel_id="custom",
                sender_id="system",
                content=TextContent(body=data),
                external_id=event_id,
            )

        source = SSESource(
            url="https://stream.example.com/chat",
            channel_id="custom",
            parser=my_parser,
        )
    """

    def __init__(
        self,
        url: str,
        channel_id: str,
        *,
        parser: SSEEventParser | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        last_event_id: str | None = None,
        reconnect: bool = False,
        max_reconnect_backoff: float = 30.0,
    ) -> None:
        """Initialize SSE source.

        Args:
            url: SSE endpoint URL.
            channel_id: Channel ID for emitted messages.
            parser: Function to parse SSE events into InboundMessage.
                Receives (event_type, data, event_id) and returns InboundMessage or None.
                If None, uses default JSON parser.
            headers: HTTP headers for the request (e.g., Authorization).
            params: Query parameters for the URL.
            timeout: Connection timeout in seconds.
            last_event_id: Resume from this event ID (sent as Last-Event-ID header).
            reconnect: Automatically reconnect with exponential backoff on errors.
                When enabled, ``Last-Event-ID`` is sent on reconnection for resumption.
            max_reconnect_backoff: Maximum backoff between reconnect attempts in seconds.
        """
        super().__init__()
        self._url = url
        self._channel_id = channel_id
        self._parser = parser or default_json_parser(channel_id)
        self._headers = headers or {}
        self._params = params or {}
        self._reconnect = reconnect
        self._max_reconnect_backoff = max_reconnect_backoff
        self._timeout = timeout
        self._last_event_id = last_event_id
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"sse:{self._url}"

    async def start(self, emit: EmitCallback) -> None:
        """Connect and start receiving SSE events."""
        if not HAS_SSE:
            raise ImportError(
                "httpx and httpx-sse are required for SSESource. "
                "Install with: pip install roomkit[sse]"
            )

        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)

        backoff = 1.0
        while True:
            # Build headers with Last-Event-ID for resumption
            headers = dict(self._headers)
            if self._last_event_id:
                headers["Last-Event-ID"] = self._last_event_id

            try:
                # Use read_timeout=None so the SSE stream isn't closed
                # during idle periods between events.
                timeout = httpx.Timeout(self._timeout, read=None)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    self._client = client

                    async with aconnect_sse(
                        client,
                        "GET",
                        self._url,
                        headers=headers,
                        params=self._params,
                    ) as event_source:
                        self._set_status(SourceStatus.CONNECTED)
                        logger.info("Connected to SSE endpoint: %s", self._url)
                        backoff = 1.0  # Reset on successful connection

                        await self._receive_loop(event_source, emit)
                        # Normal exit (stop signaled)
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
                    "SSE connection lost (%s), reconnecting in %.1fs",
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_reconnect_backoff)
                self._set_status(SourceStatus.CONNECTING)
            finally:
                self._client = None

    async def _receive_loop(self, event_source: Any, emit: EmitCallback) -> None:
        """Main receive loop - reads SSE events and emits them."""
        async for sse in event_source.aiter_sse():
            if self._should_stop():
                break

            # Track last event ID for potential reconnection
            if sse.id:
                self._last_event_id = sse.id

            # Parse the event
            message = self._parser(sse.event, sse.data, sse.id)
            if message is not None:
                result = await emit(message)
                self._record_message()

                if result.blocked:
                    logger.debug("Message blocked: %s", result.reason)

    async def stop(self) -> None:
        """Stop receiving and close the connection."""
        await super().stop()
        logger.info("SSE source stopped")

    @property
    def last_event_id(self) -> str | None:
        """Get the last received event ID for resumption."""
        return self._last_event_id

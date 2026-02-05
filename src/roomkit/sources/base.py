"""Base abstraction for event-driven message sources."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from roomkit.models.delivery import InboundMessage, InboundResult

logger = logging.getLogger("roomkit.sources")


@unique
class SourceStatus(StrEnum):
    """Connection status for an event source."""

    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class SourceHealth(BaseModel):
    """Health information for an event source."""

    status: SourceStatus = SourceStatus.STOPPED
    connected_at: datetime | None = None
    last_message_at: datetime | None = None
    messages_received: int = 0
    error: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


# Type alias for the emit callback
EmitCallback = Callable[["InboundMessage"], Awaitable["InboundResult"]]


class SourceProvider(ABC):
    """Base class for event-driven message sources.

    A SourceProvider actively listens for inbound messages from an external
    system (WebSocket, NATS, SSE, WhatsApp via neonize, etc.) and emits them
    into RoomKit's inbound pipeline.

    Unlike webhook-based providers that receive HTTP POST requests, source
    providers maintain persistent connections and push messages as they arrive.

    Lifecycle:
        1. Create source instance with configuration
        2. Call `start(emit)` - source connects and begins listening
        3. Source calls `emit(message)` for each inbound message
        4. Call `stop()` to disconnect and cleanup

    Example:
        class MyWebSocketSource(SourceProvider):
            def __init__(self, url: str):
                self._url = url
                self._ws = None
                self._status = SourceStatus.STOPPED

            @property
            def name(self) -> str:
                return f"websocket:{self._url}"

            async def start(self, emit: EmitCallback) -> None:
                self._status = SourceStatus.CONNECTING
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    self._status = SourceStatus.CONNECTED
                    async for message in ws:
                        inbound = self._parse(message)
                        await emit(inbound)

            async def stop(self) -> None:
                self._status = SourceStatus.STOPPED
                if self._ws:
                    await self._ws.close()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source instance.

        Used for logging and framework events. Should be descriptive,
        e.g. "neonize:session.db" or "nats:events.inbound".
        """
        ...

    @abstractmethod
    async def start(self, emit: EmitCallback) -> None:
        """Start receiving messages and emit them via the callback.

        This method should:
        1. Establish connection to the external system
        2. Listen for incoming messages in a loop
        3. Parse each message into an InboundMessage
        4. Call `await emit(message)` for each message
        5. Handle reconnection internally if the connection drops

        The method should run until `stop()` is called or an unrecoverable
        error occurs. For recoverable errors (network issues), implement
        reconnection with backoff.

        Args:
            emit: Callback to emit messages into RoomKit. Returns InboundResult
                  indicating whether the message was processed or blocked.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving messages and release resources.

        This method should:
        1. Signal the start() loop to exit
        2. Close any open connections
        3. Cancel any pending tasks
        4. Release any held resources

        After stop() returns, start() should be safe to call again.
        """
        ...

    @property
    def status(self) -> SourceStatus:
        """Current connection status.

        Subclasses should override this to return the actual status.
        Default implementation returns STOPPED.
        """
        return SourceStatus.STOPPED

    async def healthcheck(self) -> SourceHealth:
        """Return health information for monitoring.

        Subclasses should override this to provide detailed health info
        including message counts, timestamps, and any error details.
        """
        return SourceHealth(status=self.status)


class BaseSourceProvider(SourceProvider):
    """Convenience base class with common source functionality.

    Provides:
    - Status tracking
    - Message counting
    - Timestamp tracking
    - Stop signal via asyncio.Event
    """

    def __init__(self) -> None:
        self._status = SourceStatus.STOPPED
        self._connected_at: datetime | None = None
        self._last_message_at: datetime | None = None
        self._messages_received: int = 0
        self._error: str | None = None
        self._stop_event = asyncio.Event()

    @property
    def status(self) -> SourceStatus:
        return self._status

    async def healthcheck(self) -> SourceHealth:
        return SourceHealth(
            status=self._status,
            connected_at=self._connected_at,
            last_message_at=self._last_message_at,
            messages_received=self._messages_received,
            error=self._error,
        )

    def _set_status(self, status: SourceStatus, error: str | None = None) -> None:
        """Update status and optionally set error message."""
        self._status = status
        self._error = error
        if status == SourceStatus.CONNECTED:
            self._connected_at = datetime.now(UTC)
            self._error = None

    def _record_message(self) -> None:
        """Record that a message was received."""
        self._messages_received += 1
        self._last_message_at = datetime.now(UTC)

    async def stop(self) -> None:
        """Signal the source to stop."""
        self._stop_event.set()
        self._status = SourceStatus.STOPPED

    def _should_stop(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()

    def _reset_stop(self) -> None:
        """Reset stop signal for restart."""
        self._stop_event.clear()

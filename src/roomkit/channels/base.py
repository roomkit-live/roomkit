"""Abstract base class for channels."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.models.trace import ProtocolTrace

# Callback type for protocol trace observers
TraceCallback = Callable[["ProtocolTrace"], Any]

_trace_logger = logging.getLogger("roomkit.trace")


def _safe_invoke(cb: TraceCallback, trace: ProtocolTrace) -> None:
    """Invoke a trace callback, scheduling coroutines as tasks."""
    try:
        result = cb(trace)
        if asyncio.coroutines.iscoroutine(result):
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(result)
    except Exception:
        _trace_logger.exception("Trace callback error")


class Channel(ABC):
    """Base class for all channels."""

    channel_type: ChannelType
    category: ChannelCategory = ChannelCategory.TRANSPORT
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL

    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._provider: Any = None
        self._trace_callbacks: list[tuple[TraceCallback, frozenset[str] | None]] = []
        self._trace_framework_handler: TraceCallback | None = None

    # -------------------------------------------------------------------------
    # Protocol trace
    # -------------------------------------------------------------------------

    @property
    def trace_enabled(self) -> bool:
        """Whether any trace observers are registered."""
        return bool(self._trace_callbacks) or self._trace_framework_handler is not None

    def on_trace(
        self,
        callback: TraceCallback,
        *,
        protocols: list[str] | None = None,
    ) -> None:
        """Register a protocol trace observer.

        Args:
            callback: Called with each :class:`ProtocolTrace`.  May be sync
                or async (coroutines are scheduled as tasks).
            protocols: Optional allowlist of protocol names (e.g.
                ``["sip"]``).  ``None`` means all protocols.
        """
        self._trace_callbacks.append((callback, frozenset(protocols) if protocols else None))

    def emit_trace(self, trace: ProtocolTrace) -> None:
        """Emit a protocol trace to all registered observers."""
        for cb, protocols in self._trace_callbacks:
            if protocols is None or trace.protocol in protocols:
                _safe_invoke(cb, trace)
        if self._trace_framework_handler is not None:
            _safe_invoke(self._trace_framework_handler, trace)

    def resolve_trace_room(self, session_id: str | None) -> str | None:  # noqa: B027
        """Resolve a room ID for a trace with the given session.

        Override in session-based channels (voice, realtime voice) to
        map session IDs to room IDs.  Returns ``None`` by default.
        """
        return None

    @property
    def provider_name(self) -> str | None:
        """Provider or backend name for event attribution."""
        p = self._provider
        return p.name if p is not None and hasattr(p, "name") else None

    # -------------------------------------------------------------------------
    # Channel metadata
    # -------------------------------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Return channel metadata. Override in subclasses."""
        return {}

    @abstractmethod
    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Process an inbound message into a RoomEvent."""

    @abstractmethod
    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Deliver an event to this channel."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to an event. Default: no-op for transport channels."""
        return ChannelOutput.empty()

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether this channel can accept streaming text delivery."""
        return False

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Deliver a streaming text response to this channel.

        Default: accumulate text, deliver as complete event.
        """
        chunks: list[str] = []
        async for chunk in text_stream:
            chunks.append(chunk)
        updated = event.model_copy(update={"content": TextContent(body="".join(chunks))})
        return await self.deliver(updated, binding, context)

    async def connect_session(  # noqa: B027
        self,
        session: Any,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Accept a long-lived session after inbound processing.

        Called by ``process_inbound`` when ``message.session`` is present
        and hooks did not block.  Override in session-based channels
        (voice, persistent WebSocket, etc.).  Default: no-op.
        """

    async def disconnect_session(  # noqa: B027
        self,
        session: Any,
        room_id: str,
    ) -> None:
        """Clean up a session on remote disconnect.

        Override in session-based channels to release resources.
        Default: no-op.
        """

    def capabilities(self) -> ChannelCapabilities:
        """Return channel capabilities."""
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])

    async def close(self) -> None:
        """Close the channel and its provider."""
        if self._provider is not None:
            await self._provider.close()

    @staticmethod
    def extract_text(event: RoomEvent) -> str:
        """Extract plain text from an event's content."""
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""

"""Abstract base class for channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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


class Channel(ABC):
    """Base class for all channels."""

    channel_type: ChannelType
    category: ChannelCategory = ChannelCategory.TRANSPORT
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL

    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._provider: Any = None

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

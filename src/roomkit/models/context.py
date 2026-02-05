"""Room context model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant
from roomkit.models.room import Room


class RoomContext(BaseModel):
    """Contextual information about a room for hook and channel processing."""

    room: Room
    bindings: list[ChannelBinding] = Field(default_factory=list)
    participants: list[Participant] = Field(default_factory=list)
    recent_events: list[RoomEvent] = Field(default_factory=list)

    def other_channels(self, exclude_channel_id: str) -> list[ChannelBinding]:
        """Get all bindings except the specified channel."""
        return [b for b in self.bindings if b.channel_id != exclude_channel_id]

    def channels_by_type(self, channel_type: ChannelType) -> list[ChannelBinding]:
        """Get all bindings of a specific channel type."""
        return [b for b in self.bindings if b.channel_type == channel_type]

    def get_binding(self, channel_id: str) -> ChannelBinding | None:
        """Get the binding for a specific channel."""
        for b in self.bindings:
            if b.channel_id == channel_id:
                return b
        return None

"""Room context model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant
from roomkit.models.room import Room


class RoomContext(BaseModel):
    """Contextual information about a room for hook and channel processing.

    ``recent_events`` is the room's **tail** in ascending order: its last
    element is the most recent event the context knows about. Every consumer
    reads it that way — memory providers slice ``[-N:]``, the inbound pipeline
    appends the freshly committed event at the end, and hooks scanning for
    "the latest X" iterate it reversed. A producer building a ``RoomContext``
    by hand owes it the same shape.

    The room, bindings, participants and events it holds are the store's
    snapshots (RFC §14.4): a hook or a channel treats them as frozen, and
    changes the room through the kit (``update_room_metadata()``), never in
    place. ``InMemoryStore`` shares them, so a change made in place would
    reach the store behind the write interface's back.
    """

    room: Room
    bindings: list[ChannelBinding] = Field(default_factory=list)
    participants: list[Participant] = Field(default_factory=list)
    recent_events: list[RoomEvent] = Field(default_factory=list)
    history_start_index: int | None = Field(default=None, ge=0)
    """Lower index from which the loaded conversation tail is complete.

    Zero means the whole conversation was loaded, even when lifecycle records
    precede its first message. A positive value marks a bounded read's oldest
    retained conversation event. Visibility filtering preserves this boundary:
    removing a hidden message does not make the loaded window smaller.
    ``None`` means its producer supplied no boundary (older or hand-built
    contexts); readers may retain their conservative index-based fallback.
    """
    author_register: dict[str, Any] | None = None
    """The room's register of authors (RFC §6.4, §10.1 step 12), which the
    labels of its turns read their ranks from, as the store keeps it apart
    from the room's metadata: its digests name the room's senders to whoever
    tries every phone number or name, so it reaches the runtime's readers and
    a remote executor here, never a client through ``room.metadata``.
    ``None`` when the room holds none or the context was built without it."""

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

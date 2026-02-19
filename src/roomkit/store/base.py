"""Abstract base class for conversation storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import RoomEvent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task


class ConversationStore(ABC):
    """Persistent storage for rooms, events, bindings, and participants.

    Implement this ABC to plug in any storage backend (SQL, Redis, etc.).
    The library ships with `InMemoryStore` for development and testing.
    """

    # Room operations

    @abstractmethod
    async def create_room(self, room: Room) -> Room:
        """Persist a new room."""
        ...

    @abstractmethod
    async def get_room(self, room_id: str) -> Room | None:
        """Get a room by ID, or ``None`` if it doesn't exist."""
        ...

    @abstractmethod
    async def update_room(self, room: Room) -> Room:
        """Update an existing room."""
        ...

    @abstractmethod
    async def delete_room(self, room_id: str) -> bool:
        """Delete a room. Returns ``True`` if the room existed."""
        ...

    @abstractmethod
    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        """List rooms with pagination."""
        ...

    @abstractmethod
    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Room]:
        """Find rooms matching the given filters."""
        ...

    @abstractmethod
    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        """Find the most recent room for a participant."""
        ...

    @abstractmethod
    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        """Find a room ID that has a binding for the given channel_id."""
        ...

    # Event operations

    @abstractmethod
    async def add_event(self, event: RoomEvent) -> RoomEvent:
        """Store a new event."""
        ...

    @abstractmethod
    async def get_event(self, event_id: str) -> RoomEvent | None:
        """Get an event by ID."""
        ...

    @abstractmethod
    async def update_event(self, event: RoomEvent) -> RoomEvent:
        """Update an existing event (e.g., mark as edited or deleted)."""
        ...

    @abstractmethod
    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
    ) -> list[RoomEvent]:
        """List events in a room with pagination and optional visibility filter."""
        ...

    @abstractmethod
    async def check_idempotency(self, room_id: str, key: str) -> bool:
        """Check if an idempotency key has been seen. Returns ``True`` if duplicate."""
        ...

    @abstractmethod
    async def get_event_count(self, room_id: str) -> int:
        """Return the total number of events in a room."""
        ...

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomically assign the next index and store the event.

        The default implementation reads the count and writes in two steps.
        Backends should override this with an atomic implementation (e.g.
        a single SQL transaction) to prevent race conditions on the index.
        """
        count = await self.get_event_count(room_id)
        indexed = event.model_copy(update={"index": count})
        return await self.add_event(indexed)

    # Binding operations

    @abstractmethod
    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        """Attach a channel binding to a room."""
        ...

    @abstractmethod
    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        """Get a channel binding, or ``None`` if not attached."""
        ...

    @abstractmethod
    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        """Update an existing channel binding."""
        ...

    @abstractmethod
    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        """Detach a channel from a room. Returns ``True`` if it was attached."""
        ...

    @abstractmethod
    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        """List all channel bindings for a room."""
        ...

    # Participant operations

    @abstractmethod
    async def add_participant(self, participant: Participant) -> Participant:
        """Add a participant to a room."""
        ...

    @abstractmethod
    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        """Get a participant by ID within a room."""
        ...

    @abstractmethod
    async def update_participant(self, participant: Participant) -> Participant:
        """Update a participant."""
        ...

    @abstractmethod
    async def list_participants(self, room_id: str) -> list[Participant]:
        """List all participants in a room."""
        ...

    # Identity operations

    @abstractmethod
    async def create_identity(self, identity: Identity) -> Identity:
        """Create a new identity record."""
        ...

    @abstractmethod
    async def get_identity(self, identity_id: str) -> Identity | None:
        """Get an identity by ID."""
        ...

    @abstractmethod
    async def resolve_identity(self, channel_type: str, address: str) -> Identity | None:
        """Look up an identity by channel type and address."""
        ...

    @abstractmethod
    async def link_address(self, identity_id: str, channel_type: str, address: str) -> None:
        """Link a channel address to an identity."""
        ...

    # Task operations

    @abstractmethod
    async def add_task(self, task: Task) -> Task:
        """Store a new task."""
        ...

    @abstractmethod
    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        ...

    @abstractmethod
    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        """List tasks for a room, optionally filtered by status."""
        ...

    @abstractmethod
    async def update_task(self, task: Task) -> Task:
        """Update a task."""
        ...

    # Observation operations

    @abstractmethod
    async def add_observation(self, observation: Observation) -> Observation:
        """Store a new observation."""
        ...

    @abstractmethod
    async def list_observations(self, room_id: str) -> list[Observation]:
        """List all observations for a room."""
        ...

    # Read tracking

    @abstractmethod
    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        """Mark an event as read for a channel."""
        ...

    @abstractmethod
    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        """Mark all events as read for a channel."""
        ...

    @abstractmethod
    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        """Return the number of unread events for a channel in a room."""
        ...

"""In-memory implementation of ConversationStore."""

from __future__ import annotations

from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import RoomEvent
from roomkit.models.identity import Identity
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from roomkit.store.base import ConversationStore


class InMemoryStore(ConversationStore):
    """Dict-based in-memory store for development and testing."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._events: dict[str, RoomEvent] = {}
        self._room_events: dict[str, list[str]] = {}
        self._bindings: dict[str, dict[str, ChannelBinding]] = {}
        self._participants: dict[str, dict[str, Participant]] = {}
        self._idempotency: dict[str, set[str]] = {}
        self._read_markers: dict[str, dict[str, str]] = {}
        self._identities: dict[str, Identity] = {}
        self._address_index: dict[tuple[str, str], str] = {}
        self._tasks: dict[str, Task] = {}
        self._room_tasks: dict[str, list[str]] = {}
        self._observations: dict[str, Observation] = {}
        self._room_observations: dict[str, list[str]] = {}

    # Room operations

    async def create_room(self, room: Room) -> Room:
        self._rooms[room.id] = room
        self._room_events.setdefault(room.id, [])
        self._bindings.setdefault(room.id, {})
        self._participants.setdefault(room.id, {})
        self._idempotency.setdefault(room.id, set())
        self._read_markers.setdefault(room.id, {})
        return room

    async def get_room(self, room_id: str) -> Room | None:
        room = self._rooms.get(room_id)
        return room.model_copy() if room is not None else None

    async def update_room(self, room: Room) -> Room:
        if room.id not in self._rooms:
            from roomkit.core.framework import RoomNotFoundError

            raise RoomNotFoundError(room.id)
        self._rooms[room.id] = room
        return room

    async def delete_room(self, room_id: str) -> bool:
        if room_id not in self._rooms:
            return False
        del self._rooms[room_id]
        # Clean up events
        event_ids = self._room_events.pop(room_id, [])
        for eid in event_ids:
            self._events.pop(eid, None)
        # Clean up tasks
        task_ids = self._room_tasks.pop(room_id, [])
        for tid in task_ids:
            self._tasks.pop(tid, None)
        # Clean up observations
        obs_ids = self._room_observations.pop(room_id, [])
        for oid in obs_ids:
            self._observations.pop(oid, None)
        self._bindings.pop(room_id, None)
        self._participants.pop(room_id, None)
        self._idempotency.pop(room_id, None)
        self._read_markers.pop(room_id, None)
        return True

    async def list_rooms(self, offset: int = 0, limit: int = 50) -> list[Room]:
        rooms = list(self._rooms.values())
        return [r.model_copy() for r in rooms[offset : offset + limit]]

    async def find_rooms(
        self,
        organization_id: str | None = None,
        status: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Room]:
        results: list[Room] = []
        for room in self._rooms.values():
            if organization_id is not None and room.organization_id != organization_id:
                continue
            if status is not None and room.status.value != status:
                continue
            if metadata_filter and not all(
                room.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            results.append(room.model_copy())
        return results[offset : offset + limit]

    async def find_latest_room(
        self,
        participant_id: str,
        channel_type: str | None = None,
        status: str | None = None,
    ) -> Room | None:
        best: Room | None = None
        for room in self._rooms.values():
            if status is not None and room.status.value != status:
                continue
            # Check if participant is in this room
            if participant_id not in self._participants.get(room.id, {}):
                # Also check bindings for participant_id
                found = False
                for binding in self._bindings.get(room.id, {}).values():
                    if binding.participant_id == participant_id and (
                        channel_type is None or binding.channel_type.value == channel_type
                    ):
                        found = True
                        break
                if not found:
                    continue
            if best is None or room.created_at > best.created_at:
                best = room
        return best.model_copy() if best is not None else None

    async def find_room_id_by_channel(
        self, channel_id: str, status: str | None = None
    ) -> str | None:
        for room_id, bindings in self._bindings.items():
            if channel_id in bindings:
                if status is not None:
                    room = self._rooms.get(room_id)
                    if room is None or room.status.value != status:
                        continue
                return room_id
        return None

    # Event operations

    async def add_event(self, event: RoomEvent) -> RoomEvent:
        self._events[event.id] = event
        self._room_events.setdefault(event.room_id, []).append(event.id)
        if event.idempotency_key:
            self._idempotency.setdefault(event.room_id, set()).add(event.idempotency_key)
        return event

    async def get_event(self, event_id: str) -> RoomEvent | None:
        event = self._events.get(event_id)
        return event.model_copy() if event is not None else None

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        self._events[event.id] = event
        return event

    async def list_events(
        self,
        room_id: str,
        offset: int = 0,
        limit: int = 50,
        visibility_filter: str | None = None,
    ) -> list[RoomEvent]:
        event_ids = self._room_events.get(room_id, [])
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        if visibility_filter is not None:
            events = [e for e in events if e.visibility == visibility_filter]
        return [e.model_copy() for e in events[offset : offset + limit]]

    async def check_idempotency(self, room_id: str, key: str) -> bool:
        return key in self._idempotency.get(room_id, set())

    async def get_event_count(self, room_id: str) -> int:
        return len(self._room_events.get(room_id, []))

    async def add_event_auto_index(self, room_id: str, event: RoomEvent) -> RoomEvent:
        """Atomically assign index = len(room_events) and append."""
        events = self._room_events.setdefault(room_id, [])
        indexed = event.model_copy(update={"index": len(events)})
        self._events[indexed.id] = indexed
        events.append(indexed.id)
        if indexed.idempotency_key:
            self._idempotency.setdefault(room_id, set()).add(indexed.idempotency_key)
        return indexed

    # Binding operations

    async def add_binding(self, binding: ChannelBinding) -> ChannelBinding:
        self._bindings.setdefault(binding.room_id, {})[binding.channel_id] = binding
        return binding

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        binding = self._bindings.get(room_id, {}).get(channel_id)
        return binding.model_copy() if binding is not None else None

    async def update_binding(self, binding: ChannelBinding) -> ChannelBinding:
        self._bindings.setdefault(binding.room_id, {})[binding.channel_id] = binding
        return binding

    async def remove_binding(self, room_id: str, channel_id: str) -> bool:
        room_bindings = self._bindings.get(room_id, {})
        if channel_id not in room_bindings:
            return False
        del room_bindings[channel_id]
        return True

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        return [b.model_copy() for b in self._bindings.get(room_id, {}).values()]

    # Participant operations

    async def add_participant(self, participant: Participant) -> Participant:
        self._participants.setdefault(participant.room_id, {})[participant.id] = participant
        return participant

    async def get_participant(self, room_id: str, participant_id: str) -> Participant | None:
        participant = self._participants.get(room_id, {}).get(participant_id)
        return participant.model_copy() if participant is not None else None

    async def update_participant(self, participant: Participant) -> Participant:
        self._participants.setdefault(participant.room_id, {})[participant.id] = participant
        return participant

    async def list_participants(self, room_id: str) -> list[Participant]:
        return [p.model_copy() for p in self._participants.get(room_id, {}).values()]

    # Read tracking

    async def mark_read(self, room_id: str, channel_id: str, event_id: str) -> None:
        self._read_markers.setdefault(room_id, {})[channel_id] = event_id

    async def mark_all_read(self, room_id: str, channel_id: str) -> None:
        event_ids = self._room_events.get(room_id, [])
        if event_ids:
            self._read_markers.setdefault(room_id, {})[channel_id] = event_ids[-1]

    async def get_unread_count(self, room_id: str, channel_id: str) -> int:
        event_ids = self._room_events.get(room_id, [])
        last_read = self._read_markers.get(room_id, {}).get(channel_id)
        if last_read is None:
            return len(event_ids)
        try:
            idx = event_ids.index(last_read)
            return len(event_ids) - idx - 1
        except ValueError:
            return len(event_ids)

    # Identity operations

    async def create_identity(self, identity: Identity) -> Identity:
        self._identities[identity.id] = identity
        for ch_type, addresses in identity.channel_addresses.items():
            for addr in addresses:
                self._address_index[(ch_type, addr)] = identity.id
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        identity = self._identities.get(identity_id)
        return identity.model_copy() if identity is not None else None

    async def resolve_identity(self, channel_type: str, address: str) -> Identity | None:
        identity_id = self._address_index.get((channel_type, address))
        if identity_id is None:
            return None
        identity = self._identities.get(identity_id)
        return identity.model_copy() if identity is not None else None

    async def link_address(self, identity_id: str, channel_type: str, address: str) -> None:
        identity = self._identities.get(identity_id)
        if identity is None:
            return
        current = identity.channel_addresses.get(channel_type, [])
        if address not in current:
            new_addresses = {**identity.channel_addresses, channel_type: [*current, address]}
            self._identities[identity_id] = identity.model_copy(
                update={"channel_addresses": new_addresses}
            )
        self._address_index[(channel_type, address)] = identity_id

    # Task operations

    async def add_task(self, task: Task) -> Task:
        self._tasks[task.id] = task
        self._room_tasks.setdefault(task.room_id, []).append(task.id)
        return task

    async def get_task(self, task_id: str) -> Task | None:
        task = self._tasks.get(task_id)
        return task.model_copy() if task is not None else None

    async def list_tasks(self, room_id: str, status: str | None = None) -> list[Task]:
        task_ids = self._room_tasks.get(room_id, [])
        tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return [t.model_copy() for t in tasks]

    async def update_task(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    # Observation operations

    async def add_observation(self, observation: Observation) -> Observation:
        self._observations[observation.id] = observation
        self._room_observations.setdefault(observation.room_id, []).append(observation.id)
        return observation

    async def list_observations(self, room_id: str) -> list[Observation]:
        obs_ids = self._room_observations.get(room_id, [])
        return [
            self._observations[oid].model_copy() for oid in obs_ids if oid in self._observations
        ]

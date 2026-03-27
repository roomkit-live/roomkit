"""RoomLifecycleMixin — room CRUD and participant management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core.exceptions import (
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomNotFoundError,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    ChannelCategory,
    EventType,
    HookTrigger,
    IdentificationStatus,
    RoomStatus,
)
from roomkit.models.participant import Participant
from roomkit.models.room import Room

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.locks import RoomLockManager
    from roomkit.orchestration.base import Orchestration
    from roomkit.recorder._room_recorder_manager import RoomRecorderManager
    from roomkit.recorder.base import RoomRecorderBinding
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

_ORCHESTRATION_UNSET: Any = object()


@runtime_checkable
class RoomLifecycleHost(Protocol):
    """Contract: capabilities a host class must provide for RoomLifecycleMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _lock_manager: Per-room lock for serialised mutation.
        _room_recorder_mgr: Manager for room-level media recording.
        _default_orchestration: Kit-level default orchestration strategy.
        _channels: Registry of channel-id to :class:`Channel` instances.

    Cross-mixin methods (provided by other mixins in the MRO):
        register_channel: From :class:`ChannelOpsMixin`.
        attach_channel: From :class:`ChannelOpsMixin`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _room_recorder_mgr: RoomRecorderManager
    _default_orchestration: Orchestration | None
    _channels: dict[str, Channel]


class RoomLifecycleMixin(HelpersMixin):
    """Room lifecycle operations: create, close, timers, participants.

    Host contract: :class:`RoomLifecycleHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _room_recorder_mgr: RoomRecorderManager
    _default_orchestration: Orchestration | None
    _channels: dict[str, Channel]

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    register_channel: Any  # see RoomLifecycleHost
    attach_channel: Any  # see RoomLifecycleHost

    async def create_room(
        self,
        room_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        recorders: list[RoomRecorderBinding] | None = None,
        orchestration: Orchestration | None | Any = _ORCHESTRATION_UNSET,
    ) -> Room:
        """Create a new room.

        Args:
            room_id: Optional explicit room ID (auto-generated if omitted).
            metadata: Optional room metadata dict.
            recorders: Optional list of :class:`RoomRecorderBinding` for
                room-level media recording.
            orchestration: Orchestration strategy for this room. Overrides
                the kit-level default. Pass ``None`` to explicitly disable.
        """
        room = Room(
            id=room_id or uuid4().hex,
            metadata=metadata or {},
        )
        result = await self._store.create_room(room)
        # Start room-level media recorders
        if recorders:
            self._room_recorder_mgr.register(room.id, recorders)

        # Apply orchestration strategy
        orch = (
            orchestration
            if orchestration is not _ORCHESTRATION_UNSET
            else self._default_orchestration
        )
        if orch is not None:
            for agent in orch.agents():
                if agent.channel_id not in self._channels:
                    self.register_channel(agent)
            for agent in orch.agents():
                await self.attach_channel(
                    room.id,
                    agent.channel_id,
                    category=ChannelCategory.INTELLIGENCE,
                )
            await orch.install(self, room.id)  # type: ignore[arg-type]

        await self._fire_lifecycle_hook(
            room.id,
            HookTrigger.ON_ROOM_CREATED,
            EventType.SYSTEM,
            code="room_created",
            message=f"Room {room.id} created",
            data={"room_id": room.id},
        )
        await self._emit_framework_event(
            "room_created", room_id=room.id, data={"room_id": room.id}
        )
        return result

    async def get_room(self, room_id: str) -> Room:
        """Get a room by ID. Raises RoomNotFoundError if missing."""
        room = await self._store.get_room(room_id)
        if room is None:
            raise RoomNotFoundError(f"Room {room_id} not found")
        return room

    async def close_room(self, room_id: str) -> Room:
        """Close a room."""
        async with self._lock_manager.locked(room_id):
            # Stop room-level media recorders before closing
            self._room_recorder_mgr.stop_room(room_id)
            room = await self.get_room(room_id)
            room = room.model_copy(
                update={"status": RoomStatus.CLOSED, "closed_at": datetime.now(UTC)}
            )
            result = await self._store.update_room(room)
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_ROOM_CLOSED,
                EventType.SYSTEM,
                code="room_closed",
                message=f"Room {room_id} closed",
                data={"room_id": room_id},
            )
            await self._emit_framework_event(
                "room_closed", room_id=room_id, data={"room_id": room_id}
            )
            return result

    async def check_room_timers(self, room_id: str) -> Room:
        """Check and apply timer-based transitions for a single room.

        Returns the room (possibly transitioned to PAUSED or CLOSED).
        """
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id)

            if room.status in (RoomStatus.CLOSED, RoomStatus.ARCHIVED):
                return room

            timers = room.timers
            if timers.last_activity_at is None:
                return room

            elapsed = (datetime.now(UTC) - timers.last_activity_at).total_seconds()

            # Check closed threshold first (supersedes pause)
            if timers.closed_after_seconds is not None and elapsed > timers.closed_after_seconds:
                if room.status != RoomStatus.CLOSED:
                    room = room.model_copy(
                        update={"status": RoomStatus.CLOSED, "closed_at": datetime.now(UTC)}
                    )
                    await self._store.update_room(room)
                    await self._emit_system_event(
                        room_id,
                        EventType.SYSTEM,
                        code="room_closed_by_timer",
                        message=f"Room {room_id} closed after {elapsed:.0f}s inactivity",
                        data={
                            "elapsed_seconds": elapsed,
                            "threshold": timers.closed_after_seconds,
                        },
                    )
                    await self._fire_lifecycle_hook(
                        room_id,
                        HookTrigger.ON_ROOM_CLOSED,
                        EventType.SYSTEM,
                        code="room_closed_by_timer",
                        message=f"Room {room_id} closed by timer",
                        data={"elapsed_seconds": elapsed},
                    )
                    await self._emit_framework_event(
                        "room_closed", room_id=room_id, data={"reason": "timer"}
                    )
                return room

            # Check pause threshold (only for ACTIVE rooms)
            if (
                room.status == RoomStatus.ACTIVE
                and timers.inactive_after_seconds is not None
                and elapsed > timers.inactive_after_seconds
            ):
                room = room.model_copy(
                    update={"status": RoomStatus.PAUSED, "updated_at": datetime.now(UTC)}
                )
                await self._store.update_room(room)
                await self._emit_system_event(
                    room_id,
                    EventType.SYSTEM,
                    code="room_paused_by_timer",
                    message=f"Room {room_id} paused after {elapsed:.0f}s inactivity",
                    data={"elapsed_seconds": elapsed, "threshold": timers.inactive_after_seconds},
                )
                await self._fire_lifecycle_hook(
                    room_id,
                    HookTrigger.ON_ROOM_PAUSED,
                    EventType.SYSTEM,
                    code="room_paused_by_timer",
                    message=f"Room {room_id} paused by timer",
                    data={"elapsed_seconds": elapsed},
                )
                await self._emit_framework_event(
                    "room_paused", room_id=room_id, data={"reason": "timer"}
                )

            return room

    async def check_all_timers(self) -> list[Room]:
        """Check timers on all active/paused rooms. Returns rooms that transitioned."""
        transitioned: list[Room] = []
        for status in (RoomStatus.ACTIVE, RoomStatus.PAUSED):
            rooms = await self._store.find_rooms(status=status.value)
            for room in rooms:
                old_status = room.status
                updated = await self.check_room_timers(room.id)
                if updated.status != old_status:
                    transitioned.append(updated)
        return transitioned

    async def update_room_metadata(self, room_id: str, metadata: dict[str, Any]) -> Room:
        """Update room metadata."""
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id)
            room = room.model_copy(
                update={"metadata": {**room.metadata, **metadata}, "updated_at": datetime.now(UTC)}
            )
            return await self._store.update_room(room)

    async def ensure_participant(
        self,
        room_id: str,
        channel_id: str,
        participant_id: str,
        display_name: str | None = None,
    ) -> Participant:
        """Get an existing participant or create one."""
        existing = await self._store.get_participant(room_id, participant_id)
        if existing:
            return existing
        participant = Participant(
            id=participant_id,
            room_id=room_id,
            channel_id=channel_id,
            display_name=display_name,
        )
        return await self._store.add_participant(participant)

    async def resolve_participant(
        self,
        room_id: str,
        participant_id: str,
        identity_id: str,
        resolved_by: str = "manual",
    ) -> Participant:
        """Resolve a pending participant to a known identity (RFC 7.4).

        Called by an advisor or automated process when a pending/ambiguous
        participant has been identified.
        """
        async with self._lock_manager.locked(room_id):
            participant = await self._store.get_participant(room_id, participant_id)
            if participant is None:
                raise ParticipantNotFoundError(
                    f"Participant {participant_id} not found in room {room_id}"
                )

            identity = await self._store.get_identity(identity_id)
            if identity is None:
                raise IdentityNotFoundError(f"Identity {identity_id} not found")

            # Update participant fields
            participant = participant.model_copy(
                update={
                    "identification": IdentificationStatus.IDENTIFIED,
                    "identity_id": identity_id,
                    "resolved_at": datetime.now(UTC),
                    "resolved_by": resolved_by,
                    "candidates": None,
                    "display_name": identity.display_name or participant.display_name,
                }
            )
            await self._store.update_participant(participant)

            # Update binding if present
            binding = await self._store.get_binding(room_id, participant.channel_id)
            if binding:
                binding = binding.model_copy(update={"participant_id": identity_id})
                await self._store.update_binding(binding)

            # Emit system event
            await self._emit_system_event(
                room_id,
                EventType.PARTICIPANT_IDENTIFIED,
                code="participant_identified",
                message=f"Participant {participant_id} identified as {identity.display_name}",
                data={
                    "participant_id": participant_id,
                    "identity_id": identity_id,
                    "resolved_by": resolved_by,
                },
            )

            # Fire lifecycle hook
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_PARTICIPANT_IDENTIFIED,
                EventType.PARTICIPANT_IDENTIFIED,
                code="participant_identified",
                message="Participant identified",
                data={
                    "participant_id": participant_id,
                    "identity_id": identity_id,
                },
            )

            return participant

"""RoomLifecycleMixin — room CRUD and participant management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._participant_channels import channels_reached, warn_cross_channel
from roomkit.core.exceptions import (
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomNotFoundError,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    AgentResponsePolicy,
    ChannelCategory,
    EventType,
    HookTrigger,
    IdentificationStatus,
    RoomStatus,
)
from roomkit.models.participant import Participant
from roomkit.models.room import Room, RoomTimers

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
    _default_agent_response_policy: AgentResponsePolicy
    _channels: dict[str, Channel]


class RoomLifecycleMixin(HelpersMixin):
    """Room lifecycle operations: create, close, timers, participants.

    Host contract: :class:`RoomLifecycleHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _room_recorder_mgr: RoomRecorderManager
    _default_orchestration: Orchestration | None
    _default_agent_response_policy: AgentResponsePolicy
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
        organization_id: str | None = None,
        timers: RoomTimers | None = None,
        agent_response_policy: AgentResponsePolicy | None = None,
    ) -> Room:
        """Create a new room.

        Args:
            room_id: Optional explicit room ID (auto-generated if omitted).
            metadata: Optional room metadata dict.
            recorders: Optional list of :class:`RoomRecorderBinding` for
                room-level media recording.
            orchestration: Orchestration strategy for this room. Overrides
                the kit-level default. Pass ``None`` to explicitly disable.
            organization_id: Optional organization/tenant ID for multi-tenant
                isolation. Stored on the room and used for org-level queries.
            timers: Optional lifecycle timers (auto-pause / auto-close on
                inactivity). When the timers omit ``last_activity_at``, the
                idle clock starts at creation time. See
                :meth:`check_room_timers` / :meth:`check_all_timers` for
                evaluating the thresholds.
        """
        if timers is not None and timers.last_activity_at is None:
            timers = timers.model_copy(update={"last_activity_at": datetime.now(UTC)})
        room = Room(
            id=room_id or uuid4().hex,
            organization_id=organization_id,
            metadata=metadata or {},
            timers=timers or RoomTimers(),
            agent_response_policy=(
                agent_response_policy
                if agent_response_policy is not None
                else self._default_agent_response_policy
            ),
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
            await orch.install(self, room.id)  # ty: ignore[invalid-argument-type]

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

    async def get_room(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Get a room by ID. Raises RoomNotFoundError if missing.

        Pass *organization_id* to scope the read to one tenant (RFC §17.2). A
        room belonging to another organization is reported as **not found**
        rather than refused: a distinct "wrong organization" error would let a
        caller probe which room ids exist outside its own tenant, which is the
        thing scoping is meant to prevent.

        A library has no caller or auth context of its own, so the scope has to
        come from the caller. Left unset, the read is unscoped and behaves as it
        always has.
        """
        room = await self._store.get_room(room_id)
        if room is None or (
            organization_id is not None and room.organization_id != organization_id
        ):
            raise RoomNotFoundError(f"Room {room_id} not found")
        return room

    async def close_room(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Close a room.

        *organization_id* scopes the operation to one tenant (RFC §17.2); a
        room belonging to another organization is reported as not found.
        """
        async with self._lock_manager.locked(room_id):
            # Stop room-level media recorders before closing
            self._room_recorder_mgr.stop_room(room_id)
            room = await self.get_room(room_id, organization_id=organization_id)
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

    async def archive_room(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Archive a room — the terminal storage state (RFC §5.1).

        ARCHIVED refuses new events exactly as CLOSED does; the difference is
        intent. A closed room MAY be reopened by returning it to ACTIVE; an
        archived one is done. History, participants and bindings stay readable
        at every status, archived included.

        This is the only path to the status, and therefore the only source of
        the ``room_archived`` framework event (§8.2).
        """
        async with self._lock_manager.locked(room_id):
            self._room_recorder_mgr.stop_room(room_id)
            room = await self.get_room(room_id, organization_id=organization_id)
            if room.status == RoomStatus.ARCHIVED:
                return room
            room = room.model_copy(
                update={
                    "status": RoomStatus.ARCHIVED,
                    "closed_at": room.closed_at or datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }
            )
            result = await self._store.update_room(room)
            await self._emit_framework_event(
                "room_archived", room_id=room_id, data={"room_id": room_id}
            )
            return result

    async def set_room_timers(self, room_id: str, timers: RoomTimers) -> Room:
        """Set or replace the lifecycle timers for an existing room.

        When ``timers.last_activity_at`` is unset, the room's existing activity
        timestamp is preserved (or the current time is used for a room that has
        seen no activity yet), so adjusting thresholds never resets the idle
        clock. Use :meth:`check_room_timers` / :meth:`check_all_timers` to apply
        the thresholds.
        """
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id)
            last_activity = (
                timers.last_activity_at or room.timers.last_activity_at or datetime.now(UTC)
            )
            timers = timers.model_copy(update={"last_activity_at": last_activity})
            room = room.model_copy(update={"timers": timers, "updated_at": datetime.now(UTC)})
            return await self._store.update_room(room)

    async def set_agent_response_policy(self, room_id: str, policy: AgentResponsePolicy) -> Room:
        """Set what an agent's own output solicits in this room (RFC §19.3.1).

        The policy is chosen at creation, but a room rarely knows then how many
        agents it will end up holding: one that gains a second agent is a
        different room, because under ``AGENT_CHAIN`` the first answer solicits
        the other agent, whose answer comes back, down to ``max_chain_depth``.
        Switching to ``ADDRESSED_ONLY`` at that moment is what keeps a room of
        independent agents from answering itself.

        Applies to events processed after it — an event already broadcast is
        not reconsidered. No-op when the room already holds *policy*.
        """
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id)
            if room.agent_response_policy is policy:
                return room
            room = room.model_copy(
                update={"agent_response_policy": policy, "updated_at": datetime.now(UTC)}
            )
            logger.info(
                "Room %s agent response policy set to %s",
                room_id,
                policy.value,
                extra={"room_id": room_id},
            )
            return await self._store.update_room(room)

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
                        # The record OF the closing, written once the status has
                        # flipped — the one write a CLOSED room still owes its
                        # timeline (RFC §5.1).
                        records_transition=True,
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
        """Get an existing participant or create one.

        A participant is one record per (room, id), so ``channel_id`` names the
        channel a record being *created* is primarily reached on — it is not a
        filter. A participant the room already has is returned as they stand,
        primary channel included, even when that channel is not the one asked
        for: the same person reached by SMS and then by email is one participant,
        not two (RFC §5.5). The channel asked for is recorded in
        ``connected_via``, and a channel that is not the record's primary one is
        logged, because nothing in the returned record would otherwise say so —
        a caller keeping a lifecycle or a status on it is keeping it on a record
        another channel also drives. Use :meth:`add_member` for a deliberate join.

        Bookkeeping only: no ``PARTICIPANT_UPDATED`` event, no
        ``ON_PARTICIPANT_UPDATED`` hook. Runs under the room lock — recording a
        channel is a read-modify-write, and ``add_member`` / ``remove_member``
        hold that lock while they write the same record.
        """
        async with self._lock_manager.locked(room_id):
            existing = await self._store.get_participant(room_id, participant_id)
            if existing:
                warn_cross_channel(existing, channel_id, rehomed=False)
                channels = channels_reached(existing, channel_id)
                if channels is None:
                    return existing
                return await self._store.update_participant(
                    existing.model_copy(update={"connected_via": channels})
                )
            participant = Participant(
                id=participant_id,
                room_id=room_id,
                channel_id=channel_id,
                connected_via=[channel_id],
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

"""RoomLifecycleMixin — room CRUD and participant management."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._participant_channels import channels_reached, warn_cross_channel
from roomkit.core._room_metadata import without_framework_keys
from roomkit.core.exceptions import (
    IdentityNotFoundError,
    ParticipantNotFoundError,
    RoomNotFoundError,
    TransitionRefusedError,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    AgentResponsePolicy,
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
        _file_announced: From :class:`RoomRecordingMixin`.
        _stop_room_recordings: From :class:`RoomRecordingMixin`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _room_recorder_mgr: RoomRecorderManager
    _default_orchestration: Orchestration | None
    _default_agent_response_policy: AgentResponsePolicy
    _channels: dict[str, Channel]


class RoomLifecycleMixin(HelpersMixin):
    """Room lifecycle operations: create, close, archive, timers, participants.

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
    _file_announced: Any  # see RoomLifecycleHost
    _stop_room_recordings: Any  # see RoomLifecycleHost
    _install_strategy: Any  # see StrategyMixin
    _forget_strategy: Any  # see StrategyMixin

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
            metadata: Optional room metadata dict; a key only the
                framework writes, under ``roomkit:`` or of the room's private
                state (``author_register``), is dropped with a warning.
            recorders: Optional list of :class:`RoomRecorderBinding` for
                room-level media recording.
            orchestration: Orchestration strategy for this room. Overrides
                the kit-level default. Pass ``None`` to explicitly disable.
            organization_id: Optional organization/tenant ID for multi-tenant
                isolation. Stored on the room and used for org-level queries.
            timers: Optional lifecycle timers (auto-pause / auto-close on
                inactivity, RFC §5.1). While the room has had no activity,
                ``last_activity_at`` stays unset and inactivity counts from
                its ``created_at``. Every locked section applies the
                transitions due, and :meth:`check_room_timers` /
                :meth:`check_all_timers` apply them to a room nobody writes
                to.

        Room ids are global, across organizations: a room is created once
        (RFC §14.3), so an id the store holds is refused whichever
        organization holds it, and the room it holds is kept. The refusal
        says the id is taken and nothing of the room. A host that lets its
        tenants choose room ids namespaces them (``f"{tenant}:{name}"``) or
        generates them (omit *room_id*): a refused id is one another tenant
        may hold.

        Raises:
            RoomAlreadyExistsError: The store holds a room under *room_id*;
                nothing is written, and the recordings started for it are
                stopped.
            Exception: A recorder's refusal to start (``ValueError`` from a
                file recorder without encryption at rest, RFC §17.6), raised
                as the recorder raised it. The recorders start before the room
                is written, so no room exists afterwards and the recordings
                already started are stopped (RFC §12.11).
        """
        metadata = without_framework_keys(metadata or {}, room_id=room_id, writer="create_room")
        room = Room(
            id=room_id or uuid4().hex,
            organization_id=organization_id,
            metadata=metadata,
            timers=timers or RoomTimers(),
            agent_response_policy=(
                agent_response_policy
                if agent_response_policy is not None
                else self._default_agent_response_policy
            ),
        )
        result = await self._store_recorded_room(room, recorders or [])
        # A strategy recorded for this id belonged to a room that is gone.
        await self._forget_strategy(room.id)

        # Apply orchestration strategy
        orch = (
            orchestration
            if orchestration is not _ORCHESTRATION_UNSET
            else self._default_orchestration
        )
        if orch is not None:
            # The path install_strategy() takes: claimed, recorded (RFC §19.7).
            await self._install_strategy(room.id, orch, organization_id=room.organization_id)

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

    async def _store_recorded_room(self, room: Room, recorders: list[RoomRecorderBinding]) -> Room:
        """Store a new room with its recorders running and announced (RFC §12.11).

        The recorders start first, all or nothing, so one that refuses fails
        the creation before anything is written; a store write that fails
        stops them. Once the room exists they are filed under it and each is
        announced: ON_RECORDING_STARTED is the consent point (RFC §17.6), and a
        room recorder captures nothing until a track is added, so the
        announcement still precedes any audio.
        """
        started = self._room_recorder_mgr.start(room.id, recorders)
        try:
            stored = await self._store.create_room(room)
        except BaseException:
            self._room_recorder_mgr.discard(started)
            raise
        await self._file_announced(room.id, started, room.organization_id)
        return stored

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

        The room is the store's snapshot (RFC §14.4): change it through a copy
        and a write (``room.model_copy(update=...)``), never in place.
        """
        room = await self._store.get_room(room_id)
        if room is None or (
            organization_id is not None and room.organization_id != organization_id
        ):
            raise RoomNotFoundError(f"Room {room_id} not found")
        return room

    async def close_room(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Close a room (RFC §5.1).

        The room's due timer transitions apply first. A room already CLOSED
        is returned as it is: nothing is written, no hook fires and no
        framework event is emitted.

        *organization_id* scopes the operation to one tenant (RFC §17.2); a
        room belonging to another organization is reported as not found.

        Raises:
            TransitionRefusedError: the room is ARCHIVED, which no call
                leaves; nothing is written.
        """
        hooks: list[asyncio.Task[None]] = []
        try:
            async with self._lock_manager.locked(room_id):
                room = await self.get_room(room_id, organization_id=organization_id)
                room = await self._room_timers.apply_due(room, hooks=hooks)
                if room.status is RoomStatus.CLOSED:
                    return room
                if room.status is RoomStatus.ARCHIVED:
                    raise TransitionRefusedError(room_id, room.status, "close")
                return await self._close(room)
        finally:
            await self._room_timers.settle(room_id, hooks)

    async def _close(self, room: Room) -> Room:
        """Close *room*, an ACTIVE or a PAUSED one, under its lock."""
        room_id = room.id
        # Stop room-level media recorders before closing, once the scoped
        # read has found the room: a refused call stops nothing (§12.11).
        await self._stop_room_recordings(room_id)
        room = room.model_copy(
            update={"status": RoomStatus.CLOSED, "closed_at": datetime.now(UTC), "paused_at": None}
        )
        result = await self._store_refusing_room(room)
        await self._fire_lifecycle_hook(
            room_id,
            HookTrigger.ON_ROOM_CLOSED,
            EventType.SYSTEM,
            code="room_closed",
            message=f"Room {room_id} closed",
            data={"room_id": room_id},
        )
        await self._emit_framework_event("room_closed", room_id=room_id, data={"room_id": room_id})
        return result

    async def archive_room(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Archive a room — the terminal storage state (RFC §5.1).

        ARCHIVED refuses new events exactly as CLOSED does; the difference is
        intent. A closed room MAY be reopened by returning it to ACTIVE; an
        archived one is done. History, participants and bindings stay readable
        at every status, archived included.

        This is the only path to the status, and therefore the only source of
        the ``room_archived`` framework event (§8.2). The room's due timer
        transitions apply first. A room already ARCHIVED is returned as it
        is: nothing is written or emitted.
        """
        hooks: list[asyncio.Task[None]] = []
        try:
            async with self._lock_manager.locked(room_id):
                room = await self.get_room(room_id, organization_id=organization_id)
                room = await self._room_timers.apply_due(room, hooks=hooks)
                if room.status is RoomStatus.ARCHIVED:
                    return room
                await self._stop_room_recordings(room_id)
                room = room.model_copy(
                    update={
                        "status": RoomStatus.ARCHIVED,
                        "closed_at": room.closed_at or datetime.now(UTC),
                        "paused_at": None,
                        "updated_at": datetime.now(UTC),
                    }
                )
                result = await self._store_refusing_room(room)
                await self._emit_framework_event(
                    "room_archived", room_id=room_id, data={"room_id": room_id}
                )
                return result
        finally:
            await self._room_timers.settle(room_id, hooks)

    async def set_room_timers(
        self,
        room_id: str,
        timers: RoomTimers,
        *,
        organization_id: str | None = None,
    ) -> Room:
        """Set or replace the lifecycle timers for an existing room.

        When ``timers.last_activity_at`` is unset, the room's existing activity
        timestamp is preserved, so adjusting thresholds never resets the idle
        clock; a room that has had no activity keeps none, and its
        inactivity counts from its creation (RFC §5.1). The thresholds apply
        from the next locked section on, or the next
        :meth:`check_room_timers` / :meth:`check_all_timers`.
        """
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
            last_activity = timers.last_activity_at or room.timers.last_activity_at
            timers = timers.model_copy(update={"last_activity_at": last_activity})
            room = room.model_copy(update={"timers": timers, "updated_at": datetime.now(UTC)})
            return await self._store.update_room(room)

    async def set_agent_response_policy(
        self,
        room_id: str,
        policy: AgentResponsePolicy,
        *,
        organization_id: str | None = None,
    ) -> Room:
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
            room = await self.get_room(room_id, organization_id=organization_id)
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

    async def check_room_timers(self, room_id: str, *, organization_id: str | None = None) -> Room:
        """Apply the room's due timer transitions (RFC §5.1), for a room
        nobody writes to.

        An ACTIVE room pauses once ``inactive_after_seconds`` passed since its
        last activity, or since its creation before any; a PAUSED room
        closes once ``closed_after_seconds`` passed since ``paused_at``. Each
        transition takes the instant it fell due, however late the check:
        one long after both pauses the room at the first instant and closes
        it at the second, in one write. Each emits ``room_paused`` or
        ``room_closed`` (``reason = timer``), then fires ``ON_ROOM_PAUSED``
        or ``ON_ROOM_CLOSED`` once the room lock is released: their hooks
        have started when this returns, which does not wait for them (RFC
        §9.3). Every locked operation on the room applies them as well.

        Returns the room as stored once its due transitions are applied.
        *organization_id* scopes the check to one tenant (RFC §17.2).
        """
        return await self._room_timers.check(room_id, organization_id=organization_id)

    async def check_all_timers(self, *, organization_id: str | None = None) -> list[Room]:
        """Check the timers of every ACTIVE and PAUSED room
        (:meth:`check_room_timers`) and return the rooms that transitioned.

        *organization_id* limits the check to one tenant's rooms; with none,
        every room is checked.
        """
        return await self._room_timers.check_all(organization_id=organization_id)

    async def update_room_metadata(
        self,
        room_id: str,
        metadata: dict[str, Any],
        *,
        organization_id: str | None = None,
    ) -> Room:
        """Update room metadata.

        A key only the framework writes is dropped with a warning, the rest
        written: one under ``roomkit:`` (RFC §5.1), a detach's revocation
        (``roomkit:detached_channels``) and what the room keeps for a channel
        (``roomkit:channel_state:``) among them, or a key of the room's
        private state (``author_register``, ``_speak_queue_results``, RFC
        §14.1). So a read-modify-write (``{**room.metadata, ...}``) writes
        its own keys and leaves the framework's as they are.
        """
        metadata = without_framework_keys(metadata, room_id=room_id, writer="update_room_metadata")
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
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
        *,
        organization_id: str | None = None,
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
            if organization_id is not None:
                await self.get_room(room_id, organization_id=organization_id)
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
        *,
        organization_id: str | None = None,
    ) -> Participant:
        """Resolve a pending participant to a known identity (RFC 7.4).

        Called by an advisor or automated process when a pending/ambiguous
        participant has been identified.
        """
        async with self._lock_manager.locked(room_id):
            if organization_id is not None:
                await self.get_room(room_id, organization_id=organization_id)
            participant = await self._store.get_participant(room_id, participant_id)
            if participant is None:
                raise ParticipantNotFoundError(
                    f"Participant {participant_id} not found in room {room_id}"
                )

            identity = await self._store.get_identity(identity_id)
            if identity is None or (
                organization_id is not None and identity.organization_id != organization_id
            ):
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

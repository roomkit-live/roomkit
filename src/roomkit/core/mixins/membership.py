"""MembershipMixin — explicit room membership (join/leave) and roster reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import ParticipantNotFoundError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    EventType,
    HookTrigger,
    IdentificationStatus,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.participant import Participant

if TYPE_CHECKING:
    from roomkit.core.locks import RoomLockManager
    from roomkit.store.base import ConversationStore


@runtime_checkable
class MembershipHost(Protocol):
    """Contract: capabilities a host class must provide for MembershipMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _lock_manager: Per-room lock for serialised mutation.

    Cross-mixin methods (provided by other mixins in the MRO):
        get_room: From :class:`RoomLifecycleMixin`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager


class MembershipMixin(HelpersMixin):
    """Explicit room membership: join, leave, and roster reads.

    Host contract: :class:`MembershipHost`.
    """

    _store: ConversationStore
    _lock_manager: RoomLockManager

    # Cross-mixin method — attribute annotation avoids MRO shadowing
    get_room: Any  # see MembershipHost / RoomLifecycleMixin

    async def add_member(
        self,
        room_id: str,
        channel_id: str,
        participant_id: str,
        *,
        identity_id: str | None = None,
        display_name: str | None = None,
        role: ParticipantRole = ParticipantRole.MEMBER,
    ) -> Participant:
        """Explicitly add a member to a room (an intentional join).

        Unlike :meth:`ensure_participant` — which lazily materialises a sender
        the first time they speak — this is a deliberate join. It is safe to
        call on every room open: joining someone who is already an ``ACTIVE``
        member is a no-op (no write, no event). A genuine join — a brand-new
        member, or re-adding someone who previously left — upserts the
        participant as ``ACTIVE``, emits ``PARTICIPANT_JOINED`` and fires the
        ``ON_PARTICIPANT_JOINED`` hook. A re-join preserves ``joined_at``.

        When ``identity_id`` is given the participant is marked ``IDENTIFIED``
        (the caller already knows who they are).

        A join through a channel other than the record's primary one moves the
        primary to the channel being joined through — that is what "primary
        channel used to join" means (RFC §5.5) — without forking a second
        record: the channel it replaces stays in ``connected_via``, and the move
        is logged naming both, since a caller reading ``channel_id`` afterwards
        sees only the new one.
        """
        async with self._lock_manager.locked(room_id):
            await self.get_room(room_id)
            existing = await self._store.get_participant(room_id, participant_id)
            if existing is not None and existing.status == ParticipantStatus.ACTIVE:
                # Already a member — idempotent no-op on the hot path.
                return existing

            identification = (
                IdentificationStatus.IDENTIFIED if identity_id else IdentificationStatus.PENDING
            )
            if existing is not None:
                channels = self._record_channel_use(existing, channel_id, rehomes=True)
                participant = existing.model_copy(
                    update={
                        "status": ParticipantStatus.ACTIVE,
                        "channel_id": channel_id,
                        "connected_via": channels or existing.connected_via,
                        "role": role,
                        "identity_id": identity_id or existing.identity_id,
                        "identification": identification,
                        "display_name": display_name or existing.display_name,
                    }
                )
                await self._store.update_participant(participant)
            else:
                participant = Participant(
                    id=participant_id,
                    room_id=room_id,
                    channel_id=channel_id,
                    connected_via=[channel_id],
                    display_name=display_name,
                    role=role,
                    status=ParticipantStatus.ACTIVE,
                    identification=identification,
                    identity_id=identity_id,
                )
                await self._store.add_participant(participant)

            await self._emit_system_event(
                room_id,
                EventType.PARTICIPANT_JOINED,
                code="participant_joined",
                message=f"Participant {participant_id} joined",
                data={"participant_id": participant_id, "identity_id": identity_id},
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_PARTICIPANT_JOINED,
                EventType.PARTICIPANT_JOINED,
                code="participant_joined",
                message="Participant joined",
                data={"participant_id": participant_id, "identity_id": identity_id},
            )
            return participant

    async def remove_member(
        self,
        room_id: str,
        participant_id: str,
        *,
        status: ParticipantStatus = ParticipantStatus.LEFT,
    ) -> Participant:
        """Remove a member from a room (leave) via a soft status flip.

        Sets the participant's ``status`` (default ``LEFT``; pass ``BANNED`` to
        ban) rather than deleting the row, so membership history and read
        markers survive. Emits ``PARTICIPANT_LEFT`` and fires the
        ``ON_PARTICIPANT_LEFT`` hook. Raises :class:`ParticipantNotFoundError`
        when the participant is unknown.
        """
        async with self._lock_manager.locked(room_id):
            participant = await self._store.get_participant(room_id, participant_id)
            if participant is None:
                raise ParticipantNotFoundError(
                    f"Participant {participant_id} not found in room {room_id}"
                )
            participant = participant.model_copy(update={"status": status})
            await self._store.update_participant(participant)

            await self._emit_system_event(
                room_id,
                EventType.PARTICIPANT_LEFT,
                code="participant_left",
                message=f"Participant {participant_id} left",
                data={"participant_id": participant_id, "status": status.value},
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_PARTICIPANT_LEFT,
                EventType.PARTICIPANT_LEFT,
                code="participant_left",
                message="Participant left",
                data={"participant_id": participant_id, "status": status.value},
            )
            return participant

    async def rename_member(
        self,
        room_id: str,
        participant_id: str,
        display_name: str,
    ) -> Participant:
        """Change what a member is called — never who they are (RFC §5.5).

        ``add_member()`` on an ACTIVE member is deliberately a no-op, so a
        display name set at join stays put; this is the verb that changes it
        in place. Presentation only: ``id`` and ``identity_id`` are what
        attribution, correlation and moderation stand on, and no rename
        touches them. Emits ``PARTICIPANT_UPDATED`` and fires
        ``ON_PARTICIPANT_UPDATED`` so an interface reflecting the room learns
        of the change the same way it learned of the join. A rename to the
        name already held is a no-op — no write, no event. Raises
        :class:`ParticipantNotFoundError` when the participant is unknown.
        """
        async with self._lock_manager.locked(room_id):
            participant = await self._store.get_participant(room_id, participant_id)
            if participant is None:
                raise ParticipantNotFoundError(
                    f"Participant {participant_id} not found in room {room_id}"
                )
            if participant.display_name == display_name:
                return participant
            participant = participant.model_copy(update={"display_name": display_name})
            await self._store.update_participant(participant)

            data = {"participant_id": participant_id, "display_name": display_name}
            await self._emit_system_event(
                room_id,
                EventType.PARTICIPANT_UPDATED,
                code="participant_updated",
                message=f"Participant {participant_id} is now displayed as {display_name!r}",
                data=data,
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_PARTICIPANT_UPDATED,
                EventType.PARTICIPANT_UPDATED,
                code="participant_updated",
                message="Participant renamed",
                data=data,
            )
            return participant

    async def list_members(
        self,
        room_id: str,
        *,
        include_left: bool = False,
    ) -> list[Participant]:
        """List a room's members.

        Returns only ``ACTIVE`` participants (the current roster) by default.
        Pass ``include_left=True`` to also include those who left or were
        removed/banned.
        """
        participants = await self._store.list_participants(room_id)
        if include_left:
            return participants
        return [p for p in participants if p.status == ParticipantStatus.ACTIVE]

    async def is_member(self, room_id: str, identity_id: str) -> bool:
        """Return ``True`` if ``identity_id`` is an active member of the room."""
        for participant in await self._store.list_participants(room_id):
            if (
                participant.identity_id == identity_id
                and participant.status == ParticipantStatus.ACTIVE
            ):
                return True
        return False

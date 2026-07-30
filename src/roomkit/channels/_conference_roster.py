"""The room roster behind a conference's participants.

An SFU reports arrivals and departures on its own identities; the framework
keeps a Room roster that everything else reads. Reconciling the two is
bookkeeping against the store: no channel, no backend and no bot session take
part, which is what makes it exercisable on its own.

See RFC section 12.10.4.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_identity import participant_update
from roomkit.channels._conference_metadata import CONFERENCE_METADATA_KEY, provider_record
from roomkit.models.enums import IdentificationStatus, ParticipantStatus
from roomkit.models.participant import Participant

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager, AbstractContextManager

    from roomkit.conference.models import ConferenceParticipant
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.identity import IdentityResult
    from roomkit.store.base import ConversationStore

PRESENT_STATUSES = frozenset({ParticipantStatus.ACTIVE, ParticipantStatus.INACTIVE})
"""Statuses a participant leaves *from*.

Leaving is a transition out of being present, so only what means present makes
it. ``LEFT`` is already there; ``BANNED`` is a decision the room took, and the
SFU reporting a disconnection — which a banned participant produces on their way
out, by definition — is not the room taking it back.
"""


class ConferenceRoster:
    """Keeps a room's participants in step with the conference.

    Inert until a store arrives: a channel is constructed before it is
    registered with a framework, and there is nothing to record in between.
    """

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._store: ConversationStore | None = None
        self._locks: RoomLockManager | None = None
        self._lease: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext

    def set_store(
        self,
        store: ConversationStore,
        locks: RoomLockManager | None = None,
        lease: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        """Wire the store this writes to, and the lock its writers hold.

        The lock is the framework's per-room one, which ``add_member()`` and
        ``remove_member()`` take while they write. A roster is written from two
        directions — the integrator's decisions and the SFU's observations —
        and read-modify-write from one of them while the other commits is how a
        ban gets overwritten by what was read before it.

        ``lease`` is the framework's hold on both resources. Every operation
        this class starts on them runs under it — the reads included, and the
        lock from the moment its acquisition begins — because the framework's
        ``close()`` promise is about operations in flight, not only about
        writes: a read suspended in the store when the store is released
        resumes inside a resource that no longer exists (RFC 12.10.4). Nothing
        else runs under it — everything between taking a lease and releasing
        it is this class's own code and the resource calls themselves.
        """
        self._store = store
        self._locks = locks
        self._lease = lease if lease is not None else contextlib.nullcontext

    def _locked(self, room_id: str) -> AbstractAsyncContextManager[None]:
        """The queue this room's roster writes go through, if there is one."""
        if self._locks is None:
            return contextlib.nullcontext()
        return self._locks.locked(room_id)

    async def knows(self, room_id: str, participant_id: str) -> bool:
        """Whether the room has a participant under this identity, in any status.

        Asked about the *record*, not about admission: what it settles is
        whether the framework has met this participant before, which is what
        says an arrival was named by the framework rather than by the SFU (RFC
        12.10.2, rule 3). A banned participant is one the room has met, and
        re-identifying them from an SFU's attributes would be as wrong as doing
        it to anyone else. Admission is :meth:`standing`.
        """
        if self._store is None:
            return False
        with self._lease():
            return await self._store.get_participant(room_id, participant_id) is not None

    async def standing(self, room_id: str, participant_id: str) -> ParticipantStatus | None:
        """The status this room's roster holds for a participant, or ``None``.

        What admission is checked against, and it is a status rather than a
        yes/no because the two refusals are different answers: a room that has
        never heard of the participant is a caller who has yet to admit them,
        and a room that has is a caller being told no.

        ``None`` when there is no store either. A channel with nothing to read
        cannot say anyone belongs to the room, and a credential is not the thing
        to fail open on.
        """
        if self._store is None:
            return None
        with self._lease():
            participant = await self._store.get_participant(room_id, participant_id)
        return None if participant is None else participant.status

    async def record(
        self,
        room_id: str,
        participant: ConferenceParticipant,
        identity: IdentityResult | None = None,
    ) -> None:
        """Create or update the Room participant behind a conference identity.

        An identity the framework minted is already the participant id. One it
        did not — a dial-in, or an admission arranged out of band — keeps the
        backend's identity as ``external_id`` and would stay unidentified, so
        whoever resolved the address its provider attached hands the answer in
        as ``identity``.

        The record keeps the backend's identity as its ``id`` whether or not it
        was identified: an Identity is *linked* to it through ``identity_id``,
        never substituted for it. Re-keying would leave the transcript
        attributed to one identifier while the recording, the interruption
        allowlist and this roster still used another — the correlation RFC
        12.10.2 exists to hold.

        An identity applies to the record being created, and only then. A
        participant the room already has is one the framework named or has
        already met, and neither is re-identified from an SFU's attributes on
        the way back in — which is why the caller hands in ``None`` for it.

        What the provider attached goes under one key of the record's metadata,
        with its provenance kept and its size bounded
        (:mod:`roomkit.channels._conference_metadata`). A re-join refreshes that
        key and touches nothing else on the record.

        An arrival does not lift a ban either. Only a participant who *left*
        comes back as ``ACTIVE``: ``BANNED`` is a decision the room made about
        them, and the SFU reporting them connected is not the room changing its
        mind. The record still takes the attributes their provider attached —
        what is refused is admission, not the fact that this is who arrived.

        Written under the room lock, like every roster write, so a ban landing
        between the read and the write is not overwritten by what was read
        before it.
        """
        if self._store is None:
            return
        with self._lease():
            async with self._locked(room_id):
                existing = await self._store.get_participant(room_id, participant.participant_id)
                if existing is not None:
                    await self._refresh(existing, participant)
                    return
                attributes = _attributes(participant)
                record = Participant(
                    id=participant.participant_id,
                    room_id=room_id,
                    channel_id=self._channel_id,
                    external_id=participant.participant_id,
                    identification=IdentificationStatus.UNKNOWN,
                    metadata={} if attributes is None else {CONFERENCE_METADATA_KEY: attributes},
                )
                resolved = participant_update(identity)
                if resolved:
                    record = record.model_copy(update=resolved)
                await self._store.add_participant(record)

    async def _refresh(self, existing: Participant, participant: ConferenceParticipant) -> None:
        """Bring a participant the room already has up to date with an arrival."""
        if self._store is None:  # pragma: no cover — guarded by the caller
            return
        update: dict[str, Any] = {}
        if existing.status is ParticipantStatus.LEFT:
            update["status"] = ParticipantStatus.ACTIVE
        attributes = _attributes(participant, existing.metadata.get(CONFERENCE_METADATA_KEY))
        if attributes is not None:
            # One key replaced, never a flat merge: what the integrator put
            # on this participant is theirs, and a conference is where
            # strangers get to propose keys (RFC 12.10.2).
            update["metadata"] = {**existing.metadata, CONFERENCE_METADATA_KEY: attributes}
        if update:
            await self._store.update_participant(existing.model_copy(update=update))

    async def mark_left(self, room_id: str, participant_id: str) -> None:
        """Record the departure on the room roster.

        Firing a hook is not enough: a participant left behind as ACTIVE makes
        the roster lie to everything that reads it.

        A departure is not an amnesty. ``BANNED`` says the room removed and
        blocked someone (RFC section 5.5), and writing ``LEFT`` over it turns
        the SFU's own "they disconnected" — which a banned participant produces
        by definition, on their way out — into the thing that lifts the ban.
        Leaving is a transition out of being present, so only a status that
        means present transitions.
        """
        await self._transition(room_id, participant_id, ParticipantStatus.LEFT)

    async def _transition(
        self, room_id: str, participant_id: str, status: ParticipantStatus
    ) -> None:
        """Move a participant to a status the conference observed, if it may.

        Read-modify-write under the room lock, like every other write to a
        roster: ``remove_member()`` holds it while it bans, and a departure that
        read the record before that ban and wrote after it would put the ban
        back to ``LEFT`` — the same bypass by a slower route.
        """
        if self._store is None:
            return
        with self._lease():
            async with self._locked(room_id):
                participant = await self._store.get_participant(room_id, participant_id)
                if participant is None or participant.status not in PRESENT_STATUSES:
                    return
                await self._store.update_participant(
                    participant.model_copy(update={"status": status})
                )


def _attributes(
    participant: ConferenceParticipant, previous: object = None
) -> dict[str, dict[str, Any]] | None:
    """What the provider attached, or ``None`` when it attached nothing.

    A participant the framework named carries no provider attributes at all, and
    most do: writing two empty bags onto every one of them would say the SFU
    reported something when it reported nothing, on every record in the room.
    """
    attributes = provider_record(participant, previous)
    return attributes if any(attributes.values()) else None

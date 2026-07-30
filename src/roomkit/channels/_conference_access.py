"""Who a conference channel admits, and what it refuses to admit them to.

What is minted here is admission to a live media session, valid for as long as
it says it is, against a conference the framework may since have left. That is
what makes this the one piece of in-flight work a teardown takes back rather
than leaves behind: everything else the drain protects degrades gracefully past
its deadline — a chunk lands late, an event arrives out of order — and a
credential does not.

Refusing is not revocation and does not claim to be. Cancelling a request only
reaches backends that let cancellation through, so both endings say the
credential was not returned and the backend may have issued one anyway.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC sections 12.10.2 and 12.10.4.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from roomkit.conference.models import ConferenceAccess, ConferenceGrants
from roomkit.core.exceptions import (
    ParticipantNotAdmittedError,
    ParticipantNotFoundError,
    RoomNotAttachedError,
)
from roomkit.models.enums import ParticipantStatus

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.channels._conference_roster import ConferenceRoster
    from roomkit.conference.base import ConferenceBackend
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.channels.conference")

ADMISSIBLE_STATUSES = frozenset(
    {ParticipantStatus.ACTIVE, ParticipantStatus.INACTIVE, ParticipantStatus.LEFT}
)
"""Roster statuses a conference credential may be minted for.

An allowlist rather than a denylist, because the two fail in opposite
directions: a status added to the enum later is refused admission until someone
decides it should have it, where a denylist would admit it silently and nothing
would say so.

``LEFT`` is on it. Departure is a status rather than a removal here, and being
readmitted to a conference is ordinary — a participant whose connection dropped
asks for a second credential, not for a second identity. ``BANNED`` is not:
RFC section 5.5 defines it as "removed and blocked", and a credential is exactly
the thing a block has to reach.
"""


class ConferenceAccessMixin:
    """Minting conference credentials, and taking them back.

    Host contract — what ConferenceChannel provides:
        channel_id, _backend, _activity, _roster: the channel and what it
            talks to.
        _bot_identity, _default_grants: what the admission checks read.
        _abandoned_mints: requests a teardown took back, kept on the channel
            because `_mint` reads it after the room's record has stopped
            listing the request.
        _framework: wired for completeness; the lock itself is reached through
            `_locked_room`.
        _locked_room: the framework's per-room lock, which every membership
            change is written under and the admission decision is ordered
            against.
        _resource_lease: the framework's hold on the store and the lock
            manager, taken around every operation this mixin starts on them
            (RFC 12.10.4).
        _room / _attached_room: the per-room record (ConferenceRoomState).
    """

    channel_id: str
    _backend: ConferenceBackend
    _activity: RoomActivity
    _roster: ConferenceRoster
    _bot_identity: str
    _default_grants: ConferenceGrants
    _abandoned_mints: set[asyncio.Task[ConferenceAccess]]
    _framework: RoomKit | None
    _locked_room: Any
    _resource_lease: Any

    # Provided by ConferenceChannel — see the host contract above
    _room: Any
    _attached_room: Any

    async def mint_access(
        self, room_id: str, participant_id: str, *, grants: ConferenceGrants | None = None
    ) -> ConferenceAccess:
        """Mint credentials for a participant to join the conference.

        The integrator delivers the result to its client application; the
        framework does not serve it.

        What is minted here is admission to a live media session, so the
        preconditions are checked rather than assumed. A credential is only
        ever as narrow as the identity it names, and this is the last place the
        framework can still tell that identity apart from a typo — or from
        someone the room has removed and blocked.

        And the only one of those checks that cannot be made good afterwards.
        Consulting the roster is an await, and a detach landing in it left the
        room check describing a conference the channel had already left — with
        a token handed out that the SFU will honour and no revocation in the
        backend contract to take it back. So the whole of it is work the
        teardown drains, with the room read again on the far side of the roster:
        either the credential is issued into a conference the channel is still
        part of and the detach waits for it, or the detach got there first and
        nothing is minted.

        The drain alone is not enough for this one. It is bounded — it has to
        be, since it waits on code the channel does not own — and everything
        else it protects degrades gracefully past the deadline: a chunk arrives
        late, an event lands out of order. A credential does not degrade. It is
        valid for as long as it says it is, against a conference the framework
        has left. So this is the one piece of in-flight work a teardown takes
        back rather than leaves behind, and the answer is refused rather than
        returned if it arrives anyway. See :meth:`_mint`.

        The same is true of the participant, and it is why the decision is not
        made out of two reads. Being admissible is a fact about the roster and
        the roster is written by ``remove_member()``, so a check that merely
        reads it twice is asking a question whose answer can be out of date
        before it arrives: a store answers about the moment the query reached
        it, and a ban committed a moment later comes back as ``ACTIVE``. No
        arrangement of independent reads fixes that — only being in the same
        queue as the writer does.

        So the decision is taken under the framework's per-room lock, which is
        the lock ``remove_member()`` and ``add_member()`` hold while they write.
        A ban either completes before this acquires it, and is seen; or waits
        for it, and lands after a credential that was legitimate when it was
        issued. Nothing in between.

        The lock is taken *outside* the drain, on purpose. Inside it, this call
        would hold work a teardown waits for while waiting for a lock that
        teardown holds — the two would clear each other only when the drain
        timed out, which is a five-second stall on every detach that races a
        mint. Outside, the ordering still holds: a detach that got the lock
        first has set ``attached`` to false by the time this reads it.
        """
        async with self._activity.track(room_id):
            await self._check_admissible(room_id, participant_id)
            self._require_attached(room_id)
            generation = self._room(room_id).generation
            access = await self._mint(
                room_id, participant_id, grants or self._default_grants, generation
            )
        return await self._hand_over(room_id, participant_id, access, generation)

    async def _hand_over(
        self, room_id: str, participant_id: str, access: ConferenceAccess, generation: int
    ) -> ConferenceAccess:
        """Decide, in the writers' own queue, whether the credential goes out.

        The framework's per-room lock is what ``remove_member()`` and
        ``add_member()`` hold while they change a roster, so taking it here puts
        this decision in the same order as theirs rather than merely after a
        read of what they wrote. It is the only way the two can be one decision:
        a store read answers about the moment it reached the store, and a ban
        committed a moment later comes back as ``ACTIVE``.

        Reentrant, as the lock manager is: a caller minting from inside a hook
        that already holds the room — an ``ON_PARTICIPANT_JOINED`` handler
        admitting the participant it was told about is the obvious one — passes
        straight through rather than waiting for itself.

        The room is read here too, and last: it costs nothing to read, and under
        this lock a detach that got here first has already finished writing what
        this is about to read.

        Under the framework's resource lease from before the acquisition
        begins to after the lock is let go — an acquisition is already an
        operation the lock manager is running, and everything inside the block
        is this mixin's own code and the resource calls themselves, so the
        lease never covers integrator code (RFC 12.10.4).
        """
        with self._resource_lease():
            async with self._locked_room(room_id):
                standing = await self._roster.standing(room_id, participant_id)
                # No await from here to the return, so the two conditions are one
                # answer about one moment — and no writer can run between them,
                # because every writer of either is waiting on this lock.
                room = self._room(room_id)
                if room.generation != generation or not room.attached:
                    raise self._refuse_mint(room_id, participant_id)
                if standing not in ADMISSIBLE_STATUSES:
                    raise self._refuse_barred_mint(room_id, participant_id)
                return access

    async def _mint(
        self, room_id: str, participant_id: str, grants: ConferenceGrants, generation: int
    ) -> ConferenceAccess:
        """Ask the backend for the credential, on a request a detach can take back.

        Two things stand between a wedged backend and a token for a conference
        the channel has left. The request runs on a task the channel owns, so a
        teardown that has run out of patience cancels it rather than racing it —
        the one cancellation this channel can offer, and enough to stop most
        backends completing the call at all. And the room is read once more
        when the answer arrives, so a backend that shielded its request still
        does not get its credential handed on.

        Refusing after the backend has answered is not revocation, and this does
        not pretend otherwise: the credential may exist on the SFU's books. What
        is guaranteed is that nobody receives it, which is the part the framework
        controls. It is said out loud, because a token nobody was given is still
        something an operator may want to know exists.

        Which is why cancelling is not reported as having prevented anything.
        The request coming back cancelled says only that *this* call gave up on
        it — a backend that shielded its network operation goes on to mint, and
        a caller told the credential was never issued would be told something
        the framework has no way to know. Both endings say the same thing, and
        both warn.

        The room is read when the answer arrives, which is this method's own
        refusal: it is synchronous, it is inside the drain, and it stops a
        credential for a conference the channel has left from travelling any
        further. Whether it is *handed over* is decided in
        :meth:`_hand_over`, in the same queue the roster's writers are in.
        """
        room = self._room(room_id)
        request = asyncio.ensure_future(self._backend.mint_access(room_id, participant_id, grants))
        room.mints.add(request)
        try:
            access = await request
        except asyncio.CancelledError:
            if request not in self._abandoned_mints:
                raise
            raise self._refuse_mint(room_id, participant_id) from None
        finally:
            # A caller cancelled while waiting here leaves the request with
            # nobody to receive it — and it is still a credential being minted.
            if not request.done():
                request.cancel()
            self._abandoned_mints.discard(request)
            room.mints.discard(request)
        if room.generation != generation or not room.attached:
            raise self._refuse_mint(room_id, participant_id)
        return access

    def _refuse_mint(self, room_id: str, participant_id: str) -> RoomNotAttachedError:
        """Withhold a credential for a conference the channel has left."""
        return RoomNotAttachedError(
            self._withhold(
                room_id,
                participant_id,
                f"Channel {self.channel_id!r} left room {room_id!r}",
            )
        )

    def _refuse_barred_mint(
        self, room_id: str, participant_id: str
    ) -> ParticipantNotAdmittedError:
        """Withhold a credential the room stopped admitting while it was minted.

        A different answer from a room the channel has left, and a different
        exception, exactly as the two are before the mint starts: this one says
        the conference is fine and the participant is not.
        """
        return ParticipantNotAdmittedError(
            self._withhold(
                room_id,
                participant_id,
                f"Room {room_id!r} stopped admitting {participant_id!r}",
            )
        )

    def _withhold(self, room_id: str, participant_id: str, because: str) -> str:
        """Say a credential is being kept back, and that one may exist regardless.

        The framework can guarantee that nobody receives it. It cannot guarantee
        that the backend did not make one — cancelling a request only reaches
        backends that let cancellation through, and a shielded network call
        finishes and mints whatever this call does with the answer. So the
        refusal claims only what is true, and the warning names the room and
        the participant, because a credential nobody was handed is still a
        credential an operator may need to revoke at the SFU.
        """
        logger.warning(
            "%s while the backend was minting conference access for %s. The credential is "
            "not being returned, but the backend may have issued one: check whether it "
            "needs revoking there",
            because,
            participant_id,
        )
        return (
            f"{because} while minting access for {participant_id!r}; the credential is not "
            "being returned, and the backend may have issued one that needs revoking there"
        )

    def _abandon_mints(self, room_id: str) -> None:
        """Take back every credential still being minted for a room that has gone.

        Called once the drain has given whatever is in flight its chance to
        finish. Anything left is a request whose answer would be admission to a
        conference the channel is leaving.
        """
        room = self._room(room_id)
        requests, room.mints = room.mints, set()
        for request in requests:
            if request.done():
                continue
            self._abandoned_mints.add(request)
            request.cancel()

    def _require_attached(self, room_id: str) -> None:
        """Refuse to admit anyone to a conference this channel is not in."""
        if self._attached_room(room_id) is None:
            raise RoomNotAttachedError(
                f"Channel {self.channel_id!r} is not attached to room {room_id!r}, "
                "so it cannot mint access to its conference"
            )

    async def _check_admissible(self, room_id: str, participant_id: str) -> None:
        """Refuse to mint for a room, or an identity, that has no claim to one.

        Four refusals, in this order:

        The room must be one this channel is attached to. Minting for a room it
        left admits someone to a conference the framework is no longer part of,
        and it is a credential that outlives the check. Read first so that a
        room already gone is named as the problem rather than the roster it no
        longer has, and read again by the caller once the roster has answered.

        The bot's identity is reserved. The channel has to recognise its own bot
        by identity (see ``_is_own_bot``), so a participant sharing it cannot be
        told apart from the bot: they would be excluded from the roster, the
        hooks and the transcript. Refusing at the door is the only place that
        collision can still be prevented rather than reported.

        The participant must be one the room knows. RFC 12.10.2 has the
        framework pass a Room ``Participant.id``, and every attribution
        guarantee downstream is written in those terms; a mistyped identifier
        otherwise mints a perfectly valid token for someone with no place in the
        room. Creating the participant instead would only record the mistake and
        hand out the credential anyway — an integrator admitting someone new
        says so with ``ensure_participant()``.

        And their standing on the roster must permit it. Being on the roster is
        not the same as being welcome on it: ``remove_member(..., BANNED)``
        leaves the record in place, so a check that asked only whether the room
        had heard of them would go on minting for someone the room removed and
        blocked. Read here, before the backend is asked for anything, because
        that is the last point at which a credential can still be not-issued —
        the ConferenceBackend contract offers no revocation, and an SFU honours
        what it minted.
        """
        self._require_attached(room_id)
        # Before the roster is consulted: the bot has no participant record, so
        # the check below would refuse it as unknown and name the wrong problem.
        if participant_id == self._bot_identity:
            raise ValueError(
                f"Participant id {participant_id!r} is reserved: it is channel "
                f"{self.channel_id!r}'s bot identity. A participant sharing it cannot be "
                "told apart from the bot, and would be excluded from the roster, the "
                "conference hooks and transcription. Mint under another id, or give the "
                "bot an identity no participant can hold (bot_identity=)."
            )
        standing = await self._roster.standing(room_id, participant_id)
        if standing is None:
            raise ParticipantNotFoundError(
                f"Participant {participant_id!r} is not in room {room_id!r}. Conference "
                "access is minted for a participant of the room (RFC 12.10.2): admit them "
                "first with ensure_participant(), so that transcriptions and hooks have a "
                "participant to attribute to."
            )
        if standing not in ADMISSIBLE_STATUSES:
            raise ParticipantNotAdmittedError(
                f"Participant {participant_id!r} is {standing.value} in room {room_id!r}, "
                "so no conference credential is being minted for them. Readmit them with "
                "add_member() if that is meant to be undone — ensure_participant() returns "
                "the record as it stands and would leave the block in place."
            )

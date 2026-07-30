"""The bot session a conference channel holds, and what it costs to open one.

The bot joins on first need — a room can exist long before anyone confers — and
the join is serialised per room, because participants arriving together is the
normal way a meeting starts and two concurrent joins would publish the AI on two
tracks.

The announcements are the delicate part. The join lock is released before the
conference is announced, so integrator code cannot hold every other room's joins
behind it, which leaves the announcement itself exposed to a detach landing
mid-way. Each one is registered as room activity with its own check inside:
either it is registered before the detach and the detach waits for it, or the
detach got there first and nothing is announced.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC sections 12.10.4 and 17.7.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from typing import TYPE_CHECKING, Any

# Read through the module rather than bound into this one: the deadline a
# deployment or a test sets on it is the one that has to apply, and a name
# imported here would be a second copy to remember — see `_budget`.
from roomkit.channels import _conference_activity
from roomkit.conference.models import BotSession, ConferenceGrants
from roomkit.core.exceptions import RoomNotAttachedError
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.session_event import SessionStartedEvent

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.conference.base import ConferenceBackend
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.channels.conference")


def _settle_clock(bot: BotSession, backend: str) -> None:
    """Make a session's join time comparable, whatever the backend set.

    ``duration_ms`` subtracts ``joined_at`` from an aware *now*, and Python
    refuses to subtract a naive datetime from an aware one. That TypeError would
    be raised inside the teardown, where the hook engine logs it and the detach
    still reports success — so a backend reaching for ``datetime.now()`` instead
    of ``datetime.now(UTC)`` would cost the conference its end announcement
    entirely, and leave nothing but a stack trace to say why.

    Settled here, on the way in, rather than defended against at the
    subtraction: this is the one place that knows which backend to name. UTC is
    the assumption because it is what the field's own default uses; a backend
    that means another zone says so with a tzinfo.

    Set in place because the backend keeps its own reference to this session and
    compares against it — a copy would be a different session to everything on
    the far side of the boundary.
    """
    if bot.joined_at.tzinfo is not None:
        return
    logger.warning(
        "Conference backend %r returned bot session %s with a naive joined_at; reading it "
        "as UTC. Set a timezone-aware value to decide how conference_ended's duration_ms "
        "is measured",
        backend,
        bot.id,
    )
    bot.joined_at = bot.joined_at.replace(tzinfo=UTC)


class ConferenceSessionMixin:
    """Opening, announcing and closing the channel's bot session.

    Host contract — what ConferenceChannel provides:
        channel_id, channel_type, _backend, _activity, _framework: the channel
            and what it talks to.
        _bot_identity, _bot_grants: what the bot joins as.
        _backend_closed: set once the backend is gone, so nothing that outlived
            a close's budget calls into it afterwards.
        _rooms / _room: the per-room records (ConferenceRoomState).
        _emit_framework_event: how the conference start is announced.
        _announce_end: how a detach's end is announced, for the session a
            close finally manages to take out — see ConferenceAttachmentMixin.
    """

    channel_id: str
    channel_type: ChannelType
    _backend: ConferenceBackend
    _activity: RoomActivity
    _framework: RoomKit | None
    _bot_identity: str
    _bot_grants: ConferenceGrants
    _backend_closed: bool

    # Provided by ConferenceChannel — see the host contract above
    _rooms: Any
    _room: Any
    _emit_framework_event: Any
    _announce_end: Any

    async def _ensure_bot(self, room_id: str) -> BotSession:
        """Join the conference on first need.

        A room can exist long before anyone confers, so the bot connection is
        opened when there is finally something to listen to. Serialised, because
        participants arriving together is the normal way a meeting starts and
        two concurrent joins would publish the AI on two tracks.
        """
        room = self._room(room_id)
        async with room.lock:
            if (bot := room.bot) is not None:
                return bot
            if not room.attached:
                raise RoomNotAttachedError(
                    f"Channel {self.channel_id!r} is not attached to room {room_id!r}"
                )
            generation = room.generation
            # Marked for the whole of the join, not just the network call: the
            # bot is not recognisable by its session until the record carries
            # it, and a backend that echoes the bot back mid-join arrives inside
            # this window.
            room.joining = True
            try:
                bot = await self._backend.join_as_bot(
                    room_id, self._bot_identity, self._bot_grants
                )
                _settle_clock(bot, self._backend.name)
                if room.generation != generation:
                    # Recorded rather than plain, because the detach that closed
                    # this generation had no bot to put on `leaving` — this one
                    # did not exist yet — so a `leave()` that fails here would
                    # strand a session no part of the room could account for.
                    await self._leave_and_record(room_id, bot)
                    raise RoomNotAttachedError(
                        f"Channel {self.channel_id!r} was detached from room "
                        f"{room_id!r} while joining"
                    )
                room.bot = bot
            finally:
                room.joining = False
        # The lock is released before the conference is announced, because
        # announcing it runs integrator code that may await for as long as it
        # likes and a detach must not queue behind that.
        #
        # Which leaves the announcement itself exposed: a detach can land while
        # a listener of "conference_started" is still running, and then
        # "conference_ended" reaches observers first — a conference that ended
        # before it began, to anyone with no way to tell the two apart. Checking
        # the generation before emitting does not fix that; the emission is
        # itself an await. So the announcement registers as activity the detach
        # drains, with the check inside: either this passes and the detach waits
        # for it, or the detach got there first and this abandons the join.
        async with self._activity.track(room_id):
            if not room.is_current(generation, bot):
                raise RoomNotAttachedError(
                    f"Channel {self.channel_id!r} was detached from room {room_id!r} while joining"
                )
            await self._emit_framework_event("conference_started", room_id, bot_session_id=bot.id)
        # A second activity rather than the same one, so the check is made
        # twice: a listener of the announcement above may itself have detached
        # the channel, and this must see that.
        #
        # Announcing a session that has already ended is worse than not
        # announcing it — the hook hands out a bot session that has left the
        # conference, and code reading `event.session` to greet the room would
        # speak into nothing. Which is why the check cannot stand outside the
        # block: dispatching the hook is an await of its own, and a detach
        # landing in it fires the greeting after the end. Either this is
        # registered before the detach and the detach waits for it, or the
        # detach got there first and no session is announced at all.
        #
        # The cost is a detach waiting behind an integrator's greeting — the
        # head-of-line block the join lock is otherwise shaped to avoid. Bounded
        # by the drain budget, and the price of the guarantee being true rather
        # than usually true.
        async with self._activity.track(room_id):
            if not room.is_current(generation, bot):
                raise RoomNotAttachedError(
                    f"Channel {self.channel_id!r} was detached from room {room_id!r} while joining"
                )
            await self._fire_session_started(room_id, bot)
        return bot

    async def _ensure_bot_for_arrival(self, room_id: str) -> bool:
        """Bring the bot in for an arriving participant, and say whether to record them.

        The roster does not depend on the bot. RFC section 12.10.4 makes
        recording an arrival an unconditional MUST, and makes the join itself a
        SHOULD the channel is free to do lazily — so a participant is in the
        conference whether or not the framework managed to get its own session
        into it, and the room has to say so. A ``join_as_bot`` that the SFU
        refuses is the likeliest failure a conference has: it is the first
        network call of the whole meeting.

        Which is why the failure is swallowed rather than left to reach the
        backend's emission loop, where it would be logged as a callback error
        and the arrival dropped — no Participant, no identity, no
        ``ON_CONFERENCE_PARTICIPANT_JOINED``. The roster is what the disclosure
        obligations of section 17.7 are read from, and a roster missing someone
        who is in the meeting is the one answer it must never give.

        ``RoomNotAttachedError`` is the exception, and returns ``False``: it does
        not mean the join failed, it means this channel is no longer in the
        conference the arrival belongs to. Writing then would leave a record
        behind for an attachment that is gone.

        Nothing is retried here. The bot is brought in on first need from
        several places — the next arrival, a published track, a delivery — and
        each of them finds ``room.bot`` still unset and tries again.
        """
        try:
            await self._ensure_bot(room_id)
        except RoomNotAttachedError:
            return False
        except Exception:
            logger.exception(
                "Conference channel %r could not bring its bot into room %s. The arrival is "
                "being recorded anyway, and the conference runs without the framework's own "
                "media session — no transcription and no AI voice — until a later join succeeds",
                self.channel_id,
                room_id,
            )
        return True

    async def _fire_session_started(self, room_id: str, bot: BotSession) -> None:
        """Announce the bot connection on the framework-wide session contract.

        ON_SESSION_STARTED is consumed by code that reaches straight into the
        payload — auto-greeting reads ``event.session`` — so the shared
        SessionStartedEvent is what has to arrive, not a synthetic room event
        shaped like one.
        """
        if self._framework is None:
            return
        context = await self._framework._build_context(room_id)
        await self._framework.hook_engine.run_async_hooks(
            room_id,
            HookTrigger.ON_SESSION_STARTED,
            SessionStartedEvent(
                room_id=room_id,
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=bot.identity,
                session=bot,
            ),
            context,
            skip_event_filter=True,
        )

    def _is_own_bot(self, room_id: str, participant_id: str) -> bool:
        """Whether an identity is the channel's own bot.

        Some backends report the bot back through the callbacks it registered,
        and they do it while the connection is being established — before there
        is a session to compare against. So the configured identity stands in
        for the session's, which closes the window in which the bot would be
        taken for a human, recorded as a participant, and have its own speech
        transcribed back into the room it came from.

        The session identity is checked too, for backends that normalise the
        identity they were given.

        The cost of that stand-in is that a *human* carrying the configured
        identity is excluded as well, and the two are genuinely
        indistinguishable once the bot is in: they arrive on the same
        ``participant_id``, and treating that as a human is how the AI ends up
        transcribing itself. So the exclusion stands — but it is reported,
        because being dropped from the roster, the hooks and the transcript
        without a word is the failure mode this channel refuses. What the
        framework can still tell apart is *when* it happened: outside the join
        window and with no session to attribute it to, nothing about this
        identity is the bot's.
        """
        room = self._room(room_id)
        if room.bot is not None and participant_id == room.bot.identity:
            return True
        if participant_id != self._bot_identity:
            return False
        if room.bot is None and not room.joining:
            self._report_identity_collision(room_id, participant_id)
        return True

    def _report_identity_collision(self, room_id: str, participant_id: str) -> None:
        """Announce an identity the channel had to exclude but cannot claim.

        Once per room: a participant the channel keeps refusing goes on
        publishing tracks, and the point is to be heard once, not to fill the
        log with the same line.
        """
        room = self._room(room_id)
        if room.collision_reported:
            return
        room.collision_reported = True
        logger.error(
            "Participant %r in room %s carries channel %r's configured bot identity, and no "
            "bot session of ours accounts for it. It is being excluded from the roster, the "
            "conference hooks and transcription, because a participant sharing the bot's "
            "identity cannot be told apart from the bot itself. Give the bot an identity no "
            "participant can hold (bot_identity=), or admit this participant under another id.",
            participant_id,
            room_id,
            self.channel_id,
        )

    async def _leave_and_record(self, room_id: str, bot: BotSession) -> bool:
        """Take the bot out of the conference, and say whether it is out.

        A ``leave()`` the SFU refuses is the failure this exists for: the bot is
        still in the meeting, and until this the framework said the opposite —
        the session was dropped from the room's books and ``info()`` reported
        the conference unattended. A bot listening to a meeting the framework
        calls empty is a disclosure problem (RFC 17.7) before it is an
        operational one.

        So a session that does not come out stays on ``leaving`` with the
        reason, which is what ``bot_present`` and ``leave_failed`` are read
        from, and the failure is logged rather than raised: the teardown that
        called this has a conference room to destroy afterwards, and that
        destruction is the one compensation left — it evicts the bot the
        ``leave()`` could not.

        A backend that is already closed is the other way to not come out.
        Every wait a close makes is bounded, so a join or a teardown can outlive
        one: the budget passes, the backend goes, and the work that was still in
        flight arrives here with nothing to send the call on. Making it anyway
        is not a recovery — the transport it would use is the one that was
        closed — so it is refused and said out loud, and the session is kept on
        the books like any other the channel could not remove.
        """
        room = self._room(room_id)
        if self._backend_closed:
            logger.warning(
                "Conference channel %r cannot take bot session %s out of room %s: its backend "
                "is already closed, so the bot may remain in the conference",
                self.channel_id,
                bot.id,
                room_id,
            )
            room.record_leave_failure(bot, "the channel's backend was already closed")
            return False
        try:
            await self._backend.leave(bot)
        except asyncio.CancelledError:
            # The closing budget expired while the call was in flight. The
            # cancellation must travel on — it is how the budget stops the
            # step — but the session's standing cannot travel with it: nothing
            # confirmed the bot is out, so it goes on the books first, where
            # info() and the close's final raise will both find it.
            room.record_leave_failure(bot, "the closing budget expired while leave() ran")
            raise
        except Exception as exc:
            room.record_leave_failure(bot, f"{type(exc).__name__}: {exc}")
            logger.exception(
                "Conference channel %r could not take bot session %s out of room %s. The bot "
                "may still be in the conference, so it stays reported as present — see "
                "info()['rooms'][%r]['leave_failed'] — and no conference_ended is announced "
                "for it",
                self.channel_id,
                bot.id,
                room_id,
                room_id,
            )
            return False
        room.forget_leaving(bot)
        return True

    async def _leave_all(self, room_id: str) -> None:
        """Take every session a room still has in the conference out of it.

        The sessions a previous ``leave()`` refused to remove first, then the
        bot in the room now. That retry is the only one there is: a detach has
        no later moment of its own, and a close is the last thing the channel
        will ever do, so a session still stuck after this is one an operator has
        to go and remove — which is why it stays on the books rather than being
        forgotten.

        A session that does come out here is announced, because its detach owed
        it a ``conference_ended`` and held the announcement back for as long as
        the bot was in the meeting. The bot in ``room.bot`` is not: closing a
        channel is not a detach and has never announced ends for the
        conferences it was still in.
        """
        room = self._room(room_id)
        for entry in room.stuck_sessions():
            if await self._leave_and_record(room_id, entry.bot) and entry.owed_an_end:
                await self._announce_end(room_id, entry.bot)
        if room.bot is not None:
            await self._leave_and_record(room_id, room.bot)

    async def _close_room(self, room_id: str) -> None:
        """Destroy the conference room, unless the backend is already gone."""
        if self._backend_closed:
            logger.warning(
                "Conference channel %r cannot close the conference room %s: its backend is "
                "already closed",
                self.channel_id,
                room_id,
            )
            return
        await self._backend.close_room(room_id)

    async def _settle_joins(self) -> None:
        """Wait for any join still holding a room's lock to have let go.

        Taking the lock and releasing it is the whole operation: it cannot be
        acquired until the join inside it has finished, which by then means the
        join has seen the closed generation and left the session it opened.

        One budget for all of them, not one each. A channel serving twenty
        conferences would otherwise spend twenty times the drain budget on a
        close, and closing is not where a channel is allowed to be slow; the
        rooms are also independent, so there is nothing to gain by asking them
        in turn. Rooms that do not settle inside it are named, because what is
        left behind is a bot in a conference nobody will leave.
        """
        if not self._rooms:
            return
        waiters = {
            asyncio.ensure_future(self._settle_one(room.lock)): room_id
            for room_id, room in list(self._rooms.items())
        }
        _, unfinished = await asyncio.wait(waiters, timeout=_conference_activity.DRAIN_TIMEOUT_S)
        for waiter in unfinished:
            waiter.cancel()
        if unfinished:
            logger.warning(
                "Closing conference channel %r with a join still in flight for room(s) %s "
                "after %.0fs; the bot it is opening may be left in the conference",
                self.channel_id,
                ", ".join(sorted(waiters[waiter] for waiter in unfinished)),
                _conference_activity.DRAIN_TIMEOUT_S,
            )

    @staticmethod
    async def _settle_one(lock: asyncio.Lock) -> None:
        """Hold a room's join lock for as long as it takes to get it."""
        async with lock:
            return

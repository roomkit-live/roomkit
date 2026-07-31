"""The bot session a conference channel holds, and what it costs to open one.

The bot joins on first need — a room can exist long before anyone confers — and
the join is serialised per room, because participants arriving together is the
normal way a meeting starts and two concurrent joins would publish the AI on two
tracks.

First need is anything that says the conference is about to matter, and its
triggers cannot all be backend callbacks: presence is observable only through
a connection, so until the bot holds one, no arrival can report itself and
nothing callback-shaped can start the first join (RFC 12.10.3). Two triggers
reach the channel without the backend's help. The mint — a credential going
out is the framework's own notice that a human is about to connect — covers
the conference nobody has been admitted to yet; the attach's occupancy probe
covers the one already underway, because a channel restarted mid-meeting
re-attaches with no mint left to wait for and asks ``list_participants()``
instead. Deliveries, and presence or track events from a backend able to
observe them, remain triggers after them.

A need is only a need when something configured on the channel can use the
connection — an stt or a recording to consume with, a tts to speak with. A
channel with none of these is pure transport: the join exists for the
intelligence (RFC 12.10.1 principle 4), so its mint, arrival and probe
triggers stand down, and the channel stays what such a deployment asks it to
be — the room's admission gate and roster, with no bot in the meeting (RFC
12.10.4 step 1). The delivery and track-published triggers need no guard of
their own: each already answers to what is configured.

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
from roomkit.channels._conference_operations import ConferenceResource
from roomkit.channels._conference_room_state import LeavingSession
from roomkit.conference.models import BotSession, ConferenceGrants
from roomkit.core.exceptions import RoomNotAttachedError
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.session_event import SessionStartedEvent

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.channels._conference_operations import ConferenceOperations
    from roomkit.conference.base import ConferenceBackend
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.channels.conference")

# How long to wait before each re-join attempt after the SFU ended the bot's
# session, and how many to make. Bounded with backoff: a healthy SFU takes the
# bot back on the first try, an outage should not be hammered, and past the
# last attempt the lazy join remains — the next mint, delivery or arrival
# still re-joins. Read through the module so a test's monkeypatch applies.
REJOIN_DELAYS_S: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)


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
        _transport_only: nothing configured consumes or speaks, so the mint,
            arrival and probe triggers stand down (RFC 12.10.4 step 1).
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
    _operations: ConferenceOperations
    _framework: RoomKit | None
    _bot_identity: str
    _bot_grants: ConferenceGrants
    _transport_only: bool
    _backend_closed: bool

    # Provided by ConferenceChannel — see the host contract above
    _rooms: Any
    _room: Any
    _attached_room: Any
    _voice: Any
    _stop_consuming: Any
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
            # Read once, before the join, and recorded beside the session it
            # was applied to: derived grants follow the configuration (RFC
            # 12.10.4), so what the SFU holds for this session is this
            # snapshot, not whatever the configuration says later.
            grants = self._bot_grants
            # Marked for the whole of the join, not just the network call: the
            # bot is not recognisable by its session until the record carries
            # it, and a backend that echoes the bot back mid-join arrives inside
            # this window.
            room.joining = True
            try:
                # One lease for the join *and* its compensation: the backend
                # must not close between the two, or the session the abandoned
                # join opened would have nothing to leave through.
                with self._operations.use(
                    ConferenceResource.BACKEND, what=f"joining room {room_id} as the bot"
                ):
                    bot = await self._backend.join_as_bot(room_id, self._bot_identity, grants)
                    _settle_clock(bot, self._backend.name)
                    if room.generation != generation:
                        # Recorded rather than plain, because the detach that
                        # closed this generation had no bot to put on `leaving`
                        # — this one did not exist yet — so a `leave()` that
                        # fails here would strand a session no part of the room
                        # could account for.
                        await self._leave_and_record(room_id, bot)
                        raise RoomNotAttachedError(
                            f"Channel {self.channel_id!r} was detached from room "
                            f"{room_id!r} while joining"
                        )
                    room.bot = bot
                    room.bot_grants = grants
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
        several places — a mint, the next arrival, a published track, a
        delivery — and each of them finds ``room.bot`` still unset and tries
        again.

        A pure-transport channel does not try at all, and still answers
        ``True``: the arrival is recorded — that is the unconditional MUST —
        and the join it would have started has no function to serve (RFC
        12.10.4 step 1).
        """
        if self._transport_only:
            return True
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

    async def _ensure_bot_for_mint(self, room_id: str) -> None:
        """Bring the bot in because a credential just went out.

        A successful mint is the framework's advance notice that a human is
        about to connect, and it is the one trigger of the lazy join that does
        not depend on the backend's callbacks: presence is observable only
        through a connection (RFC 12.10.3), so until the bot holds one, no
        arrival can report itself and nothing callback-shaped can make the
        first join happen. Without this trigger a meeting where humans speak
        and the AI is meant to listen never gets a bot at all — the framework
        would have to speak first.

        Runs as a room task off the mint's own path, so nothing here reaches
        back to the mint: the caller is owed its token now, not after a media
        connection settles, and the credential belongs to the participant
        whether or not the framework got its own session into the room (RFC
        12.10.4).

        ``RoomNotAttachedError`` is silence rather than a report: a detach won
        the race and the join was abandoned exactly as it should be. Any other
        failure is logged and swallowed — the next trigger finds ``room.bot``
        still unset and tries again.

        A pure-transport channel never joins on it: the credential is the
        participant's either way, and the session the join would open has
        nothing to consume and nothing to say (RFC 12.10.4 step 1).
        """
        if self._transport_only:
            logger.debug(
                "Conference channel %r is not joining room %s on this mint: nothing "
                "configured on the channel consumes or speaks (pure transport)",
                self.channel_id,
                room_id,
            )
            return
        try:
            await self._ensure_bot(room_id)
        except RoomNotAttachedError:
            return
        except Exception:
            logger.exception(
                "Conference channel %r could not bring its bot into room %s after minting "
                "conference access. The credential was returned regardless, and the "
                "conference runs without the framework's own media session — no "
                "transcription and no AI voice — until a later join succeeds",
                self.channel_id,
                room_id,
            )

    async def _ensure_bot_for_resume(
        self, room_id: str, generation: int, *, trigger: str = "attach"
    ) -> None:
        """Bring the bot in when the attach landed over a conference already underway.

        ``trigger`` names what asked, in the logs alone: a hot-plug re-runs
        this very probe — plugging a need makes the join once more the
        consequence a probe can have (RFC 12.10.4) — and an operator reading
        "at attach" about a join no attach started would go looking for the
        wrong event.

        The mint bootstraps a conference nobody has been admitted to yet; it
        cannot resume one already running. A channel restarted mid-meeting
        re-attaches above participants an earlier life admitted, and every
        other trigger is out of reach — any re-join supervisor died with the
        process, no callback can arrive without a connection (RFC 12.10.3),
        and the humans already in the room may never mint again nor be
        delivered to. So the attach asks the one question that needs no
        connection: ``list_participants()``, control-plane, one call per
        attach. Anyone in there who is not the channel's own bot — a session
        an earlier life left behind is not occupancy — is first need, and the
        join happens exactly as it would for a mint (RFC 12.10.4). An empty
        conference stays unjoined: the laziness is preserved, and the probe
        is all an idle room ever costs.

        Nothing is written to the roster from here. The join's own catch-up
        (RFC 12.10.3) redelivers everyone through the ordinary callbacks; the
        probe only decides whether there is a meeting to join.

        Runs as a room task off the attach's own path: the attach is owed its
        answer now, and a detach cancels this like any other room work. The
        probe answers for the attachment that spawned it — ``generation`` is
        handed in by the attach rather than read here, because a spawned task
        can sit unscheduled while the world moves on. It re-reads that world
        before every step, the re-join supervisor's own discipline: a bumped
        generation or a bot already in means a loss, a re-attach or a faster
        trigger got there first, and this probe is someone else's late
        duplicate. It stands down; whoever moved the world owns the join now.

        ``RoomNotAttachedError`` is silence rather than a report: a detach won
        the race and the join was abandoned exactly as it should be. Any other
        failure — the probe's or the join's — is logged and swallowed, and the
        lazy join remains: the next mint, delivery or arrival tries again.

        A pure-transport channel skips the probe entirely, not merely the
        join: the join is the only consequence a probe can have, so with
        nothing configured to consume or speak there is nothing to ask the
        control plane (RFC 12.10.4 step 1).
        """
        if self._transport_only:
            logger.debug(
                "Conference channel %r is not probing room %s's occupancy at attach: "
                "nothing configured on the channel consumes or speaks (pure transport)",
                self.channel_id,
                room_id,
            )
            return
        room = self._room(room_id)
        if room.generation != generation or room.bot is not None:
            return
        try:
            with self._operations.use(
                ConferenceResource.BACKEND, what=f"probing room {room_id} for participants"
            ):
                participants = await self._backend.list_participants(room_id)
        except Exception:
            logger.exception(
                "Conference channel %r could not ask who is in room %s's conference at "
                "%s. If a meeting is already underway there, it runs without the "
                "framework's own media session — no transcription and no AI voice — until "
                "a mint, delivery or arrival triggers the lazy join",
                self.channel_id,
                room_id,
                trigger,
            )
            return
        occupants = [p for p in participants if p.participant_id != self._bot_identity]
        if not occupants:
            return
        if room.generation != generation or room.bot is not None:
            return
        # Said out loud because nothing else explains what follows: a bot
        # joining a meeting where nobody has minted, delivered or spoken looks
        # spontaneous to an operator unless the probe names its reason.
        logger.info(
            "Conference channel %r found %d participant(s) already in room %s's conference "
            "at %s; resuming the meeting and joining as the bot",
            self.channel_id,
            len(occupants),
            room_id,
            trigger,
        )
        try:
            await self._ensure_bot(room_id)
        except RoomNotAttachedError:
            return
        except Exception:
            logger.exception(
                "Conference channel %r found a conference already underway in room %s at "
                "%s but could not bring its bot in. The conference runs without the "
                "framework's own media session — no transcription and no AI voice — until "
                "a later join succeeds",
                self.channel_id,
                room_id,
                trigger,
            )

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
        """Take the bot out of the conference, exactly once, and say whether it is out.

        The single funnel every departure goes through — a detach's teardown,
        an abandoned join's compensation, the channel close's sweep. A session
        has at most one ``leave()`` in flight (RFC 12.10.4): a caller that
        finds a departure already running joins it and takes its answer,
        because a second concurrent ``leave()`` would ask the backend to
        remove a participant twice, and whichever answer arrived second would
        be about a session that no longer exists.

        The departure itself runs on a task of its own, so a joiner being
        cancelled cancels nothing but its own wait. The caller that *started*
        the departure owns it: its cancellation — the closing budget expiring
        — travels on to the backend call, which is how the budget stops the
        step; a backend that swallows it keeps its lease on itself until it
        truly ends, and the close retains the backend accordingly.
        """
        room = self._room(room_id)
        entry = room.leaving.get(bot.id)
        if entry is not None and entry.task is not None and not entry.task.done():
            # Join the departure in flight. Shielded: this caller's
            # cancellation must not reach a task another path owns.
            try:
                return await asyncio.shield(entry.task)
            except Exception:
                return False
        if entry is None:
            entry = room.leaving[bot.id] = LeavingSession(bot=bot)
        task = asyncio.create_task(
            self._depart(room_id, bot), name=f"roomkit-conference-leave-{bot.id}"
        )
        entry.task = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Ours to own: the budget cancelled this caller, and the departure
            # it started is cancelled with it. The task's own handler puts the
            # session back on the books before the cancellation travels on.
            task.cancel()
            raise
        except Exception:
            return False

    async def _depart(self, room_id: str, bot: BotSession) -> bool:
        """The one ``leave()`` a departure makes. See :meth:`_leave_and_record`.

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
        try:
            if self._backend_closed:
                logger.warning(
                    "Conference channel %r cannot take bot session %s out of room %s: its "
                    "backend is already closed, so the bot may remain in the conference",
                    self.channel_id,
                    bot.id,
                    room_id,
                )
                room.record_leave_failure(bot, "the channel's backend was already closed")
                return False
            with self._operations.use(
                ConferenceResource.BACKEND, what=f"leave() for session {bot.id}"
            ):
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

    async def _on_bot_session_ended(self, bot: BotSession, reason: str) -> None:
        """React to the SFU ending the bot's session without a ``leave()``.

        The report is the session's end in fact (RFC 12.10.3): the connection
        is gone, whatever the books say. So the books are corrected — the
        session comes off them, ``bot_present`` stops answering yes for a
        connection that no longer exists — the session's lanes are closed and
        its recordings finalized (their tracks stopped delivering with the
        connection), its playbacks are abandoned, and the end is announced.
        The next need re-joins lazily, exactly as the first one did: the
        generation bump is what makes the dead session's leftover background
        work abandon itself instead of adopting the new bot.

        A stale report — a session already replaced by a re-join, or one a
        detach is already taking out — corrects nothing: whoever owns that
        session's end owns its announcements too.
        """
        room = self._attached_room(bot.room_id)
        if room is None or room.bot is not bot:
            return
        logger.warning(
            "Conference channel %r lost bot session %s in room %s without a leave(): %s. "
            "The session is off the books; the channel re-joins on the next need",
            self.channel_id,
            bot.id,
            bot.room_id,
            reason,
        )
        room.bump()
        room.bot = None
        self._voice.forget_room(bot.room_id)
        for track_id in room.forget_subscriptions():
            await self._stop_consuming(track_id)
        await self._announce_end(bot.room_id, bot)
        # The meeting is still running and the room still wants its audio
        # collected, but nothing left in it can create the "next need" — the
        # dead session received the frames and the events that used to. So
        # the need is manufactured: a bounded re-join with backoff, abandoned
        # the moment anything contradicts it. Registered as a room task, so a
        # detach or a close cancels it like any other background work.
        if room.may_collect():
            room.spawn(self._rejoin_after_loss(bot.room_id, room.generation))

    async def _rejoin_after_loss(self, room_id: str, generation: int) -> None:
        """Bring a replacement bot in after the SFU dropped the last one.

        Each attempt re-reads the world first: a re-attach bumped the
        generation, a lazy join already brought a bot in, a detach turned
        collection off — any of them makes this supervisor someone else's
        late duplicate, and it stands down. Exhausting the attempts is
        reported, not fatal: the lazy join remains, and the next delivery or
        arrival still re-joins.
        """
        room = self._room(room_id)
        for delay in REJOIN_DELAYS_S:
            await asyncio.sleep(delay)
            if room.generation != generation or room.bot is not None or not room.may_collect():
                return
            try:
                await self._ensure_bot(room_id)
            except RoomNotAttachedError:
                return
            except Exception:
                logger.warning(
                    "Conference channel %r could not re-join room %s after losing its bot "
                    "session; retrying",
                    self.channel_id,
                    room_id,
                    exc_info=True,
                )
                continue
            logger.info(
                "Conference channel %r re-joined room %s after losing its bot session",
                self.channel_id,
                room_id,
            )
            return
        logger.error(
            "Conference channel %r gave up re-joining room %s after %d attempt(s). The "
            "conference runs untranscribed and unrecorded until the next mint, delivery "
            "or arrival triggers the lazy join",
            self.channel_id,
            room_id,
            len(REJOIN_DELAYS_S),
        )

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
        if room.bot is not None:
            room.start_closing(room.bot)
            room.bot = None
        for entry in room.closing_sessions():
            if await self._leave_and_record(room_id, entry.bot) and entry.owed_an_end:
                await self._announce_end(room_id, entry.bot)

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
        with self._operations.use(
            ConferenceResource.BACKEND, what=f"closing conference room {room_id}"
        ):
            await self._backend.close_room(room_id)

    async def _settle_joins(self) -> set[str]:
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
            return set()
        waiters = {
            asyncio.ensure_future(self._settle_one(room.lock)): room_id
            for room_id, room in list(self._rooms.items())
        }
        _, unfinished = await asyncio.wait(waiters, timeout=_conference_activity.DRAIN_TIMEOUT_S)
        for waiter in unfinished:
            waiter.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)
        room_ids = {waiters[waiter] for waiter in unfinished}
        if unfinished:
            logger.warning(
                "Closing conference channel %r with a join still in flight for room(s) %s "
                "after %.0fs; the bot it is opening may be left in the conference",
                self.channel_id,
                ", ".join(sorted(room_ids)),
                _conference_activity.DRAIN_TIMEOUT_S,
            )
        return room_ids

    @staticmethod
    async def _settle_one(lock: asyncio.Lock) -> None:
        """Hold a room's join lock for as long as it takes to get it."""
        async with lock:
            return

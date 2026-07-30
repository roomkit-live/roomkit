"""Attaching a conference channel to a room, and taking it back off.

Both are reached through the ``Channel`` contract — ``on_room_attached`` and
``on_room_detached``, awaited by the framework's attach and detach — and not
through the lifecycle hooks. A hook is observation: it runs alongside the
integrator's own handlers and its errors are logged rather than raised, and
neither is true of creating the conference the binding claims exists. So a
backend that refuses takes the attach down with it, and an integrator's
``ON_CHANNEL_DETACHED`` handler runs after the channel has let go.

A detach has two halves, and only the first can always happen now. Closing
admission is bookkeeping and never blocks. Destroying — leaving the conference,
announcing the end — has to come after the work that still describes a live
conference, and when the detach was itself triggered from inside that work,
"after" is not somewhere the call can wait for. So it is deferred, onto a task
the next attach to that room waits for.

Which is where the ordering guarantees live: the lanes stop before the bot
leaves, the end is announced last, and a teardown deferred past a re-attach
destroys the attachment it closed rather than the one that replaced it.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC sections 12.10.4 and 17.7.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

# Read through the module rather than bound into this one: the deadline a
# deployment or a test sets on it is the one that has to apply, and a name
# imported here would be a second copy to remember — see `_budget`.
from roomkit.channels import _conference_activity
from roomkit.core.exceptions import RoomNotAttachedError
from roomkit.core.task_utils import log_task_exception

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.channels._conference_lane import ConferenceLane
    from roomkit.channels._conference_recording import ConferenceRecording, TrackRecording
    from roomkit.channels._conference_recording_events import ConferenceRecordingEvents
    from roomkit.channels._conference_voice import ConferenceVoice
    from roomkit.conference.base import ConferenceBackend
    from roomkit.conference.models import BotSession
    from roomkit.models.channel import ChannelBinding

logger = logging.getLogger("roomkit.channels.conference")

# Rooms whose teardown the current context is running inside. An attach reached
# from in there — a ``conference_ended`` handler re-attaching the channel is the
# realistic way — cannot wait for that teardown to finish without waiting for
# itself. Carried on a ContextVar rather than read off the running task for the
# same reason the activity marker is: a task inherits a copy of the context that
# created it, so the marker follows the causal chain wherever the hook engine
# schedules it.
_tearing_down: ContextVar[frozenset[str]] = ContextVar(
    "roomkit_conference_teardown", default=frozenset()
)


def _since(moment: datetime) -> int:
    """Milliseconds from a moment to now, never negative.

    A backend may set ``joined_at`` from a clock the framework does not share,
    and a duration reported as negative is worse than one reported as zero.
    """
    return max(0, int((datetime.now(UTC) - moment).total_seconds() * 1000))


class ConferenceAttachmentMixin:
    """Attach, detach and teardown for a conference channel.

    Host contract — what ConferenceChannel provides:
        channel_id, _backend, _activity, _voice: the channel and what it talks
            to.
        _recorder, _lanes: what a detach takes out by value, so a teardown
            deferred past a re-attach cannot finalize what the new attachment
            has since opened.
        _recording_events: announces each finalized recording, before the end
            of the conference is announced.
        _teardowns: detaches still finishing on their own tasks, channel-wide
            because `close()` waits for all of them.
        _close_room_on_detach, _e2ee: what the attachment asks the SFU for.
        _room: the per-room record (ConferenceRoomState).
        _apply_collection_state, _abandon_mints, _leave_and_record, _close_room,
            _emit_framework_event: what the two halves of a detach reach for.
    """

    channel_id: str
    _backend: ConferenceBackend
    _activity: RoomActivity
    _voice: ConferenceVoice
    _recorder: ConferenceRecording | None
    _recording_events: ConferenceRecordingEvents
    _lanes: dict[str, ConferenceLane]
    _teardowns: set[asyncio.Task[None]]
    _close_room_on_detach: bool
    _e2ee: bool

    # Provided by ConferenceChannel and its other mixins — see above
    _room: Any
    _apply_collection_state: Any
    _abandon_mints: Any
    _leave_and_record: Any
    _close_room: Any
    _emit_framework_event: Any

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        """React to the room binding changing.

        A binding the integrator has closed must stop the channel collecting,
        not merely stop it writing. Frames that keep arriving are still decoded,
        still sent to speech recognition, and still cost what they cost — so the
        gate closes immediately here, and the subscriptions are dropped on the
        loop right after, since unsubscribing is asynchronous and this is not.
        """
        room = self._room(room_id)
        was_collecting = room.may_collect()
        room.binding = binding
        if room.may_collect() == was_collecting:
            return
        room.spawn(self._apply_collection_state(room_id))

    async def on_room_attached(self, room_id: str, binding: ChannelBinding) -> None:
        """Create the conference the binding says exists — RFC §12.10.4 step 1.

        A backend that refuses takes the attachment down with it: nothing here
        is guarded, so ``attach_channel()`` raises and the binding is taken
        back. That is the point of doing this on the channel contract rather
        than on the ``ON_CHANNEL_ATTACHED`` hook, where a hook engine that logs
        and never raises would leave the room believing it was conferenced.

        Creating the room and destroying it are serialised against each other,
        which is what makes waiting for the previous teardown a courtesy rather
        than the guarantee. That wait is bounded — it has to be — and past its
        deadline the old teardown is somewhere unknown: possibly still ahead of
        its ``close_room``, possibly inside it. Holding the lock across the
        create *and* the generation bump is what settles the order either way:
        the teardown reads the generation on the far side of this, sees the room
        it is closing is no longer the current one, and leaves the new
        conference alone.
        """
        await self._settle_previous_attachment(room_id)
        room = self._room(room_id)
        await self._hold_sfu_room(room_id)
        try:
            await self._backend.ensure_room(room_id, e2ee=self._e2ee)
            room.bump()
            room.attached = True
            room.binding = binding
        finally:
            room.sfu_room.release()

    async def _hold_sfu_room(self, room_id: str) -> None:
        """Take the room's create/destroy lock, or refuse the attachment.

        Bounded, and a refusal rather than a wait, because the thing on the
        other side of this lock is a ``close_room`` against the same name. Going
        ahead without it is the race itself; waiting for it without a deadline
        makes a wedged SFU into an attach that never returns. Refusing is the
        third answer, and the only one that is both safe and finite: the caller
        is told the conference could not be created, and ``attach_channel()``
        takes the binding back.
        """
        try:
            await asyncio.wait_for(
                self._room(room_id).sfu_room.acquire(), _conference_activity.DRAIN_TIMEOUT_S
            )
        except TimeoutError:
            raise RoomNotAttachedError(
                f"Channel {self.channel_id!r} could not attach to room {room_id!r}: the "
                f"conference room it detached from is still being destroyed after "
                f"{_conference_activity.DRAIN_TIMEOUT_S:.0f}s, and creating it now would "
                "hand the new conference to that teardown. Retry once it has finished."
            ) from None

    async def _settle_previous_attachment(self, room_id: str) -> None:
        """Let the attachment this one replaces finish being destroyed.

        A detach triggered from inside the channel's own work returns before its
        destructive half has run, and that gap is somewhere an integrator can
        re-attach: ``detach()`` then ``attach()`` in one handler is two calls
        that both appear to have completed. What lands afterwards belongs to the
        generation it closed, not to this one — it announces an end after this
        generation's start, and with ``close_room_on_detach`` it destroys the
        conference this attach has just created.

        So the new generation waits for the old one to finish dying, and the
        detach/re-attach pair reads as what it is: ended, then started.

        Except when the wait would be circular. A teardown deferred out of an
        announcement is waiting for that announcement, so a handler that
        re-attaches from inside it would be waiting for itself; the same is true
        of one re-attaching from a ``conference_ended`` handler, which runs
        inside the teardown. Both are refused the wait and left to the
        generation the teardown carries, which is what keeps it from destroying
        an attachment that is not the one it closed.
        """
        task = self._room(room_id).pending_teardown
        if task is None or task.done():
            return
        if room_id in _tearing_down.get() or self._activity.enclosing(room_id):
            return
        _, unfinished = await asyncio.wait([task], timeout=_conference_activity.DRAIN_TIMEOUT_S)
        if unfinished:
            logger.warning(
                "Channel %r is re-attaching to room %s while its previous detach is still "
                "finishing after %.0fs: the end of the previous conference may be announced "
                "after the start of this one",
                self.channel_id,
                room_id,
                _conference_activity.DRAIN_TIMEOUT_S,
            )

    async def on_room_detached(self, room_id: str) -> None:
        """Close the room to new work, then destroy what it was doing.

        The two halves are separate because only the first can always happen
        now. Closing admission is bookkeeping and never blocks. Destroying —
        leaving the conference, announcing the end — has to come after the work
        that still describes a live conference, and when the detach was itself
        triggered from inside that work, "after" is not somewhere this call can
        wait for. So it is deferred rather than skipped.
        """
        room = self._room(room_id)
        # Bumped before taking the lock, so a join already holding it sees the
        # detach when it re-reads the generation after connecting.
        #
        # What it returns is the generation this detach closes. A teardown that
        # runs later destroys what belonged to *this* attachment, and nothing
        # that has since come to belong to the next one.
        generation = room.bump()
        room.cancel_tasks()
        async with room.lock:
            room.attached = False
            bot, room.bot = room.bot, None
        room.binding = None
        # Before the bot leaves, not after: the synthesis loop publishes on
        # this session, and a loop still running would go on speaking into a
        # conference the channel has left.
        self._voice.forget_room(room_id)
        # Out of routing immediately — `_on_track_audio` finds nothing to hand a
        # frame to from here on. They are closed in the destructive phase, since
        # closing is what releases the pipeline stage state and finalizes the
        # recordings, and that can wait a moment; being fed cannot.
        #
        # Taken out by value rather than looked up again later, and the
        # recordings for the same reason as the lanes: a teardown deferred past
        # a re-attach would otherwise finalize a recording the new attachment
        # has since opened under the same track id.
        track_ids = room.forget_subscriptions()
        lanes = [lane for tid in track_ids if (lane := self._lanes.pop(tid, None)) is not None]
        recordings = self._recorder.detach_room(room_id) if self._recorder is not None else []
        if bot is not None:
            # The bot is out of `_bots` but still in the conference, and RFC 17.7
            # asks for its presence to be observable *at any time*. This is what
            # `info()` reads until `leave()` has actually happened.
            #
            # Keyed by session, because a room can be leaving more than once: a
            # teardown held open in `leave()` is still running when a re-attach
            # brings a second bot in and a second detach sends it out behind the
            # first. One entry per room loses the older of the two while it is
            # still sitting in the conference.
            room.start_leaving(bot)
        enclosing = self._activity.enclosing(room_id)
        if enclosing:
            self._defer_teardown(room_id, bot, lanes, recordings, generation, enclosing)
            return
        await self._teardown(room_id, bot, lanes, recordings, generation)

    def _defer_teardown(
        self,
        room_id: str,
        bot: BotSession | None,
        lanes: list[ConferenceLane],
        recordings: list[TrackRecording],
        generation: int,
        enclosing: list[asyncio.Event],
    ) -> None:
        """Finish the detach once the work that triggered it has finished.

        A ``conference_started`` handler that detaches the channel is ordinary
        integrator code, and it leaves this call nested inside the very
        announcement the teardown must not overtake. Waiting inline would
        deadlock; carrying on inline would put the end in front of the rest of
        that announcement's observers, which is the inversion the drain exists
        to prevent. Neither, then: admission is already closed, and the
        destroying happens on its own task the moment the announcement is done.

        Tracked separately from ``_tasks``: those are cancelled by the next
        detach, and a teardown that is cancelled leaves the bot in the
        conference. ``close()`` waits for these instead. Kept per room as well,
        because the next *attach* to that room has to wait for this one — see
        :meth:`_settle_previous_attachment`.
        """
        task = asyncio.create_task(
            self._teardown(room_id, bot, lanes, recordings, generation, after=enclosing)
        )
        self._teardowns.add(task)
        self._room(room_id).pending_teardown = task
        task.add_done_callback(partial(self._forget_teardown, room_id))
        task.add_done_callback(log_task_exception)

    def _forget_teardown(self, room_id: str, task: asyncio.Task[None]) -> None:
        """Drop a finished teardown, without dropping the one that replaced it."""
        self._teardowns.discard(task)
        room = self._room(room_id)
        if room.pending_teardown is task:
            room.pending_teardown = None

    async def _teardown(
        self,
        room_id: str,
        bot: BotSession | None,
        lanes: list[ConferenceLane],
        recordings: list[TrackRecording],
        generation: int,
        *,
        after: list[asyncio.Event] = [],  # noqa: B006 — read-only, never mutated
    ) -> None:
        """Leave the conference and announce the end, after everything else.

        Order matters twice over: the lanes stop before the bot leaves, so a
        transcription cannot arrive during the end announcement; and the end is
        announced last, so it is the final thing any observer hears.

        The recordings are finalized here rather than at the detach for the same
        reason as the lanes: an observer told the conference is over must not
        then find a file still open. Nothing is still being written by then —
        the subscriptions were forgotten before the drain — so this is about
        where the closing lands, not about what it captures. Their results are
        announced at the same point and for the same reason: where the files
        went is part of what the conference was, and it belongs in front of the
        announcement that it is over.

        ``generation`` is the attachment this is destroying. Everything reached
        through the arguments belongs to it and is destroyed unconditionally —
        those lanes, those recordings, that bot session. Everything reached
        through the room's name does not: a re-attach the deferral could not be
        made to wait for has left a *live* conference under that name, and
        closing its SFU room or resetting its track generations would destroy an
        attachment this detach never touched.

        Every destructive step is best-effort and independent of the others.
        They are the last chance anything here has to be undone, and a lane that
        will not close has nothing to do with the bot still sitting in the
        conference: written as one happy sequence, a recorder that raised on a
        full disk skipped the ``leave()`` behind it and left the bot in the
        meeting. The detach itself never fails either — the framework awaits
        this between removing the binding and announcing the detach, and an
        exception here would cost the room its ON_CHANNEL_DETACHED.
        """
        marker = _tearing_down.set(_tearing_down.get() | {room_id})
        # Whether the conference has no bot left in it. A session `leave()`
        # could not remove stays on the room's books, and destroying the SFU
        # room below is the one compensation left for it.
        left = bot is None
        try:
            if after:
                await self._activity.wait_for(after)
            # Admission is closed — the generation is bumped and the room is out
            # of `_attached` — so nothing new starts. What is already running
            # still holds the truth of a live conference: an announcement being
            # made, an arrival being recorded, a chunk on its way to the bot
            # track. Draining lets each finish before this contradicts it.
            await self._activity.drain(room_id)
            # The drain is bounded, and one kind of work must not be left behind
            # when it runs out: a credential still being minted would admit
            # someone to the conference this is about to leave.
            self._abandon_mints(room_id)
            for lane in lanes:
                await self._best_effort(
                    lane.aclose(),
                    "Conference channel %r could not close a lane of room %s. The rest of the "
                    "teardown is going ahead: the lane's pipeline state is leaked, the bot is "
                    "not",
                    self.channel_id,
                    room_id,
                )
            if self._recorder is not None:
                await self._best_effort(
                    self._finish_recordings(recordings),
                    "Conference channel %r could not finalize the recordings of room %s. The "
                    "rest of the teardown is going ahead: an unclosed container is a file, the "
                    "bot is a participant",
                    self.channel_id,
                    room_id,
                )
            if bot is not None:
                # Which forgets the session when it worked and keeps it, with
                # the reason, when it did not.
                left = await self._leave_and_record(room_id, bot)
                if left:
                    await self._announce_end(room_id, bot)
        finally:
            _tearing_down.reset(marker)
            room = self._room(room_id)
            if room.generation == generation:
                room.track_epochs.clear()
                room.collision_reported = False
        # The join lock itself outlives the attachment, like the generation
        # does. A join can be queued on it right now, and dropping it would let
        # a re-attach mint a second lock: two joins for one room, holding
        # different locks, publishing the AI on two tracks — the thing the lock
        # exists to prevent. One lock per room the channel ever served is the
        # same footprint as the generation counter beside it.
        if not self._close_room_on_detach:
            return
        if not await self._destroy_sfu_room(room_id, generation):
            return
        if not left and bot is not None:
            await self._end_after_eviction(room_id, bot)

    async def _destroy_sfu_room(self, room_id: str, generation: int) -> bool:
        """Close the conference room this detach opened, if it is still that one.

        Returns whether the room was destroyed, which is what says a bot
        ``leave()`` could not remove has been evicted with it.

        The generation is read under the create/destroy lock and acted on
        without letting go, which is the whole of what makes the answer usable.
        Read outside it, it would be read before a re-attach's ``ensure_room``
        and acted on after — and RFC 12.10.4 step 5's MUST would destroy a
        conference that is live, with its participants in it. A re-attach the
        deferral could not be made to wait for is exactly the case this arrives
        in.
        """
        room = self._room(room_id)
        try:
            await asyncio.wait_for(room.sfu_room.acquire(), _conference_activity.DRAIN_TIMEOUT_S)
        except TimeoutError:
            logger.warning(
                "Conference channel %r is not closing the conference room %s it detached "
                "from: an attach has been holding it for %.0fs, so whatever exists under "
                "that name is not what this detach closed",
                self.channel_id,
                room_id,
                _conference_activity.DRAIN_TIMEOUT_S,
            )
            return False
        try:
            if room.generation != generation:
                logger.info(
                    "Channel %r is not closing the conference room %s it detached from: the "
                    "room has been re-attached since, so the conference under that name is a "
                    "live one",
                    self.channel_id,
                    room_id,
                )
                return False
            # Whatever became of `leave()`. RFC 12.10.4 step 5 makes this a MUST
            # "whether or not a bot ever joined", and a bot the SFU would not
            # let go of is the case where it matters most: destroying the room
            # is what evicts it.
            return await self._best_effort(
                self._close_room(room_id),
                "Conference channel %r could not close the conference room %s it detached from",
                self.channel_id,
                room_id,
            )
        finally:
            room.sfu_room.release()

    async def _finish_recordings(self, recordings: list[TrackRecording]) -> None:
        """Close a detached room's recordings and say where they went.

        One step of the teardown rather than two, so the pair can be guarded as
        one: what a recording was is not reportable until it is finalized, and
        neither half is a reason to leave the bot in the conference.
        """
        if self._recorder is None:
            return
        await self._recording_events.stopped_all(await self._recorder.finish(recordings))

    async def _announce_end(self, room_id: str, bot: BotSession) -> None:
        """Say that the conference is over, and which conference that was.

        Named and measured. A detach deferred past a re-attach announces an end
        after the next start, and nothing else in the event says which
        conference it is the end of; `conference_started` has carried its
        session all along.

        Only ever reached once the bot is genuinely out — RFC 12.10.7 puts the
        event at `leave()` completing. A session still sitting in the conference
        has no end to announce, and gets one at the moment something does take
        it out.
        """
        await self._emit_framework_event(
            "conference_ended",
            room_id,
            bot_session_id=bot.id,
            duration_ms=_since(bot.joined_at),
        )

    async def _end_after_eviction(self, room_id: str, bot: BotSession) -> None:
        """Account for a bot that destroying its conference room took out.

        `leave()` refused, so the session stayed on the room's books and no end
        was announced for it. Closing the room is a blunter instrument and a
        thorough one: the conference no longer exists, so nobody is in it. That
        is the point at which the departure became true, so it is the point at
        which it is recorded and announced.
        """
        self._room(room_id).forget_leaving(bot)
        logger.info(
            "Conference channel %r could not take bot session %s out of room %s, but closing "
            "the conference room has evicted it",
            self.channel_id,
            bot.id,
            room_id,
        )
        await self._announce_end(room_id, bot)

    async def _best_effort(self, step: Awaitable[None], message: str, *args: object) -> bool:
        """Run one step of a teardown, and let the others happen if it fails.

        Returns whether it worked, for the steps whose failure changes what the
        rest of the teardown owes the room. Cancellation still propagates: a
        teardown being cancelled is not one that should carry on.
        """
        try:
            await step
        except Exception:
            logger.exception(message, *args)
            return False
        return True

    async def _await_teardowns(self) -> None:
        """Wait for detaches still finishing on their own tasks."""
        pending = list(self._teardowns)
        if not pending:
            return
        done, unfinished = await asyncio.wait(
            pending, timeout=_conference_activity.DRAIN_TIMEOUT_S
        )
        del done
        if unfinished:
            logger.warning(
                "Closing conference channel %r with %d detach(es) unfinished",
                self.channel_id,
                len(unfinished),
            )

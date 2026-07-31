"""The AI's voice in a conference, and the policy that decides who may cut it off.

What the bot is saying, who is allowed to talk over it, and the loop that stops
sending when someone does are one question asked at three moments, so one
collaborator holds all three. A lane asks whether the speaker it carries
interrupts the bot and gets an answer; it never touches the playback state that
answer is drawn from.

The channel keeps the decision a channel owns — whether a conference is willing
to read an event aloud at all — and delegates the rest here.

See RFC section 12.10.5.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from roomkit.channels import _conference_activity
from roomkit.channels._conference_activity import RoomActivity
from roomkit.channels._conference_lane import ConferenceBargeIn
from roomkit.channels._conference_operations import (
    ConferenceOperations,
    ConferenceResource,
    OperationLease,
)
from roomkit.conference.models import ConferenceInterruptionScope
from roomkit.core.task_utils import log_task_exception
from roomkit.models.enums import HookTrigger
from roomkit.voice.base import AudioChunk
from roomkit.voice.interruption import (
    InterruptionConfig,
    InterruptionHandler,
    InterruptionStrategy,
)

if TYPE_CHECKING:
    from roomkit.channels._conference_lane import ConferenceLane
    from roomkit.conference.base import ConferenceBackend
    from roomkit.conference.models import BotSession, ConferenceInterruptionConfig
    from roomkit.core.framework import RoomKit
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.channels.conference")

# Opens the bot's media connection for a room, joining the conference if this is
# the first thing that needs it.
EnsureBot = Callable[[str], Awaitable["BotSession"]]

# Called with each chunk the bot has published into a room. The publication is
# the only place the bot's own audio exists on this side of the backend, and it
# is what a recording of the conference would otherwise be missing.
#
# Awaited, because opening the bot's recording is what announces it, and an
# announcement runs integrator hooks. It happens on the first chunk of a bot
# session and on no other, so what the publishing loop pays for it is bounded
# by the number of sessions, not by the number of chunks.
PublishedCallback = Callable[[str, "AudioChunk"], Awaitable[None]]


@dataclass
class ConferencePlayback:
    """The bot's current utterance in a room.

    Interruption policy is a question about a moment: whether the bot is
    speaking, and how far into what it is saying. This is what holds the
    answer, and it is what makes ``may_interrupt`` answerable at all.
    """

    room_id: str
    text: str
    started_at: float | None = None
    """When the utterance took the floor, or ``None`` while it still waits.

    Not the moment it was queued: an answer waiting its turn has not been heard
    by anyone, and a position measured from the queue would report a barge-in
    landing deep into speech nobody has heard yet.
    """

    interrupted: bool = False
    abandoned: bool = False
    """The channel left the room, or closed, mid-utterance.

    Distinct from ``interrupted``: a barge-in is something that happened
    *inside* a conference the bot is still part of, and the exchange completes.
    This is the conference going away underneath the utterance, and nothing
    downstream of it should still run.
    """

    bot: BotSession | None = None
    """The session this utterance publishes on, once it has taken the floor.

    What a barge-in stops playback against: the gesture goes to the session
    that queued the audio, not to whatever session the room holds by the time
    the latch is read. ``None`` while the utterance still waits its turn —
    nothing of it has been published anywhere, so there is nothing to stop.
    """

    published: AudioChunk | None = None
    """The last chunk the backend *accepted*, or ``None`` before the first.

    What the terminal chunk is built from — it carries the format of the audio
    it closes — and what says a boundary is owed at all. Kept on the record
    rather than in the publishing loop's frame because closing the utterance is
    not always that loop's to do: a cancelled publication is closed from a task
    of its own, and it needs to know what was said. Cleared once the boundary
    has been published, so nothing can close the same utterance twice.

    Written only when the publication has *finished*, which is why the two are
    separate fields: a chunk still in flight is not yet part of what the
    conference heard, and a chunk the backend refused never will be.
    """

    publishing: tuple[AudioChunk, asyncio.Future[None]] | None = None
    """The chunk currently on its way to the backend, and the call carrying it.

    A publication outlives the caller's interest in it — a backend that shields
    its send finishes whatever this side does about being cancelled — so it is
    held here until it is known to have arrived or not. The boundary waits for
    it: a terminal chunk that overtakes the audio it closes ends the utterance
    before its last words.
    """

    closing: bool = False
    """Whether a task is still publishing this utterance's terminal chunk.

    Only a cancelled publication sets it, and it is what keeps the record in
    the room while that lasts: an utterance forgotten before its boundary is out
    is one a detach can no longer abandon, and the chunk would be published into
    a session on its way out of the conference.
    """

    def begin(self) -> None:
        """Take the floor: the utterance is about to be published."""
        self.started_at = time.monotonic()

    def settle_publication(self) -> None:
        """Take what became of the chunk in flight, if it has finished.

        A publication that succeeded joins what the conference heard; one the
        backend refused does not, and the utterance stays whatever the chunks
        before it made it. One still running is left alone — it is not yet
        either.
        """
        in_flight = self.publishing
        if in_flight is None:
            return
        chunk, publishing = in_flight
        if not publishing.done():
            return
        self.publishing = None
        if not publishing.cancelled() and publishing.exception() is None:
            self.published = chunk

    @property
    def speaking(self) -> bool:
        """Whether this utterance has taken the floor rather than waiting for it."""
        return self.started_at is not None

    @property
    def position_ms(self) -> int:
        """Milliseconds since the utterance started, or 0 before it did.

        Elapsed time rather than audio position: the bot publishes to an SFU
        that plays out on its own clock, so the framework never learns where
        playback actually is.
        """
        if self.started_at is None:
            return 0
        return int((time.monotonic() - self.started_at) * 1000)


@dataclass
class _RoomVoice:
    """One room's share of the bot's voice: what it is saying, and whose turn.

    The two have exactly the same lifetime — a room has a turn to give for as
    long as anything is speaking in it or waiting to — so they are one record
    rather than two dictionaries that would drift.
    """

    playbacks: list[ConferencePlayback] = field(default_factory=list)
    """Every utterance the room owns: the one speaking, and any behind it.

    A list, not one entry: nothing stops two events being delivered to a room
    at once, and a single slot would let the second utterance evict the first —
    leaving a loop nobody can reach, still publishing, when the channel comes
    to abandon what it thinks is the only playback.
    """

    floor: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Held for the whole of an utterance, so only one is ever on the track.

    The bot has one track and the SFU mixes nothing for it, so two utterances
    publishing at once do not arrive as two voices — they arrive as one
    stream of alternating chunks that is audible as neither, and whose
    ``is_final`` marks boundaries that were never there.
    """

    unterminated: str | None = None
    """Why nothing further may be published on this room's bot track.

    Set when an utterance could not be closed — the backend refused the terminal
    chunk, or nothing could say whether the last audio arrived. It is the track
    that is unusable rather than the utterance: RFC section 12.10.4 asks both
    that utterances never interleave and that each one ends on ``is_final``, and
    publishing onto a track with an unclosed utterance breaks both at once. It
    outlives the playback it came from and is cleared only when the bot session
    is replaced, which is what :meth:`ConferenceVoice.forget_room` does.
    """


class ConferenceVoice:
    """Synthesis, publication and barge-in for one conference channel.

    Owns the synthesizer, the interruption policy, and what the bot is saying
    in each room. Two entry points face outwards: :meth:`speak`, called by the
    channel's ``deliver``, and :meth:`consider_interruption`, which is the
    lane's speech callback — a lane says *someone is speaking*, and this
    decides what that means for the bot.
    """

    def __init__(
        self,
        *,
        backend: ConferenceBackend,
        tts: TTSProvider | None,
        interruption: ConferenceInterruptionConfig,
        ensure_bot: EnsureBot,
        activity: RoomActivity,
        operations: ConferenceOperations,
        on_published: PublishedCallback | None = None,
    ) -> None:
        self._backend = backend
        self._tts = tts
        self._interruption = interruption
        self._ensure_bot = ensure_bot
        self._activity = activity
        self._operations = operations
        self._on_published = on_published
        self._handler = InterruptionHandler(
            InterruptionConfig(strategy=self._require_supported_strategy())
        )
        self._speaking: dict[str, _RoomVoice] = {}
        self._framework: RoomKit | None = None
        # Awaited when a barge-in lands, with the room it landed in. This is
        # how an interruption reaches a speech-to-speech provider: the latch
        # stops the chunk stream and stop_playback drops what the transport
        # holds, but the provider generating the response knows neither — the
        # tap is where the cancellation crosses to it (RFC 12.10.12).
        self._on_interrupted: Callable[[str], Awaitable[None]] | None = None
        # Terminal chunks being published for utterances whose own task was
        # cancelled, by room. Kept because two things wait for them: the next
        # utterance in that room, which must not start before the previous one
        # has ended, and the channel closing, which must not release the
        # synthesizer with a chunk still on its way out.
        self._closings: dict[str, set[asyncio.Future[None]]] = {}

    def set_framework(self, framework: RoomKit) -> None:
        """Wire the hooks this fires — BEFORE_TTS, AFTER_TTS, ON_BARGE_IN."""
        self._framework = framework

    @property
    def tts(self) -> TTSProvider | None:
        """The synthesizer currently plugged in, if any."""
        return self._tts

    def set_tts(self, tts: TTSProvider | None) -> TTSProvider | None:
        """Swap the synthesizer, returning the one it replaces.

        The hot-plug seam (RFC 12.10.4). Setting it does not touch utterances
        already in flight — each ``speak`` captured its provider on entry — so
        an unplug latches them separately (:meth:`interrupt_all`); what this
        changes is every ``speak`` from now on.
        """
        previous, self._tts = self._tts, tts
        return previous

    def set_on_published(self, callback: PublishedCallback | None) -> None:
        """Point the published-audio tap at a recorder, or at nothing.

        Follows the recording being plugged and unplugged: the tap is how the
        bot's own speech reaches the conference recording, and a callback left
        wired to an unplugged recorder would feed chunks to a closed file.
        """
        self._on_published = callback

    def set_on_interrupted(self, callback: Callable[[str], Awaitable[None]] | None) -> None:
        """Point the barge-in tap at a speech-to-speech provider, or at nothing.

        Follows the provider being plugged and unplugged. The latch and
        ``stop_playback`` end the utterance on this side of the backend; the
        tap is what tells the provider to stop *generating* it (RFC 12.10.12).
        Best-effort by construction: it is awaited inside the barge-in path,
        and a callback that fails there has lost only the upstream
        cancellation — the room has already gone quiet.
        """
        self._on_interrupted = callback

    def _require_supported_strategy(self) -> InterruptionStrategy:
        """Reject an interruption strategy a lane cannot honour."""
        strategy = self._interruption.strategy
        if strategy is InterruptionStrategy.SEMANTIC:
            raise ValueError(
                "InterruptionStrategy.SEMANTIC is not supported in a conference. "
                "Classifying a backchannel needs the transcript, which a lane only "
                "has once the utterance has ended — too late to interrupt the bot. "
                "Use IMMEDIATE or CONFIRMED."
            )
        return strategy

    def may_interrupt(self, participant_id: str) -> bool:
        """Whether a participant is allowed to interrupt the bot."""
        scope = self._interruption.scope
        if scope is ConferenceInterruptionScope.NONE:
            return False
        if scope is ConferenceInterruptionScope.ALLOWLIST:
            return participant_id in self._interruption.allowlist
        return True

    # -------------------------------------------------------------------------
    # Speaking
    # -------------------------------------------------------------------------

    async def speak(self, room_id: str, text: str) -> None:
        """Synthesize one utterance and publish it on the bot track.

        Synthesized once and published once: the SFU distributes it to
        everyone, so there is no per-participant audio to produce.

        One utterance at a time. A room has a single bot track and the SFU
        mixes nothing for it, so a second answer arriving mid-sentence waits
        for the floor rather than publishing into the same stream — see
        :attr:`_RoomVoice.floor`. It waits rather than preempting because both
        answers were produced for this room and the pipeline delivered both;
        cutting the bot off is what a *participant* does, and it has its own
        path.

        Publication is interruptible. A lane that detects an allowed speaker
        talking over the bot latches the playback, the loop stops sending at
        the next chunk, and the backend is told to drop what it has already
        queued (``stop_playback``, RFC section 12.10.3) — the latch stops the
        stream, the stop silences the transport, and together they are the
        barge-in mechanism and the reason the bot's utterance is tracked at
        all. An utterance still waiting its turn is latched too: taking the
        floor from the bot means the room goes quiet, not that the queue
        starts draining into it.

        AFTER_TTS fires either way, so BEFORE_TTS and AFTER_TTS stay a matched
        pair — an utterance stopped before it ever spoke runs neither. What was
        actually heard is on the ON_BARGE_IN event, which carries the text and
        how far into it the interruption landed.

        The exception is a channel that leaves the room while the bot is
        speaking. That is not an utterance that ended, it is a conference that
        stopped: the loop drops the rest of the audio rather than publishing
        into a session the channel has left, and AFTER_TTS does not fire,
        because firing a room's hooks after detaching from it is the thing
        detaching was supposed to stop.

        A channel with no synthesizer returns before any of it: joining the
        conference and running BEFORE_TTS for audio that will never exist is
        work with no output.
        """
        tts = self._tts
        if tts is None:
            return
        room = self._speaking.get(room_id)
        if room is None:
            room = self._speaking[room_id] = _RoomVoice()
        # Registered before the wait, not after it: this is what makes an
        # utterance queued behind another reachable by a detach and by a
        # barge-in. Registered after, it would be invisible for exactly as long
        # as it is powerless to notice anything itself.
        playback = ConferencePlayback(room_id=room_id, text=text)
        room.playbacks.append(playback)
        try:
            spoken = await self._take_the_floor(room, playback, tts)
        finally:
            self._forget(playback)
        if spoken is None or playback.abandoned:
            return
        await self._run_after_tts(room_id, spoken)

    async def speak_stream(
        self,
        room_id: str,
        chunks: AsyncIterator[AudioChunk],
        *,
        on_playback: Callable[[ConferencePlayback], None] | None = None,
    ) -> None:
        """Publish one ready-made utterance on the bot track.

        The speech-to-speech entry point (RFC 12.10.12): the audio arrives
        synthesized — a realtime provider's response — so there is no text to
        run BEFORE_TTS over and nothing for AFTER_TTS to report; those are
        text-synthesis hooks and this utterance never was text. Everything
        else is :meth:`speak`: the same floor, the same closings wait, the
        same latch and terminal chunk, so a provider response and a TTS
        answer are indistinguishable to the backend and to a barge-in.

        ``on_playback`` hands the caller the utterance's record as soon as it
        exists — before the floor, so a barge-in landing while the response
        still waits its turn latches a record the caller already holds. The
        caller uses it to keep ``text`` abreast of the provider's transcript,
        which is what ON_BARGE_IN reports as ``interrupted_text``; absent a
        transcript it stays ``""``, and the event says nothing was known to
        have been heard.
        """
        room = self._speaking.get(room_id)
        if room is None:
            room = self._speaking[room_id] = _RoomVoice()
        playback = ConferencePlayback(room_id=room_id, text="")
        room.playbacks.append(playback)
        if on_playback is not None:
            on_playback(playback)
        try:
            async with room.floor:
                if not await self._clear_to_publish(room, playback):
                    return
                bot = await self._ensure_bot(room_id)
                playback.bot = bot
                playback.begin()
                lease = self._operations.acquire(
                    ConferenceResource.REALTIME,
                    what=f"provider utterance for room {room_id}",
                )
                await self._pump(playback, bot, chunks, lease)
        finally:
            self._forget(playback)

    async def _take_the_floor(
        self, room: _RoomVoice, playback: ConferencePlayback, tts: TTSProvider
    ) -> str | None:
        """Wait for the room's turn, then say one thing. Returns what was said.

        ``None`` where nothing was: a hook blocked it, or it was stopped while
        it waited. BEFORE_TTS runs *inside* the turn rather than before the
        wait, because it is where orchestration holds the bot silent — a hook
        that answered while the previous utterance was still going would be
        answering about a room that has since moved on.
        """
        room_id = playback.room_id
        async with room.floor:
            if not await self._clear_to_publish(room, playback):
                return None
            text = await self._run_before_tts(room_id, playback.text)
            if not text:
                return None
            playback.text = text
            bot = await self._ensure_bot(room_id)
            playback.bot = bot
            playback.begin()
            await self._publish(playback, bot, tts)
            return text

    async def _clear_to_publish(self, room: _RoomVoice, playback: ConferencePlayback) -> bool:
        """Inside the floor: whether this utterance may go out on the track.

        An utterance stopped while it waited — abandoned by a detach, latched
        by a barge-in — is dropped here, before anything of it is published.

        The rest is the previous turn. An utterance a cancellation left to be
        closed is publishing its boundary on a task of its own, and it no
        longer holds the floor to keep this one off. Waiting here is what
        keeps the two from interleaving — the end of the previous turn goes
        out before the start of this one.

        And when the wait runs out, this answer is dropped rather than
        published. The previous utterance has no boundary yet, so anything
        sent now is heard as its continuation and the boundary still to
        come lands in the middle of it — the interleaving RFC section
        12.10.4 forbids outright, arrived at by waiting instead of by
        racing. Both endings are within what the RFC leaves to the
        implementation, and the one that goes unheard is the one that
        cannot corrupt the track.
        """
        room_id = playback.room_id
        if playback.abandoned or playback.interrupted:
            logger.info("Conference answer dropped before its turn in room %s", room_id)
            return False
        if not await self._settle_closings(room_id) or room.unterminated is not None:
            logger.warning(
                "Conference answer dropped in room %s: the previous utterance has not "
                "been closed (%s), and publishing this one would be heard as its "
                "continuation",
                room_id,
                room.unterminated or "still closing",
            )
            return False
        return True

    async def _publish(
        self, playback: ConferencePlayback, bot: BotSession, tts: TTSProvider
    ) -> None:
        """Synthesize one utterance and pump it onto the bot track."""
        # The synthesizer is in use for the whole of the loop — the iterator
        # is suspended inside the provider between chunks — so the lease
        # covers it all: a close must not free the synthesizer under a stream
        # a provider is still producing.
        lease = self._operations.acquire(
            ConferenceResource.TTS, what=f"synthesis for room {playback.room_id}"
        )
        await self._pump(playback, bot, tts.synthesize_stream(playback.text), lease)

    async def _pump(
        self,
        playback: ConferencePlayback,
        bot: BotSession,
        chunks: AsyncIterator[AudioChunk],
        lease: OperationLease,
    ) -> None:
        """Publish one utterance's chunks on the bot track, until it ends or is stopped.

        Cancelled counts as ended, and still owes the backend a boundary. The
        conference is live and the bot is in it — an orchestration that dropped
        this answer, a caller that gave up on ``deliver()`` — so the exception
        RFC section 12.10.4 makes for a session on its way out does not apply:
        nothing is going away, and an utterance left open is one the next answer
        is heard as the continuation of.

        The boundary is therefore published from a task this object owns, on
        both endings rather than only on the cancelled one. A cancellation
        landing *inside* the closing is the case that makes it worth doing
        twice over: awaited plainly it would take the terminal chunk down with
        it, and the utterance the cancellation was supposed to end would end
        nowhere. Shielded, this call gives up on the wait and the closing does
        not give up on the chunk.

        The lease arrives held — taken by the caller against the resource the
        chunks are drawn from, the synthesizer or the realtime provider — and
        is released when the loop ends: a close must not free that resource
        under a stream still producing into it. The terminal chunk needs no
        part of it; publishing the boundary leases the backend on its own.
        """
        room_id = playback.room_id
        try:
            async for chunk in chunks:
                # Re-read every chunk: synthesis awaits between them, and that
                # is the window a detach lands in.
                if playback.abandoned:
                    logger.info(
                        "Conference playback dropped after %d ms: channel left room %s",
                        playback.position_ms,
                        room_id,
                    )
                    return
                if playback.interrupted:
                    logger.info(
                        "Conference playback interrupted in room %s after %d ms",
                        room_id,
                        playback.position_ms,
                    )
                    break
                # The flag is read before the publication, but the publication
                # is an await of its own: a detach landing inside it would take
                # the bot out of the conference while this chunk is on its way
                # to that very session. Registering it as room activity is what
                # holds `leave()` back until the chunk is out — the flag decides
                # whether there is a next one.
                async with self._activity.track(room_id):
                    if playback.abandoned:
                        return
                    await self._publish_chunk(playback, bot, chunk)
                    # After the publication, and deliberately not before: what a
                    # recording of the conference holds is what the conference
                    # heard, and a chunk the backend refused was never said.
                    if self._on_published is not None:
                        await self._on_published(room_id, chunk)
        except asyncio.CancelledError:
            self._closing_task(playback, bot)
            raise
        except Exception:
            # A backend that refused a chunk has not ended the utterance: the
            # ones it accepted are still open on the track, and the caller
            # hearing about the failure does not close them. The boundary goes
            # out first and the refusal is reported after it — and if the
            # boundary is refused too, that is recorded against the room rather
            # than substituted for the failure the caller is owed.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._closing_task(playback, bot))
            raise
        finally:
            lease.release()
        # Shielded rather than awaited: a cancellation arriving here propagates
        # to the caller, and the chunk that ends the utterance still goes out.
        await asyncio.shield(self._closing_task(playback, bot))

    async def _publish_chunk(
        self, playback: ConferencePlayback, bot: BotSession, chunk: AudioChunk
    ) -> None:
        """Put one chunk on the bot track, on a task this object owns.

        Owned, because the answer matters after this call has stopped waiting
        for it. A backend that shields its own send publishes whatever this side
        does about being cancelled, so "was it published" is not a question the
        caller's own cancellation answers — and the two things that depend on
        the answer, whether a boundary is owed and in what format, are decided
        after the utterance has ended.

        Kept on the playback rather than awaited to completion here, because
        that is what lets the boundary wait for it: a terminal chunk that
        overtakes the audio it closes ends the utterance before its last words.

        Registered as the room's in-flight work in its own right, not through
        the caller's block. The caller's registration ends when the caller does,
        so a cancelled ``speak`` released it while this was still in the
        backend — and the teardown then drained a room it believed quiet and
        took the bot out from under a chunk on its way to that very session.
        """
        publishing = self._activity.spawn(playback.room_id, self._publish_on_backend(bot, chunk))
        playback.publishing = (chunk, publishing)
        try:
            await asyncio.shield(publishing)
        finally:
            playback.settle_publication()

    async def _settle_publication(self, playback: ConferencePlayback) -> bool:
        """Wait for the chunk still in flight, and take what became of it.

        Says whether the answer is known. A publication still running when the
        budget expires is one nothing can say happened or did not, and the
        boundary that follows depends on which: published, it would end an
        utterance whose last chunk has not arrived; withheld, it would leave the
        utterance open. Neither is safe to guess, so the caller is told the
        track is in an unknown state rather than given a wrong answer.
        """
        in_flight = playback.publishing
        if in_flight is None:
            return True
        _, publishing = in_flight
        try:
            await asyncio.wait_for(
                asyncio.shield(publishing), _conference_activity.DRAIN_TIMEOUT_S
            )
        except TimeoutError:
            logger.error(
                "A chunk of the conference bot's utterance in room %s is still being "
                "published after %.0fs. Nothing can say whether it arrived, so the "
                "utterance is not being closed on a guess",
                playback.room_id,
                _conference_activity.DRAIN_TIMEOUT_S,
            )
            return False
        except Exception:
            # A publication the backend refused. The utterance is whatever the
            # chunks before it made it, which `settle_publication` has kept.
            logger.warning(
                "The conference bot's last chunk in room %s was refused by the backend",
                playback.room_id,
            )
        playback.settle_publication()
        return True

    def _closing_task(self, playback: ConferencePlayback, bot: BotSession) -> asyncio.Future[None]:
        """Start publishing the chunk that ends an utterance, on a task we own.

        Owned rather than awaited inline because the caller may be cancelled at
        any point around it — during the synthesis, or during this very closing
        — and a boundary that is cancelled with it is one the backend never
        gets. The task outlives whatever happens to the caller.

        Marked ``closing`` for as long as it takes, which keeps the utterance on
        the room's books: a record forgotten now is one a detach can no longer
        abandon, and the boundary would be published into a session on its way
        out. It is dropped when the closing finishes, which is also what the
        next utterance in the room waits for.

        Registered as the room's in-flight work from the moment it exists, so a
        detach drains the boundary before taking the bot out rather than racing
        it — the caller that asked for this may be being cancelled, and its own
        registration goes with it.
        """
        playback.closing = True
        closing = self._activity.spawn(playback.room_id, self._close_utterance(playback, bot))
        self._closings.setdefault(playback.room_id, set()).add(closing)
        closing.add_done_callback(partial(self._closed, playback))
        closing.add_done_callback(log_task_exception)
        return closing

    def _closed(self, playback: ConferencePlayback, closing: asyncio.Future[None]) -> None:
        """Take what became of a closing, and drop the utterance it belonged to.

        A closing that *failed* is not a closing. Read as one — which dropping
        the task without looking at it did — the bot track goes back into use
        with an utterance still open on it, and the next answer arrives as its
        continuation: exactly the two obligations RFC section 12.10.4 states
        together, broken together. So the failure is kept on the room instead of
        on this playback, which is about to be forgotten, and it is what holds
        the track out of use until the session is replaced.
        """
        pending = self._closings.get(playback.room_id)
        if pending is not None:
            pending.discard(closing)
            if not pending:
                del self._closings[playback.room_id]
        if not closing.cancelled() and closing.exception() is not None:
            self._leave_unterminated(
                playback.room_id, f"{type(closing.exception()).__name__}: {closing.exception()}"
            )
        playback.closing = False
        self._forget(playback)

    def unterminated(self, room_id: str) -> str | None:
        """Why nothing further may be published on a room's bot track.

        ``None`` when it is usable. What an integrator whose AI has gone quiet
        reads instead of interpreting the silence — see
        :attr:`_RoomVoice.unterminated`.
        """
        room = self._speaking.get(room_id)
        return None if room is None else room.unterminated

    def _leave_unterminated(self, room_id: str, reason: str) -> None:
        """Record that the bot track has an utterance nothing managed to end.

        Durable, and per room rather than per utterance: what is unusable is the
        *track*, and it stays unusable until the bot session that owns it is
        replaced. An answer published onto it would be heard as the continuation
        of the one that never ended, and the boundary — if it ever lands — would
        cut the new one in half.
        """
        room = self._speaking.get(room_id)
        if room is None:
            room = self._speaking[room_id] = _RoomVoice()
        if room.unterminated is not None:
            return
        room.unterminated = reason
        logger.error(
            "The conference bot's utterance in room %s could not be closed (%s). Nothing "
            "further will be published on that track: what followed would be heard as the "
            "continuation of an utterance that never ended. Re-attaching the channel starts "
            "a new bot session and clears this",
            room_id,
            reason,
        )

    async def _close_utterance(self, playback: ConferencePlayback, bot: BotSession) -> None:
        """Tell the backend the utterance is over.

        ``is_final`` is the only thing that says so, and it is what a backend
        reconstructs the AI's turns from. Leaving the loop without it — which
        is what a barge-in did — leaves the SFU believing the bot is still
        mid-sentence, with no cancellation to tell it otherwise. An empty chunk
        in the format of the audio it closes carries the whole message, and
        costs the interface nothing.

        Not handed to ``on_published``: a recording holds what the conference
        heard, and there is no audio here to hear.

        Nothing to close where nothing was published, nothing to add where the
        synthesizer already ended on a final chunk, and nothing at all for a
        room the channel has left. That last one is the exception RFC section
        12.10.4 makes to the guarantee, and it makes it because the guarantee
        cannot be kept: a conference going away is not an utterance ending, the
        session this would name is on its way out, and no marker published into
        it would outrun the ``leave()`` behind it — any more than one would
        survive the process crashing. A backend ends the utterance on the
        session going away, which is the only place the boundary can honestly
        come from.

        Every other way an utterance can stop reaches here, cancellation
        included: the conference is still live in all of them, so the boundary
        is publishable and therefore owed.

        What is owed is decided only once the chunk still in flight has landed
        or failed. Decided before, the boundary is built from a chunk that may
        never arrive — and worse, published ahead of it: the utterance would end
        before its own last words. And a final chunk whose publication failed
        reads as an utterance already closed, which is how one stayed open with
        nothing left meaning to close it.
        """
        if not await self._settle_publication(playback):
            raise RuntimeError(
                f"a chunk of the utterance in room {playback.room_id!r} is still being "
                "published, so nothing can say what boundary the track is owed"
            )
        last = playback.published
        if last is None or last.is_final:
            return
        async with self._activity.track(playback.room_id):
            if playback.abandoned:
                return
            await self._publish_on_backend(
                bot,
                AudioChunk(
                    data=b"",
                    sample_rate=last.sample_rate,
                    channels=last.channels,
                    format=last.format,
                    is_final=True,
                ),
            )
            playback.published = None

    async def _publish_on_backend(self, bot: BotSession, chunk: AudioChunk) -> None:
        """Publish one chunk under a lease on the backend.

        The lease is what a close reads: a publication still inside the
        backend — a wedged network call, an SDK that shields its send — keeps
        the backend open until the call truly ends, however the caller's own
        waits and budgets fared. Closing the backend under it was the
        use-after-close every other resource is protected from (RFC 12.10.4).
        """
        with self._operations.use(
            ConferenceResource.BACKEND, what=f"publishing bot audio for session {bot.id}"
        ):
            await self._backend.publish_audio(bot, chunk)

    def _forget(self, playback: ConferencePlayback) -> None:
        """Drop a finished utterance, and the room's record once it is idle.

        Idle means no utterance speaking *and* none waiting: a room whose
        record went while an answer was still queued for its floor would hand
        the next arrival a different lock, and the two would publish together.

        An utterance still publishing its boundary is not finished. It stays
        until it is, so that a detach can still reach it — see
        :meth:`_close_on_cancel`.
        """
        if playback.closing:
            return
        room = self._speaking.get(playback.room_id)
        if room is None:
            return
        if playback in room.playbacks:
            room.playbacks.remove(playback)
        # A room whose track is unusable keeps its record even when idle: what
        # it holds is why, and the next answer has to be able to read it.
        if not room.playbacks and room.unterminated is None:
            del self._speaking[playback.room_id]

    async def _run_before_tts(self, room_id: str, text: str) -> str:
        """Let hooks block or rewrite what the bot is about to say.

        Synchronous by nature: this is where orchestration holds the bot silent
        while a handoff is pending. Deciding after the audio is on the wire
        would be too late.

        A hook may block or return a rewritten string, and one that raises
        blocks — BEFORE_TTS fails closed in the hook engine.
        """
        if self._framework is None:
            return text
        context = await self._framework._build_context(room_id)
        result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.BEFORE_TTS,
            text,
            context,
            skip_event_filter=True,
        )
        if not result.allowed:
            logger.info("Conference TTS blocked by hook: %s", result.reason)
            return ""
        return result.event if isinstance(result.event, str) else text

    async def _run_after_tts(self, room_id: str, text: str) -> None:
        """Notify hooks that the bot has spoken."""
        if self._framework is None:
            return
        context = await self._framework._build_context(room_id)
        await self._framework.hook_engine.run_async_hooks(
            room_id,
            HookTrigger.AFTER_TTS,
            text,
            context,
            skip_event_filter=True,
        )

    # -------------------------------------------------------------------------
    # Being interrupted
    # -------------------------------------------------------------------------

    async def consider_interruption(self, lane: ConferenceLane, speech_ms: float) -> None:
        """Decide whether a speaking participant interrupts the bot.

        Called for every frame of a participant's speech, because a strategy
        such as CONFIRMED only knows the answer once enough of it has
        accumulated. Once an utterance has interrupted, the playback is
        latched and the remaining frames cost a dictionary lookup.

        Only an utterance that has taken the floor can be interrupted, and the
        floor admits one at a time — so there is exactly one thing here to
        decide about. A room holding an answer that has not begun is a room
        where the bot is silent: between two utterances, or while BEFORE_TTS
        decides, and conversation in that gap is not somebody talking over
        anyone.

        A barge-in that does land silences the whole room, the answers waiting
        their turn included: a participant taking the floor is asking the room
        to go quiet, and letting the queue drain into the silence they just
        made is not what they asked for. They are silenced without an
        ON_BARGE_IN of their own — the event says what was cut off, and nothing
        was cut off from words nobody heard.
        """
        room = self._speaking.get(lane.room_id)
        if room is None:
            return
        pending = [p for p in room.playbacks if not p.interrupted]
        speaking = next((p for p in pending if p.speaking), None)
        if speaking is None:
            return
        if not self.may_interrupt(lane.participant_id):
            return
        decision = self._handler.evaluate(
            playback_position_ms=speaking.position_ms,
            speech_duration_ms=int(speech_ms),
        )
        if not decision.should_interrupt:
            return
        for playback in pending:
            playback.interrupted = True
        await self._stop_playback(speaking)
        await self._interrupt_upstream(lane.room_id)
        await self._fire_barge_in(lane, speaking)

    async def _interrupt_upstream(self, room_id: str) -> None:
        """Carry a landed barge-in to the speech-to-speech provider, if one is wired.

        Contained like :meth:`_stop_playback`, and for the same reason: the
        barge-in has already landed on this side of the backend, and a
        provider that could not cancel its generation has lost only audio the
        latch will never publish. ON_BARGE_IN still owes the room its event.
        """
        if self._on_interrupted is None:
            return
        try:
            await self._on_interrupted(room_id)
        except Exception:
            logger.warning(
                "The speech-to-speech provider could not be told about the barge-in in "
                "room %s; the response it is still generating will be discarded unheard",
                room_id,
                exc_info=True,
            )

    async def _stop_playback(self, playback: ConferencePlayback) -> None:
        """Tell the backend to drop the interrupted utterance's queued audio.

        The latch stops the loop at the next chunk; this is what stops the
        audio already past it, queued in the transport ahead of playout (RFC
        section 12.10.3). Best-effort: a backend that fails here has only kept
        its own queue, whose size bounds the overrun — the barge-in has
        already landed, and the closing chunk still follows either way.
        """
        bot = playback.bot
        if bot is None:
            return
        try:
            with self._operations.use(
                ConferenceResource.BACKEND, what=f"stopping bot playback for session {bot.id}"
            ):
                await self._backend.stop_playback(bot)
        except Exception:
            logger.warning(
                "The conference backend could not stop the bot's playback in room %s; the "
                "audio it had already queued will play out to the end of its buffer",
                playback.room_id,
                exc_info=True,
            )

    async def _fire_barge_in(self, lane: ConferenceLane, playback: ConferencePlayback) -> None:
        """Announce that a named participant cut the bot off.

        Registered as room activity because this runs on the lane's own task and
        hands integrator code the room: a handler that detaches the channel —
        "stop the meeting when someone talks over me" is a real policy — would
        otherwise run the teardown from inside the lane it is about to close,
        cancelling the chain it is standing on. Nested, the detach recognises
        itself and defers instead.
        """
        if self._framework is None:
            return
        context = await self._framework._build_context(lane.room_id)
        async with self._activity.track(lane.room_id):
            await self._framework.hook_engine.run_async_hooks(
                lane.room_id,
                HookTrigger.ON_BARGE_IN,
                ConferenceBargeIn(
                    room_id=lane.room_id,
                    track_id=lane.track_id,
                    participant_id=lane.participant_id,
                    interrupted_text=playback.text,
                    audio_position_ms=playback.position_ms,
                ),
                context,
                skip_event_filter=True,
            )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def forget_room(self, room_id: str) -> None:
        """Stop and drop every playback in a room when the channel leaves it.

        Dropping the entries alone would not stop anything: ``speak`` holds its
        own reference to the playback and to the bot session, so the loops would
        go on synthesizing and publishing into a conference the channel is no
        longer part of. Marking them is what the loops read — every one of them,
        the answers still waiting for the floor included, since one of those
        would otherwise be handed a room the channel has left.

        This is also where a track nothing could close becomes usable again.
        What made it unusable was an utterance left open on *that bot session*,
        and the channel is leaving the conference: the next attachment joins as
        a new session with a track of its own, on which nothing is open.
        """
        room = self._speaking.pop(room_id, None)
        if room is None:
            return
        for playback in room.playbacks:
            playback.abandoned = True

    async def interrupt_all(self) -> None:
        """Stop every room's playbacks the way a barge-in does, and wait it out.

        The unplug's ending, distinct from :meth:`abandon_all` on exactly the
        point RFC 12.10.4 turns on: the conference is live and the bot stays
        in it, so every utterance cut here still owes the backend its
        boundary. The latch stops each loop at its next chunk and the loop's
        own closing publishes the terminal chunk; ``stop_playback`` drops what
        the transport had already queued. The closings are then waited for on
        the usual budget, so the caller gets a track that is genuinely quiet
        rather than one still publishing its endings.
        """
        for room_id, room in list(self._speaking.items()):
            pending = [p for p in room.playbacks if not p.interrupted]
            speaking = next((p for p in pending if p.speaking), None)
            for playback in pending:
                playback.interrupted = True
            if speaking is not None:
                await self._stop_playback(speaking)
                await self._interrupt_upstream(room_id)
        await self._settle_closings()

    def abandon_all(self) -> None:
        """Stop every room's playbacks, without closing anything.

        Separate from :meth:`aclose` because the channel needs it *first* — the
        loops must stop publishing before the bot sessions go — while closing
        the synthesizer belongs at the end, after everything that might still
        be drawing on it.
        """
        for room in self._speaking.values():
            for playback in room.playbacks:
                playback.abandoned = True
        self._speaking.clear()

    async def aclose(self, *, close_provider: bool) -> None:
        """Stop every playback, and close the synthesizer when it is ours.

        A caller sharing one synthesizer across channels closes it itself,
        which is why the channel's ownership answer is passed in rather than
        assumed. The channel's own close passes ``False`` and closes the
        synthesizer through its shutdown coordinator instead, so the close
        waits on the synthesizer's leases like on every other resource's.

        The utterances a cancellation left to be closed are waited for first,
        on the usual budget. They are publishing into a conference the channel
        is about to leave, and one that lands after the bot has gone is a chunk
        addressed to nothing.
        """
        await self._settle_closings()
        self.abandon_all()
        if close_provider and self._tts is not None:
            await self._tts.close()

    async def close_tts(self) -> None:
        """Close the synthesizer. The coordinator's closer for the TTS resource."""
        if self._tts is not None:
            await self._tts.close()

    async def _settle_closings(self, room_id: str | None = None) -> bool:
        """Let the terminal chunks of cancelled utterances go out.

        One room's, or every room's when the channel is closing. Says whether
        all of them made it, which is what the next utterance in a room needs:
        the answer decides whether the track is safe to publish on.

        Bounded like every other wait this channel makes on code it does not
        own — the next answer must not queue behind a wedged backend for ever —
        and the deadline passing is a fact reported rather than glossed over.
        """
        groups = (
            list(self._closings.values())
            if room_id is None
            else [self._closings.get(room_id, set())]
        )
        pending = [closing for group in groups for closing in group if not closing.done()]
        if not pending:
            return True
        _, unfinished = await asyncio.wait(pending, timeout=_conference_activity.DRAIN_TIMEOUT_S)
        if unfinished:
            logger.warning(
                "%d conference utterance(s) were still being closed after %.0fs; the backend "
                "has not been told where they ended",
                len(unfinished),
                _conference_activity.DRAIN_TIMEOUT_S,
            )
        return not unfinished

"""The framework's single connection to one LiveKit room.

The session owns an ``rtc.Room`` and the state that hangs off it: which tracks
the framework asked for, which pumps are running, and the audio source the AI's
voice goes out on. The frame delivery itself is in ``_livekit_media``, which is
the one piece that reads none of this. The backend keeps the control plane and
one session per conference.

The **event bridge** here is load-bearing rather than incidental. LiveKit calls
its handlers synchronously and discards whatever they return, while the
framework's fanout is awaited — so handlers only enqueue, and a single consumer
task awaits the emissions in the order they were enqueued. Scheduling a task per
event instead would let a track's arrival overtake its publisher's, and a roster
asked to open a lane for a participant it has never seen has no good answer.

The pumps are deliberately *not* on that queue: see ``_livekit_media``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from roomkit.conference._livekit_mapping import (
    participant_record,
    quality_label,
    track_record,
)
from roomkit.conference._livekit_media import AudioSink, VideoSink, pump_audio, pump_video
from roomkit.conference._livekit_voice import BotVoiceTrack
from roomkit.conference.models import (
    BotSession,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.conference.livekit")


def _consume_exception(task: asyncio.Task[Any]) -> None:
    """Take a shared task's parting error so no waiter-less run warns."""
    if not task.cancelled():
        task.exception()


# How many control-plane events the bridge holds before declaring the session
# unhealthy. State events (active speaker, connection quality) coalesce to one
# entry per key and never accumulate; lifecycle events carry facts the roster
# and the lanes must not miss, so past this bound the session *ends* — through
# the same ``bot_session_ended`` contract as a dropped connection — rather
# than lose an arrival or a track silently. The channel's supervisor re-joins,
# and the new session's catch-up rebuilds a consistent view of the *current*
# state; what happened entirely inside the outage window is discarded with
# the session, counted, and named in the report (RFC 12.10.3).
MAX_QUEUED_EVENTS = 512

# How long to wait before re-attempting the disconnect an unhealthy-session
# teardown needs. Unlike the SFU-drop path, an overflow ends a session whose
# connection is still live: the end must not be *reported* until the old bot
# is genuinely out, or the re-join would seat a replacement beside it. The
# first attempt is immediate; these pace the retries.
OVERFLOW_DISCONNECT_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ConferenceEmissions:
    """The backend's callback fanout, handed to a session as plain functions.

    A session emits without holding the backend, so what it needs from the
    backend is stated here rather than discovered by reaching into it.
    """

    participant_joined: Callable[[str, ConferenceParticipant], Awaitable[None]]
    participant_left: Callable[[str, ConferenceParticipant], Awaitable[None]]
    track_published: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_unpublished: Callable[[str, ConferenceTrack], Awaitable[None]]
    track_audio: AudioSink
    track_video: VideoSink
    active_speaker_changed: Callable[[str, str], Awaitable[None]]
    connection_quality: Callable[[str, str, str], Awaitable[None]]
    bot_session_ended: Callable[[BotSession, str], Awaitable[None]]


class LiveKitBotSession:
    """One bot participant in one LiveKit room."""

    def __init__(
        self,
        *,
        rtc: Any,
        session: BotSession,
        config: Any,
        emissions: ConferenceEmissions,
    ) -> None:
        self._rtc = rtc
        self.session = session
        self._config = config
        self._emissions = emissions
        self._room: Any = rtc.Room()
        # The event bridge is bounded. The consumer awaits the framework's
        # fanout — identity resolution, hooks — so a participant generating
        # events faster than the fanout returns would otherwise grow this
        # without limit. State events never queue more than one entry each
        # (see `_put_state`); a lifecycle event that would not fit ends the
        # session rather than being lost (see `_overflow`).
        self._events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._pending_state: dict[Any, tuple[Callable[..., Awaitable[None]], tuple[Any, ...]]] = {}
        self._consumer: asyncio.Task[None] | None = None
        self._announced: set[str] = set()
        self._tracks: dict[str, ConferenceTrack] = {}
        self._publications: dict[str, Any] = {}
        self._wanted: set[str] = set()
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._voice = BotVoiceTrack(
            rtc=rtc,
            room=self._room,
            identity=session.identity,
            room_id=session.room_id,
            queue_ms=config.publish_queue_ms,
        )
        self._dominant_speaker: str | None = None
        self._left = False
        self._disconnected = False
        # The one disconnect in flight, whoever asked for it — see
        # `_disconnect_once` — and whether a *requested* leave() has asked:
        # an unhealthy end that loses that race reports nothing, because the
        # caller that requested the departure owns the books.
        self._disconnecting: asyncio.Task[None] | None = None
        self._leave_requested = False
        # The teardown task an SFU-side disconnect runs on; the callback that
        # learns of the loss is synchronous. Kept referenced until it ends.
        self._ender: asyncio.Task[None] | None = None

    @property
    def room_id(self) -> str:
        return self.session.room_id

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, url: str, token: str) -> None:
        """Join the room and report what is already there.

        Handlers and the consumer task are in place *before* the connect, so an
        event that lands during it is queued rather than dropped. The catch-up
        that follows is enqueued through the same queue for the same reason:
        ordering is what the queue is for, and a catch-up emitted directly would
        be the one thing that jumps it.
        """
        self._register_handlers()
        self._consumer = asyncio.create_task(self._consume())
        options = self._rtc.RoomOptions(auto_subscribe=False)
        await self._room.connect(url, token, options)
        local = self._room.local_participant
        self.session.id = local.sid or self.session.id
        if (joined_at := local.joined_at) is not None:
            self.session.joined_at = joined_at
        self._catch_up()

    def _catch_up(self) -> None:
        """Announce the participants that were here before the bot was.

        ``participant_connected`` fires for arrivals, and everyone already in
        the room is not an arrival — so without this a bot joining a meeting in
        progress sees an empty conference and subscribes to nothing. This is the
        first half of the race the mock cannot stage: the second half is a
        publisher that leaves between its track being announced and the
        subscription reaching the server, which :meth:`subscribe` handles.
        """
        for participant in self._room.remote_participants.values():
            self._enqueue_participant_joined(participant)
            for publication in participant.track_publications.values():
                self._enqueue_track_published(publication, participant)

    async def leave(self) -> None:
        """Disconnect, and take the in-flight utterance with the session.

        RFC section 12.10.4: an utterance the channel abandons because it is
        leaving is not closed by a terminal chunk — the session that chunk would
        name is on its way out. So the session going away *is* the boundary, and
        what that means here is that the queued audio goes unplayed and the
        track goes with it. Nothing is published on the way out.

        A disconnect the SDK refuses *propagates*. Swallowing it here reported
        a success to a channel whose entire departure bookkeeping — the leaving
        ledger, ``info()``'s bot_present, the close's final raise — exists to
        never misstate whether the bot is out of the meeting (RFC 12.10.4:
        failing to remove a session is failing to close). The local teardown
        that already ran stays torn down; a retry reattempts the disconnect
        alone, and only a disconnect that returned makes later calls no-ops.

        ``_leave_requested`` is set before the first await: an unhealthy end
        may be mid-disconnect on its own task, and the flag is how it learns —
        after the one shared disconnect returns — that a *requested* leave now
        owns the books and nothing spontaneous is to be reported.
        """
        self._leave_requested = True
        if self._disconnected:
            return
        if not self._left:
            self._left = True
            if self._voice.abandon_utterance():
                logger.debug(
                    "Conference bot %s left room %s mid-utterance; the session ends it",
                    self.session.identity,
                    self.room_id,
                )
            await self._stop_pumps()
            await self._voice.close()
        await self._disconnect_once()
        if self._consumer is not None:
            self._consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer
            self._consumer = None
        self._drain_events()

    async def _disconnect_once(self) -> None:
        """One disconnect on the wire at a time, shared by every path.

        ``leave()`` and an unhealthy end can both need the disconnect, and
        each may be suspended in it when the other arrives — two concurrent
        calls into the SDK, and two owners for one outcome. Single-flight:
        the first caller starts the call, later callers await the same one
        (shielded, so a budget cancelling *a caller* does not cancel the call
        the other still awaits). Success marks the session disconnected for
        everyone; failure propagates to every waiter and is not terminal —
        the next caller starts a fresh attempt.
        """
        if self._disconnected:
            return
        if self._disconnecting is None or self._disconnecting.done():
            self._disconnecting = asyncio.create_task(self._room.disconnect())
            self._disconnecting.add_done_callback(_consume_exception)
        await asyncio.shield(self._disconnecting)
        self._disconnected = True

    def _drain_events(self) -> int:
        """Drop whatever the consumer will never get to, and say how much.

        A departing session's remaining events describe a conference the
        framework has stopped listening to, and emitting them after the bot has
        gone would announce arrivals into a room that is being torn down. The
        count is returned because on the unhealthy path it is part of the
        report: these were facts, and they are being discarded.
        """
        undelivered = 0
        while not self._events.empty():
            self._events.get_nowait()
            undelivered += 1
        self._pending_state.clear()
        return undelivered

    async def _stop_pumps(self) -> None:
        pumps = list(self._pumps.values())
        self._pumps.clear()
        for pump in pumps:
            pump.cancel()
        for pump in pumps:
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    # -------------------------------------------------------------------------
    # Subscription — the framework's set is the authoritative one
    # -------------------------------------------------------------------------

    async def subscribe(self, track_id: str) -> None:
        """Ask LiveKit to start forwarding a track to the bot.

        A track that is no longer published is recorded as wanted and nothing
        else: the publisher left between the announcement and this call, which
        is ordinary in a conference and not a failure the channel should have to
        handle. Raising would turn every such race into an error, and the
        publication is gone for good — a republish arrives under a new sid.
        """
        self._wanted.add(track_id)
        publication = self._publications.get(track_id)
        if publication is None:
            logger.debug(
                "Track %s in room %s is not published, so there is nothing to subscribe to yet",
                track_id,
                self.room_id,
            )
            return
        publication.set_subscribed(True)

    async def unsubscribe(self, track_id: str) -> None:
        self._wanted.discard(track_id)
        if (publication := self._publications.get(track_id)) is not None:
            publication.set_subscribed(False)
        await self._stop_pump(track_id)

    async def _stop_pump(self, track_id: str) -> None:
        pump = self._pumps.pop(track_id, None)
        if pump is None:
            return
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump

    def publisher_identity(self, track_id: str) -> str | None:
        """Who publishes a track, for the moderation calls that need it.

        LiveKit's mute API is keyed on the participant *and* the track, while the
        interface passes only the track — so the answer has to come from
        somewhere, and this session watched it be published.
        """
        track = self._tracks.get(track_id)
        return None if track is None else track.participant_id

    def tracks(self) -> Mapping[str, ConferenceTrack]:
        return self._tracks

    # -------------------------------------------------------------------------
    # Publishing the AI's voice
    # -------------------------------------------------------------------------

    async def publish(self, chunk: AudioChunk) -> None:
        """Put one chunk of the AI's speech on the bot's track.

        The track keeps the publishing contract; what the session adds is that
        there has to be a session at all — a chunk arriving after the bot left
        has no track to go on, and the SDK's own error for that would say
        nothing about why.
        """
        if self._left:
            raise RuntimeError(
                f"the bot has left room {self.room_id!r}, so there is no track to publish on"
            )
        await self._voice.publish(chunk)

    # -------------------------------------------------------------------------
    # Event bridge — sync handlers in, ordered emissions out
    # -------------------------------------------------------------------------

    def _register_handlers(self) -> None:
        room = self._room
        room.on("participant_connected", self._enqueue_participant_joined)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("track_published", self._enqueue_track_published)
        room.on("track_unpublished", self._on_track_unpublished)
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("track_muted", self._on_track_muted)
        room.on("track_unmuted", self._on_track_unmuted)
        room.on("active_speakers_changed", self._on_active_speakers_changed)
        room.on("connection_quality_changed", self._on_connection_quality_changed)
        room.on("disconnected", self._on_disconnected)

    def _put(self, emit: Callable[..., Awaitable[None]], *args: Any) -> None:
        """Queue a lifecycle event — an arrival, a departure, a track.

        These carry facts the roster and the lanes must not miss, so they are
        never coalesced and never silently dropped: a queue that can no
        longer hold them ends the session instead (see :meth:`_overflow`).
        """
        if self._left:
            return
        if self._events.qsize() >= MAX_QUEUED_EVENTS:
            self._overflow()
            return
        self._events.put_nowait(("event", (emit, args)))

    def _put_state(self, key: Any, emit: Callable[..., Awaitable[None]], *args: Any) -> None:
        """Queue a state event, keeping only the latest value per key.

        Active speaker and connection quality are *states*, not facts: only
        the current value matters, and a consumer that fell behind should say
        the newest one, not replay the history. One marker per key sits in
        the queue; further updates replace the stored value in place, so a
        participant flapping quality cannot grow the queue at all.
        """
        if self._left:
            return
        already_queued = key in self._pending_state
        self._pending_state[key] = (emit, args)
        if already_queued:
            return
        if self._events.qsize() >= MAX_QUEUED_EVENTS:
            self._overflow()
            return
        self._events.put_nowait(("state", key))

    def _overflow(self) -> None:
        """The consumer has fallen unrecoverably behind: end the session.

        Evicting a lifecycle event would be worse than the memory it saves —
        an arrival or a publication lost here is a roster that lies and a
        track never transcribed, with nothing anywhere to say so, against the
        lifecycle MUSTs of RFC 12.10.4. A session whose view of the
        conference can no longer be trusted has one honest exit, and the
        contract already names it: the session ends, the loss is reported
        through ``bot_session_ended``, and the channel's re-join builds a
        fresh session whose catch-up announces the *current* truth.
        """
        logger.error(
            "The LiveKit event bridge for room %s overflowed at %d queued event(s): the "
            "framework's fanout is not keeping up with the conference, and going on would "
            "mean losing lifecycle facts silently. The bot session is being ended and "
            "re-joined for a consistent view",
            self.room_id,
            MAX_QUEUED_EVENTS,
        )
        # Admission closes, but the session is NOT marked disconnected: unlike
        # the SFU-drop path, this connection is still live, and the end must
        # not be reported — nor the registry emptied, nor a replacement
        # seated — until the disconnect has actually happened.
        self._left = True
        self._ender = asyncio.create_task(
            self._end_unhealthy(
                f"event queue overflow at {MAX_QUEUED_EVENTS} events; the session's view "
                "of the conference can no longer be trusted"
            )
        )

    async def _consume(self) -> None:
        """Await the queued emissions, in order, until cancelled.

        A subscriber that raises is the backend's own fanout problem and is
        logged there; anything that escapes it would otherwise kill this task
        and take the room's whole event stream with it, so it is caught here too.
        """
        while True:
            kind, payload = await self._events.get()
            if kind == "state":
                entry = self._pending_state.pop(payload, None)
                if entry is None:
                    continue
                emit, args = entry
            else:
                emit, args = payload
            try:
                await emit(*args)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Emitting a LiveKit conference event for room %s failed", self.room_id
                )

    def _enqueue_participant_joined(self, participant: Any) -> None:
        """Announce an arrival, once.

        The catch-up and the arrival event overlap: a participant that connects
        while :meth:`connect` is still returning is both reported as an arrival
        and already in the room the catch-up walks. Announcing it twice would
        have the roster resolve one person's identity twice and open their lane
        against a second announcement of the same track.
        """
        if participant.identity in self._announced:
            return
        self._announced.add(participant.identity)
        self._put(self._emissions.participant_joined, self.room_id, self._participant(participant))

    def _on_participant_disconnected(self, participant: Any) -> None:
        self._announced.discard(participant.identity)
        self._forget_tracks_of(participant.identity)
        self._put(self._emissions.participant_left, self.room_id, self._participant(participant))

    def _forget_tracks_of(self, identity: str) -> None:
        """Drop what a departing participant published.

        Nothing is emitted for it: LiveKit reports the tracks unpublished on its
        own where it does, and inventing the event where it does not would close
        a lane the channel already closed on the departure. What this is for is
        the bookkeeping — a stale entry here would answer
        :meth:`publisher_identity` for a track that no longer exists, and send a
        moderation call after someone who has left.
        """
        for track_id, record in list(self._tracks.items()):
            if record.participant_id != identity:
                continue
            self._tracks.pop(track_id, None)
            self._publications.pop(track_id, None)
            self._wanted.discard(track_id)
            if (pump := self._pumps.pop(track_id, None)) is not None:
                pump.cancel()

    def _participant(self, participant: Any) -> ConferenceParticipant:
        return participant_record(
            identity=participant.identity,
            sid=participant.sid,
            kind_name=self._rtc.ParticipantKind.Name(participant.kind),
            name=participant.name or "",
            metadata=participant.metadata or "",
            attributes=participant.attributes or {},
            connected_at=participant.joined_at,
        )

    def _enqueue_track_published(self, publication: Any, participant: Any) -> None:
        """Announce a published track, once — same overlap as an arrival."""
        if publication.sid in self._publications:
            return
        record = self._record_track(publication, participant)
        if record is None:
            return
        self._publications[record.id] = publication
        self._put(self._emissions.track_published, self.room_id, record)

    def _record_track(self, publication: Any, participant: Any) -> ConferenceTrack | None:
        try:
            record = track_record(
                sid=publication.sid,
                room_id=self.room_id,
                participant_id=participant.identity,
                kind_name=self._rtc.TrackKind.Name(publication.kind),
                source_name=self._rtc.TrackSource.Name(publication.source),
                muted=publication.muted,
                name=publication.name or "",
                mime_type=publication.mime_type or "",
            )
        except ValueError:
            logger.warning(
                "Ignoring LiveKit track %s in room %s: RoomKit has no kind for it",
                publication.sid,
                self.room_id,
                exc_info=True,
            )
            return None
        self._tracks[record.id] = record
        return record

    def _on_track_unpublished(self, publication: Any, participant: Any) -> None:
        track_id = publication.sid
        self._publications.pop(track_id, None)
        self._wanted.discard(track_id)
        record = self._tracks.pop(track_id, None)
        if record is None:
            return
        self._put(self._emissions.track_unpublished, self.room_id, record)

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        """Start a pump, but only for a track the framework asked for.

        The bot joins with ``auto_subscribe`` off, so this should only ever fire
        behind a :meth:`subscribe`. Should is not must — a subscription the SDK
        arranged on its own would deliver frames the framework never requested,
        which RFC section 12.10.3 forbids — so it is undone here rather than
        trusted.
        """
        track_id = publication.sid
        if track_id not in self._wanted:
            logger.warning(
                "LiveKit subscribed the bot to track %s in room %s without being asked; "
                "undoing it",
                track_id,
                self.room_id,
            )
            publication.set_subscribed(False)
            return
        record = self._tracks.get(track_id) or self._record_track(publication, participant)
        if record is None:
            return
        if track_id in self._pumps:
            return
        self._pumps[track_id] = asyncio.create_task(self._pump(record, track))

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        pump = self._pumps.pop(publication.sid, None)
        if pump is not None:
            pump.cancel()

    def _on_track_muted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, True)

    def _on_track_unmuted(self, participant: Any, publication: Any) -> None:
        self._set_muted(publication.sid, False)

    def _set_muted(self, track_id: str, muted: bool) -> None:
        """Keep the record's mute flag true to the publisher's own state.

        The interface has no mute event, so nothing is emitted. What this buys
        is that a caller reading ``ConferenceTrack.muted`` — a roster, a
        moderation view — reads the publisher's current state rather than
        whatever it was when the track appeared.
        """
        if (record := self._tracks.get(track_id)) is not None:
            record.muted = muted

    def _on_active_speakers_changed(self, speakers: list[Any]) -> None:
        """Report the dominant speaker, when it is a different one.

        LiveKit sends the whole active set, loudest first, and the interface
        carries one identity — so the loudest is the dominant one. An empty set
        means nobody is speaking, which the interface has no way to say, so it
        says nothing rather than naming a speaker who has stopped.
        """
        dominant = speakers[0].identity if speakers else None
        if dominant is None or dominant == self._dominant_speaker:
            self._dominant_speaker = dominant
            return
        self._dominant_speaker = dominant
        self._put_state("speaker", self._emissions.active_speaker_changed, self.room_id, dominant)

    def _on_connection_quality_changed(self, participant: Any, quality: Any) -> None:
        label = quality_label(getattr(quality, "name", str(quality)))
        if label is None:
            return
        self._put_state(
            ("quality", participant.identity),
            self._emissions.connection_quality,
            self.room_id,
            participant.identity,
            label,
        )

    def _on_disconnected(self, reason: Any = None) -> None:
        """The SFU dropped the bot, which ends the session in fact.

        Not the same as :meth:`leave`: nothing here was asked for, and there
        is nowhere to publish a boundary — a dropped connection announces
        nothing, any more than a crashed process would (RFC section 12.10.4).
        What there *is* somewhere to send is the fact itself: the session's
        local state is torn down on a task of its own — this callback is
        synchronous — and the end is reported through ``bot_session_ended``,
        because a loss the framework never hears about is a bot it reports
        present forever.
        """
        if self._voice.abandon_utterance():
            logger.warning(
                "LiveKit disconnected the conference bot in room %s mid-utterance (%s)",
                self.room_id,
                reason,
            )
        else:
            logger.info(
                "LiveKit disconnected the conference bot in room %s (%s)", self.room_id, reason
            )
        if self._left:
            return
        # Admission closes here, in the synchronous callback: a leave() racing
        # this teardown must find the session already ended, not tear it down
        # a second time beside it.
        self._left = True
        self._disconnected = True
        self._ender = asyncio.create_task(self._ended_by_sfu(str(reason)))

    async def _ended_by_sfu(self, reason: str) -> None:
        """Tear the session down after the SFU dropped it, and say so.

        The connection is already gone, so the disconnect attempt is a
        harmless best-effort no-op; what matters is the report. The overflow
        path is deliberately not this one — see :meth:`_end_unhealthy` — its
        connection is still live and the report has to wait for the
        disconnect.
        """
        await self._stop_pumps()
        await self._voice.close()
        with contextlib.suppress(Exception):
            await self._room.disconnect()
        await self._finish_end(reason)

    async def _end_unhealthy(self, reason: str) -> None:
        """End a live session whose view can no longer be trusted.

        The report comes *after* the disconnect, never before: reporting the
        session ended empties the backend's registry and seats a replacement,
        and doing that while the old connection is still up is two bots in
        one meeting. The disconnect is retried on a short backoff; a session
        whose disconnect will not go through is *kept* — on the backend's
        registry, on the channel's books, refusing a replacement — and said
        out loud. A later ``leave()`` (a detach, the close) retries the
        disconnect: failure is not terminal, exactly as in :meth:`leave`.
        """
        await self._stop_pumps()
        await self._voice.close()
        for attempt, delay in enumerate((0.0, *OVERFLOW_DISCONNECT_DELAYS_S)):
            if self._leave_requested or self._disconnected:
                # A requested leave() owns the books from the moment it asks;
                # nothing spontaneous is reported over it.
                return
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._disconnect_once()
            except Exception:
                logger.warning(
                    "Disconnecting the overflowed conference bot in room %s failed "
                    "(attempt %d); the session is not reported ended while the bot may "
                    "still be connected",
                    self.room_id,
                    attempt + 1,
                    exc_info=True,
                )
                continue
            # Re-read the owner *after* the shared disconnect: a leave() that
            # arrived while the call was on the wire shared its outcome, and
            # the outcome is the leave's to book, not this path's to report.
            if self._leave_requested:
                return
            await self._finish_end(reason)
            return
        logger.error(
            "The overflowed conference bot in room %s could not be disconnected after "
            "%d attempt(s). The session is being kept — on the backend's registry and "
            "the channel's books, holding the room's conference slot — rather than "
            "reported ended beside a live connection. A detach or the channel close "
            "retries the disconnect",
            self.room_id,
            len(OVERFLOW_DISCONNECT_DELAYS_S) + 1,
        )

    async def _finish_end(self, reason: str) -> None:
        """Stop the bridge and report the session's end, loss counted."""
        if self._consumer is not None:
            self._consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer
            self._consumer = None
        undelivered = self._drain_events()
        if undelivered:
            reason = (
                f"{reason}; {undelivered} queued event(s) were discarded undelivered — "
                "what happened while the consumer was stalled is not recoverable"
            )
        await self._emissions.bot_session_ended(self.session, reason)

    # -------------------------------------------------------------------------
    # Media — one pump task per subscribed track, delivered by _livekit_media
    # -------------------------------------------------------------------------

    async def _pump(self, record: ConferenceTrack, track: Any) -> None:
        """Run a track's pump, and survive its ending either way.

        A pump that raises is one track's stream failing, and it must not take
        the session's other tracks or its event bridge with it — so it is
        reported here and the task ends.
        """
        try:
            if record.kind is TrackKind.AUDIO:
                await pump_audio(
                    rtc=self._rtc,
                    track=track,
                    record=record,
                    sink=self._emissions.track_audio,
                    sample_rate=self._config.audio_sample_rate,
                    channels=self._config.audio_channels,
                )
            else:
                await pump_video(
                    rtc=self._rtc,
                    track=track,
                    record=record,
                    sink=self._emissions.track_video,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "The pump for conference track %s in room %s stopped", record.id, self.room_id
            )

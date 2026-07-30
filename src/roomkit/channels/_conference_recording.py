"""Per-track recording for ConferenceChannel.

One recording per track, opened on that track's first frame and attributed to
the participant publishing it. A conference gains and loses participants while
it runs, so a single recording holding every track would have to admit one
mid-write — which the usual containers refuse. Per-track recordings never ask
that of the recorder, and they make "who said what" a property of the output
rather than of a mixing decision.

The bot's own track is one of them. It is the only track that resembles the
outbound direction of a call, and recording it as one would be the single place
the framework mixes media it was handed separately.

See RFC section 12.10.8.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from roomkit.channels import _conference_activity
from roomkit.channels._conference_recording_writer import TrackWriter
from roomkit.recorder.base import (
    MediaRecordingConfig,
    MediaRecordingResult,
    RecordingTrack,
    pcm_codec,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.conference.models import ConferenceRecordingConfig, ConferenceTrack
    from roomkit.recorder.base import MediaRecorder, MediaRecordingHandle
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.channels.conference")


@dataclass(frozen=True)
class TrackFormat:
    """The PCM format one track's audio is in.

    Carried alongside the bytes because nothing else says: a recorder is handed
    frames it cannot tell apart by inspection, and a conference is exactly where
    two tracks need not agree — participants negotiate their own format with the
    SFU (RFC section 12.10.3), so a meeting of three may carry three.

    One type for the two things a conference records. A participant's audio
    arrives as an :class:`~roomkit.voice.audio_frame.AudioFrame`, which counts
    bytes per sample; the bot's own arrives as an
    :class:`~roomkit.voice.base.AudioChunk`, which names its codec outright.
    They are the same statement made twice, and the recording only needs it once.
    """

    sample_rate: int
    channels: int
    codec: str

    @classmethod
    def of_frame(cls, frame: AudioFrame) -> TrackFormat:
        """The format of a subscribed track's frame."""
        return cls(
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            codec=pcm_codec(frame.sample_width),
        )

    @classmethod
    def of_chunk(cls, chunk: AudioChunk) -> TrackFormat:
        """The format of a chunk the bot published, as the chunk declares it."""
        return cls(sample_rate=chunk.sample_rate, channels=chunk.channels, codec=chunk.format)

    @classmethod
    def of_track(cls, track: RecordingTrack) -> TrackFormat:
        """The format a recording was opened on, read back off its track."""
        return cls(
            sample_rate=track.sample_rate or 0,
            channels=track.channels or 1,
            codec=track.codec,
        )

    def describes(self, track: RecordingTrack) -> bool:
        """Whether a recording opened on ``track`` can carry audio in this format."""
        return (
            track.sample_rate == self.sample_rate
            and track.channels == self.channels
            and track.codec == self.codec
        )

    def __str__(self) -> str:
        return f"{self.sample_rate} Hz, {self.channels} ch, {self.codec}"


@dataclass
class TrackRecording:
    """One track's recording, and what the recorder needs to be told about it."""

    room_id: str
    track: RecordingTrack
    writer: TrackWriter
    """The recording itself: what opens it, writes it and closes it.

    Carried on this record rather than looked up by track, so it travels with
    the value :meth:`ConferenceRecording.detach_room` hands to a deferred
    teardown — which is what keeps that teardown from closing a recording a
    re-attach has since opened under the same track id.
    """

    @property
    def handle(self) -> MediaRecordingHandle | None:
        """What the recorder opened, or ``None`` while it still might not.

        Opening is asynchronous — it blocks for as long as the storage takes,
        so it does not happen where the frames arrive — and a recording is a
        thing this holds before it is a thing the recorder has.
        """
        return self.writer.handle

    @property
    def name(self) -> str:
        """What to call this recording in a log, whether or not it has an id yet."""
        handle = self.handle
        return handle.id if handle is not None else f"(opening for track {self.track.id})"


@dataclass
class FinishedRecording:
    """A closed recording, paired with the track it held.

    The result names where the media was written; the track names whose it
    was. They are carried together because the second is not reliably in the
    first: ``MediaRecordingResult.tracks`` is what a recorder chose to report,
    and the interface does not oblige it to report anything, while this
    collaborator knew the attribution from the moment it opened the recording.
    """

    room_id: str
    track: RecordingTrack
    result: MediaRecordingResult


@dataclass(frozen=True)
class RecordingCloseReport:
    """What a recording close achieved, and what it has to say out loud.

    ``finished`` is what was finalized and is announceable. ``unfinished``
    names the recordings that were not — a finalization that failed, or one
    still running inside the recorder past its budget. The recorder's own
    fate is separate: ``recorder_retained`` says it was kept alive because
    calls the framework gave up on are still inside it, and
    ``recorder_close_error`` carries what its close raised or timed out on.
    None of these stay in the log alone: the channel's close reports them as
    its own failures (RFC 12.10.4).
    """

    finished: list[FinishedRecording]
    unfinished: list[str]
    recorder_retained: bool
    recorder_close_error: str | None


class ConferenceRecording:
    """The recordings a conference channel has open, keyed by track.

    Detaching and finishing are separate, exactly as they are for lanes: a
    detach takes a room's recordings out of the collection so nothing feeds
    them any more, and the teardown finishes them at the point the conference
    is destroyed rather than the point it stopped being fed. Handing the popped
    recordings to :meth:`finish` is what keeps a deferred teardown from
    finishing a recording that a re-attach has since opened under the same
    track id — which is exactly why they are passed by value and not looked up
    again by room.

    :meth:`feed` is synchronous and never awaits: it runs inside the backend's
    emission loop, which awaits its subscribers one after another, so anything
    slower there would delay the frames of every other participant. Nothing it
    reaches asks the recorder for anything — opening a recording blocks exactly
    as writing to it does, so the open happens on the writer's own task like the
    writes do (:mod:`roomkit.channels._conference_recording_writer`), and none
    of it on the loop's thread. The methods that end a recording do await,
    because they wait for those writes to land before the container closes over
    them (RFC section 12.10.8).

    Nothing here raises at a caller. :meth:`feed` is called from the frame
    callback, where the transcription lane is fed on the next line: an
    unwritable path or a full disk that propagated from here would stop a
    conference being transcribed because it could not be recorded, and the two
    have nothing to do with each other.

    ``on_opened`` is awaited when a track's recording has actually opened, which
    is where the started announcement is made from. It is a callback rather than
    a return value because there is no longer a moment on the delivery path at
    which the answer is known.
    """

    def __init__(
        self,
        *,
        recorder: MediaRecorder,
        config: ConferenceRecordingConfig,
        channel_id: str,
        max_queued_frames: int = 100,
        on_opened: Callable[[TrackRecording], Awaitable[None]] | None = None,
    ) -> None:
        self._recorder = recorder
        self._config = config
        self._channel_id = channel_id
        self._max_queued_frames = max_queued_frames
        self._on_opened = on_opened
        self._open: dict[str, TrackRecording] = {}
        # Writers whose closed recordings still have a call running inside the
        # recorder. Kept until they settle, because releasing the recorder is
        # the one thing that must not happen while any of them does — see
        # `_release_recorder`.
        self._unsettled: set[TrackWriter] = set()
        # Whether the recorder has been given back. A channel can be closed
        # twice, and the second must not free what the first decided to leak.
        self._released = False
        # Closing runs one at a time. Two concurrent closes each saw the other's
        # emptied collections as "nothing is running" — see `close`.
        self._closing = asyncio.Lock()
        # Tracks that have already been reported for changing format mid-stream,
        # so the log carries the fact once rather than fifty times a second.
        self._renegotiated: set[str] = set()
        # Frames never written, by room. Accumulated as recordings close, so
        # the answer survives the track that lost them: a participant who left
        # took their writer with them, and what it dropped is still part of
        # what this conference failed to record.
        self._dropped: dict[str, int] = {}

    def feed(self, track: ConferenceTrack, data: bytes, audio_format: TrackFormat) -> None:
        """Record one frame of a track, starting its recording on the first.

        Started here rather than at subscription because this is where the
        track's format is known: participants negotiate their own with the SFU,
        so two tracks in one conference need not agree, and a recording that had
        to guess would either resample or mislabel. What it costs is that a
        track nobody publishes on produces no file, which is the right outcome
        anyway.

        That format is declared once, and a frame arriving in another one is
        refused rather than written — see :meth:`_refuse`.

        The frame itself is queued rather than written, and the recording it is
        queued for may not exist yet: opening it is the recorder's work and the
        recorder blocks. It is stamped here, at arrival, because a recording is
        a timeline and neither the writer's lateness nor the container's is part
        of it.
        """
        recording = self._open.get(track.id)
        if recording is None:
            recording = self._start_recording(track, audio_format)
        elif not audio_format.describes(recording.track):
            self._refuse(recording, audio_format)
            return
        if recording.writer.refused:
            # A recorder that would not open this one is not asked again, and
            # the frames it will never take are not queued for it: the realistic
            # causes do not clear between two frames, and a backlog nothing
            # drains is a backlog that only drops.
            return
        recording.writer.submit(data, time.monotonic() * 1000)

    def _refuse(self, recording: TrackRecording, audio_format: TrackFormat) -> None:
        """Drop a frame whose format is not the one its recording was opened on.

        A container fixes its streams at the first write, so there is nowhere
        honest to put a frame that renegotiated: written anyway it is decoded as
        the format the header claims — stereo read as mono, 8-bit read as 16 —
        and the result is a file that opens, plays wrong and reports nothing.
        Refusing leaves a gap instead, which is a defect anyone can see (RFC
        section 12.10.8).

        Counted as loss for the same reason the backlog's evictions are: what a
        recording failed to write is the integrator's question, and the log is
        not an interface.
        """
        self._dropped[recording.room_id] = self._dropped.get(recording.room_id, 0) + 1
        if recording.track.id in self._renegotiated:
            return
        self._renegotiated.add(recording.track.id)
        logger.error(
            "Conference recording %s was opened on %s and track %s is now delivering %s. "
            "Those frames are not being recorded: a recording carries one format, and "
            "writing them into this one would produce a file that plays wrong and says "
            "nothing. They are counted as dropped frames",
            recording.name,
            TrackFormat.of_track(recording.track),
            recording.track.id,
            audio_format,
        )

    def _start_recording(
        self, track: ConferenceTrack, audio_format: TrackFormat
    ) -> TrackRecording:
        """Start the recording that carries one track. Asks the recorder nothing.

        What it decides is what the recording will be — the format is only known
        here, on the track's first frame — and the writer does the asking, on a
        task and then a thread of its own. Nothing is left on the frame
        callback's path: a container created there costs every other
        participant's delivery the time the storage takes, which is the same
        objection that moved the writes off it (RFC sections 12.10.4, 12.10.8).

        A record exists from this moment whether or not the recorder will
        accept it. That is what keeps a refusal from being retried on every
        frame: the track has its recording, the recording knows it was refused,
        and a re-published track is a new subscription with a new one.
        """
        media_track = RecordingTrack(
            id=track.id,
            kind=track.kind.value,
            channel_id=self._channel_id,
            participant_id=track.participant_id,
            codec=audio_format.codec,
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
        )
        recording = TrackRecording(
            room_id=track.room_id,
            track=media_track,
            writer=TrackWriter(
                recorder=self._recorder,
                config=MediaRecordingConfig(
                    storage=self._config.storage,
                    format=self._config.format,
                    audio_sample_rate=audio_format.sample_rate,
                ),
                track=media_track,
                room_id=track.room_id,
                max_queued_frames=self._max_queued_frames,
            ),
        )
        self._open[track.id] = recording
        recording.writer.start(partial(self._opened, recording))
        return recording

    async def _opened(self, recording: TrackRecording) -> None:
        """Say that a track's recording is open, now that it actually is."""
        logger.info(
            "Conference recording started: %s (track=%s, participant=%s, room=%s)",
            recording.name,
            recording.track.id,
            recording.track.participant_id,
            recording.room_id,
        )
        if self._on_opened is not None:
            await self._on_opened(recording)

    def detach_room(self, room_id: str) -> list[TrackRecording]:
        """Take a room's recordings out of the collection without finishing them."""
        return [self._open.pop(track_id) for track_id in self._track_ids(room_id)]

    async def finish(self, recordings: list[TrackRecording]) -> list[FinishedRecording]:
        """Finalize recordings already taken out of the collection.

        Each one on its own account. A recorder raises for reasons that belong
        to one track — a container that will not close, a disk that filled
        between two writes — and stopping at the first left every recording
        behind it open *and* took down the teardown that called this, which ends
        in the bot leaving the conference. So what could be finalized is
        returned, and what could not is logged and left out: a recording nothing
        managed to close has no result to announce.

        All of them at once, because each one carries its own budget and a
        detach waits for the whole of this before its bot leaves the conference:
        finalized in turn, a room of ten tracks holds the bot for ten times the
        deadline, which is the failure the deadline exists to bound. The handles
        are independent — RFC section 12.11 promises ordering *per* handle and
        nothing across them — so nothing is given up by closing them together,
        and the wait becomes one deadline rather than N.
        """
        finished = await asyncio.gather(*(self._finish_one(recording) for recording in recordings))
        return [one for one in finished if one is not None]

    async def close_track(self, track_id: str) -> FinishedRecording | None:
        """Finalize one track's recording, if it ever had one.

        Returns ``None`` for a track that was subscribed but never carried a
        frame, which is not an error: the recording is opened by the first
        frame, so a participant who stayed silent leaves nothing to close. A
        recording the recorder could not close returns ``None`` too — for the
        caller they are the same, since neither leaves anything to announce.
        """
        recording = self._open.pop(track_id, None)
        if recording is None:
            return None
        return await self._finish_one(recording)

    async def _finish_one(self, recording: TrackRecording) -> FinishedRecording | None:
        """Close one track's recording, and say where it went if it says.

        The whole of it belongs to the writer, which owns every call this
        recording's handle receives: the queued writes land first within a
        bounded budget, then the track is flushed and the container closed, each
        on the worker thread and never overlapping. What comes back is where the
        media was written, or ``None`` for a recording that never opened or
        would not close — the log has already said which.
        """
        # Listed as in use *before* the finalization rather than after it. A
        # writer being closed is a writer with calls in the recorder, and the
        # window between taking the recording out of `_open` and putting its
        # writer here is one a concurrent `close()` saw as "nothing is running"
        # — and freed the recorder in the middle of this call.
        self._unsettled.add(recording.writer)
        try:
            result = await recording.writer.aclose(timeout=_conference_activity.DRAIN_TIMEOUT_S)
        finally:
            # Kept only if it left something running. What a budget gave up on
            # is still inside the recorder, and a provider released while a
            # call is in it is a crash rather than an error where that provider
            # is a native muxer (RFC section 12.11) — but a recording that
            # closed cleanly has nothing to wait for, and keeping every writer
            # a long meeting ever had is a leak that grows with the number of
            # tracks it has seen.
            if not recording.writer.unsettled:
                self._unsettled.discard(recording.writer)
        # Forgotten with the recording it belonged to: a track published again
        # opens a recording on whatever format it carries then, and deserves to
        # be told about a mismatch of its own.
        self._renegotiated.discard(recording.track.id)
        self._dropped[recording.room_id] = (
            self._dropped.get(recording.room_id, 0) + recording.writer.dropped
        )
        if result is None:
            return None
        logger.info(
            "Conference recording stopped: %s (track=%s, %.1fs, %d bytes)",
            result.id,
            recording.track.id,
            result.duration_seconds,
            result.size_bytes,
        )
        return FinishedRecording(room_id=recording.room_id, track=recording.track, result=result)

    def _track_ids(self, room_id: str) -> list[str]:
        """Tracks recording in a room, as a snapshot safe to iterate over."""
        return [
            track_id for track_id, recording in self._open.items() if recording.room_id == room_id
        ]

    async def drain(self) -> None:
        """Wait until every open recording has written what it was given.

        The counterpart of :meth:`ConferenceLane.drain`, and there for the same
        reason: writing is asynchronous now, so "has this frame reached the
        recorder yet" is a question with a moment attached, and an observer —
        a test above all — needs somewhere to wait for that moment.
        """
        for recording in list(self._open.values()):
            await recording.writer.drain()

    def dropped_frames(self, room_id: str) -> int:
        """Frames a room's recordings never wrote, evicted or lost to a write.

        RFC sections 12.10.4 and 12.10.8 require the loss to be exposed rather
        than only logged: a recording with a hole in it that nothing reports
        reads as a defective recorder. What it counts is both causes together —
        a backlog that filled because the storage could not keep up, and writes
        the recorder refused. Which of the two it was is in the log; how much
        went missing is the question an integrator asks first.
        """
        live = sum(
            recording.writer.dropped
            for recording in self._open.values()
            if recording.room_id == room_id
        )
        return self._dropped.get(room_id, 0) + live

    async def close(self, *, close_recorder: bool) -> RecordingCloseReport:
        """Finalize everything still open, and release the recorder if it is ours.

        Finalizing is unconditional whatever the ownership answer is: a
        recording left open is a file nothing ever closed, and a container
        finalized by no one is not a recording. Releasing the recorder is the
        part a caller sharing it across channels keeps for itself.

        The report carries everything the caller has to say out loud rather
        than leave in this module's logs: what was finalized (the last moment
        anything can be said about those files), what could not be, and what
        became of the recorder itself — a close that only logged those turned
        a retained provider into a success (RFC 12.10.4).

        One at a time, because two of these at once is not a hypothetical: a
        caller closing a channel while a framework shutdown closes the same one
        is the ordinary way it happens. Run together, the second read the
        collections the first had already emptied — the recordings taken out to
        be finalized, the writers not yet listed as busy — concluded that
        nothing was running, and released the recorder into the middle of the
        first one's work. A flag cannot express that: what is needed is that
        the second close *waits*, and then finds the answers the first left.
        """
        async with self._closing:
            open_recordings = list(self._open.values())
            self._open.clear()
            results = await asyncio.gather(
                *(self._finish_one(recording) for recording in open_recordings)
            )
            finished: list[FinishedRecording] = []
            unfinished: list[str] = []
            for recording, result in zip(open_recordings, results, strict=True):
                if result is not None:
                    finished.append(result)
                elif recording.writer.unsettled:
                    unfinished.append(
                        f"track {recording.track.id}: its finalization is still running "
                        "inside the recorder"
                    )
                elif not recording.writer.refused:
                    unfinished.append(
                        f"track {recording.track.id}: the recorder could not finalize it"
                    )
            retained = False
            release_error: str | None = None
            if close_recorder:
                retained, release_error = await self._release_recorder()
            return RecordingCloseReport(
                finished=finished,
                unfinished=unfinished,
                recorder_retained=retained,
                recorder_close_error=release_error,
            )

    async def _release_recorder(self) -> tuple[bool, str | None]:
        """Give the recorder back, once nothing is still running inside it.

        Every wait a recording makes is bounded, because a teardown held open is
        a bot left in a conference — but what those budgets gave up on is still
        executing on a worker thread, and releasing the provider underneath it
        is a different kind of failure: for a native muxer, freeing the context
        a call is inside is a crash rather than an exception.

        So the two waits are separated. The recording's budget belongs to the
        teardown and ends when the bot leaves; this one belongs to the recorder
        and runs afterwards, when there is no longer a bot waiting on it. A
        recorder that still will not come back is left unreleased and said out
        loud: leaking it is a bounded cost, and calling ``close()`` into a
        running call is not.

        Released at most once, and only ever from the same decision that made
        it: a second ``close()`` on a channel is ordinary — a caller closing
        what a framework shutdown has already closed — and it must not be the
        one that finds an emptied set and frees a recorder the first close
        refused to.

        Returns ``(retained, error)``: whether the recorder was kept alive
        under calls still running inside it, and what its own close raised or
        timed out on when it did — the caller's close report says both, since
        a log line is not a result.
        """
        if self._released:
            return False, None
        if not await self._settle_calls():
            logger.error(
                "Not releasing the recorder of channel %r: calls the framework gave up on "
                "are still running inside it. It is being leaked rather than freed while "
                "something is using it",
                self._channel_id,
            )
            return True, None
        self._released = True
        try:
            # On a worker thread like everything else the recorder is asked:
            # releasing what a recording held is a file operation, and the loop
            # this runs on has a channel to finish closing. Bounded like every
            # other wait on this interface (RFC section 12.11): a recorder that
            # will not let go must not be what stops a channel closing.
            await asyncio.wait_for(
                asyncio.to_thread(self._recorder.close), _conference_activity.DRAIN_TIMEOUT_S
            )
        except TimeoutError:
            logger.error(
                "Releasing the recorder of channel %r is still running after %.1fs; the "
                "channel is closing without it",
                self._channel_id,
                _conference_activity.DRAIN_TIMEOUT_S,
            )
            budget = _conference_activity.DRAIN_TIMEOUT_S
            return False, f"recorder.close() did not return within {budget:.1f}s"
        except Exception as exc:
            # The files are written by now, and the channel closing behind
            # this still has a bot to take out of a conference.
            logger.exception("Could not release the recorder of channel %r", self._channel_id)
            return False, f"recorder.close() failed: {type(exc).__name__}: {exc}"
        return False, None

    async def _settle_calls(self) -> bool:
        """Wait for every abandoned recorder call. Says whether all of them returned.

        One budget for all of them rather than one each: a channel that recorded
        twenty tracks would otherwise pay twenty times over to close, and the
        writers are independent, so there is nothing to gain by asking them in
        turn.

        What did not settle is kept. Dropping it would make the *next* close
        find nothing outstanding and free the recorder on the strength of a
        question this one already answered no to — which is the call into a
        running muxer this exists to prevent, arrived at one step later.
        """
        writers = list(self._unsettled)
        if not writers:
            return True
        settled = await asyncio.gather(
            *(writer.settle(timeout=_conference_activity.DRAIN_TIMEOUT_S) for writer in writers)
        )
        self._unsettled = {
            writer for writer, done in zip(writers, settled, strict=True) if not done
        }
        return not self._unsettled

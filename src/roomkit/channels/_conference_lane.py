"""Per-track processing lanes for ConferenceChannel.

One lane per subscribed AUDIO track. A lane owns a queue and the task that
drains it, runs the frames through the shared AudioPipeline under its own
stream identity, and hands complete utterances back to the channel.

Two properties are the reason this exists rather than living inline in the
channel's ``on_track_audio`` callback:

- The backend awaits each callback in sequence (``ConferenceBackend._emit``),
  so any await in the callback delays the frames of *every* participant. A
  lane accepts a frame synchronously and returns.
- Speech is segmented by VAD before it reaches speech recognition, so one
  utterance produces one transcription rather than one per 20 ms frame.

See RFC section 12.10.4.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.channels import _conference_activity
from roomkit.channels._conference_backlog import TrackBacklog
from roomkit.voice.pipeline.vad.base import VADEventType

if TYPE_CHECKING:
    from roomkit.channels._conference_operations import OperationLease
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.pipeline.engine import AudioPipeline

logger = logging.getLogger("roomkit.channels.conference")

# Lane tasks that outlived their cancellation grace. Held strongly so a
# runaway provider's task cannot be garbage-collected while still pending —
# that ends as a "task was destroyed but it is pending" warning instead of
# the deferred stage-state release the abandonment promised.
_ABANDONED_TASKS: set[asyncio.Task[None]] = set()


@dataclass
class ConferenceTranscription:
    """What a lane produced, before it enters the room.

    Carried to ON_TRANSCRIPTION so a hook can identify the track and the
    participant, block the text, or rewrite it.
    """

    track_id: str
    participant_id: str
    room_id: str
    text: str


@dataclass
class ConferenceBargeIn:
    """A participant spoke over the bot and was allowed to interrupt it.

    Carried to ON_BARGE_IN. The interrupting participant is named, which is
    what distinguishes a conference barge-in from a 1:1 one (RFC 12.10.5) —
    ``BargeInEvent`` identifies a voice session, and a conference lane has
    none.
    """

    room_id: str
    track_id: str
    participant_id: str
    interrupted_text: str
    audio_position_ms: int


# Called for every frame a participant is speaking, with the speech duration so
# far. The channel decides whether that speech interrupts the bot.
SpeechCallback = Callable[["ConferenceLane", float], Awaitable[None]]

# Called once per utterance, with the accumulated speech and its sample rate.
UtteranceCallback = Callable[["ConferenceLane", bytes, int], Awaitable[None]]

# Called at the VAD's utterance boundaries: once when the track goes from
# silence to speech, once when the utterance closes. The channel announces
# them on the speech hooks — the real-time "who is speaking right now" a
# management interface reads (RFC 12.10.4).
EdgeCallback = Callable[["ConferenceLane"], Awaitable[None]]

# Called with every processed frame, synchronously — it runs on the lane's own
# task inside the frame path, so anything slow here stalls this lane's VAD and
# recognition behind it. The one consumer is the speech-to-speech mix (RFC
# 12.10.12), which appends to a ring buffer and returns.
FrameCallback = Callable[["ConferenceLane", "AudioFrame"], None]


class ConferenceLane:
    """One AUDIO track's processing lane.

    Frames enter through :meth:`submit`, which never blocks and never awaits.
    The lane's own task runs them through the pipeline stages keyed on the
    track id, and calls back into the channel when a participant is speaking
    and when an utterance completes.

    Backpressure: the backlog is bounded, and a full one drops its *oldest*
    frame to make room. A lane that falls behind — a slow speech recognizer is
    the realistic cause — stays close to live rather than accumulating an
    ever-growing delay, at the cost of a gap in the audio it hands on. Drops
    are counted on ``dropped_frames`` and logged. The bound and the policy live
    in :class:`~roomkit.channels._conference_backlog.TrackBacklog`, which the
    track's recording is fed through as well: a conference has one answer to
    "what does it discard under overload", not one per collaborator.
    """

    def __init__(
        self,
        *,
        track_id: str,
        room_id: str,
        participant_id: str,
        pipeline: AudioPipeline,
        on_speech: SpeechCallback,
        on_utterance: UtteranceCallback,
        on_speech_start: EdgeCallback,
        on_speech_end: EdgeCallback,
        on_frame: FrameCallback | None = None,
        max_queued_frames: int = 100,
        lease: OperationLease | None = None,
    ) -> None:
        self.track_id = track_id
        self.room_id = room_id
        self.participant_id = participant_id

        # The lane's hold on the shared pipeline and recognizer, released the
        # moment no task of this lane's can still be inside either — which for
        # an abandoned task is when it truly ends, not when it was given up
        # on. What keeps the channel from closing those providers under a
        # runaway recognizer call is this lease and nothing else.
        self._lease = lease
        self._pipeline = pipeline
        self._on_speech = on_speech
        self._on_utterance = on_utterance
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_frame = on_frame
        self._backlog: TrackBacklog[AudioFrame] = TrackBacklog(
            maxsize=max_queued_frames, on_overflow=self._report_overflow
        )
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._released = asyncio.Event()
        self._speaking = False
        self._speech_ms = 0.0

    @property
    def running(self) -> bool:
        """Whether the lane is still draining its queue."""
        return self._task is not None

    @property
    def dropped_frames(self) -> int:
        """Frames this lane never processed, because it was behind."""
        return self._backlog.dropped

    @property
    def released(self) -> bool:
        """Whether no task can still be using this lane's provider state."""
        return self._released.is_set()

    def start(self) -> None:
        """Begin draining the queue."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def submit(self, frame: AudioFrame) -> None:
        """Hand a frame to the lane. Never blocks, never awaits.

        This runs inside the backend's emission loop, which awaits its
        subscribers one after another: anything slower here would stall frame
        delivery for every other track in the conference.
        """
        self._backlog.submit(frame)

    def _report_overflow(self, dropped: int) -> None:
        logger.warning(
            "Conference lane %s is behind: dropped %d frame(s) for participant %s",
            self.track_id,
            dropped,
            self.participant_id,
        )

    async def drain(self) -> None:
        """Wait until every queued frame has been processed."""
        await self._backlog.join()

    async def aclose(self) -> bool:
        """Stop the lane and release the stage state its stream held.

        Stage state is keyed by stream and some of it is native memory, so a
        lane that ends without releasing leaks for every track the conference
        ever carried.

        A lane can be asked to close from inside its own task. The callbacks it
        makes run integrator code — ON_TRANSCRIPTION, ON_BARGE_IN — and a
        handler that detaches the channel puts the whole teardown on the causal
        chain that started in :meth:`_run`. Cancelling there would cancel the
        teardown along with the lane, leaving the bot in the conference;
        awaiting the task would be awaiting the caller. So that case stops the
        lane cooperatively instead: by the time the flag is read the frame has
        made its last pipeline call, and the loop ends rather than taking
        another.

        The wait after the cancellation is bounded. The task may be inside a
        provider — a recogniser mid-stream — and cancellation is a request a
        provider can swallow, so waiting until it lands is a teardown held for
        as long as the provider cares to take, and every channel behind this
        one held with it. A task that outlives the grace is abandoned and
        reported; the stream's stage state is *not* released underneath it —
        some of that state is native memory the task may still be touching —
        but the moment the task does end, it is.

        Returns:
            ``True`` when the task outlived its cancellation grace. Such a
            lane keeps its lease on the shared providers until the task truly
            ends, which is what holds their close off.
        """
        self._closed = True
        if self._released.is_set():
            return False
        task, self._task = self._task, None
        if task is None or task is asyncio.current_task():
            self._release_stream()
            return False
        task.cancel()
        _, pending = await asyncio.wait({task}, timeout=_conference_activity.CANCEL_GRACE_S)
        if not pending:
            self._release_stream()
            return False
        logger.error(
            "Conference lane %s did not stop within %.1fs of being cancelled — a provider "
            "inside it is not honouring cancellation. The lane is abandoned; its stage "
            "state stays held until the task actually ends, and is released then",
            self.track_id,
            _conference_activity.CANCEL_GRACE_S,
        )
        _ABANDONED_TASKS.add(task)
        task.add_done_callback(_ABANDONED_TASKS.discard)
        task.add_done_callback(self._release_when_the_task_ends)
        return True

    def _release_stream(self) -> None:
        """Release stage state once and open the provider-safety barrier."""
        if self._released.is_set():
            return
        try:
            self._pipeline.release_stream(self.track_id)
        finally:
            self._open_barrier()

    def _release_when_the_task_ends(self, task: asyncio.Task[None]) -> None:
        """Free the stream's stage state once its runaway task is truly over.

        The deferred half of :meth:`aclose`: releasing while the task still
        ran would free native stage memory under code that may be touching it.
        Best-effort by then — the pipeline itself may have been closed in the
        meantime — and the task's parting exception is consumed here so an
        abandoned lane does not end as an unretrieved-exception warning.
        """
        with contextlib.suppress(BaseException):
            if not task.cancelled():
                task.exception()
            self._release_stream()
        # A failing reset is already best-effort at this boundary. The lane
        # task has nevertheless ended, so no provider call is still in flight.
        self._open_barrier()

    def _open_barrier(self) -> None:
        """Say that nothing of this lane's is inside the shared providers."""
        self._released.set()
        if self._lease is not None:
            self._lease.release()

    async def _run(self) -> None:
        while not self._closed:
            frame = await self._backlog.get()
            try:
                if self._closed:
                    continue
                await self._process(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad frame must not silence the lane for the rest of the
                # meeting.
                logger.exception("Conference lane %s failed on a frame", self.track_id)
            finally:
                self._backlog.task_done()

    async def _process(self, frame: AudioFrame) -> None:
        """Run one frame through the stages and act on what the VAD said."""
        result = self._pipeline.process_inbound_stream(self.track_id, frame)
        event = result.vad_event

        # After the stages, so every track contributes to the mix in the
        # contract format; before the VAD gates anything, because the mix
        # wants the audio itself, not only the speech the VAD keeps.
        if self._on_frame is not None:
            self._on_frame(self, result.frame)

        if event is not None and event.type is VADEventType.SPEECH_START:
            self._speaking = True
            self._speech_ms = 0.0
            await self._on_speech_start(self)

        if self._speaking:
            self._speech_ms += _frame_duration_ms(result.frame)
            await self._on_speech(self, self._speech_ms)

        if event is None or event.type is not VADEventType.SPEECH_END:
            return

        self._speaking = False
        self._speech_ms = 0.0
        # Before the utterance is handed on, not after: recognition is a round
        # trip, and "they stopped speaking" is true now.
        await self._on_speech_end(self)
        if event.audio_bytes:
            await self._on_utterance(self, event.audio_bytes, result.frame.sample_rate)


def _frame_duration_ms(frame: AudioFrame) -> float:
    """How much time a frame carries, in milliseconds.

    AudioFrame validates its own format on construction, so the divisors here
    are known non-zero.
    """
    bytes_per_sample = frame.sample_width * frame.channels
    return (len(frame.data) / bytes_per_sample) / frame.sample_rate * 1000.0

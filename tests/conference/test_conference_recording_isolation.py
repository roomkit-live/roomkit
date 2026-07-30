"""What a conference recording is not allowed to cost (RFC §12.10.4, §12.10.8).

Recording and transcription are two things done with the same frame, and until
the write left the delivery callback they were one thing: the write ran where
the frame arrived, so a slow disk delayed every other participant's audio, and a
failing disk stopped the transcription of a conference it had nothing to say
about.

These are the properties that separate them again, and RFC §12.10.4 says the
first is checkable from outside — "by delaying recognition on one track and
measuring frame delivery on another". A blocking recorder is the delay; the
measurement is the wall time another track's frame spends waiting for the loop,
which is where a write done on it would put that frame.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import threading
from typing import TYPE_CHECKING

import pytest

from roomkit.channels import _conference_activity as activity_module
from roomkit.channels._conference_recording_writer import TrackWriter
from roomkit.core.exceptions import ConferenceCloseError
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.recorder.base import MediaRecordingConfig, MediaRecordingHandle, RecordingTrack
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conference.lane_audio import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    drain,
    drain_recordings,
    say,
    speech_frame,
)
from tests.conference.test_conference_recording import ROOM, _recording_conference

if TYPE_CHECKING:
    from collections.abc import Callable

SLOW = 0.25
"""Long enough that a delivery held behind a write is unmistakable."""


def _numbered_frame(marker: int) -> AudioFrame:
    """A frame whose bytes say which one it is, so order is checkable."""
    return AudioFrame(data=struct.pack("<h", marker) * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def _until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Yield to the loop until a predicate holds, or fail the test saying so."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"condition still false after {timeout}s")
        await asyncio.sleep(0.005)


class _BlockingRecorder(MockMediaRecorder):
    """A recorder whose writes block, and remember which thread they blocked.

    Two ways to block: a fixed sleep, or a gate the test opens when it is ready.
    Both are what a synchronous ``MediaRecorder`` looks like on slow storage —
    the interface offers no other kind.
    """

    def __init__(self, *, seconds: float = 0.0, gate: threading.Event | None = None) -> None:
        super().__init__()
        self._seconds = seconds
        self._gate = gate
        self.entered = threading.Event()
        self.threads: set[int] = set()

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        self.threads.add(threading.get_ident())
        self.entered.set()
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        elif self._seconds:
            # Blocking, not awaiting: the point is that it cannot yield.
            threading.Event().wait(self._seconds)
        super().on_data(handle, track, data, timestamp_ms)


class _BlockingLifecycle(MockMediaRecorder):
    """A recorder that blocks on opening and closing rather than on writing.

    The other half of the interface, and the half a queue for the frames does
    nothing about: creating a container and finalizing one are file operations
    that block for as long as the storage takes, exactly as a write does.
    """

    def __init__(self, *, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds
        self.threads: set[int] = set()

    def _block(self) -> None:
        self.threads.add(threading.get_ident())
        threading.Event().wait(self._seconds)

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        self._block()
        return super().on_recording_start(config)

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self._block()
        super().on_track_added(handle, track)

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self._block()
        super().on_track_removed(handle, track)

    def on_recording_stop(self, handle: MediaRecordingHandle):  # type: ignore[no-untyped-def]
        self._block()
        return super().on_recording_stop(handle)


class _FailingWrites(MockMediaRecorder):
    """A recorder that accepts a recording and then refuses every frame."""

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        raise OSError("No space left on device")


class _RefusingOpen(MockMediaRecorder):
    """A recorder that cannot open a recording at all."""

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        self.starts += 1
        raise OSError("No such file or directory: /nope")


class _RefusingTrack(MockMediaRecorder):
    """A recorder that opens a recording and then refuses its track."""

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        raise RuntimeError("this container will not carry that")


class _BlockingStop(MockMediaRecorder):
    """A recorder that will not finalize, and says when it was asked to.

    A muxer draining a deadlocked encoder, a network filesystem that stopped
    answering: from the framework there is one shape, ``on_recording_stop``
    that does not return.
    """

    def __init__(self, *, gate: threading.Event) -> None:
        super().__init__()
        self._gate = gate
        self.stopping = threading.Event()
        self.closed = False
        self.closed_while_stopping = False

    def on_recording_stop(self, handle: MediaRecordingHandle):  # type: ignore[no-untyped-def]
        self.stopping.set()
        self._gate.wait(timeout=10.0)
        return super().on_recording_stop(handle)

    def close(self) -> None:
        self.closed = True
        # The observation that matters: released while a call is inside it,
        # which for a native muxer is a crash rather than an error.
        self.closed_while_stopping = self.stopping.is_set() and not self._gate.is_set()


class _BlockingRelease(MockMediaRecorder):
    """A recorder that will not let go when the channel releases it."""

    def __init__(self, *, gate: threading.Event) -> None:
        super().__init__()
        self._gate = gate
        self.releasing = threading.Event()

    def close(self) -> None:
        self.releasing.set()
        self._gate.wait(timeout=10.0)


class _CountingClose(MockMediaRecorder):
    """A recorder that counts how many times it was released."""

    def __init__(self) -> None:
        super().__init__()
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


class _SlowOpenRecorder(MockMediaRecorder):
    """A recorder whose open blocks, and which records the order it was called in.

    The ordering this exists to check is between one recording's late open and
    the *provider's* own release: a container being created inside a recorder
    that has been freed is a crash rather than an error where the recorder wraps
    a native muxer.
    """

    def __init__(self, *, gate: threading.Event) -> None:
        super().__init__()
        self._gate = gate
        self.entered = threading.Event()
        self.order: list[str] = []

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        self.order.append("open_enter")
        self.entered.set()
        self._gate.wait(timeout=10.0)
        handle = super().on_recording_start(config)
        self.order.append("open_return")
        return handle

    def on_recording_stop(self, handle: MediaRecordingHandle):  # type: ignore[no-untyped-def]
        self.order.append("recording_stop")
        return super().on_recording_stop(handle)

    def close(self) -> None:
        self.order.append("recorder_close")


class _StopOrder(MockMediaRecorder):
    """Remembers how much had been written when each recording was closed."""

    def __init__(self) -> None:
        super().__init__()
        self.written_at_stop: list[int] = []

    def on_recording_stop(self, handle: MediaRecordingHandle):  # type: ignore[no-untyped-def]
        self.written_at_stop.append(len(self.chunks))
        return super().on_recording_stop(handle)


class TestLaneIsolation:
    async def test_a_slow_write_does_not_delay_another_tracks_frames(self) -> None:
        """The measurement RFC §12.10.4 names, applied to the recorder: one
        track's storage latency must not become every participant's delivery
        latency.

        Measured from the moment Bob's frame is *scheduled*, not from the moment
        its emission finally begins. A blocking write does not slow a delivery
        down, it stops it from starting: what a lane's frames lose to it is
        waiting for the loop, and a stopwatch started inside the emission would
        read zero while the conference stuttered.
        """
        recorder = _BlockingRecorder(seconds=SLOW)
        _, _, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        loop = asyncio.get_running_loop()

        # Alice's frame leaves a write to be done. Bob's is queued behind it in
        # the loop's own ready queue, which is exactly where the conference
        # would stall if the write were done there.
        await backend.simulate_audio(alice, speech_frame())
        started = loop.time()
        await asyncio.create_task(backend.simulate_audio(bob, speech_frame()))

        assert loop.time() - started < SLOW
        await _until(recorder.entered.is_set)
        assert recorder.threads

    async def test_a_write_does_not_run_on_the_loops_thread(self) -> None:
        """RFC §12.11 permits it and §12.10.8 requires it: every call in the
        recorder interface blocks, so a write made on the loop's thread delays
        everything else on that loop however it is queued.
        """
        recorder = _BlockingRecorder()
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        assert recorder.threads
        assert threading.get_ident() not in recorder.threads

    async def test_opening_a_recording_does_not_delay_another_tracks_frames(self) -> None:
        """The same measurement, applied to the other half of the interface.

        Queueing the writes was not the whole of it: the recording still had to
        be *created*, and it was created where the first frame arrived. A
        container opened on the delivery path costs every other participant the
        time the storage takes, once per track — which on a meeting that starts
        with everyone unmuting at once is once per participant, back to back.
        """
        recorder = _BlockingLifecycle(seconds=SLOW)
        _, _, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")
        loop = asyncio.get_running_loop()

        # Alice's first frame is the one that opens her recording. Bob's is
        # queued behind it in the loop's own ready queue, which is where the
        # conference would stall if the open were done there.
        await backend.simulate_audio(alice, speech_frame())
        started = loop.time()
        await asyncio.create_task(backend.simulate_audio(bob, speech_frame()))

        assert loop.time() - started < SLOW
        await _until(lambda: bool(recorder.threads), timeout=5.0)
        assert threading.get_ident() not in recorder.threads

    async def test_closing_a_recording_does_not_run_on_the_loops_thread(self) -> None:
        """A recording ends while the meeting runs on — a participant leaving
        halfway through is the ordinary case — so flushing the track and
        finalizing the container block the loop everything else is on.
        """
        recorder = _BlockingLifecycle(seconds=SLOW)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)
        recorder.threads.clear()

        await backend.simulate_track_unpublished(alice.id)

        assert recorder.results != [], "the recording was never closed"
        assert recorder.threads
        assert threading.get_ident() not in recorder.threads

    async def test_a_tracks_frames_are_written_in_the_order_they_arrived(self) -> None:
        """One writer per recording is what lets the recorder be promised its
        calls never overlap and never reorder (RFC §12.11) — a recording is a
        timeline, and a batch written out of order is not one.
        """
        _, channel, backend, recorder = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        for marker in range(20):
            await backend.simulate_audio(alice, _numbered_frame(marker))
        await drain_recordings(channel)

        assert [struct.unpack("<h", c.data[:2])[0] for c in recorder.chunks] == list(range(20))


class TestRecordingFailuresStayWithTheRecording:
    async def test_a_write_that_fails_does_not_stop_the_transcription(self) -> None:
        """The defect this suite exists for: an unwritable disk used to reach
        the frame callback before the lane did, so the meeting stopped being
        transcribed because it could not be recorded.
        """
        kit, channel, backend, _ = await _recording_conference(
            recorder=_FailingWrites(), stt=MockSTTProvider(transcripts=["bonjour"])
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, alice)
        await drain(channel, alice.id)

        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "bonjour"]

    async def test_what_a_failed_write_lost_is_counted_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Loss an integrator cannot see reads as a bad recorder rather than as
        a disk that filled up (RFC §12.10.8).
        """
        _, channel, backend, _ = await _recording_conference(recorder=_FailingWrites())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
            for _ in range(4):
                await backend.simulate_audio(alice, speech_frame())
            await drain_recordings(channel)

        assert channel.info()["rooms"][ROOM]["recording_dropped_frames"] == 4
        assert "Could not write" in caplog.text

    async def test_an_open_that_fails_is_not_retried_on_every_frame(self) -> None:
        """A path that cannot be written does not become writable between two
        frames, and retrying puts a failing file open back on the delivery path
        fifty times a second.
        """
        recorder = _RefusingOpen()
        kit, channel, backend, _ = await _recording_conference(
            recorder=recorder, stt=MockSTTProvider(transcripts=["bonjour"])
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, alice)
        await drain(channel, alice.id)

        assert recorder.starts == 1
        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "bonjour"]

    async def test_a_recording_opened_for_a_refused_track_is_closed_again(self) -> None:
        """Otherwise it is a container the recorder holds open for a track that
        will never arrive.
        """
        recorder = _RefusingTrack()
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        assert len(recorder.handles) == 1
        assert [handle.state for handle in recorder.handles] == ["stopped"]
        assert recorder.chunks == []

    async def test_a_track_gets_a_fresh_attempt_after_it_is_republished(self) -> None:
        """The failure belonged to a recording, not to the track for ever."""
        recorder = _RefusingOpen()
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        await backend.simulate_track_unpublished(alice.id)
        again = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(again, speech_frame())
        await drain_recordings(channel)

        assert recorder.starts == 2


class TestOverload:
    async def test_a_writer_that_falls_behind_drops_and_says_how_much(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bounded, oldest-first, counted — the three obligations RFC §12.10.8
        places on the write backlog, the same three §12.10.4 places on a lane's.
        """
        gate = threading.Event()
        recorder = _BlockingRecorder(gate=gate)
        _, channel, backend, _ = await _recording_conference(
            recorder=recorder, max_queued_frames=2
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        with caplog.at_level(logging.WARNING, logger="roomkit.channels.conference"):
            for marker in range(12):
                await backend.simulate_audio(alice, _numbered_frame(marker))
            await _until(recorder.entered.is_set)
            dropped = channel.info()["rooms"][ROOM]["recording_dropped_frames"]
            gate.set()
            await drain_recordings(channel)

        assert dropped > 0
        assert "is behind" in caplog.text
        # The oldest went, so what survived ends with the newest frames.
        written = [struct.unpack("<h", c.data[:2])[0] for c in recorder.chunks]
        assert written[-1] == 11
        assert len(written) == 12 - dropped

    async def test_a_recording_goes_on_after_it_has_dropped(self) -> None:
        """A gap is a gap, not the end of the recording."""
        gate = threading.Event()
        recorder = _BlockingRecorder(gate=gate)
        _, channel, backend, _ = await _recording_conference(
            recorder=recorder, max_queued_frames=2
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")

        for marker in range(12):
            await backend.simulate_audio(alice, _numbered_frame(marker))
        await _until(recorder.entered.is_set)
        gate.set()
        await drain_recordings(channel)

        await backend.simulate_audio(alice, _numbered_frame(99))
        await drain_recordings(channel)

        assert struct.unpack("<h", recorder.chunks[-1].data[:2])[0] == 99


class TestFlushBeforeFinalizing:
    async def test_what_is_queued_is_written_before_the_container_closes(self) -> None:
        """A recording closed over frames still in flight ends early and says
        nothing about it — worse than a truncated one, because it reads as
        complete.
        """
        recorder = _StopOrder()
        kit, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        for marker in range(10):
            await backend.simulate_audio(alice, _numbered_frame(marker))
        # Let the recording be announced and live first: a detach racing the
        # start announcement reads as a refusal, and drops rather than writes.
        await _until(lambda: recorder.chunks != [], timeout=5.0)

        await kit.detach_channel(ROOM, "conf")

        assert recorder.written_at_stop == [10]

    async def test_a_track_that_ends_flushes_before_its_recording_is_reported(self) -> None:
        """Same rule on the other path out: a participant who leaves mid-meeting
        has their recording finalized while the conference runs on.
        """
        recorder = _StopOrder()
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        for marker in range(10):
            await backend.simulate_audio(alice, _numbered_frame(marker))

        await backend.simulate_track_unpublished(alice.id)

        assert recorder.written_at_stop == [10]
        assert channel.info()["rooms"][ROOM]["recording_dropped_frames"] == 0


class TestClosingIsBoundedToo:
    """Every wait a recording makes is bounded, and the finalization was not.

    The teardown waits for a recording to close before the bot leaves the
    conference (RFC §12.10.4 step 5 runs after §12.10.8's flush), so a muxer
    that will not finalize kept the bot sitting in the meeting for as long as it
    took — which is to say, until the process ended. There is no length of time
    for which that is the better failure.
    """

    async def test_a_recorder_that_will_not_close_does_not_hold_the_bot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        kit, _, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        try:
            await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=5.0)

            assert recorder.stopping.is_set(), "the recording was never finalized"
            assert backend.bots == [], "a recorder that would not close kept the bot in"
        finally:
            gate.set()

    async def test_what_the_budget_gave_up_on_is_said(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A recording with no reported location is a file an integrator has to
        go looking for; saying so is the difference between that and a defect.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        kit, _, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())

        try:
            with caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"):
                await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=5.0)

            assert "is still running" in caplog.text
        finally:
            gate.set()


class TestTheRecorderIsNotFreedWhileItIsWorking:
    """A budget stops the framework waiting; it does not stop the thread.

    Whatever a budget gave up on is still executing inside the recorder, and
    ``close()`` is the one call that must not overtake it: freeing the context a
    call is inside is a crash rather than an exception where the recorder wraps
    a native muxer. So the recording's budget ends when the bot leaves, and the
    recorder's own wait happens afterwards, when nothing is queued behind it.
    """

    async def test_a_late_open_is_closed_before_the_recorder_is_released(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _SlowOpenRecorder(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await _until(recorder.entered.is_set)
        assert channel._recorder is not None
        writer = channel._recorder._open[alice.id].writer

        closing = asyncio.create_task(channel.close())
        # Released only once the framework has genuinely given up on the open,
        # which is the path this is about: the budget has expired, the thread
        # has not, and the recorder must not be freed underneath it. The late
        # close is the second call to enter the recorder for this writer.
        await _until(lambda: len(writer._in_recorder) > 1, timeout=5.0)
        gate.set()
        await asyncio.wait_for(closing, timeout=10.0)

        assert recorder.order == [
            "open_enter",
            "open_return",
            "recording_stop",
            "recorder_close",
        ], recorder.order

    async def test_a_close_that_will_not_settle_leaks_rather_than_frees(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Leaking a recorder is a bounded cost. Calling ``close()`` into a
        running call is not, so the framework refuses to and says why.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            with (
                caplog.at_level(logging.ERROR, logger="roomkit.channels.conference"),
                pytest.raises(ConferenceCloseError) as failure,
            ):
                await asyncio.wait_for(channel.close(), timeout=5.0)

            assert recorder.closed is False, "the recorder was freed while a call was in it"
            assert "not being released" in caplog.text
            # And not in the log alone: a retained recorder is a close that
            # failed, said in the close's own result (RFC 12.10.4).
            assert "recorder" in str(failure.value)
        finally:
            gate.set()


class TestClosingFromTheOpeningAnnouncement:
    """``ON_RECORDING_STARTED`` runs integrator code, and closing the channel
    from it is a real policy — a disclosure rule that will not be recorded ends
    the meeting.

    Announced from the task that owned the recording, that close came back to
    cancel and await the very task it was running on: a task awaiting itself,
    which surfaced as a RecursionError, a destroyed task and a frame that was
    never written.
    """

    async def test_closing_from_the_handler_does_not_recurse(self) -> None:
        kit, channel, backend, recorder = await _recording_conference()
        failures: list[BaseException] = []

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _close_it(event: object, ctx: object) -> None:
            try:
                await channel.close()
            except BaseException as error:  # noqa: BLE001 — the point is to see it
                failures.append(error)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await _until(lambda: recorder.results != [] or failures != [], timeout=10.0)

        assert failures == []
        assert [result.id for result in recorder.results] == [
            handle.id for handle in recorder.handles
        ]

    async def test_the_frame_that_opened_the_recording_is_still_written(self) -> None:
        """The frame waits out the announcement rather than racing it — no
        audio is captured before ON_RECORDING_STARTED has been heard
        (RFC 17.6) — and a handler that takes its time costs the recording
        nothing: the frame is buffered, and written once consent stood.
        """
        kit, channel, backend, recorder = await _recording_conference()
        released = asyncio.Event()

        @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
        async def _hold(event: object, ctx: object) -> None:
            await released.wait()

        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, _numbered_frame(7))
        for _ in range(20):
            await asyncio.sleep(0)
        assert recorder.chunks == [], "audio was captured before the announcement was heard"
        released.set()
        await _until(lambda: recorder.chunks != [], timeout=5.0)

        assert [struct.unpack("<h", c.data[:2])[0] for c in recorder.chunks] == [7]


class TestClosingTwice:
    """Closing a channel twice is ordinary — a caller closing what a framework
    shutdown has already closed — and the second one must not undo the first
    one's decision.

    The set of outstanding calls was emptied by the attempt rather than by the
    answer, so a close that *refused* to release the recorder left nothing
    behind for the next one to refuse on: the second close found no outstanding
    call and freed a recorder with a call still running in it.
    """

    async def test_the_second_close_does_not_free_a_working_recorder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            with pytest.raises(ConferenceCloseError) as first:
                await asyncio.wait_for(channel.close(), timeout=5.0)
            # The second close joins the first's terminal result — the one
            # shutdown already decided not to free the recorder, and a replay
            # must not be the call that finds an emptied ledger and frees it.
            with pytest.raises(ConferenceCloseError) as second:
                await asyncio.wait_for(channel.close(), timeout=5.0)

            assert second.value is first.value
            assert recorder.closed is False, "the second close freed it anyway"
        finally:
            gate.set()

    async def test_a_recorder_that_settles_is_released_once(self) -> None:
        """And the guard is not "never release": a clean close releases, and a
        second one does not do it again.
        """
        recorder = _CountingClose()
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        await channel.close()
        await channel.close()

        assert recorder.closes == 1


class TestTheChannelCloseIsBoundedToo:
    """RFC §12.11: every wait on this interface is bounded, the close included.

    `MediaRecorder.close()` was moved to a worker thread and then awaited
    without a deadline, so a recorder that would not let go stopped the channel
    — and the framework behind it — from closing at all.
    """

    async def test_a_recorder_that_will_not_be_released_does_not_hang_the_close(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.1)
        gate = threading.Event()
        recorder = _BlockingRelease(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            with pytest.raises(ConferenceCloseError) as failure:
                await asyncio.wait_for(channel.close(), timeout=3.0)

            assert recorder.releasing.is_set(), "the recorder was never asked to let go"
            # A close that gave up on the recorder's own close is not a
            # success, and says so rather than only logging it.
            assert "recorder.close() did not return" in str(failure.value)
        finally:
            gate.set()


class TestWhatTheChannelHoldsOnTo:
    """A writer is kept only while it has something running inside the recorder.

    Kept unconditionally, a long meeting accumulated one dead writer per track
    it had ever carried — a leak that grows with the history of the room rather
    than with what is happening in it.
    """

    async def test_finished_recordings_are_not_retained(self) -> None:
        _, channel, backend, _ = await _recording_conference()
        await backend.simulate_participant_joined(ROOM, "p-alice")

        for _ in range(8):
            track = await backend.simulate_track_published(ROOM, "p-alice")
            await backend.simulate_audio(track, speech_frame())
            await drain_recordings(channel)
            await backend.simulate_track_unpublished(track.id)

        assert channel._recorder is not None
        assert channel._recorder._unsettled == set()


class TestTheCloseBudgetIsNotPerTrack:
    """A detach waits for every recording to be finalized before its bot leaves.

    Finalized in turn, each one carrying its own budget, a room of N tracks held
    the bot for N times the deadline — which is the failure the deadline exists
    to bound, reached by adding up the bounds. The handles are independent (RFC
    §12.11 orders calls per handle and says nothing across them), so they close
    together.
    """

    async def test_many_tracks_cost_one_budget_not_one_each(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.2)
        tracks = 6
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        kit, channel, backend, _ = await _recording_conference(recorder=recorder)
        for index in range(tracks):
            await backend.simulate_participant_joined(ROOM, f"p-{index}")
            track = await backend.simulate_track_published(ROOM, f"p-{index}")
            await backend.simulate_audio(track, speech_frame())
        await drain_recordings(channel)

        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await asyncio.wait_for(kit.detach_channel(ROOM, "conf"), timeout=10.0)
            elapsed = loop.time() - started

            assert elapsed < 0.2 * tracks, f"the budget was paid per track: {elapsed:.2f}s"
            assert backend.bots == []
        finally:
            gate.set()


class TestTwoClosesAtOnce:
    """Two closes running *together*, which is how it actually happens: a
    caller closing a channel while a framework shutdown closes the same one.

    The first takes the recordings out of the collection before finalizing them,
    and a writer was only listed as busy once its finalization returned. Between
    those two moments a concurrent close saw an empty collection and an empty
    busy list, concluded nothing was running, and freed the recorder into the
    middle of the first one's `on_recording_stop`. A flag guards two closes in
    sequence; it cannot express "wait for the one already deciding".
    """

    async def test_a_concurrent_close_does_not_free_a_working_recorder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.2)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            first = asyncio.create_task(channel.close())
            await _until(recorder.stopping.is_set, timeout=5.0)
            # Inside the first close's finalization, which is exactly where the
            # second one used to find "nothing is running".
            second = asyncio.create_task(channel.close())
            # Real time rather than loop turns: what must not happen happens on
            # a worker thread, so the window has to be wide enough for one.
            await asyncio.sleep(0.2)

            assert recorder.closed_while_stopping is False, (
                "a concurrent close freed the recorder while a call was inside it"
            )

            gate.set()
            results = await asyncio.wait_for(
                asyncio.gather(first, second, return_exceptions=True), timeout=10.0
            )
            # The wedged finalization made the one shutdown a failed close,
            # and both callers were handed that same terminal result.
            assert all(isinstance(r, ConferenceCloseError) for r in results)
        finally:
            gate.set()

    async def test_a_track_finalization_holds_the_recorder_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A track ending mid-meeting while the channel closes is two
        finalizations, and the second must see the first.

        Holds the property rather than a past defect: the busy list used to be
        read only for calls the framework had *given up* on, and this path
        happened to be covered anyway — the writer is claimed before its
        finalization starts, and a finalization past its budget becomes an
        abandoned call. It is the cancelled-close case below that slipped
        through. This one is here so the coverage stops being accidental.

        The budget is generous on purpose: what must be reached is a
        finalization the framework is *still waiting for*, not one already past
        its deadline.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 2.0)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            # The track ends on its own, which finalizes its recording outside
            # any channel close.
            ending = asyncio.create_task(backend.simulate_track_unpublished(alice.id))
            await _until(recorder.stopping.is_set, timeout=5.0)
            closing = asyncio.create_task(channel.close())
            # Waited on the outcome rather than on a duration: the close either
            # decides not to release — which is the whole point — or releases,
            # and a fixed sleep only asks whether it got there in time.
            await _until(lambda: closing.done() or recorder.closed, timeout=10.0)

            assert recorder.closed_while_stopping is False, (
                "the recorder was freed while a track's finalization was inside it"
            )

            gate.set()
            results = await asyncio.wait_for(
                asyncio.gather(ending, closing, return_exceptions=True), timeout=10.0
            )
            assert not isinstance(results[0], BaseException)
            assert isinstance(results[1], ConferenceCloseError)
        finally:
            gate.set()

    async def test_a_cancelled_close_does_not_leave_the_recorder_free_to_take(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling a close does not cancel the thread it was waiting on. The
        call is still inside the recorder, so the next close must still see it.
        """
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", 0.2)
        gate = threading.Event()
        recorder = _BlockingStop(gate=gate)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await drain_recordings(channel)

        try:
            first = asyncio.create_task(channel.close())
            await _until(recorder.stopping.is_set, timeout=5.0)
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first
            # The cancelled caller abandoned only its wait — the one shutdown
            # is still running, and the second close joins it.
            second = asyncio.create_task(channel.close())
            await _until(lambda: second.done() or recorder.closed, timeout=5.0)

            assert recorder.closed_while_stopping is False, (
                "a close after a cancelled one freed the recorder mid-call"
            )

            gate.set()
            with pytest.raises(ConferenceCloseError):
                await asyncio.wait_for(second, timeout=10.0)
        finally:
            gate.set()


class _StageBlockingRecorder(MockMediaRecorder):
    """Blocks the open on one gate and the finalization on another.

    Which is what makes the *transition* between two stages of one close
    observable: the framework can be waiting on the open, watch it return, and
    be asked whether the recorder is idle in the instant before the finalization
    enters it.
    """

    def __init__(self, *, opening: threading.Event, stopping: threading.Event) -> None:
        super().__init__()
        self._opening = opening
        self._stopping = stopping
        self.entered_open = threading.Event()
        self.entered_stop = threading.Event()
        self.closed = False

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        self.entered_open.set()
        self._opening.wait(timeout=10.0)
        return super().on_recording_start(config)

    def on_recording_stop(self, handle: MediaRecordingHandle):  # type: ignore[no-untyped-def]
        self.entered_stop.set()
        self._stopping.wait(timeout=10.0)
        return super().on_recording_stop(handle)

    def close(self) -> None:
        self.closed = True


class TestSettlingIsQuiescenceNotASnapshot:
    """A close hands the recorder one call after another, so what is running
    changes *because* something finished.

    Waiting on the set as it stood and then returning is how a caller concludes
    the recorder is idle at the exact moment the next stage enters it: the open
    returns, the wait is satisfied, and the finalization starts into a provider
    that is about to be freed.

    The first test drives that where it is deterministic — the wait itself,
    asked while a close is between two stages. The second is the same property
    through the channel, and it passes either way: lining ``channel.close()`` up
    so that it is inside that wait at the moment the open returns is not
    something the public surface can be made to do on purpose. It is here so the
    coverage does not depend on that scheduling staying as it is.
    """

    async def test_the_wait_does_not_return_between_two_stages(self) -> None:
        opening = threading.Event()
        stopping = threading.Event()
        recorder = _StageBlockingRecorder(opening=opening, stopping=stopping)
        writer = TrackWriter(
            recorder=recorder,
            config=MediaRecordingConfig(storage="", format="wav"),
            track=RecordingTrack(id="tr-1", kind="audio", channel_id="conf"),
            room_id=ROOM,
            max_queued_frames=4,
        )
        writer.start()
        await _until(recorder.entered_open.is_set, timeout=5.0)

        try:
            # Nothing queued, so the close goes straight to waiting on the open
            # — and with a budget it will not exhaust, so it goes on to the
            # finalization rather than giving up and closing the recording late.
            closing = asyncio.ensure_future(writer.aclose(timeout=5.0))
            await asyncio.sleep(0)
            # Asked here, the only call in the recorder is that open — which is
            # the snapshot the old wait took and then trusted.
            settling = asyncio.ensure_future(writer.settle(timeout=5.0))
            await asyncio.sleep(0)
            opening.set()
            await _until(recorder.entered_stop.is_set, timeout=5.0)

            assert not settling.done(), (
                "the wait returned while the finalization was inside the recorder"
            )

            stopping.set()
            assert await asyncio.wait_for(settling, timeout=5.0) is True
            await asyncio.wait_for(closing, timeout=5.0)
        finally:
            opening.set()
            stopping.set()

    async def test_a_channel_close_does_not_free_the_recorder_mid_close(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        budget = 0.3
        monkeypatch.setattr(activity_module, "DRAIN_TIMEOUT_S", budget)
        opening = threading.Event()
        stopping = threading.Event()
        recorder = _StageBlockingRecorder(opening=opening, stopping=stopping)
        _, channel, backend, _ = await _recording_conference(recorder=recorder)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(alice, speech_frame())
        await _until(recorder.entered_open.is_set, timeout=5.0)

        try:
            # The track ends while the open is still blocked. Its close gives up
            # on the queued frame after the budget and then waits on the open,
            # which is the stage the channel's own close will be told about.
            ending = asyncio.create_task(backend.simulate_track_unpublished(alice.id))
            await asyncio.sleep(budget * 2)

            closing = asyncio.create_task(channel.close())
            await asyncio.sleep(0)
            # Letting the open through moves that close on to its finalization.
            # Both waits resolve on the same future, so this is the instant the
            # snapshot was wrong about.
            opening.set()
            await _until(recorder.entered_stop.is_set, timeout=5.0)

            assert recorder.closed is False, (
                "the recorder was freed between two stages of one close"
            )

            stopping.set()
            await asyncio.wait_for(asyncio.gather(ending, closing), timeout=10.0)
        finally:
            opening.set()
            stopping.set()

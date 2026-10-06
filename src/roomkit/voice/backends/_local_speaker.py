"""The local speaker as one persistent output stream (RFC §12.3.4).

:class:`~roomkit.voice.backends.local.LocalAudioBackend` plays every response,
realtime or streamed TTS, through one PortAudio output stream opened once and
kept open: its thread pulls queued PCM each block and plays silence when there
is none. An output stream opened per response starts at a new render-to-capture
delay each time, which the audio server then takes seconds to settle — an echo
canceller cannot follow a delay moving under it (measured on PipeWire: a quarter
of the echo left above -50 dBFS, against one twentieth with the delay held
still).

The speaker knows nothing of sessions or echo cancellation: after each block it
hands the backend a :class:`SpeakerBlock` saying what was played.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("roomkit.voice.local")

MAX_BUFFER_SECONDS = 30
_MAX_UNDERRUN_WARNINGS = 5


@dataclass(frozen=True, slots=True)
class SpeakerBlock:
    """One block the speaker played.

    Attributes:
        data: The whole block: the queued audio and the silence completing it.
        written: Bytes of queued audio in the block (0: silence only).
        drained: The response's last audio left the queue in this block, or a
            response that had no audio completed.
        flushed: The queue was flushed (a cut) since the previous block.
        queued: Audio is still queued after this block.
    """

    data: bytes
    written: int
    drained: bool
    flushed: bool
    queued: bool


class LocalSpeaker:
    """A persistent callback-driven output stream fed from a byte queue.

    Playback is primed: the speaker holds silence until ``prebuffer_ms`` of audio
    is queued, so the burst jitter of a realtime provider or a streamed TTS does
    not scatter gaps through a sentence. Priming is released early when the
    response is complete (short responses) or after ~100 ms without new audio,
    and re-armed on every underrun, so a starvation makes one re-prime.

    Args:
        sd: The ``sounddevice`` module.
        sample_rate: Playback sample rate (Hz).
        channels: Number of channels.
        block_duration_ms: Duration of each output block.
        device: Output device index or name (None = default).
        low_latency: Ask PortAudio for a low-latency stream (with an AEC: the
            reference is fed when a block is handed over, so the less waits
            between that and the speaker, the better). macOS always gets "high":
            CoreAudio's tiny "low" buffers underrun from Python callback jitter.
        prebuffer_ms: Audio to accumulate before playing (``0``: the first byte).
        on_block: Called on the audio thread after each block is filled.
        on_finished: Called on the audio thread when the stream stops, with
            whether this speaker stopped it (False: PortAudio did, e.g. the device
            went away).
    """

    def __init__(
        self,
        sd: Any,
        *,
        sample_rate: int,
        channels: int,
        block_duration_ms: int,
        device: int | str | None,
        low_latency: bool,
        prebuffer_ms: int,
        on_block: Callable[[SpeakerBlock], None],
        on_finished: Callable[[bool], None] | None = None,
    ) -> None:
        self._sd = sd
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = int(sample_rate * block_duration_ms / 1000)
        self._device = device
        self._latency = "low" if low_latency and sys.platform != "darwin" else "high"
        self._frame_width = channels * 2  # int16
        self._on_block = on_block
        self._on_finished = on_finished

        self._stream: Any = None  # sd.RawOutputStream
        # Which stream is current: a finished callback of a replaced stream,
        # late from PortAudio's thread, must not mark the new one finished.
        self._generation = 0
        self._finished = False
        self._closing = False

        self._lock = threading.Lock()
        self._queue: deque[bytes] = deque()
        self._offset = 0  # bytes consumed in the front chunk
        # Running total of queued bytes: an O(1) check in the callback.
        self._buffered = 0
        self._max_bytes = sample_rate * self._frame_width * MAX_BUFFER_SECONDS
        self._prebuffer_bytes = int(sample_rate * self._frame_width * prebuffer_ms / 1000)
        self._priming = True
        # Set by end_response(): drains a final buffer smaller than the prebuffer.
        self._response_complete = False
        # Missing end-of-response valve: after ~100 ms of priming with no new
        # audio, drain whatever is queued (the SIP pacer's accumulate timeout).
        self._prime_idle_blocks = 0
        self._prime_max_idle_blocks = max(1, 100 // block_duration_ms)
        # Set by flush(): silence until the next append, which a provider still
        # streaming the cut response must not refill.
        self._interrupted = False
        self._flushed = False
        self._underruns = 0

    # -- lifecycle -------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the stream is open and still running."""
        return self._stream is not None and not self._finished

    def open(self) -> None:
        """Open and start the stream; a no-op while it runs.

        A new stream starts from a clean queue, so a second caller while
        another response plays never clobbers the live buffer.
        """
        if self.is_open:
            return
        self._discard_stream()
        self._reset_queue(interrupted=False)
        self._generation += 1
        generation = self._generation
        self._finished = False
        self._closing = False
        stream = self._sd.RawOutputStream(
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._device,
            latency=self._latency,
            callback=self._callback,
            finished_callback=lambda: self._stream_finished(generation),
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream
        logger.info(
            "Speaker stream: rate=%dHz blocksize=%d latency=%s device=%s",
            self._sample_rate,
            self._blocksize,
            self._latency,
            self._device if self._device is not None else "default",
        )

    def close(self) -> None:
        """Stop the stream and drop the queued audio."""
        self._reset_queue(interrupted=self._interrupted)
        self._closing = True
        self._discard_stream()

    def _discard_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()
            stream.close()
        except Exception:  # noqa: S110
            logger.debug("Error closing the speaker stream", exc_info=True)

    def _stream_finished(self, generation: int) -> None:
        if generation != self._generation:
            return
        self._finished = True
        if self._on_finished is not None:
            self._on_finished(self._closing)

    @property
    def latency(self) -> float:
        """Seconds between a block leaving the queue and the device playing it."""
        if self._stream is None:
            return 0.0
        try:
            return max(0.0, float(self._stream.latency))
        except Exception:
            return 0.0

    # -- the queue -------------------------------------------------------------

    @property
    def underruns(self) -> int:
        """Mid-response starvations: the queue ran dry before the response ended."""
        return self._underruns

    @property
    def buffered_bytes(self) -> int:
        return self._buffered

    @property
    def max_bytes(self) -> int:
        """The queue's bound: 30 s of audio."""
        return self._max_bytes

    def append(self, data: bytes) -> tuple[int, bool]:
        """Queue as much of *data* as the bound allows.

        Returns the bytes queued, and whether this ended a cut's silence.
        """
        with self._lock:
            resumed = self._interrupted
            self._interrupted = False
            available = max(0, self._max_bytes - self._buffered)
            available -= available % self._frame_width
            accepted = data[:available]
            if accepted:
                self._queue.append(accepted)
                self._buffered += len(accepted)
            # New audio means a response is in flight: a stale end-of-response
            # must not release the priming gate early.
            self._response_complete = False
            self._prime_idle_blocks = 0
        return len(accepted), resumed

    def begin_response(self) -> None:
        """A new response starts: a previous cut no longer silences the speaker."""
        with self._lock:
            self._interrupted = False
            self._response_complete = False

    def end_response(self) -> None:
        """The response is complete: release a partial prebuffer.

        Ignored after a cut: providers end a response on barge-in too (e.g.
        Gemini's ``interrupted``), and that stale signal must not release the
        priming gate for the next one.
        """
        with self._lock:
            if not self._interrupted:
                self._response_complete = True

    def flush(self) -> int:
        """Drop the queued audio and play silence until the next append.

        Returns the number of chunks dropped. The next block reports
        ``flushed``.
        """
        with self._lock:
            queued = len(self._queue)
            self._reset_queue_locked(interrupted=True)
            self._flushed = True
        return queued

    def _reset_queue(self, *, interrupted: bool) -> None:
        with self._lock:
            self._reset_queue_locked(interrupted=interrupted)

    def _reset_queue_locked(self, *, interrupted: bool) -> None:
        self._queue.clear()
        self._offset = 0
        self._buffered = 0
        self._priming = True
        self._response_complete = False
        self._prime_idle_blocks = 0
        self._interrupted = interrupted

    # -- the audio thread ------------------------------------------------------

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Fill one block from the queue, silence for the rest, then report it."""
        if status:
            logger.warning("Speaker callback status: %s", status)
        needed = frames * self._frame_width
        with self._lock:
            flushed, self._flushed = self._flushed, False
            written, drained, underrun_no = self._fill(outdata, needed)
            queued = bool(self._queue)
        if written < needed:
            outdata[written:] = b"\x00" * (needed - written)
        if underrun_no and underrun_no <= _MAX_UNDERRUN_WARNINGS:
            logger.warning(
                "Speaker underrun #%d — buffer starved mid-response, re-priming",
                underrun_no,
            )
        self._on_block(SpeakerBlock(bytes(outdata), written, drained, flushed, queued))

    def _fill(self, outdata: Any, needed: int) -> tuple[int, bool, int]:
        """Copy queued audio into *outdata* (lock held).

        Returns the bytes written, whether the response drained, and the
        underrun's number when the queue starved mid-response (else 0).
        """
        drained = False
        # After a cut, never drain. While priming, hold silence until enough
        # audio is queued; both fall through so the block is still reported.
        draining = not self._interrupted
        if draining and self._priming:
            if self._response_complete and self._buffered == 0:
                # A response with no audio still ends.
                self._response_complete = False
                drained = True
            release = (
                self._buffered >= max(self._prebuffer_bytes, 1)
                or (self._response_complete and self._buffered > 0)
                or (self._buffered > 0 and self._prime_idle_blocks >= self._prime_max_idle_blocks)
            )
            if release:
                # Drain starts in this same callback — no wasted block.
                self._priming = False
                self._prime_idle_blocks = 0
            else:
                self._prime_idle_blocks += 1
                draining = False
        if not draining:
            return 0, drained, 0

        written = 0
        while written < needed and self._queue:
            chunk = self._queue[0]
            n = min(len(chunk) - self._offset, needed - written)
            outdata[written : written + n] = chunk[self._offset : self._offset + n]
            written += n
            self._offset += n
            if self._offset >= len(chunk):
                self._queue.popleft()
                self._offset = 0
        self._buffered -= written
        if written == needed:
            return written, drained, 0

        # The queue ran dry mid-block: re-arm the prebuffer. A complete response
        # ends here; anything else is a starvation, counted.
        self._priming = True
        self._prime_idle_blocks = 0
        if self._response_complete:
            self._response_complete = False
            return written, True, 0
        self._underruns += 1
        return written, drained, self._underruns

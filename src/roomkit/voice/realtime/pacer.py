"""OutboundAudioPacer — wall-clock pacing for RTP-based transports.

Absorbs jitter from AI providers that stream audio faster than real-time.
Pre-buffers ~80 ms at the start of each response, then paces subsequent
chunks slightly ahead of wall-clock rate with interruptible sleeps.

The pacer maintains a configurable *jitter headroom* — audio is sent a
fixed amount ahead of real-time so the remote endpoint's jitter buffer
always has a safety margin.  Without headroom, asyncio scheduling jitter
can cause the remote jitter buffer to underrun, producing choppy audio.

Pacing sleeps use ``asyncio.sleep()`` (a single ``call_later``) rather
than ``asyncio.wait_for(event.wait(), timeout)`` which creates and
cancels a Task per sleep.  At 50 frames/second the overhead difference
is significant — ``sleep()`` is ~10× lighter.  Interrupts are detected
at frame boundaries (≤ 20 ms latency), which is acceptable for barge-in.

Reusable by any transport that needs outbound pacing (SIP, WebRTC, etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger("roomkit.voice.realtime.pacer")

# Sentinel strings used as control signals on the queue
_RESPONSE_END = "RESPONSE_END"
_STOP = "STOP"


class OutboundAudioPacer:
    """Paces outbound audio to wall-clock rate with pre-buffering.

    Absorbs jitter from AI providers that stream faster than real-time.
    Reusable by any RTP-based transport.

    Args:
        send_fn: Async callable that sends audio bytes to the remote peer.
        sample_rate: Audio sample rate in Hz (e.g. 8000, 16000).
        channels: Number of audio channels (default 1).
        sample_width: Bytes per sample (default 2 for 16-bit PCM).
        prebuffer_ms: Milliseconds of audio to accumulate before first send.
        jitter_headroom_ms: Milliseconds of lead to maintain over wall-clock.
            The pacer sends audio this far ahead of real-time so the remote
            endpoint's jitter buffer always has a safety margin.  Set to 0
            for strict wall-clock pacing (not recommended for RTP).
    """

    def __init__(
        self,
        send_fn: Callable[[bytes], Awaitable[None]],
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        prebuffer_ms: float = 80,
        jitter_headroom_ms: float = 60,
    ) -> None:
        self._send_fn = send_fn
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._prebuffer_bytes = int(sample_rate * channels * sample_width * prebuffer_ms / 1000)
        self._bytes_per_second = sample_rate * channels * sample_width
        # 20 ms frame — the smallest chunk we pace individually
        self._frame_bytes = int(sample_rate * channels * sample_width * 0.02)
        # Jitter buffer headroom — audio stays this far ahead of wall-clock
        self._jitter_headroom = jitter_headroom_ms / 1000

        self._queue: asyncio.Queue[bytes | str] = asyncio.Queue()
        self._interrupt_event = asyncio.Event()
        self._response_done = asyncio.Event()
        self._response_done.set()  # No response in flight initially
        self._task: asyncio.Task[None] | None = None

    def push(self, audio: bytes) -> None:
        """Enqueue audio (non-blocking). Called from provider callback."""
        if audio:
            self._queue.put_nowait(audio)

    def end_of_response(self) -> None:
        """Signal end of AI response. Resets pacing for next response."""
        self._response_done.clear()
        self._queue.put_nowait(_RESPONSE_END)

    async def wait_for_response_done(self) -> None:
        """Wait until the current response has been fully flushed."""
        await self._response_done.wait()

    def interrupt(self) -> None:
        """Drain queue + wake sender. Called on speech_start for barge-in."""
        # Swap with a fresh queue to avoid busy-loop drain
        old_queue = self._queue
        self._queue = asyncio.Queue()
        # Drain the old queue to call task_done on pending items
        while not old_queue.empty():
            try:
                old_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._interrupt_event.set()
        self._response_done.set()  # Unblock any wait_for_response_done() waiters

    async def start(self) -> None:
        """Start the background sender task."""
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="outbound_audio_pacer"
        )

    async def stop(self) -> None:
        """Signal shutdown, await task with timeout, cancel if needed."""
        if self._task is None:
            return
        self._queue.put_nowait(_STOP)
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    # -- Internal sender loop --

    async def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep for *seconds*, returning True if interrupted.

        Uses ``asyncio.sleep()`` (single ``call_later``) instead of
        ``asyncio.wait_for(event.wait(), timeout)`` which creates and
        cancels a Task per call — ~10× lighter at 50 frames/sec.

        Interrupt is detected after waking, so worst-case latency is
        one sleep duration (~20 ms for frame pacing).
        """
        await asyncio.sleep(seconds)
        return self._interrupt_event.is_set()

    async def _send_paced(
        self,
        data: bytes,
        pace_start: float,
        cumulative: float,
    ) -> tuple[float, bool]:
        """Send *data* in frame-sized chunks with wall-clock pacing.

        Maintains ``_jitter_headroom`` seconds of lead over wall-clock so
        the remote jitter buffer always has a safety margin.

        Returns ``(updated_cumulative, interrupted)``.
        """
        headroom = self._jitter_headroom
        offset = 0
        while offset < len(data):
            if self._interrupt_event.is_set():
                return cumulative, True
            end = min(offset + self._frame_bytes, len(data))
            chunk = data[offset:end]
            offset = end
            try:
                await self._send_fn(chunk)
            except Exception:
                logger.exception("Error sending paced audio chunk")
                continue
            cumulative += len(chunk) / self._bytes_per_second
            ahead = cumulative - (time.monotonic() - pace_start)
            sleep_time = ahead - headroom
            if sleep_time > 0.002 and await self._interruptible_sleep(sleep_time):
                return cumulative, True
        return cumulative, False

    async def _run(self) -> None:
        """Background task: pre-buffer → burst → pace to wall-clock."""
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return

            if item == _STOP:
                return

            if item == _RESPONSE_END:
                # Nothing to flush — channel handles resampler flush
                self._response_done.set()
                continue

            # Clear interrupt flag at the start of each new audio burst
            self._interrupt_event.clear()

            audio: bytes = item  # type: ignore[assignment]

            # --- Pre-buffer phase: accumulate ~prebuffer_ms before first send ---
            buf = bytearray(audio)
            next_item: bytes | str | None = None
            interrupted = False

            while len(buf) < self._prebuffer_bytes:
                if self._interrupt_event.is_set():
                    interrupted = True
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except (TimeoutError, asyncio.CancelledError):
                    break
                if isinstance(next_item, str):
                    if next_item == _STOP:
                        # Send what we have, then exit
                        if buf:
                            with contextlib.suppress(Exception):
                                await self._send_fn(bytes(buf))
                        return
                    if next_item == _RESPONSE_END:
                        break
                else:
                    if next_item:
                        buf.extend(next_item)

            # If interrupted during pre-buffer, discard and wait for next response
            if interrupted or self._interrupt_event.is_set():
                self._interrupt_event.clear()
                continue

            # --- Send the pre-buffer burst (capped) then pace the overflow ---
            burst_end = min(len(buf), self._prebuffer_bytes)
            burst = bytes(buf[:burst_end])
            overflow = bytes(buf[burst_end:])

            if burst:
                try:
                    await self._send_fn(burst)
                except Exception:
                    logger.exception("Error sending pre-buffered audio")
                    continue

            burst_ms = len(burst) * 1000 / self._bytes_per_second
            overflow_ms = len(overflow) * 1000 / self._bytes_per_second
            logger.debug(
                "Pacer: prebuffer burst=%.0fms, overflow=%.0fms",
                burst_ms,
                overflow_ms,
            )

            # If we broke out due to RESPONSE_END, send overflow and loop back
            if next_item == _RESPONSE_END:
                if overflow:
                    with contextlib.suppress(Exception):
                        await self._send_fn(overflow)
                self._response_done.set()
                continue

            # Start wall-clock pacing from after the burst
            pace_start = time.monotonic()
            cumulative = len(burst) / self._bytes_per_second

            # Pace any overflow from the pre-buffer (large initial chunk)
            if overflow:
                cumulative, was_interrupted = await self._send_paced(
                    overflow, pace_start, cumulative
                )
                if was_interrupted:
                    self._interrupt_event.clear()
                    continue

            # --- Paced sending phase: process queue items ---
            underruns = 0
            while True:
                # Check how far ahead of wall-clock we are
                ahead = cumulative - (time.monotonic() - pace_start)
                # Wait for the next chunk with a timeout slightly beyond
                # our lead time.  If the queue runs dry, that's an underrun.
                wait_timeout = max(ahead + 0.1, 0.1)
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=wait_timeout)
                except asyncio.CancelledError:
                    return
                except TimeoutError:
                    # Queue ran dry — TTS/provider not streaming fast enough
                    underruns += 1
                    if underruns <= 5:
                        behind = -(cumulative - (time.monotonic() - pace_start))
                        logger.warning(
                            "Pacer underrun #%d: queue empty, %.0fms behind",
                            underruns,
                            behind * 1000,
                        )
                    continue

                if isinstance(next_item, str):
                    if next_item == _STOP:
                        return
                    if next_item == _RESPONSE_END:
                        if underruns:
                            logger.info("Pacer: response done, %d underruns", underruns)
                        self._response_done.set()
                        break  # back to outer loop for next response
                    continue

                # Check interrupt before sending
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()
                    break  # back to outer loop

                assert isinstance(next_item, bytes)
                audio = next_item
                if not audio:
                    continue

                # Pace large items frame-by-frame, small items in one shot
                if len(audio) > self._frame_bytes * 2:
                    cumulative, was_interrupted = await self._send_paced(
                        audio, pace_start, cumulative
                    )
                    if was_interrupted:
                        self._interrupt_event.clear()
                        break
                else:
                    try:
                        await self._send_fn(audio)
                    except Exception:
                        logger.exception("Error sending paced audio")
                        continue

                    # Wall-clock pacing
                    chunk_s = len(audio) / self._bytes_per_second
                    cumulative += chunk_s
                    ahead = cumulative - (time.monotonic() - pace_start)
                    sleep_time = ahead - self._jitter_headroom
                    if sleep_time > 0.002 and await self._interruptible_sleep(sleep_time):
                        self._interrupt_event.clear()
                        break  # back to outer loop

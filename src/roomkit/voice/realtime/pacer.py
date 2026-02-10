"""OutboundAudioPacer — wall-clock pacing for RTP-based transports.

Absorbs jitter from AI providers that stream audio faster than real-time.
Pre-buffers ~150 ms at the start of each response, then paces subsequent
chunks to wall-clock rate with interruptible sleeps.

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
    """

    def __init__(
        self,
        send_fn: Callable[[bytes], Awaitable[None]],
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        prebuffer_ms: float = 150,
    ) -> None:
        self._send_fn = send_fn
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._prebuffer_bytes = int(sample_rate * channels * sample_width * prebuffer_ms / 1000)
        self._bytes_per_second = sample_rate * channels * sample_width

        self._queue: asyncio.Queue[bytes | str] = asyncio.Queue()
        self._interrupt_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def push(self, audio: bytes) -> None:
        """Enqueue audio (non-blocking). Called from provider callback."""
        if audio:
            self._queue.put_nowait(audio)

    def end_of_response(self) -> None:
        """Signal end of AI response. Resets pacing for next response."""
        self._queue.put_nowait(_RESPONSE_END)

    def interrupt(self) -> None:
        """Drain queue + wake sender. Called on speech_start for barge-in."""
        # Drain synchronously — no await needed from caller
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._interrupt_event.set()

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

            # Send the pre-buffered burst
            if buf:
                try:
                    await self._send_fn(bytes(buf))
                except Exception:
                    logger.exception("Error sending pre-buffered audio")
                    continue

            # If we broke out due to RESPONSE_END, loop back for next response
            if next_item == _RESPONSE_END:
                continue

            # Start wall-clock pacing from after the burst
            pace_start = time.monotonic()
            cumulative = len(buf) / self._bytes_per_second

            # --- Paced sending phase ---
            while True:
                try:
                    next_item = await self._queue.get()
                except asyncio.CancelledError:
                    return

                if isinstance(next_item, str):
                    if next_item == _STOP:
                        return
                    if next_item == _RESPONSE_END:
                        break  # back to outer loop for next response
                    continue

                # Check interrupt before sending
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()
                    break  # back to outer loop

                audio = next_item  # type: ignore[assignment]
                if not audio:
                    continue

                try:
                    await self._send_fn(audio)
                except Exception:
                    logger.exception("Error sending paced audio")
                    continue

                # Wall-clock pacing — interruptible via the event
                chunk_s = len(audio) / self._bytes_per_second
                cumulative += chunk_s
                ahead = cumulative - (time.monotonic() - pace_start)
                if ahead > 0.005:
                    try:
                        await asyncio.wait_for(self._interrupt_event.wait(), timeout=ahead)
                        # Event fired → interrupted
                        self._interrupt_event.clear()
                        break  # back to outer loop
                    except TimeoutError:
                        pass  # normal pacing timeout

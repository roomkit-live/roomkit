"""Shared microphone capture source backed by sounddevice.

Opens the input device once, for as long as the source is running, and fans
its frames out to every subscriber.  A voice session becomes one subscriber
among them instead of the device's owner — see RFC Section 12.12.

Requires the ``sounddevice`` optional dependency::

    pip install roomkit[local-audio]

Usage::

    from roomkit.voice.capture import LocalMicSource

    mic = LocalMicSource(backlog_seconds=10)
    mic.start()

    # Listen with no session in sight — enqueue only, never block the callback.
    detector = mic.subscribe(frames.put_nowait, name="wakeword")

    # … VAD reports SPEECH_START:
    mark = mic.mark()

    # … the wake word matched — the session replays from the mark, so the
    # utterance that preceded it reaches the provider intact.
    await channel.start_session(room_id, participant_id,
                                metadata={"capture_since": mark})
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.voice._sounddevice import import_sounddevice
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.capture.base import DEFAULT_BACKLOG_SECONDS, AudioCaptureSource

logger = logging.getLogger("roomkit.voice.capture")


class LocalMicSource(AudioCaptureSource):
    """Capture source owning the system microphone."""

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration_ms: int = 20,
        backlog_seconds: float = DEFAULT_BACKLOG_SECONDS,
        max_backlog_bytes: int | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=2,
            block_duration_ms=block_duration_ms,
            backlog_seconds=backlog_seconds,
            max_backlog_bytes=max_backlog_bytes,
        )
        self._sd = import_sounddevice("LocalMicSource")
        self._input_device = input_device
        self._stream: Any = None  # sd.RawInputStream

    @property
    def input_latency_ms(self) -> float | None:
        """PortAudio's reported input latency, once the stream is open."""
        if self._stream is None:
            return None
        try:
            return float(self._stream.latency) * 1000
        except Exception:
            logger.debug("Input latency unavailable", exc_info=True)
            return None

    def start(self) -> None:
        if self._stream is not None:
            return

        blocksize = int(self._sample_rate * self._block_duration_ms / 1000)
        stream = self._sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._input_device,
            callback=self._audio_callback,
        )
        stream.start()
        self._stream = stream
        logger.info(
            "Shared mic capture started: rate=%d, channels=%d, block=%dms, backlog=%dkB",
            self._sample_rate,
            self._channels,
            self._block_duration_ms,
            self._max_ring_bytes // 1024,
        )

    def stop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            stream.stop()
        except Exception:
            logger.warning("Error stopping shared mic stream", exc_info=True)
        finally:
            stream.close()
        logger.info("Shared mic capture stopped")

    def _audio_callback(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback.  Runs on the capture thread."""
        if status:
            logger.warning("Mic status: %s", status)
        self._dispatch(
            AudioFrame(
                data=bytes(indata),
                sample_rate=self._sample_rate,
                channels=self._channels,
                sample_width=2,
            )
        )

"""``ScenarioVoiceBackend`` — the simulated phone of the voice test bench."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession
from roomkit.voice.testing.wav import (
    DEFAULT_SAMPLE_RATE,
    PCMAudio,
    pcm_frames,
    read_wav,
    write_wav,
)


@dataclass
class _Capture:
    """What one session's bot sent so far, in the format of its first chunk."""

    sample_rate: int
    channels: int
    chunks: list[bytes] = field(default_factory=list)
    error: ValueError | None = None
    """A format mismatch seen on the way in, raised again on the way out:
    the channel swallows a failing send, the bench must not."""


class ScenarioVoiceBackend(MockVoiceBackend):
    """A :class:`MockVoiceBackend` that plays audio at a transport's cadence
    and keeps what the bot said.

    The mock already injects frames (``simulate_audio_received``), records
    what was sent (``sent_audio``) and simulates barge-in, session-ready and
    disconnects. This adds the two things a scenario needs and a unit test
    does not:

    - :meth:`play` cuts a WAV or a :class:`PCMAudio` into ``frame_ms`` frames
      and delivers them one per ``frame_ms`` of wall-clock time — or as fast
      as the loop allows with ``realtime=False``, the level where the VAD is
      scripted and time is not what is under test;
    - every :meth:`send_audio` / :meth:`send_audio_sync` is captured per
      session with its format, readable back as a :class:`PCMAudio`
      (:meth:`captured`) or written to a WAV (:meth:`write_capture`), so a
      failed scenario can be listened to.

    ``is_playing`` is true while a ``send_audio`` is in flight, on top of the
    mock's ``start_playing`` / ``stop_playing``; a mock TTS sends in
    microseconds, so a scenario waits on the TTS hooks for "the bot is
    speaking", not on this flag. ``capture_sample_rate`` is the format a raw
    ``bytes`` send is captured at (a chunk carries its own): the realtime
    channels send raw bytes, typically at 24 kHz. Still a pure transport
    (RFC §12): no VAD, no speech intelligence, whatever ``capabilities`` it
    is told to declare.
    """

    def __init__(
        self,
        *,
        capabilities: VoiceCapability = VoiceCapability.NONE,
        frame_ms: int = 20,
        capture_sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(capabilities=capabilities)
        if frame_ms <= 0:
            raise ValueError(f"frame_ms must be positive, got {frame_ms}")
        self._frame_ms = frame_ms
        self._capture_sample_rate = capture_sample_rate
        self._captures: dict[str, _Capture] = {}
        self._in_flight: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "ScenarioVoiceBackend"

    @property
    def frame_ms(self) -> int:
        """Frame duration :meth:`play` delivers at."""
        return self._frame_ms

    # -------------------------------------------------------------------------
    # Playing the caller's side
    # -------------------------------------------------------------------------

    async def play(
        self,
        session: VoiceSession,
        source: PCMAudio | str | Path,
        *,
        realtime: bool = True,
    ) -> int:
        """Deliver *source* to the channel as ``frame_ms`` frames.

        *source* is a :class:`PCMAudio` or the path of a WAV file (read off
        the loop). With ``realtime`` each frame is due ``frame_ms`` after the
        previous one, anchored on the first so the cadence does not drift, and
        the call returns when the last frame's slot has elapsed; without it
        the frames are delivered back to back, yielding to the loop between
        two so a streaming STT or a barge-in can interleave, and the call
        returns once the last frame has been handed to the channel. Neither is
        when the channel has finished with the audio — wait on the hooks
        (``VoiceTrace``) for that. Returns the number of frames sent.
        """
        audio = (
            source if isinstance(source, PCMAudio) else await asyncio.to_thread(read_wav, source)
        )
        frames = pcm_frames(audio, frame_ms=self._frame_ms)
        loop = asyncio.get_running_loop()
        started = loop.time()
        for i, frame in enumerate(frames):
            await self.simulate_audio_received(session, frame)
            if not realtime:
                await asyncio.sleep(0)
                continue
            due = started + (i + 1) * self._frame_ms / 1000.0
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        return len(frames)

    # -------------------------------------------------------------------------
    # Capturing the bot's side
    # -------------------------------------------------------------------------

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        self._in_flight[session.id] = self._in_flight.get(session.id, 0) + 1
        try:
            if isinstance(audio, bytes):
                self._capture(session, audio, self._capture_sample_rate, 1)
                await super().send_audio(session, audio)
            else:
                await super().send_audio(session, self._tee(session, audio))
        finally:
            self._in_flight[session.id] -= 1

    def is_playing(self, session: VoiceSession) -> bool:
        return self._in_flight.get(session.id, 0) > 0 or super().is_playing(session)

    async def _tee(
        self, session: VoiceSession, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        async for chunk in chunks:
            self._capture(session, chunk.data, chunk.sample_rate, chunk.channels)
            yield chunk

    def send_audio_sync(self, session: VoiceSession, chunk: AudioChunk) -> None:
        self._capture(session, chunk.data, chunk.sample_rate, chunk.channels)
        super().send_audio_sync(session, chunk)

    def _capture(
        self, session: VoiceSession, data: bytes, sample_rate: int, channels: int
    ) -> None:
        capture = self._captures.get(session.id)
        if capture is None:
            capture = self._captures[session.id] = _Capture(sample_rate, channels)
        elif (capture.sample_rate, capture.channels) != (sample_rate, channels):
            # A WAV has one format; a bench that wrote a mixed one would lie
            # about what the bot said. Loud on the way in, and again on the
            # way out: a channel logs a failed send and carries on.
            capture.error = capture.error or ValueError(
                f"session {session.id}: capture is {capture.sample_rate} Hz x"
                f"{capture.channels}, got a chunk at {sample_rate} Hz x{channels}"
            )
            raise capture.error
        capture.chunks.append(data)

    def captured(self, session: VoiceSession) -> PCMAudio:
        """Everything the bot sent to *session* so far, as one clip.

        Raises the ``ValueError`` of a format mismatch seen during a send,
        which the channel had swallowed.
        """
        capture = self._captures.get(session.id)
        if capture is None:
            return PCMAudio(data=b"", sample_rate=self._capture_sample_rate)
        if capture.error is not None:
            raise capture.error
        return PCMAudio(
            data=b"".join(capture.chunks),
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )

    def write_capture(self, session: VoiceSession, path: str | Path) -> Path:
        """Write the bot's audio for *session* to a WAV file."""
        return write_wav(path, self.captured(session))

    def clear_capture(self, session: VoiceSession) -> None:
        """Forget what the bot sent to *session* so far."""
        self._captures.pop(session.id, None)

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
    speaking", not on this flag. Still a pure transport (RFC §12): no VAD, no
    speech intelligence, whatever ``capabilities`` it is told to declare.
    """

    def __init__(
        self,
        *,
        capabilities: VoiceCapability = VoiceCapability.NONE,
        frame_ms: int = 20,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(capabilities=capabilities)
        if frame_ms <= 0:
            raise ValueError(f"frame_ms must be positive, got {frame_ms}")
        self._frame_ms = frame_ms
        # The format assumed for a raw-bytes send_audio, which carries none.
        self._sample_rate = sample_rate
        self._captures: dict[str, _Capture] = {}

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

        *source* is a :class:`PCMAudio` or the path of a WAV file. With
        ``realtime`` each frame is due ``frame_ms`` after the previous one,
        anchored on the first so the cadence does not drift; without it the
        frames are delivered back to back. Returns the number of frames sent.
        The call returns once the last frame has been handed to the channel,
        which is not when the channel has finished with it — wait on the
        hooks (``VoiceTrace``) for that.
        """
        audio = source if isinstance(source, PCMAudio) else read_wav(source)
        frames = pcm_frames(audio, frame_ms=self._frame_ms)
        loop = asyncio.get_running_loop()
        started = loop.time()
        for i, frame in enumerate(frames):
            await self.simulate_audio_received(session, frame)
            if realtime:
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
        held = session.id in self._playing_sessions
        self._playing_sessions.add(session.id)
        try:
            if isinstance(audio, bytes):
                self._capture(session, audio, self._sample_rate, 1)
                await super().send_audio(session, audio)
            else:
                await super().send_audio(session, self._tee(session, audio))
        finally:
            if not held:
                self._playing_sessions.discard(session.id)

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
            # about what the bot said. Loud, on purpose.
            raise ValueError(
                f"session {session.id}: capture is {capture.sample_rate} Hz x"
                f"{capture.channels}, got a chunk at {sample_rate} Hz x{channels}"
            )
        capture.chunks.append(data)

    def captured(self, session: VoiceSession) -> PCMAudio:
        """Everything the bot sent to *session* so far, as one clip."""
        capture = self._captures.get(session.id)
        if capture is None:
            return PCMAudio(data=b"", sample_rate=self._sample_rate)
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

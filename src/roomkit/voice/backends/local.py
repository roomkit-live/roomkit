"""Local audio backend using system microphone and speakers.

This backend captures audio from the local microphone and plays audio
through the system speakers.  It is designed for local testing and
development — no WebRTC or WebSocket infrastructure required.

Requires the ``sounddevice`` optional dependency::

    pip install roomkit[local-audio]

Usage::

    from roomkit.voice.backends.local import LocalAudioBackend

    backend = LocalAudioBackend()
    voice_channel = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
    kit.register_channel(voice_channel)

    # Create a session and start capturing from the mic
    session = await backend.connect("room-1", "user-1", "voice-1")
    await backend.start_listening(session)

    # ... pipeline processes mic audio, AI responds, TTS plays through speakers ...

    await backend.stop_listening(session)
    await backend.disconnect(session)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import threading
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import AudioReceivedCallback, VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)

if TYPE_CHECKING:
    import sounddevice as sd

logger = logging.getLogger("roomkit.voice.local")


def _import_sounddevice() -> Any:
    """Import sounddevice, raising a clear error if missing."""
    try:
        import sounddevice as _sd

        return _sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for LocalAudioBackend. "
            "Install it with: pip install roomkit[local-audio]"
        ) from exc


class LocalAudioBackend(VoiceBackend):
    """VoiceBackend that uses the system microphone and speakers.

    Audio captured from the microphone is delivered as ``AudioFrame`` objects
    via the ``on_audio_received`` callback.  Outbound audio (TTS) is played
    through the default output device.

    The backend supports one active listening session at a time.  Call
    :meth:`start_listening` after :meth:`connect` to begin mic capture, and
    :meth:`stop_listening` to end it.

    Args:
        input_sample_rate: Mic capture sample rate (Hz).
        output_sample_rate: Speaker playback sample rate (Hz).
        channels: Number of audio channels (1 = mono).
        block_duration_ms: Duration of each audio block in milliseconds.
            Controls how often ``on_audio_received`` fires.
        input_device: Sounddevice input device index or name (None = default).
        output_device: Sounddevice output device index or name (None = default).
    """

    def __init__(
        self,
        *,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        channels: int = 1,
        block_duration_ms: int = 20,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self._sd = _import_sounddevice()

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._channels = channels
        self._block_duration_ms = block_duration_ms
        self._input_device = input_device
        self._output_device = output_device

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []

        # Session tracking
        self._sessions: dict[str, VoiceSession] = {}

        # Active mic stream per session
        self._input_streams: dict[str, sd.RawInputStream] = {}

        # Event loop reference for dispatching callbacks from the audio thread
        self._loop: asyncio.AbstractEventLoop | None = None

        # Playback tracking for barge-in
        self._playing_sessions: set[str] = set()
        self._output_streams: dict[str, sd.RawOutputStream] = {}
        self._playback_tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def name(self) -> str:
        return "LocalAudio"

    @property
    def capabilities(self) -> VoiceCapability:
        return VoiceCapability.INTERRUPTION

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        session_id = str(uuid.uuid4())
        session_metadata = {
            "input_sample_rate": self._input_sample_rate,
            "output_sample_rate": self._output_sample_rate,
            "backend": "local_audio",
            **(metadata or {}),
        }
        session = VoiceSession(
            id=session_id,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata=session_metadata,
        )
        self._sessions[session_id] = session
        logger.info(
            "Local audio session created: session=%s, room=%s, participant=%s",
            session_id,
            room_id,
            participant_id,
        )
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        await self.stop_listening(session)
        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._playing_sessions.discard(session.id)
        logger.info("Local audio session ended: session=%s", session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        for session in list(self._sessions.values()):
            await self.disconnect(session)

    # -------------------------------------------------------------------------
    # Microphone capture
    # -------------------------------------------------------------------------

    async def start_listening(self, session: VoiceSession) -> None:
        """Start capturing audio from the microphone for a session.

        Audio frames are delivered via the ``on_audio_received`` callback.

        Args:
            session: The voice session to capture audio for.
        """
        if session.id in self._input_streams:
            logger.warning("Already listening for session %s", session.id)
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        blocksize = int(self._input_sample_rate * self._block_duration_ms / 1000)

        # Capture references as locals so the PortAudio callback thread
        # reads stable snapshots instead of mutable instance attributes.
        callback_ref = self._audio_received_callback
        loop_ref = self._loop

        def _audio_callback(
            indata: bytes, frames: int, time_info: Any, status: Any
        ) -> None:
            if status:
                logger.warning("Mic status: %s", status)
            if not callback_ref:
                return

            frame = AudioFrame(
                data=bytes(indata),
                sample_rate=self._input_sample_rate,
                channels=self._channels,
                sample_width=2,
            )

            if loop_ref is not None and loop_ref.is_running():
                loop_ref.call_soon_threadsafe(callback_ref, session, frame)
            else:
                callback_ref(session, frame)

        stream = self._sd.RawInputStream(
            samplerate=self._input_sample_rate,
            blocksize=blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._input_device,
            callback=_audio_callback,
        )
        stream.start()
        self._input_streams[session.id] = stream
        logger.info(
            "Mic capture started: session=%s, rate=%d, block=%dms",
            session.id,
            self._input_sample_rate,
            self._block_duration_ms,
        )

    async def stop_listening(self, session: VoiceSession) -> None:
        """Stop capturing audio from the microphone for a session.

        Args:
            session: The voice session to stop capturing for.
        """
        stream = self._input_streams.pop(session.id, None)
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                logger.warning("Error stopping mic stream for session %s", session.id)
            finally:
                stream.close()
            logger.info("Mic capture stopped: session=%s", session.id)

    # -------------------------------------------------------------------------
    # Speaker playback
    # -------------------------------------------------------------------------

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        """Play audio through the system speakers.

        Args:
            session: The target session.
            audio: Raw PCM-16 LE bytes or an async iterator of AudioChunks.
        """
        self._playing_sessions.add(session.id)
        try:
            if isinstance(audio, bytes):
                await self._play_pcm(audio)
            else:
                await self._play_stream(session, audio)
        except Exception:
            logger.exception("Error playing audio for session %s", session.id)
        finally:
            self._playing_sessions.discard(session.id)

    async def _play_pcm(self, pcm_data: bytes) -> None:
        """Play a complete PCM-16 LE buffer through speakers."""
        sd = self._sd

        n_samples = len(pcm_data) // 2
        if n_samples == 0:
            return

        samples = struct.unpack(f"<{n_samples}h", pcm_data[: n_samples * 2])
        import array

        buf = array.array("h", samples)

        def _play() -> None:
            import numpy as np

            data = np.frombuffer(buf, dtype=np.int16).reshape(-1, self._channels)
            sd.play(data, samplerate=self._output_sample_rate, device=self._output_device)
            sd.wait()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _play)

    async def _play_stream(
        self,
        session: VoiceSession,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        """Play a stream of AudioChunks with buffered output.

        Uses a callback-based ``RawOutputStream`` following the
        `sounddevice asyncio pattern`_: PortAudio's own audio thread
        pulls PCM data from a shared buffer and raises ``CallbackStop``
        once the buffer is fully drained, signalling an ``asyncio.Event``
        to wake the coroutine.

        Chunk consumption runs in a cancellable task so that
        ``cancel_audio()`` can abort both the TTS HTTP stream and the
        drain wait in one shot.

        .. _sounddevice asyncio pattern:
           https://python-sounddevice.readthedocs.io/en/0.5.3/examples.html
           #using-a-stream-in-an-asyncio-coroutine
        """
        sd = self._sd
        loop = asyncio.get_running_loop()
        finished = asyncio.Event()

        audio_buf = bytearray()
        buf_lock = threading.Lock()
        producer_done = threading.Event()

        def _output_callback(
            outdata: bytes, frames: int, time_info: Any, status: Any
        ) -> None:
            nbytes = frames * 2 * self._channels  # int16
            with buf_lock:
                n = min(len(audio_buf), nbytes)
                if n > 0:
                    outdata[:n] = bytes(audio_buf[:n])
                    del audio_buf[:n]
                if n < nbytes:
                    outdata[n:] = b"\x00" * (nbytes - n)
                # When producer finished and buffer drained, stop playback.
                if producer_done.is_set() and len(audio_buf) == 0:
                    loop.call_soon_threadsafe(finished.set)
                    raise sd.CallbackStop

        stream = sd.RawOutputStream(
            samplerate=self._output_sample_rate,
            channels=self._channels,
            dtype="int16",
            callback=_output_callback,
            device=self._output_device,
        )
        self._output_streams[session.id] = stream
        stream.start()

        async def _run() -> None:
            # Phase 1: consume TTS chunks into the buffer.
            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    return
                if chunk.data:
                    with buf_lock:
                        audio_buf.extend(chunk.data)

            # Phase 2: wait for the PortAudio callback to drain.
            producer_done.set()
            with buf_lock:
                if len(audio_buf) == 0:
                    return  # nothing to drain
            await finished.wait()

        task = asyncio.create_task(_run())
        self._playback_tasks[session.id] = task
        try:
            await task
        except asyncio.CancelledError:
            pass  # Normal: cancelled by cancel_audio() during barge-in
        finally:
            self._playback_tasks.pop(session.id, None)
            ostream = self._output_streams.pop(session.id, None)
            if ostream is not None:
                with contextlib.suppress(Exception):
                    ostream.abort()
                ostream.close()

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Log transcription text (no UI in local mode)."""
        label = "User" if role == "user" else "Assistant"
        logger.info("[%s] %s", label, text)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callback = callback

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)

    async def cancel_audio(self, session: VoiceSession) -> bool:
        was_playing = session.id in self._playing_sessions
        if was_playing:
            self._playing_sessions.discard(session.id)
            # Cancel the consumption task — unblocks the async-for
            # that may be waiting on the TTS HTTP stream.
            task = self._playback_tasks.pop(session.id, None)
            if task is not None:
                task.cancel()
            else:
                self._sd.stop()  # Fallback for _play_pcm() based playback
            logger.info("Audio cancelled for session %s", session.id)
        return was_playing

    def is_playing(self, session: VoiceSession) -> bool:
        return session.id in self._playing_sessions

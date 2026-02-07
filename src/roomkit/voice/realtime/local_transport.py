"""Local audio transport for realtime voice — mic input, speaker output.

Uses ``sounddevice`` to capture audio from the system microphone and
play AI audio through the system speakers.  Designed for local testing
of speech-to-speech providers (Gemini Live, OpenAI Realtime) without
any WebSocket or browser setup.

Requires the ``sounddevice`` optional dependency::

    pip install roomkit[local-audio]

Usage::

    from roomkit.voice.realtime.local_transport import LocalAudioTransport

    transport = LocalAudioTransport()
    channel = RealtimeVoiceChannel("voice", provider=provider, transport=transport)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec.base import AECProvider
from roomkit.voice.pipeline.denoiser.base import DenoiserProvider
from roomkit.voice.realtime.base import RealtimeSession
from roomkit.voice.realtime.transport import (
    RealtimeAudioTransport,
    TransportAudioCallback,
    TransportDisconnectCallback,
)

logger = logging.getLogger("roomkit.voice.realtime.local_transport")

_STATS_INTERVAL = 5.0  # seconds between stats log lines


class LocalAudioTransport(RealtimeAudioTransport):
    """Realtime audio transport using the system microphone and speakers.

    Replaces WebSocket transport for local testing — audio goes to/from
    the local sound hardware instead of a browser.

    Audio playback uses a callback-driven ``RawOutputStream``.  PortAudio
    pulls samples at the hardware rate; when no AI audio is queued the
    callback feeds silence, keeping the stream alive and avoiding the
    ALSA underrun crashes that occur with blocking ``write()`` or
    concurrent ``sd.play()`` calls.

    Args:
        input_sample_rate: Mic capture sample rate (Hz).
        output_sample_rate: Speaker playback sample rate (Hz).
        channels: Number of audio channels (1 = mono).
        block_duration_ms: Duration of each mic audio block in milliseconds.
        input_device: Sounddevice input device index or name (None = default).
        output_device: Sounddevice output device index or name (None = default).
        mute_mic_during_playback: If True, suppress mic audio while the
            AI is speaking (half-duplex).  Prevents speaker echo from
            triggering the provider's server-side VAD.  Recommended when
            using speakers instead of headphones.  Default True.
        aec: Optional AEC provider for echo cancellation.  When set,
            mic audio is processed through ``aec.process()`` and speaker
            audio is fed as reference via ``aec.feed_reference()``.
            Requires ``input_sample_rate == output_sample_rate``.
        denoiser: Optional denoiser provider for noise suppression.
            Applied to mic audio after AEC (if both are set).
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
        mute_mic_during_playback: bool = True,
        aec: AECProvider | None = None,
        denoiser: DenoiserProvider | None = None,
    ) -> None:
        self._sd = _import_sounddevice()

        if aec is not None and input_sample_rate != output_sample_rate:
            raise ValueError(
                f"AEC requires matching sample rates, got "
                f"input={input_sample_rate} output={output_sample_rate}"
            )

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._channels = channels
        self._block_duration_ms = block_duration_ms
        self._input_device = input_device
        self._output_device = output_device
        self._mute_mic_during_playback = mute_mic_during_playback

        # Callbacks
        self._audio_callbacks: list[TransportAudioCallback] = []
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []

        # Active sessions
        self._input_streams: dict[str, Any] = {}  # session_id -> RawInputStream
        self._sessions: dict[str, RealtimeSession] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing_event = threading.Event()

        # Speaker output: callback-driven stream + chunk queue
        self._output_stream: Any | None = None
        self._output_buffer: deque[bytes] = deque()
        self._output_buf_offset = 0  # bytes consumed in the front chunk
        self._buffer_lock = threading.Lock()

        # --- Diagnostics (all touched from callback thread) ---
        # CPython's GIL guarantees atomic int increments, so these
        # counters are safe to update from the PortAudio callback
        # thread without a lock.
        self._bytes_queued = 0  # total bytes pushed via send_audio
        self._bytes_played = 0  # total bytes written to speaker
        self._bytes_silence = 0  # total silence bytes (underrun)
        self._cb_count = 0  # number of speaker callback invocations
        self._cb_underruns = 0  # callbacks with partial/full silence
        self._cb_status_errors = 0  # PortAudio status flags
        self._mic_frames_suppressed = 0  # mic frames dropped (echo suppression)
        self._last_stats_time = 0.0

        # --- AEC ---
        self._aec = aec
        if aec is not None:
            self._aec_block_bytes = (
                int(input_sample_rate * block_duration_ms / 1000) * channels * 2
            )
            self._ref_buffer = bytearray()
        else:
            self._aec_block_bytes = 0
            self._ref_buffer = bytearray()

        # --- Denoiser ---
        self._denoiser = denoiser

    @property
    def name(self) -> str:
        return "LocalAudioTransport"

    async def accept(self, session: RealtimeSession, connection: Any) -> None:
        """Start mic capture and speaker output for the session.

        The ``connection`` argument is ignored — audio comes from the
        local microphone.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self._sessions[session.id] = session
        self._last_stats_time = time.monotonic()

        # --- Output stream (speakers, callback-driven) ---
        if self._output_stream is None:
            out = self._sd.RawOutputStream(
                samplerate=self._output_sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._output_device,
                latency="low" if self._aec is not None else "high",
                callback=self._speaker_callback,
            )
            out.start()
            self._output_stream = out

        # --- Input stream (microphone) ---
        blocksize = int(self._input_sample_rate * self._block_duration_ms / 1000)

        def _mic_callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            if self._closing_event.is_set():
                return
            if status:
                logger.warning("Mic status: %s", status)

            # Half-duplex echo suppression: don't send mic audio while
            # the speaker is playing AI audio, otherwise the provider's
            # server-side VAD picks up the echo and triggers barge-in.
            if self._mute_mic_during_playback and len(self._output_buffer) > 0:
                self._mic_frames_suppressed += 1
                return

            audio_bytes = bytes(indata)

            # AEC: remove speaker echo from the captured mic audio.
            if self._aec is not None:
                audio_bytes = self._aec_process(audio_bytes)

            # Denoise: suppress background noise.
            if self._denoiser is not None:
                try:
                    audio_bytes = self._denoise_process(audio_bytes)
                except Exception:
                    logger.exception("Denoiser error — passing audio through")

            for cb in self._audio_callbacks:
                if self._loop is not None and self._loop.is_running():
                    self._loop.call_soon_threadsafe(cb, session, audio_bytes)
                else:
                    cb(session, audio_bytes)

        stream = self._sd.RawInputStream(
            samplerate=self._input_sample_rate,
            blocksize=blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._input_device,
            callback=_mic_callback,
        )
        stream.start()
        self._input_streams[session.id] = stream

        logger.info(
            "Local audio started: session=%s, mic=%dHz, spk=%dHz, block=%dms",
            session.id,
            self._input_sample_rate,
            self._output_sample_rate,
            self._block_duration_ms,
        )

    async def send_audio(self, session: RealtimeSession, audio: bytes) -> None:
        """Queue audio for playback through the system speakers."""
        if not audio or self._closing_event.is_set() or len(audio) < 2:
            return

        with self._buffer_lock:
            self._output_buffer.append(audio)
        self._bytes_queued += len(audio)

        # Periodic stats (logged from asyncio thread, not the callback)
        now = time.monotonic()
        if now - self._last_stats_time >= _STATS_INTERVAL:
            self._log_stats(now)

    async def send_message(self, session: RealtimeSession, message: dict[str, Any]) -> None:
        """Log messages (no UI in local mode)."""
        msg_type = message.get("type", "unknown")

        if msg_type == "transcription":
            role = message.get("role", "?")
            text = message.get("text", "")
            is_final = message.get("is_final", False)
            marker = "" if is_final else " (partial)"
            label = "You" if role == "user" else "AI"
            logger.info("[%s]%s %s", label, marker, text)
        elif msg_type == "speaking":
            who = message.get("who", "?")
            speaking = message.get("speaking", False)
            state = "speaking" if speaking else "silent"
            logger.info("[%s] %s", who, state)
        elif msg_type == "clear_audio":
            with self._buffer_lock:
                n = len(self._output_buffer)
                self._output_buffer.clear()
                self._output_buf_offset = 0
            logger.info("[barge-in] cleared audio queue (%d chunks)", n)

    async def disconnect(self, session: RealtimeSession) -> None:
        """Stop mic capture and speaker output for the session."""
        self._closing_event.set()
        self._log_stats(time.monotonic(), final=True)
        self._stop_output()

        stream = self._input_streams.pop(session.id, None)
        if stream is not None:
            stream.abort()
            stream.close()

        self._sessions.pop(session.id, None)
        logger.info("Local audio stopped: session=%s", session.id)

    async def close(self) -> None:
        """Stop all streams."""
        self._closing_event.set()
        self._log_stats(time.monotonic(), final=True)
        self._stop_output()

        for sid in list(self._input_streams):
            stream = self._input_streams.pop(sid)
            stream.abort()
            stream.close()
        self._sessions.clear()

        if self._aec is not None:
            self._aec.close()
        if self._denoiser is not None:
            self._denoiser.close()

    # -- Speaker callback (runs in PortAudio C thread) --

    def _speaker_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Pull queued audio into the output buffer; fill gaps with silence.

        Writes directly into the PortAudio output buffer to avoid
        per-callback heap allocations that would pressure the GC.
        """
        if status:
            self._cb_status_errors += 1

        self._cb_count += 1
        bytes_needed = frames * self._channels * 2  # 2 bytes per int16 sample
        written = 0

        # Pull audio chunks into outdata
        with self._buffer_lock:
            buf = self._output_buffer
            while written < bytes_needed and buf:
                chunk = buf[0]
                avail = len(chunk) - self._output_buf_offset
                n = min(avail, bytes_needed - written)
                src_start = self._output_buf_offset
                segment = chunk[src_start : src_start + n]
                outdata[written : written + n] = segment
                written += n
                self._output_buf_offset += n
                if self._output_buf_offset >= len(chunk):
                    buf.popleft()
                    self._output_buf_offset = 0

        self._bytes_played += written

        # Fill remaining with silence
        if written < bytes_needed:
            silence_bytes = bytes_needed - written
            outdata[written:] = b"\x00" * silence_bytes
            self._bytes_silence += silence_bytes
            if written > 0:
                # Partial fill = we ran out of data mid-callback
                self._cb_underruns += 1

        # AEC: feed the COMPLETE output frame (audio + silence) as
        # reference.  The SpeexDSP split API requires playback() for
        # every output frame — skipping silence frames causes the
        # internal ring buffer to lose sync with the actual speaker
        # output and prevents the adaptive filter from converging.
        if self._aec is not None:
            self._aec_feed_played(bytearray(bytes(outdata)))

    def _stop_output(self) -> None:
        """Close the speaker output stream."""
        with self._buffer_lock:
            self._output_buffer.clear()
            self._output_buf_offset = 0
        out = self._output_stream
        if out is not None:
            self._output_stream = None
            try:
                out.abort()
                out.close()
            except Exception:  # nosec B110
                pass

    # -- Diagnostics --

    def _log_stats(self, now: float, *, final: bool = False) -> None:
        """Log playback queue diagnostics."""
        elapsed = now - self._last_stats_time
        if elapsed <= 0:
            return
        self._last_stats_time = now

        with self._buffer_lock:
            queue_bytes = sum(len(c) for c in self._output_buffer)
            queue_chunks = len(self._output_buffer)
        bps = self._output_sample_rate * self._channels * 2
        queue_ms = (queue_bytes / bps * 1000) if bps else 0

        label = "FINAL audio stats" if final else "audio stats"
        logger.debug(
            "[%s] queued=%dB played=%dB silence=%dB "
            "queue_now=%d chunks/%.0fms "
            "cb=%d underruns=%d pa_err=%d mic_suppressed=%d",
            label,
            self._bytes_queued,
            self._bytes_played,
            self._bytes_silence,
            queue_chunks,
            queue_ms,
            self._cb_count,
            self._cb_underruns,
            self._cb_status_errors,
            self._mic_frames_suppressed,
        )

    # -- AEC helpers --

    def _aec_process(self, audio_bytes: bytes) -> bytes:
        """Run mic audio through the AEC to remove speaker echo."""
        frame = AudioFrame(
            data=audio_bytes,
            sample_rate=self._input_sample_rate,
            channels=self._channels,
            sample_width=2,
        )
        result = self._aec.process(frame)  # type: ignore[union-attr]
        return result.data

    def _aec_feed_played(self, played: bytearray) -> None:
        """Feed actually-played speaker bytes to the AEC as reference.

        Called from ``_speaker_callback`` so the reference is time-aligned
        with what the speaker is outputting.
        """
        self._ref_buffer.extend(played)
        block = self._aec_block_bytes

        while len(self._ref_buffer) >= block:
            chunk = bytes(self._ref_buffer[:block])
            del self._ref_buffer[:block]
            frame = AudioFrame(
                data=chunk,
                sample_rate=self._input_sample_rate,
                channels=self._channels,
                sample_width=2,
            )
            self._aec.feed_reference(frame)  # type: ignore[union-attr]

    # -- Denoiser helper --

    def _denoise_process(self, audio_bytes: bytes) -> bytes:
        """Run mic audio through the denoiser to suppress background noise."""
        frame = AudioFrame(
            data=audio_bytes,
            sample_rate=self._input_sample_rate,
            channels=self._channels,
            sample_width=2,
        )
        result = self._denoiser.process(frame)  # type: ignore[union-attr]
        return result.data

    # -- Callback registration --

    def on_audio_received(self, callback: TransportAudioCallback) -> None:
        self._audio_callbacks.append(callback)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)


def _import_sounddevice() -> Any:
    """Import sounddevice, raising a clear error if missing."""
    try:
        import sounddevice as _sd

        return _sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for LocalAudioTransport. "
            "Install it with: pip install roomkit[local-audio]"
        ) from exc

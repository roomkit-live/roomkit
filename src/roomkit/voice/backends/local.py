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

    # ... channel pipeline processes mic audio, AI responds, TTS plays through speakers ...

    await backend.stop_listening(session)
    await backend.disconnect(session)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import sys
import threading
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from roomkit.voice._sounddevice import import_sounddevice
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import (
    AudioPlayedCallback,
    AudioReceivedCallback,
    SessionReadyCallback,
    SpeakerChangeCallback,
    TransportDisconnectCallback,
    VoiceBackend,
)
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)
from roomkit.voice.capture.base import CaptureMark

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import sounddevice as sd

    from roomkit.voice.capture.base import AudioCaptureSource, CaptureSubscription
    from roomkit.voice.pipeline.aec.base import AECProvider
    from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

logger = logging.getLogger("roomkit.voice.local")

_MAX_REALTIME_BUFFER_SECONDS = 30

_DEFAULT_INPUT_SAMPLE_RATE = 16000
_DEFAULT_CHANNELS = 1
_DEFAULT_BLOCK_DURATION_MS = 20


def _resolve_input_format(
    source: AudioCaptureSource | None,
    input_sample_rate: int | None,
    channels: int | None,
    block_duration_ms: int | None,
) -> tuple[int, int, int]:
    """Settle the input format between explicit arguments and a shared source.

    A shared source owns the device, so it owns the format too.  An explicit
    argument that contradicts it is a configuration error, not something to
    silently resample around.
    """
    if source is None:
        return (
            input_sample_rate if input_sample_rate is not None else _DEFAULT_INPUT_SAMPLE_RATE,
            channels if channels is not None else _DEFAULT_CHANNELS,
            (block_duration_ms if block_duration_ms is not None else _DEFAULT_BLOCK_DURATION_MS),
        )

    for name, requested, provided in (
        ("input_sample_rate", input_sample_rate, source.sample_rate),
        ("channels", channels, source.channels),
        ("block_duration_ms", block_duration_ms, source.block_duration_ms),
    ):
        if requested is not None and requested != provided:
            raise ValueError(
                f"{name}={requested} conflicts with the capture source "
                f"({name}={provided}). The source owns the input format; omit "
                f"the argument or configure the source instead."
            )
    return source.sample_rate, source.channels, source.block_duration_ms


class LocalAudioBackend(VoiceBackend):
    """VoiceBackend that uses the system microphone and speakers.

    Audio captured from the microphone is delivered as ``AudioFrame`` objects
    via the ``on_audio_received`` callback.  Outbound audio (TTS) is played
    through the default output device.

    Works in two modes:

    - **VoiceChannel mode** (STT/TTS): call :meth:`connect` then
      :meth:`start_listening`.  The channel's AudioPipeline handles
      all audio processing.
    - **RealtimeVoiceChannel mode** (speech-to-speech): call :meth:`accept`.
      The channel's AudioPipeline handles all audio processing — the
      backend is a pure transport.

    Args:
        input_sample_rate: Mic capture sample rate (Hz).
        output_sample_rate: Speaker playback sample rate (Hz).
        channels: Number of audio channels (1 = mono).
        block_duration_ms: Duration of each audio block in milliseconds.
            Controls how often ``on_audio_received`` fires.
        input_device: Sounddevice input device index or name (None = default).
        output_device: Sounddevice output device index or name (None = default).
        aec: Optional AEC provider for transport-level echo cancellation.
            Speaker audio is fed as reference via ``aec.feed_reference()``
            from the output callback.
        mute_mic_during_playback: If True (default), suppress mic frames
            while the speaker is playing (half-duplex).  Prevents echo
            from triggering VAD and false barge-ins when using speakers
            instead of headphones.
        rt_prebuffer_ms: Audio to accumulate before starting (or resuming
            after an underrun) realtime speaker playback.  Absorbs burst
            jitter from realtime providers the same way the SIP pacer's
            prebuffer does — without it, any momentary starvation inserts
            an audible mid-sentence gap.  ``0`` plays from the first byte.
    """

    def __init__(
        self,
        *,
        input_sample_rate: int | None = None,
        output_sample_rate: int = 24000,
        channels: int | None = None,
        block_duration_ms: int | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        aec: AECProvider | None = None,
        mute_mic_during_playback: bool = True,
        rt_prebuffer_ms: int = 120,
        source: AudioCaptureSource | None = None,
    ) -> None:
        input_sample_rate, channels, block_duration_ms = _resolve_input_format(
            source, input_sample_rate, channels, block_duration_ms
        )
        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive")
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if block_duration_ms <= 0:
            raise ValueError("block_duration_ms must be positive")
        if rt_prebuffer_ms < 0:
            raise ValueError("rt_prebuffer_ms must be non-negative")
        self._sd = import_sounddevice("LocalAudioBackend")

        # A shared source owns the input device; this backend only subscribes.
        # Its lifecycle stays the caller's — close() here never stops it.
        self._source = source
        self._subscriptions: dict[str, CaptureSubscription] = {}

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._channels = channels
        self._block_duration_ms = block_duration_ms
        self._input_device = input_device
        self._output_device = output_device

        # Callback registrations
        self._audio_received_callback: AudioReceivedCallback | None = None
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._audio_played_callbacks: list[AudioPlayedCallback] = []
        self._session_ready_callbacks: list[SessionReadyCallback] = []

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

        # Half-duplex echo suppression
        self._mute_mic_during_playback = mute_mic_during_playback

        # Realtime transport state
        self._muted_sessions: set[str] = set()
        self._gated_sessions: set[str] = set()
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []

        # Realtime mode flag — set by accept(), controls mic dispatch
        self._realtime_mode = False

        # Realtime speaker output: persistent callback-driven stream
        # with a chunk queue.  Created once in accept(), shared across
        # all send_audio() calls.
        self._rt_output_stream: Any = None  # sd.RawOutputStream
        self._rt_output_buffer: deque[bytes] = deque()
        self._rt_buf_offset = 0  # bytes consumed in the front chunk
        self._rt_buf_lock = threading.Lock()
        self._rt_closing = threading.Event()
        # When True, speaker callback outputs silence and send_audio()
        # drops incoming audio.  Set by interrupt(), cleared on next
        # response_start (via send_audio when _playing_sessions is empty).
        self._rt_interrupted = False
        # Prebuffer priming: while True, the callback outputs silence until
        # the buffer holds rt_prebuffer_ms of audio (or the response is
        # complete / the idle valve fires).  Re-armed on every underrun so
        # starvation produces one rare re-prime instead of scattered gaps.
        self._rt_prebuffer_bytes = int(output_sample_rate * channels * 2 * rt_prebuffer_ms / 1000)
        self._rt_priming = True
        # Set by end_of_response(): allows draining a final partial buffer
        # smaller than the prebuffer (short responses).
        self._rt_response_complete = False
        # Running total of queued-unplayed bytes — O(1) check in the
        # callback instead of summing the deque under the lock every block.
        self._rt_buffered_bytes = 0
        self._rt_max_buffer_bytes = (
            output_sample_rate * channels * 2 * _MAX_REALTIME_BUFFER_SECONDS
        )
        self._rt_dropped_bytes = 0
        # Mid-response starvation counter (see rt_underruns property).
        self._rt_underruns = 0
        # Missing end_of_response safety valve: after ~100ms of priming with
        # no new audio appended, drain whatever is buffered (mirrors the SIP
        # pacer's 0.1s accumulate-then-burst timeout).
        self._rt_prime_idle_blocks = 0
        self._rt_prime_max_idle_blocks = max(1, 100 // block_duration_ms)

        # --- AEC (transport-level reference feeding) ---
        self._aec = aec
        self._aec_active_sessions: set[str] = set()
        self._aec_needs_resample = aec is not None and output_sample_rate != input_sample_rate
        if aec is not None:
            # Block size in bytes at the *input* sample rate — the rate the
            # AEC expects for both capture and reference.
            self._aec_block_bytes = (
                int(input_sample_rate * block_duration_ms / 1000) * channels * 2
            )
            # When output rate differs, we accumulate output-rate bytes and
            # resample whole blocks to input rate before feeding the AEC.
            self._aec_out_block_bytes = (
                int(output_sample_rate * block_duration_ms / 1000) * channels * 2
            )
            self._ref_buffers: dict[str, bytearray] = {}
            if self._aec_needs_resample:
                from roomkit.voice.pipeline.resampler.linear import (
                    LinearResamplerProvider,
                )

                self._aec_resampler: LinearResamplerProvider | None = LinearResamplerProvider()
                logger.info(
                    "AEC transport-level reference: resampling %dHz -> %dHz",
                    output_sample_rate,
                    input_sample_rate,
                )
            else:
                self._aec_resampler = None
        else:
            self._aec_block_bytes = 0
            self._aec_out_block_bytes = 0
            self._aec_resampler = None
            self._ref_buffers = {}

    @property
    def name(self) -> str:
        return "LocalAudio"

    @property
    def auto_connect(self) -> bool:
        return True

    @property
    def capabilities(self) -> VoiceCapability:
        caps = VoiceCapability.INTERRUPTION
        # When transport-level AEC is configured, the backend applies
        # capture inline on the PortAudio thread (timing-critical).
        # Report NATIVE_AEC so the pipeline skips its own AEC stage.
        if self._aec is not None:
            caps |= VoiceCapability.NATIVE_AEC
        return caps

    @property
    def feeds_aec_reference(self) -> bool:
        return self._aec is not None

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
        self._rt_closing.set()
        await self.stop_listening(session)
        self._stop_rt_output()
        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._playing_sessions.discard(session.id)
        self._gated_sessions.discard(session.id)
        self._muted_sessions.discard(session.id)
        self._aec_end_playback(session.id)
        self._ref_buffers.pop(session.id, None)
        if self._aec is not None:
            self._aec.reset(session.id)
        logger.info("Local audio session ended: session=%s", session.id)

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        self._rt_closing.set()
        for session in list(self._sessions.values()):
            await self.disconnect(session)
        self._stop_rt_output()
        if self._aec is not None:
            self._aec.close()

    # -------------------------------------------------------------------------
    # Microphone capture
    # -------------------------------------------------------------------------

    def _make_frame_handler(self, session: VoiceSession) -> Callable[[AudioFrame], None]:
        """Build the per-session delivery path for captured frames.

        Mute, gating, half-duplex suppression and AEC are session state, so
        they stay here whether the frame came from this backend's own device
        stream or from a shared capture source.

        References are captured as locals so the capture thread reads stable
        snapshots instead of mutable instance attributes.
        """
        callback_ref = self._audio_received_callback
        loop_ref = self._loop
        aec_ref = self._aec  # Transport-level AEC (timing-critical)

        def _handle(frame: AudioFrame) -> None:
            if not callback_ref:
                return

            # Explicit mute via set_input_muted()
            if session.id in self._muted_sessions:
                return

            # Half-duplex echo suppression: suppress mic frames while the
            # speaker is playing.  Prevents echo from triggering VAD /
            # barge-in when using speakers instead of headphones.
            if self._mute_mic_during_playback and self._playing_sessions:
                return

            # Gated by primary-speaker mode
            if session.id in self._gated_sessions:
                return

            # Transport-level AEC: run capture inline on the capture thread
            # so reference and capture timing stay synchronous.
            # The channel's pipeline skips AEC (NATIVE_AEC capability).
            processed = aec_ref.process(frame, session.id) if aec_ref is not None else frame

            if loop_ref is not None and loop_ref.is_running():
                loop_ref.call_soon_threadsafe(callback_ref, session, processed)
            else:
                callback_ref(session, processed)

        return _handle

    async def start_listening(self, session: VoiceSession) -> None:
        """Start capturing audio from the microphone for a session.

        Audio frames are delivered via the ``on_audio_received`` callback.
        With a shared capture source this subscribes to it rather than opening
        a device, replaying from ``session.metadata["capture_since"]`` when a
        mark is present — so speech that preceded the session is not lost.

        Args:
            session: The voice session to capture audio for.
        """
        if session.id in self._input_streams or session.id in self._subscriptions:
            logger.warning("Already listening for session %s", session.id)
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        handler = self._make_frame_handler(session)

        source = self._source
        if source is not None:
            self._subscribe_to_source(source, session, handler)
        else:
            self._open_input_stream(session, handler)

        # Capture is live — fire session ready callbacks
        for cb in self._session_ready_callbacks:
            cb(session)

    def _subscribe_to_source(
        self,
        source: AudioCaptureSource,
        session: VoiceSession,
        handler: Callable[[AudioFrame], None],
    ) -> None:
        """Attach this session to the shared capture source."""
        latency_ms = source.input_latency_ms
        if latency_ms is not None:
            self._configure_aec_delay(latency_ms)

        subscription = source.subscribe(
            handler,
            since=self._session_capture_mark(session),
            name=f"session:{session.id}",
        )
        self._subscriptions[session.id] = subscription
        logger.info(
            "Mic capture subscribed: session=%s, rate=%d, block=%dms, replayed=%dB%s",
            session.id,
            self._input_sample_rate,
            self._block_duration_ms,
            subscription.replayed_bytes,
            " (truncated)" if subscription.truncated else "",
        )

    def _session_capture_mark(self, session: VoiceSession) -> CaptureMark | None:
        """Read the replay mark a caller passed through session metadata."""
        mark = session.metadata.get("capture_since")
        if mark is None or isinstance(mark, CaptureMark):
            return mark
        # Starting the call matters more than the backlog: warn and go live.
        logger.warning(
            "Ignoring capture_since for session %s: expected CaptureMark, got %s",
            session.id,
            type(mark).__name__,
        )
        return None

    def _open_input_stream(
        self, session: VoiceSession, handler: Callable[[AudioFrame], None]
    ) -> None:
        """Open a device stream owned by this session (no shared source)."""
        blocksize = int(self._input_sample_rate * self._block_duration_ms / 1000)
        input_sample_rate = self._input_sample_rate
        channels = self._channels
        deliver = handler
        has_callback = self._audio_received_callback is not None

        def _audio_callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Mic status: %s", status)
            if not has_callback:
                return
            deliver(
                AudioFrame(
                    data=bytes(indata),
                    sample_rate=input_sample_rate,
                    channels=channels,
                    sample_width=2,
                )
            )

        stream = self._sd.RawInputStream(
            samplerate=self._input_sample_rate,
            blocksize=blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._input_device,
            callback=_audio_callback,
        )
        self._configure_aec_delay_from_streams(stream)
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

        With a shared capture source this only detaches this session — the
        source keeps running for whoever else is listening.

        Args:
            session: The voice session to stop capturing for.
        """
        subscription = self._subscriptions.pop(session.id, None)
        if subscription is not None:
            subscription.unsubscribe()
            logger.info("Mic capture unsubscribed: session=%s", session.id)

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

        In realtime mode, bytes are queued into a persistent output buffer
        that a callback-driven PortAudio stream drains continuously.

        In VoiceChannel mode, streaming chunks use a per-session output
        stream, and raw bytes fall back to ``sd.play()``.

        Args:
            session: The target session.
            audio: Raw PCM-16 LE bytes or an async iterator of AudioChunks.
        """
        if self._realtime_mode:
            # Realtime path: queue bytes into persistent output buffer
            if isinstance(audio, bytes) and audio and not self._rt_closing.is_set():
                with self._rt_buf_lock:
                    was_interrupted = self._rt_interrupted
                    self._rt_interrupted = False
                    available = max(0, self._rt_max_buffer_bytes - self._rt_buffered_bytes)
                    frame_width = self._channels * 2
                    available -= available % frame_width
                    accepted = audio[:available]
                    dropped = len(audio) - len(accepted)
                    if accepted:
                        self._rt_output_buffer.append(accepted)
                        self._rt_buffered_bytes += len(accepted)
                    self._rt_dropped_bytes += dropped
                    # New audio means a response is in flight: a stale
                    # end-of-response must not release the priming gate early.
                    self._rt_response_complete = False
                    self._rt_prime_idle_blocks = 0
                # Add after releasing the buffer lock. If the callback just
                # observed an empty buffer and clears playing state, this final
                # write wins; adding before the append had a race that could
                # unmute capture while newly queued audio was about to play.
                if accepted:
                    self._playing_sessions.add(session.id)
                if dropped and self._rt_dropped_bytes == dropped:
                    logger.warning(
                        "Realtime speaker buffer reached its %ds bound; dropping excess audio",
                        _MAX_REALTIME_BUFFER_SECONDS,
                    )
                if was_interrupted:
                    logger.info("[INTERRUPT] cleared — buffering for resume")
            return

        # VoiceChannel path
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
        if self._aec is not None:
            logger.warning(
                "AEC reference feeding is not supported with non-streaming "
                "playback (sd.play). Use streaming TTS for AEC support."
            )
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
        # Guard: prevents the callback from accessing shared state once
        # the asyncio side begins tearing down the stream.
        stream_closing = threading.Event()

        def _output_callback(outdata: bytearray, frames: int, time_info: Any, status: Any) -> None:
            # Early exit if the stream is being torn down — avoids
            # accessing freed memory or Python objects during close().
            if stream_closing.is_set():
                nbytes = frames * 2 * self._channels
                outdata[:nbytes] = b"\x00" * nbytes
                raise sd.CallbackStop

            nbytes = frames * 2 * self._channels  # int16
            stop = False
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
                    stop = True

            # AEC: feed the COMPLETE output frame (audio + silence) as
            # reference.  The SpeexDSP split API requires playback() for
            # every output frame — skipping silence frames causes the
            # internal ring buffer to lose sync with the actual speaker
            # output and prevents the adaptive filter from converging.
            if self._aec is not None and not self._aec_capture_paused(session.id):
                if n > 0:
                    self._aec_begin_playback(session.id)
                if session.id in self._aec_active_sessions:
                    self._aec_feed_played(bytearray(bytes(outdata)), session.id)

            # Notify listeners about played audio (time-aligned reference
            # for pipeline AEC).  The frame is created once and shared.
            if self._audio_played_callbacks:
                played_frame = AudioFrame(
                    data=bytes(outdata),
                    sample_rate=self._output_sample_rate,
                    channels=self._channels,
                    sample_width=2,
                )
                for cb in self._audio_played_callbacks:
                    with contextlib.suppress(Exception):
                        cb(session, played_frame)

            if stop:
                raise sd.CallbackStop

        # Low latency when AEC is active — minimizes the time gap between
        # when reference audio is fed and when the speaker actually plays it.
        # Exception: on macOS CoreAudio, "low" yields tiny hardware buffers
        # that underrun from Python callback jitter → audible crackling.
        if sys.platform == "darwin":
            out_latency = "high"
        elif self._aec is not None:
            out_latency = "low"
        else:
            out_latency = "high"

        stream = sd.RawOutputStream(
            samplerate=self._output_sample_rate,
            channels=self._channels,
            dtype="int16",
            callback=_output_callback,
            device=self._output_device,
            latency=out_latency,
        )
        self._output_streams[session.id] = stream
        stream.start()

        async def _run() -> None:
            # Consume TTS chunks into the buffer.
            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    return
                if chunk.data:
                    with buf_lock:
                        audio_buf.extend(chunk.data)

            # Wait for the PortAudio callback to drain.
            producer_done.set()
            with buf_lock:
                if len(audio_buf) == 0:
                    return  # nothing to drain
            await finished.wait()

        task = asyncio.create_task(_run())
        self._playback_tasks[session.id] = task
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True  # cancel_audio() during barge-in
        finally:
            self._playback_tasks.pop(session.id, None)
            ostream = self._output_streams.pop(session.id, None)
            if ostream is not None:
                # Signal the callback to stop immediately so it won't
                # touch shared state while we tear down the stream.
                stream_closing.set()
                try:
                    if cancelled or not ostream.active:
                        # Barge-in or already stopped: discard buffers.
                        ostream.abort()
                    else:
                        # Normal completion: let PortAudio drain its
                        # hardware buffer so the last syllable isn't lost.
                        ostream.stop()
                except Exception:
                    with contextlib.suppress(Exception):
                        ostream.abort()
                ostream.close()
            # Transport-level AEC is owned here (NATIVE_AEC), so the pipeline
            # deliberately cannot end its playback lifecycle.  Bypass after
            # PortAudio drains, but preserve the converged hardware filter.
            self._aec_end_playback(session.id)

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
        """Register callback for raw audio received from the microphone.

        Always delivers ``(session, AudioFrame)`` — the channel's own
        AudioPipeline handles all processing.
        """
        self._audio_received_callback = callback

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)

    @property
    def supports_playback_callback(self) -> bool:
        return True

    def on_audio_played(self, callback: AudioPlayedCallback) -> None:
        self._audio_played_callbacks.append(callback)

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

    # -------------------------------------------------------------------------
    # Realtime transport methods (merged from LocalAudioTransport)
    # -------------------------------------------------------------------------

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        """Accept a session for realtime use (start mic + speaker).

        Creates a persistent callback-driven speaker output stream and
        starts mic capture.  The channel's AudioPipeline handles all
        audio processing — the backend is a pure transport.
        """
        self._realtime_mode = True
        self._sessions[session.id] = session
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Re-arm after a prior disconnect() — _rt_closing persists across
        # sessions and would silently drop every send_audio() that follows.
        self._rt_closing.clear()

        # Create persistent speaker output stream (callback-driven).
        # State reset is gated on stream creation: a second accept() while
        # another session is mid-playback must not clobber the live buffer.
        if self._rt_output_stream is None:
            with self._rt_buf_lock:
                self._rt_output_buffer.clear()
                self._rt_buf_offset = 0
                self._rt_buffered_bytes = 0
                self._rt_dropped_bytes = 0
                self._rt_priming = True
                self._rt_response_complete = False
                self._rt_prime_idle_blocks = 0
                self._rt_interrupted = False
            self._start_rt_output()

        await self.start_listening(session)

    def interrupt(self, session: VoiceSession) -> None:
        """Flush outbound queue, stop playback (sync)."""
        self._playing_sessions.discard(session.id)
        # Realtime path: flush the persistent output buffer and signal
        # the speaker callback to output silence.  The flag also prevents
        # send_audio() from refilling the buffer before the provider stops.
        with self._rt_buf_lock:
            queued = len(self._rt_output_buffer)
            self._rt_output_buffer.clear()
            self._rt_buf_offset = 0
            self._rt_buffered_bytes = 0
            self._rt_dropped_bytes = 0
            self._rt_priming = True
            self._rt_response_complete = False
            self._rt_prime_idle_blocks = 0
            self._rt_interrupted = True
        # A cancelled response never reaches the normal drained-response
        # boundary. End the playback lifecycle here so capture stops consuming
        # a stale reference timeline; preserve the converged adaptive filter.
        self._aec_end_playback(session.id)
        logger.info(
            "[INTERRUPT] flushed %d chunks, speaker muted (session %s)",
            queued,
            session.id,
        )
        # VoiceChannel path: cancel the playback task
        task = self._playback_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()

    def end_of_response(self, session: VoiceSession) -> None:
        """Mark the AI response complete — release a partial prebuffer.

        Lets the speaker callback drain a final buffer smaller than the
        prebuffer threshold (short responses).  Ignored while interrupted:
        providers fire response_end on barge-in too (e.g. Gemini's
        ``interrupted`` message), and that stale signal must not release
        the priming gate for the next response.
        """
        if not self._realtime_mode:
            return
        with self._rt_buf_lock:
            if self._rt_interrupted:
                return
            self._rt_response_complete = True

    @property
    def rt_underruns(self) -> int:
        """Mid-response speaker buffer starvations (realtime path).

        Each underrun re-arms the prebuffer, converting scattered silence
        gaps into one re-prime; a non-zero count means audio chunks are
        arriving slower than real time (event-loop or network pressure).
        """
        return self._rt_underruns

    def set_input_muted(self, session: VoiceSession, muted: bool) -> None:
        """Mute/unmute the microphone input for a session."""
        if muted:
            self._muted_sessions.add(session.id)
        else:
            self._muted_sessions.discard(session.id)
        logger.info("Input muted=%s for session %s", muted, session.id)

    def set_input_gated(self, session: VoiceSession, gated: bool) -> None:
        """Gate/un-gate audio input for primary speaker mode."""
        if gated:
            self._gated_sessions.add(session.id)
        else:
            self._gated_sessions.discard(session.id)
        logger.info("Input gated=%s for session %s", gated, session.id)

    def on_client_disconnected(self, callback: TransportDisconnectCallback) -> None:
        self._disconnect_callbacks.append(callback)

    def on_speaker_change(self, callback: SpeakerChangeCallback) -> None:
        self._speaker_change_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Realtime speaker output (persistent callback-driven stream)
    # -------------------------------------------------------------------------

    def _start_rt_output(self) -> None:
        """Create a persistent callback-driven speaker output stream.

        PortAudio's audio thread pulls PCM data from the chunk deque.
        When no AI audio is queued the callback feeds silence, keeping
        the stream alive and latency-free.
        """
        sd = self._sd
        output_blocksize = int(self._output_sample_rate * self._block_duration_ms / 1000)
        has_aec = self._aec is not None
        if sys.platform == "darwin":
            out_latency: str = "high"
        elif has_aec:
            out_latency = "low"
        else:
            out_latency = "high"

        out = sd.RawOutputStream(
            samplerate=self._output_sample_rate,
            blocksize=output_blocksize,
            channels=self._channels,
            dtype="int16",
            device=self._output_device,
            latency=out_latency,
            callback=self._rt_speaker_callback,
        )
        out.start()
        self._rt_output_stream = out
        logger.info(
            "Realtime speaker stream: rate=%dHz blocksize=%d latency=%s device=%s",
            self._output_sample_rate,
            output_blocksize,
            out_latency,
            self._output_device or "default",
        )

    def _rt_speaker_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Pull queued audio into the output buffer; fill gaps with silence."""
        if status:
            logger.warning("Speaker callback status: %s", status)

        bytes_needed = frames * self._channels * 2
        written = 0
        underrun_no = 0
        response_drained = False

        with self._rt_buf_lock:
            buf = self._rt_output_buffer
            # When interrupted, never drain — barge-in mutes playback within
            # one callback (~20ms).  While priming, hold silence until enough
            # audio is buffered to ride out provider burst jitter; released
            # early when the response is complete (short responses) or after
            # ~100ms without a new append (missing end-of-response valve).
            # Both paths fall through so the tail still runs: the AEC
            # reference and played callbacks must see silence blocks too.
            draining = not self._rt_interrupted
            if draining and self._rt_priming:
                if self._rt_response_complete and self._rt_buffered_bytes == 0:
                    # A response with no audio still has an AEC activation
                    # from response_start that must be released.
                    self._rt_response_complete = False
                    response_drained = True
                release = (
                    self._rt_buffered_bytes >= max(self._rt_prebuffer_bytes, 1)
                    or (self._rt_response_complete and self._rt_buffered_bytes > 0)
                    or (
                        self._rt_buffered_bytes > 0
                        and self._rt_prime_idle_blocks >= self._rt_prime_max_idle_blocks
                    )
                )
                if release:
                    # Drain starts in this same callback — no wasted block.
                    self._rt_priming = False
                    self._rt_prime_idle_blocks = 0
                else:
                    self._rt_prime_idle_blocks += 1
                    draining = False

            if draining:
                while written < bytes_needed and buf:
                    chunk = buf[0]
                    avail = len(chunk) - self._rt_buf_offset
                    n = min(avail, bytes_needed - written)
                    src_start = self._rt_buf_offset
                    outdata[written : written + n] = chunk[src_start : src_start + n]
                    written += n
                    self._rt_buf_offset += n
                    if self._rt_buf_offset >= len(chunk):
                        buf.popleft()
                        self._rt_buf_offset = 0
                self._rt_buffered_bytes -= written

                if written < bytes_needed:
                    # Buffer exhausted mid-block: re-arm the prebuffer.  A
                    # clean end (response complete) is expected; anything
                    # else is a mid-response starvation — count it.
                    self._rt_priming = True
                    self._rt_prime_idle_blocks = 0
                    if self._rt_response_complete:
                        self._rt_response_complete = False
                        response_drained = True
                    else:
                        self._rt_underruns += 1
                        underrun_no = self._rt_underruns

        # Fill remaining with silence (the whole block when interrupted
        # or priming)
        if written < bytes_needed:
            outdata[written:] = b"\x00" * (bytes_needed - written)

        # Log outside the lock, capped like the SIP pacer's underrun warnings.
        if underrun_no and underrun_no <= 5:
            logger.warning(
                "Speaker underrun #%d — buffer starved mid-response, re-priming",
                underrun_no,
            )

        # One physical speaker, so one stream owns this playback: the same
        # session the capture callback tags its frames with.
        session = next(iter(self._sessions.values()), None)

        # Once playback starts, feed EVERY hardware block, including silence
        # inserted for network jitter or interruption. Capture keeps advancing
        # on the mic thread, so skipping render-silence compresses AEC3's
        # reference timeline and makes it cancel the wrong point in history.
        # When capture is muted, gated, or half-duplex, both timelines pause.
        if (
            self._aec is not None
            and session is not None
            and not self._aec_capture_paused(session.id)
        ):
            if written > 0:
                self._aec_begin_playback(session.id)
            if session.id in self._aec_active_sessions:
                self._aec_feed_played(bytearray(bytes(outdata)), session.id)

        if session is not None and response_drained:
            # The persistent realtime stream stays open across responses, so
            # playback is bypassed at the drained response boundary rather
            # than at stream close. The learned hardware filter survives.
            self._aec_end_playback(session.id)

        # Notify listeners about played audio — every block, silence
        # included.  The pipeline AEC reference (wired via on_audio_played)
        # must be continuous: skipping silent blocks compresses the
        # reference timeline vs. the actual speaker output, forcing AEC3 to
        # re-estimate its delay after every gap — measured as ~1s echo-leak
        # windows at each response start, which Gemini's server VAD can
        # mistake for user speech (false barge-in).
        if self._audio_played_callbacks and session is not None:
            played_frame = AudioFrame(
                data=bytes(outdata),
                sample_rate=self._output_sample_rate,
                channels=self._channels,
                sample_width=2,
                metadata={
                    "playback_ended": response_drained,
                    "played_bytes": written,
                    # While capture is paused (mute/gate/half-duplex) the mic
                    # thread drops frames, so the pipeline-AEC reference must
                    # pause in step — the transport-AEC feed above already
                    # does.  The broadcast itself continues: playback is
                    # physically ongoing, and level/position listeners must
                    # keep seeing it.  The pipeline consumer honours the flag.
                    "capture_paused": self._aec_capture_paused(session.id),
                },
            )
            for cb in self._audio_played_callbacks:
                with contextlib.suppress(Exception):
                    cb(session, played_frame)

        # Track playing state based on buffer content
        if written > 0:
            for sid in self._sessions:
                self._playing_sessions.add(sid)
        elif not buf:
            for sid in list(self._playing_sessions):
                self._playing_sessions.discard(sid)

    def _stop_rt_output(self) -> None:
        """Close the persistent realtime speaker stream."""
        with self._rt_buf_lock:
            self._rt_output_buffer.clear()
            self._rt_buf_offset = 0
            self._rt_buffered_bytes = 0
            self._rt_dropped_bytes = 0
            self._rt_priming = True
            self._rt_response_complete = False
            self._rt_prime_idle_blocks = 0
        out = self._rt_output_stream
        if out is not None:
            self._rt_output_stream = None
            try:
                out.abort()
                out.close()
            except Exception:  # noqa: S110
                logger.debug("Error closing realtime output stream", exc_info=True)

    # -------------------------------------------------------------------------
    # AEC helpers
    # -------------------------------------------------------------------------

    def _configure_aec_delay_from_streams(self, input_stream: Any) -> None:
        """Seed an unset WebRTC delay from PortAudio's actual stream latencies."""
        if self._aec is None or self._rt_output_stream is None:
            return
        try:
            input_latency_ms = float(input_stream.latency) * 1000
        except Exception:
            logger.debug("AEC delay auto-configuration unavailable", exc_info=True)
            return
        self._configure_aec_delay(input_latency_ms)

    def _configure_aec_delay(self, input_latency_ms: float) -> None:
        """Seed an unset WebRTC delay from a known input latency.

        Split from the stream variant so a shared capture source, which owns
        the input stream this backend never sees, can report its own latency.
        """
        if self._aec is None or self._rt_output_stream is None:
            return

        setter = getattr(self._aec, "set_stream_delay_ms", None)
        configured_delay = getattr(self._aec, "stream_delay_ms", None)
        if not callable(setter) or configured_delay != 0:
            return

        try:
            output_latency_ms = float(self._rt_output_stream.latency) * 1000
        except Exception:
            logger.debug("AEC delay auto-configuration unavailable", exc_info=True)
            return

        # WebRTC clamps reported stream delay at 500 ms. The acoustic travel
        # time for a local device is negligible next to PortAudio buffering,
        # whose input + output latency is the relevant render/capture offset.
        delay_ms = min(500, max(0, round(input_latency_ms + output_latency_ms)))
        if delay_ms == 0:
            return

        try:
            setter(delay_ms)
        except Exception:
            logger.warning("AEC delay auto-configuration failed", exc_info=True)
            return

        logger.info(
            "AEC delay auto-configured from PortAudio: input=%.1fms output=%.1fms total=%dms",
            input_latency_ms,
            output_latency_ms,
            delay_ms,
        )

    def _aec_capture_paused(self, stream: str) -> bool:
        """Whether capture is paused, so reference time must pause as well."""
        return (
            stream in self._muted_sessions
            or stream in self._gated_sessions
            or (self._mute_mic_during_playback and bool(self._playing_sessions))
        )

    def _aec_begin_playback(self, stream: str) -> None:
        """Activate transport AEC once, when physical playback starts."""
        if self._aec is None or stream in self._aec_active_sessions:
            return
        try:
            self._aec.set_stream_active(stream, True)
        except Exception:
            logger.exception("Failed to activate transport AEC for stream %s", stream)
            return
        self._aec_active_sessions.add(stream)

    def _aec_end_playback(self, stream: str) -> None:
        """Pause transport AEC without destroying its learned echo path."""
        if self._aec is None or stream not in self._aec_active_sessions:
            return
        try:
            self._aec.set_stream_active(stream, False)
        except Exception:
            logger.exception("Failed to deactivate transport AEC for stream %s", stream)
            return
        self._aec_active_sessions.discard(stream)
        self._ref_buffers.pop(stream, None)

    def _aec_feed_played(self, played: bytearray, stream: str) -> None:
        """Feed actually-played speaker bytes to the AEC as reference.

        Called from ``_output_callback`` so the reference is time-aligned
        with what the speaker is outputting.  Accumulates bytes and feeds
        them in exact block-aligned chunks.  When the output and input
        sample rates differ, each block is resampled to the input rate
        before feeding the AEC.

        Args:
            played: The speaker bytes actually written this block.
            stream: The session this playback belongs to — the same key the
                capture path passes to ``aec.process()``.
        """
        ref_buffer = self._ref_buffers.setdefault(stream, bytearray())
        ref_buffer.extend(played)

        if self._aec_needs_resample:
            # Chunk at the output rate, then resample each block to input rate
            block = self._aec_out_block_bytes
            while len(ref_buffer) >= block:
                chunk = bytes(ref_buffer[:block])
                del ref_buffer[:block]
                out_frame = AudioFrame(
                    data=chunk,
                    sample_rate=self._output_sample_rate,
                    channels=self._channels,
                    sample_width=2,
                )
                ref_frame = self._aec_resampler.resample(  # ty: ignore[unresolved-attribute]
                    out_frame,
                    self._input_sample_rate,
                    self._channels,
                    2,
                    stream,
                )
                self._aec.feed_reference(ref_frame, stream)  # ty: ignore[unresolved-attribute]
        else:
            block = self._aec_block_bytes
            while len(ref_buffer) >= block:
                chunk = bytes(ref_buffer[:block])
                del ref_buffer[:block]
                frame = AudioFrame(
                    data=chunk,
                    sample_rate=self._input_sample_rate,
                    channels=self._channels,
                    sample_width=2,
                )
                self._aec.feed_reference(frame, stream)  # ty: ignore[unresolved-attribute]

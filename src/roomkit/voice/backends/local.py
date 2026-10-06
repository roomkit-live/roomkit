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
import threading
import uuid
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import await_interruptible
from roomkit.voice._sounddevice import import_sounddevice
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends._local_speaker import MAX_BUFFER_SECONDS, LocalSpeaker
from roomkit.voice.backends.base import (
    AECTapCallback,
    AudioPlayedCallback,
    AudioReceivedCallback,
    PlaybackErrors,
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
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import sounddevice as sd

    from roomkit.voice.backends._local_speaker import SpeakerBlock
    from roomkit.voice.capture.base import AudioCaptureSource, CaptureSubscription
    from roomkit.voice.pipeline.aec.base import AECProvider

logger = logging.getLogger("roomkit.voice.local")

# After playback drains or is cut, the device and the room still sound: the
# transport's AEC keeps cancelling this long on the silent reference (RFC
# §12.3.4), as a pipeline AEC does (VoiceChannel's _AEC_ECHO_TAIL_S).
_AEC_ECHO_TAIL_MS = 500

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


def _stream_latency_ms(stream: Any) -> float | None:
    """A PortAudio stream's reported latency in ms, when it reports one."""
    try:
        return float(stream.latency) * 1000
    except Exception:
        logger.debug("Stream latency unavailable", exc_info=True)
        return None


async def _one_chunk(pcm: bytes, sample_rate: int) -> AsyncIterator[AudioChunk]:
    """Raw PCM played the way a stream is: as its one chunk."""
    yield AudioChunk(data=pcm, sample_rate=sample_rate)


class LocalAudioBackend(VoiceBackend):
    """VoiceBackend that uses the system microphone and speakers.

    Audio captured from the microphone is delivered as ``AudioFrame`` objects
    via the ``on_audio_received`` callback.  Outbound audio is played through
    the output device by one stream, opened at the first playback and kept
    open, so the echo delay an AEC tracks holds still across responses
    (RFC §12.3.4).

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
        mute_mic_during_playback: If True, suppress mic frames while the
            speaker is playing (half-duplex): echo cannot trigger VAD or a
            false barge-in, and the user cannot talk over the agent either.
            The default (None) is half-duplex only without ``aec``: an AEC
            removes the echo, so the mic stays open.
        rt_prebuffer_ms: Audio to accumulate before starting (or resuming
            after an underrun) speaker playback, realtime or streamed TTS.
            Absorbs the burst jitter of a realtime provider or a TTS stream
            the same way the SIP pacer's prebuffer does — without it, any
            momentary starvation inserts an audible mid-sentence gap.  A
            response shorter than this plays once complete.  ``0`` plays
            from the first byte.
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
        mute_mic_during_playback: bool | None = None,
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
        self._playback_tasks: dict[str, asyncio.Task[None]] = {}

        # Half-duplex echo suppression: by default only when nothing cancels it
        self._mute_mic_during_playback = (
            aec is None if mute_mic_during_playback is None else mute_mic_during_playback
        )

        # Realtime transport state
        self._muted_sessions: set[str] = set()
        self._gated_sessions: set[str] = set()
        self._disconnect_callbacks: list[TransportDisconnectCallback] = []
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []

        # Realtime mode flag — set by accept(), controls mic dispatch
        self._realtime_mode = False

        # The speaker: one persistent output stream for every response, opened
        # by accept() (realtime) or the first streamed playback (VoiceChannel).
        self._speaker = LocalSpeaker(
            self._sd,
            sample_rate=output_sample_rate,
            channels=channels,
            block_duration_ms=block_duration_ms,
            device=output_device,
            low_latency=aec is not None,
            prebuffer_ms=rt_prebuffer_ms,
            on_block=self._on_speaker_block,
            on_finished=self._on_speaker_finished,
        )
        # Set by disconnect()/close(): realtime audio arriving afterwards is
        # dropped; accept() re-arms it.
        self._rt_closing = threading.Event()
        self._rt_dropped_bytes = 0
        # VoiceChannel mode: the session whose response the speaker plays, and
        # the event that response's playback waits on until it has drained.
        self._speaker_stream: str | None = None
        self._speaker_drained: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None

        # --- AEC (transport-level reference feeding) ---
        self._aec = aec
        self._aec_active_sessions: set[str] = set()
        # Blocks of echo tail left to cancel after a playback drained or was cut.
        self._aec_tail: dict[str, int] = {}
        self._aec_tail_blocks = max(1, _AEC_ECHO_TAIL_MS // block_duration_ms)
        # The capture side's latency, once known: with the speaker's, it seeds
        # an unset WebRTC delay (_configure_aec_delay).
        self._input_latency_ms: float | None = None
        # Debug taps of this transport's AEC (RFC §12.3.15): the reference fed
        # since the last captured frame, by stream, for ``aec_reference``.
        self._aec_tap_callbacks: list[AECTapCallback] = []
        self._tap_ref: dict[str, bytearray] = {}
        self._tap_ref_lock = threading.Lock()
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
        # A streamed response still playing ends with its session.
        task = self._playback_tasks.pop(session.id, None)
        if task is not None:
            task.cancel()
        await self.stop_listening(session)
        self._close_speaker()
        session.state = VoiceSessionState.ENDED
        self._sessions.pop(session.id, None)
        self._playing_sessions.discard(session.id)
        self._gated_sessions.discard(session.id)
        self._muted_sessions.discard(session.id)
        if self._speaker_stream == session.id:
            self._speaker_stream = None
        self._aec_end_playback(session.id)
        self._ref_buffers.pop(session.id, None)
        with self._tap_ref_lock:
            self._tap_ref.pop(session.id, None)
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
        self._close_speaker()
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
            if aec_ref is not None and self._aec_tap_callbacks and loop_ref is not None:
                reference = self._take_tap_reference(session.id, frame)
                loop_ref.call_soon_threadsafe(self._emit_aec_tap, session, frame, reference)
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
        self._note_input_latency(source.input_latency_ms)

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
        self._note_input_latency(_stream_latency_ms(stream))
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

        Every response goes through the one persistent speaker stream.  In
        realtime mode the bytes are queued and this returns at once: a
        provider streams a response as many calls.  In VoiceChannel mode a
        response is one call, raw bytes or a stream of chunks, which returns
        once that response has played.

        Args:
            session: The target session.
            audio: Raw PCM-16 LE bytes or an async iterator of AudioChunks.
        """
        if self._realtime_mode:
            if isinstance(audio, bytes):
                self._buffer_realtime_audio(session, audio)
            return

        # VoiceChannel path
        chunks = _one_chunk(audio, self._output_sample_rate) if isinstance(audio, bytes) else audio
        self._playing_sessions.add(session.id)
        try:
            with PlaybackErrors(logger, "Error playing audio for session %s", session.id) as play:
                await self._play_stream(session, play.watch(chunks))
        finally:
            self._playing_sessions.discard(session.id)

    def _buffer_realtime_audio(self, session: VoiceSession, audio: bytes) -> None:
        """Queue a realtime response's bytes on the speaker, dropping past its bound."""
        if not audio or self._rt_closing.is_set():
            return
        accepted, resumed = self._speaker.append(audio)
        dropped = len(audio) - accepted
        self._rt_dropped_bytes += dropped
        # Added after the queue's lock is released: when the callback has just
        # seen an empty queue and clears the playing state, this write comes
        # last, so capture stays muted while the new audio plays.
        if accepted:
            self._playing_sessions.add(session.id)
        if dropped and self._rt_dropped_bytes == dropped:
            logger.warning(
                "Realtime speaker buffer reached its %ds bound; dropping excess audio",
                MAX_BUFFER_SECONDS,
            )
        if resumed:
            logger.info("[INTERRUPT] cleared — buffering for resume")

    async def _play_stream(
        self,
        session: VoiceSession,
        chunks: AsyncIterator[AudioChunk],
    ) -> None:
        """Play one streamed response on the persistent speaker.

        The chunks are queued on the speaker, waiting for room rather than
        dropping any; this returns once the response has drained and the
        device has played it out.  Consumption runs in a cancellable task so
        that ``cancel_audio()`` aborts both the TTS stream and the wait.
        """
        speaker = self._open_speaker()
        loop = asyncio.get_running_loop()
        drained = asyncio.Event()
        self._speaker_stream = session.id
        self._speaker_drained = (loop, drained)
        speaker.begin_response()

        async def _run() -> None:
            async for chunk in chunks:
                if session.id not in self._playing_sessions:
                    return
                if chunk.data and not await self._queue_on_speaker(chunk.data):
                    return
            speaker.end_response()
            await drained.wait()
            # The last bytes left the queue; the device still plays them out.
            await asyncio.sleep(speaker.latency)

        task = asyncio.create_task(_run())
        self._playback_tasks[session.id] = task
        completed = False
        try:
            # False unless cancel_audio() cut it; a cancellation of the caller
            # itself, or the TTS stream's failure, is raised.
            completed = not await await_interruptible(task)
        finally:
            self._playback_tasks.pop(session.id, None)
            if self._speaker_drained is not None and self._speaker_drained[1] is drained:
                self._speaker_drained = None
            if not completed:
                # Cut, cancelled or failed: what is still queued never plays.
                self._cut_speaker(session.id)

    async def _queue_on_speaker(self, data: bytes) -> bool:
        """Queue streamed audio on the speaker, waiting while its queue is full.

        False when the speaker stopped (closed, or its device went away).
        """
        while True:
            accepted, _ = self._speaker.append(data)
            data = data[accepted:]
            if not data:
                return True
            if not self._speaker.is_open:
                return False
            await asyncio.sleep(self._block_duration_ms / 1000)

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

    def on_aec_tap(self, callback: AECTapCallback) -> Callable[[], None] | None:
        """Debug taps of this backend's AEC (RFC §12.3.15): ``transport_raw`` and
        ``aec_reference`` for every captured frame, on the event loop."""
        if self._aec is None:
            return None
        self._aec_tap_callbacks.append(callback)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._aec_tap_callbacks.remove(callback)

        return unsubscribe

    def _tap_reference(self, stream: str, data: bytes) -> None:
        """Keep reference bytes fed to the AEC for the next captured frames' tap."""
        if self._aec_tap_callbacks:
            with self._tap_ref_lock:
                self._tap_ref.setdefault(stream, bytearray()).extend(data)

    def _take_tap_reference(self, stream: str, frame: AudioFrame) -> AudioFrame:
        """The reference fed while *frame* was captured: its length, the bytes the
        AEC received meanwhile and silence where it received none."""
        size = len(frame.data)
        with self._tap_ref_lock:
            pending = self._tap_ref.get(stream)
            taken = bytes(pending[:size]) if pending else b""
            if pending:
                del pending[:size]
        return AudioFrame(
            data=taken + bytes(size - len(taken)),
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
        )

    def _emit_aec_tap(
        self, session: VoiceSession, frame: AudioFrame, reference: AudioFrame
    ) -> None:
        """Hand the taps their frames, on the event loop (never the audio threads)."""
        for callback in list(self._aec_tap_callbacks):
            try:
                callback(session, "transport_raw", frame)
                callback(session, "aec_reference", reference)
            except Exception:
                logger.exception("AEC debug tap failed for session %s", session.id)

    async def cancel_audio(self, session: VoiceSession) -> bool:
        was_playing = session.id in self._playing_sessions
        if was_playing:
            self._playing_sessions.discard(session.id)
            self._cut_speaker(session.id)
            # Cancel the consumption task — unblocks the async-for
            # that may be waiting on the TTS HTTP stream.
            task = self._playback_tasks.pop(session.id, None)
            if task is not None:
                task.cancel()
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

        # The persistent speaker stream.  Opening it starts from a clean queue,
        # so a second accept() while another session is mid-playback never
        # clobbers the live buffer.
        self._open_speaker()

        await self.start_listening(session)

    def interrupt(self, session: VoiceSession) -> None:
        """Flush outbound queue, stop playback (sync)."""
        self._playing_sessions.discard(session.id)
        # Flush the speaker's queue and play silence until the next response:
        # a provider still streaming the cut one must not refill it.  The AEC
        # keeps cancelling what the device and the room still play.
        queued = self._cut_speaker(session.id)
        self._rt_dropped_bytes = 0
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
        self._speaker.end_response()

    @property
    def rt_underruns(self) -> int:
        """Mid-response speaker buffer starvations, realtime or streamed TTS.

        Each underrun re-arms the prebuffer, converting scattered silence
        gaps into one re-prime; a non-zero count means audio chunks are
        arriving slower than real time (event-loop or network pressure, or
        a TTS slower than real time between sentences).
        """
        return self._speaker.underruns

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
    # The speaker: one persistent output stream (_local_speaker.py)
    # -------------------------------------------------------------------------

    def _open_speaker(self) -> LocalSpeaker:
        """Open the speaker stream unless it runs, then seed the AEC delay."""
        if not self._speaker.is_open:
            self._speaker.open()
            self._rt_dropped_bytes = 0
            self._configure_aec_delay()
        return self._speaker

    def _close_speaker(self) -> None:
        self._speaker.close()
        self._rt_dropped_bytes = 0

    def _cut_speaker(self, stream: str) -> int:
        """Drop the queued audio; returns the chunks dropped.

        The speaker's next block reports the cut and the AEC keeps cancelling
        through the echo tail.  With no stream running no block comes, so the
        AEC's playback ends here.
        """
        queued = self._speaker.flush()
        if not self._speaker.is_open:
            self._aec_end_playback(stream)
        return queued

    def _speaker_owner(self) -> str | None:
        """The session the speaker's audio belongs to.

        One physical speaker, so one session owns its playback: in realtime
        mode the session the capture callback tags its frames with, in
        VoiceChannel mode the one whose response it plays.
        """
        if self._realtime_mode:
            return next(iter(self._sessions), None)
        stream = self._speaker_stream
        return stream if stream in self._sessions else None

    def _on_speaker_block(self, block: SpeakerBlock) -> None:
        """What a played block means past the device (speaker thread): the
        AEC's reference, played-audio listeners, the playing state and the
        drained wait."""
        stream = self._speaker_owner()
        if stream is None:
            return
        self._aec_on_block(stream, block)
        if self._realtime_mode:
            # Every block, silence included: the pipeline AEC reference (wired
            # via on_audio_played) must be continuous.  Skipping silent blocks
            # compresses the reference timeline against the actual speaker
            # output, forcing AEC3 to re-estimate its delay after every gap —
            # measured as ~1 s echo-leak windows at each response start, which
            # Gemini's server VAD can mistake for user speech (false barge-in).
            self._notify_played(stream, block, ended=block.drained)
            self._track_realtime_playing(block)
            return
        # A streamed response's playback is send_audio()'s call; the channel
        # owns what follows it (VoiceChannel's AEC echo tail).
        if stream in self._playing_sessions:
            self._notify_played(stream, block, ended=False)
        if block.drained:
            self._signal_drained()

    def _on_speaker_finished(self, closed: bool) -> None:
        """The stream stopped (speaker thread): release a playback waiting on it."""
        if not closed:
            logger.warning("Speaker stream stopped by the audio device")
        self._signal_drained()

    def _signal_drained(self) -> None:
        waiter = self._speaker_drained
        if waiter is None:
            return
        loop, event = waiter
        with contextlib.suppress(RuntimeError):  # the loop has closed
            loop.call_soon_threadsafe(event.set)

    def _notify_played(self, stream: str, block: SpeakerBlock, *, ended: bool) -> None:
        """Hand listeners the played block: the time-aligned reference for a
        pipeline AEC, and the output level at playback pace."""
        session = self._sessions.get(stream)
        if not self._audio_played_callbacks or session is None:
            return
        frame = AudioFrame(
            data=block.data,
            sample_rate=self._output_sample_rate,
            channels=self._channels,
            sample_width=2,
            metadata={
                "playback_ended": ended,
                "played_bytes": block.written,
                # While capture is paused (mute/gate/half-duplex) the mic
                # thread drops frames, so the pipeline-AEC reference must
                # pause in step — the transport-AEC feed already does.  The
                # broadcast itself continues: playback is physically ongoing,
                # and level/position listeners must keep seeing it.  The
                # pipeline consumer honours the flag.
                "capture_paused": self._aec_capture_paused(stream),
            },
        )
        for cb in list(self._audio_played_callbacks):
            with contextlib.suppress(Exception):
                cb(session, frame)

    def _track_realtime_playing(self, block: SpeakerBlock) -> None:
        """Realtime mode: the sessions play while the speaker has their audio."""
        if block.written > 0:
            for sid in self._sessions:
                self._playing_sessions.add(sid)
        elif not block.queued:
            for sid in list(self._playing_sessions):
                self._playing_sessions.discard(sid)

    # -------------------------------------------------------------------------
    # AEC helpers
    # -------------------------------------------------------------------------

    def _note_input_latency(self, latency_ms: float | None) -> None:
        """Record the capture side's latency, then seed the AEC delay if it can."""
        if latency_ms is None:
            return
        self._input_latency_ms = latency_ms
        self._configure_aec_delay()

    def _configure_aec_delay(self) -> None:
        """Seed an unset WebRTC delay from PortAudio's actual stream latencies.

        It needs both: the capture side's (this backend's input stream, or a
        shared capture source's report) and the speaker's.  Called when either
        opens, so whichever opens last seeds it, in either mode.
        """
        if self._aec is None or self._input_latency_ms is None or not self._speaker.is_open:
            return

        setter = getattr(self._aec, "set_stream_delay_ms", None)
        configured_delay = getattr(self._aec, "stream_delay_ms", None)
        if not callable(setter) or configured_delay != 0:
            return

        output_latency_ms = self._speaker.latency * 1000
        if output_latency_ms <= 0:
            return  # the speaker reports none: nothing to seed from

        # WebRTC clamps reported stream delay at 500 ms. The acoustic travel
        # time for a local device is negligible next to PortAudio buffering,
        # whose input + output latency is the relevant render/capture offset.
        delay_ms = min(500, max(0, round(self._input_latency_ms + output_latency_ms)))
        if delay_ms == 0:
            return

        try:
            setter(delay_ms)
        except Exception:
            logger.warning("AEC delay auto-configuration failed", exc_info=True)
            return

        logger.info(
            "AEC delay auto-configured from PortAudio: input=%.1fms output=%.1fms total=%dms",
            self._input_latency_ms,
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
        self._aec_tail.pop(stream, None)
        if self._aec is None or stream not in self._aec_active_sessions:
            return
        try:
            self._aec.set_stream_active(stream, False)
        except Exception:
            logger.exception("Failed to deactivate transport AEC for stream %s", stream)
            return
        self._aec_active_sessions.discard(stream)
        self._ref_buffers.pop(stream, None)

    def _aec_on_block(self, stream: str, block: SpeakerBlock) -> None:
        """Run the transport AEC's playback lifecycle on one played block.

        Playback starts at the first block carrying audio.  From then every
        block is fed, silence included: capture keeps advancing on the mic
        thread, so skipping render silence would compress AEC3's reference
        timeline and make it cancel the wrong point in history.  When the
        playback drains or is cut, the device and the room still sound, so the
        AEC keeps cancelling for the echo tail on the silent reference that
        follows (RFC §12.3.4), then is bypassed with its learned filter kept.
        While capture is paused (muted, gated, half-duplex) both timelines
        pause, the tail's countdown included.
        """
        if self._aec is None:
            return
        paused = self._aec_capture_paused(stream)
        if block.written > 0 and not paused:
            self._aec_tail.pop(stream, None)
            self._aec_begin_playback(stream)
        if stream not in self._aec_active_sessions:
            return
        if not paused:
            self._aec_feed_played(bytearray(block.data), stream)
        if block.drained or block.flushed:
            self._aec_tail[stream] = self._aec_tail_blocks
        elif not paused and block.written == 0 and stream in self._aec_tail:
            self._aec_tail[stream] -= 1
            if self._aec_tail[stream] <= 0:
                self._aec_end_playback(stream)

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
                self._tap_reference(stream, ref_frame.data)
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
                self._tap_reference(stream, frame.data)

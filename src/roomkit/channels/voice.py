"""Voice channel for real-time audio communication."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.channels._voice_hooks import VoiceHooksMixin
from roomkit.channels._voice_stt import VoiceSTTMixin
from roomkit.channels._voice_tts import VoiceTTSMixin
from roomkit.channels._voice_turn import VoiceTurnMixin
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    EventType,
    HookTrigger,
)
from roomkit.voice.base import VoiceCapability
from roomkit.voice.interruption import InterruptionConfig
from roomkit.voice.utils import rms_db

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.event import RoomEvent
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization.base import DiarizationResult
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.pipeline.turn.base import TurnEntry
    from roomkit.voice.pipeline.vad.base import VADEvent
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.voice")


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


# Buffer ~100ms of audio before sending to STT stream.
# Small per-frame chunks cause resampling artifacts at frame boundaries
# when the provider resamples (e.g. 16kHz -> 24kHz).  Larger chunks
# give the stateless resampler enough context for clean interpolation.
_STT_STREAM_BUFFER_BYTES = 3200  # 100ms at 16kHz mono 16-bit

# Seconds of silence before the continuous STT audio generator closes
# the current stream.  Triggers the Gradium drain timeout which yields
# accumulated text as a final result.  Must be longer than the longest
# normal intra-utterance pause (~600ms) but short enough to feel responsive.
_STT_INACTIVITY_TIMEOUT_S = 1.0


@dataclass
class _STTStreamState:
    """Track an active streaming STT session."""

    queue: asyncio.Queue[Any]  # Queue[AudioChunk | None]
    task: asyncio.Task[Any] | None = None  # consumer task running transcribe_stream
    frame_buffer: bytearray = field(default_factory=bytearray)
    frame_buffer_rate: int = 16000
    final_text: str | None = None
    partial_text: str | None = None
    error: bool = False
    cancelled: bool = False


@dataclass
class TTSPlaybackState:
    """Track ongoing TTS playback for barge-in detection."""

    session_id: str
    text: str
    started_at: datetime = field(default_factory=_utcnow)
    total_duration_ms: int | None = None

    @property
    def position_ms(self) -> int:
        """Estimate current playback position based on elapsed time."""
        elapsed = datetime.now(UTC) - self.started_at
        return int(elapsed.total_seconds() * 1000)


class VoiceChannel(VoiceSTTMixin, VoiceTTSMixin, VoiceHooksMixin, VoiceTurnMixin, Channel):
    """Real-time voice communication channel.

    Supports three STT modes:
    - **VAD mode** (default): VAD segments speech, streaming STT during speech
      with batch fallback on SPEECH_END.
    - **Continuous mode**: No VAD + streaming STT provider — all audio streamed,
      provider handles endpointing.
    - **Batch mode** (``batch_mode=True``): No VAD, audio accumulates post-pipeline.
      Caller controls when to transcribe via :meth:`flush_stt`.  Useful for
      dictation, voicemail, and audio-file transcription with offline models.

    When a VoiceBackend and AudioPipelineConfig are configured, the channel:
    - Registers for raw audio frames from the backend via on_audio_received
    - Routes frames through the AudioPipeline inbound chain:
      [Resampler] -> [Recorder] -> [AEC] -> [AGC] -> [Denoiser] -> VAD ->
      [Diarization] + [DTMF]
    - Fires hooks based on pipeline events (speech, silence, DTMF, recording, etc.)
    - Transcribes speech using the STT provider
    - Optionally evaluates turn completion via TurnDetector
    - Synthesizes AI responses using TTS and streams to the client

    When no pipeline is configured, the channel operates without VAD — the backend
    must handle speech detection externally.
    """

    channel_type = ChannelType.VOICE
    category = ChannelCategory.TRANSPORT
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        *,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
        backend: VoiceBackend | None = None,
        pipeline: AudioPipelineConfig | None = None,
        streaming: bool = True,
        enable_barge_in: bool = True,
        barge_in_threshold_ms: int = 200,
        interruption: InterruptionConfig | None = None,
        batch_mode: bool = False,
        voice_map: dict[str, str] | None = None,
        max_audio_frames_per_second: int | None = None,
        tts_filter: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._stt = stt
        self._tts = tts
        self._backend = backend
        self._pipeline_config = pipeline
        self._streaming = streaming
        self._framework: RoomKit | None = None
        # Lock for shared state accessed from both asyncio and audio threads
        self._state_lock = threading.Lock()
        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Track TTS playback for barge-in detection
        self._playing_sessions: dict[str, TTSPlaybackState] = {}
        # Signalled when send_audio() returns for a session (before drain delay)
        self._playback_done_events: dict[str, asyncio.Event] = {}
        # The instantiated pipeline engine (if config provided)
        self._pipeline: AudioPipeline | None = None
        # Pending turns for turn detection (session_id -> list of TurnEntry)
        self._pending_turns: dict[str, list[TurnEntry]] = {}
        # Pending audio for audio-native turn detectors (session_id -> accumulated PCM)
        self._pending_audio: dict[str, bytearray] = {}
        # Active streaming STT sessions (session_id -> state)
        self._stt_streams: dict[str, _STTStreamState] = {}
        # Continuous STT mode: stream all audio to STT, no local VAD
        self._continuous_stt = False
        # Post-denoiser energy barge-in state (continuous STT mode)
        self._barge_in_energy_count = 0
        # Timestamp of last TTS playback end per session (for echo diagnostics)
        self._last_tts_ended_at: dict[str, float] = {}
        # Track scheduled fire-and-forget tasks for clean shutdown
        self._scheduled_tasks: set[asyncio.Task[Any]] = set()
        # Batch STT mode: accumulate audio, caller flushes manually
        if batch_mode and stt is None:
            raise ValueError("batch_mode=True requires an STT provider")
        if batch_mode and pipeline is not None and pipeline.vad is not None:
            raise ValueError("batch_mode=True is incompatible with VAD")
        if batch_mode and not streaming:
            raise ValueError("batch_mode=True requires streaming=True for STT")
        self._batch_mode = batch_mode
        self._batch_audio_buffers: dict[str, bytearray] = {}
        self._batch_audio_sample_rate: dict[str, int] = {}
        # Throttle audio level hooks to ~10/sec per direction per session
        self._last_input_level_at: dict[str, float] = {}
        self._last_output_level_at: dict[str, float] = {}
        # Cached event loop for cross-thread scheduling (e.g. PortAudio callback)
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Per-agent voice mapping: channel_id -> TTS voice override
        self._voice_map: dict[str, str] = voice_map or {}
        # TTS text filter: strips markers before synthesis
        self._tts_filter = tts_filter
        # Telemetry spans for voice sessions (session_id -> span_id)
        self._voice_session_spans: dict[str, str] = {}
        # Audio frame rate limiting (session_id -> (window_start, count))
        self._max_fps = max_audio_frames_per_second
        self._frame_counts: dict[str, tuple[float, int]] = {}
        # Dual-signal session ready: tracks sessions where the backend
        # has signalled ready but bind_session() hasn't run yet.
        self._session_ready_pending: set[str] = set()

        # Build InterruptionHandler: explicit config > pipeline config > legacy params
        from roomkit.voice.interruption import InterruptionHandler, InterruptionStrategy

        interruption_config = interruption
        if interruption_config is None and pipeline is not None:
            interruption_config = pipeline.interruption
        if interruption_config is None:
            # Map legacy boolean params: enable_barge_in + threshold mapped
            # to IMMEDIATE with allow_during_first_ms (old behaviour was:
            # interrupt at SPEECH_START if playback_position >= threshold).
            if not enable_barge_in:
                interruption_config = InterruptionConfig(
                    strategy=InterruptionStrategy.DISABLED,
                )
            else:
                interruption_config = InterruptionConfig(
                    strategy=InterruptionStrategy.IMMEDIATE,
                    allow_during_first_ms=barge_in_threshold_ms,
                )
        # Preserve legacy attrs for backwards compat in existing barge-in path
        self._enable_barge_in = interruption_config.strategy != InterruptionStrategy.DISABLED
        self._barge_in_threshold_ms = interruption_config.min_speech_ms

        backchannel_det = pipeline.backchannel_detector if pipeline else None
        self._interruption_handler = InterruptionHandler(
            interruption_config, backchannel_detector=backchannel_det
        )

        # Wire up pipeline: use explicit config or create a default one when a
        # backend is provided.  A pipeline is required for audio to flow from
        # the backend through STT processing.
        if backend and not pipeline:
            from roomkit.voice.pipeline.config import AudioPipelineConfig as _PipelineCfg

            pipeline = _PipelineCfg()
        if backend and pipeline:
            self._setup_pipeline(backend, pipeline)
        elif backend and VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)
        # Wire session ready callback regardless of pipeline
        if backend:
            backend.on_session_ready(self._on_session_ready)

    def _setup_pipeline(self, backend: VoiceBackend, config: AudioPipelineConfig) -> None:
        """Create AudioPipeline and wire backend -> pipeline -> callbacks."""
        from roomkit.voice.pipeline.engine import AudioPipeline

        self._pipeline = AudioPipeline(
            config,
            backend_capabilities=backend.capabilities,
            backend_feeds_aec_reference=backend.feeds_aec_reference,
        )

        # Backend delivers raw audio -> pipeline processes it
        backend.on_audio_received(self._on_audio_received)

        # Continuous STT: no local VAD, stream all audio to STT provider.
        # Batch mode takes priority — audio accumulates for manual flush.
        self._continuous_stt = (
            not self._batch_mode
            and config.vad is None
            and self._stt is not None
            and self._stt.supports_streaming
        )

        # Pipeline events -> VoiceChannel hooks
        if self._continuous_stt:
            self._pipeline.on_processed_frame(self._on_processed_frame_for_stt)
        elif self._batch_mode and self._stt is not None:
            self._pipeline.on_processed_frame(self._on_processed_frame_for_batch)
        else:
            self._pipeline.on_speech_end(self._on_pipeline_speech_end)
            self._pipeline.on_vad_event(self._on_pipeline_vad_event)
            self._pipeline.on_speech_frame(self._on_pipeline_speech_frame)
        if config.diarization is not None:
            self._pipeline.on_speaker_change(self._on_pipeline_speaker_change)
        if config.dtmf is not None:
            self._pipeline.on_dtmf(self._on_pipeline_dtmf)
        if config.recorder is not None:
            self._pipeline.on_recording_started(self._on_pipeline_recording_started)
            self._pipeline.on_recording_stopped(self._on_pipeline_recording_stopped)

        # Audio level hooks:
        # - Input: fires from pipeline processed-frame callback.
        # - Output: fires from _wrap_outbound() in the TTS mixin (all backends),
        #   AND from on_audio_played at real playback pace (backends that support it).
        #   Both share _last_output_level_at so they naturally deduplicate.
        self._pipeline.on_processed_frame(self._on_processed_frame_for_level)
        if backend.supports_playback_callback:
            backend.on_audio_played(self._on_audio_played_for_level)

        # Wire speaker output to pipeline AEC for time-aligned reference.
        # Only when the backend doesn't already feed AEC reference at
        # the transport level (same AEC instance) — otherwise double-
        # feeding corrupts the AEC's internal ring buffer.
        if (
            config.aec is not None
            and backend.supports_playback_callback
            and not backend.feeds_aec_reference
        ):

            def _on_audio_played(session: VoiceSession, frame: AudioFrame) -> None:
                if self._pipeline is not None:
                    self._pipeline.feed_aec_reference(frame)

            backend.on_audio_played(_on_audio_played)
            self._pipeline.enable_playback_aec_feed()

        # Barge-in from backend (transport-level)
        if VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

        # Out-of-band DTMF from backend (e.g. RFC 4733 via RTP)
        if VoiceCapability.DTMF_SIGNALING in backend.capabilities and hasattr(
            backend, "on_dtmf_received"
        ):
            backend.on_dtmf_received(self._on_pipeline_dtmf)

    # -------------------------------------------------------------------------
    # Session ready (dual-signal)
    # -------------------------------------------------------------------------

    def _on_session_ready(self, session: VoiceSession) -> None:
        """Handle backend signalling that a session's audio path is live.

        If ``bind_session()`` has already been called (session is in
        ``_session_bindings``), fire the hook immediately.  Otherwise
        record the session ID in ``_session_ready_pending`` so
        ``bind_session()`` can fire the hook when it runs.
        """
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
            if binding_info is None:
                self._session_ready_pending.add(session.id)
        if binding_info is not None:
            room_id, _ = binding_info
            self._schedule(
                self._fire_session_started_hook(session, room_id),
                name=f"session_started:{session.id}",
            )

    # -------------------------------------------------------------------------
    # Pipeline event handlers (wiring layer)
    # -------------------------------------------------------------------------

    def _on_audio_received(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Handle raw audio frame from backend — feed into pipeline.

        Enforces ``ChannelBinding.access`` and ``muted`` per RFC §7.5:
        audio is dropped when the binding is READ_ONLY, NONE, or muted.
        """
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if binding_info is not None:
            binding = binding_info[1]
            if binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted:
                return

        # Audio frame rate limiting — drop excess frames (thread-safe:
        # _on_audio_received is called from audio callback threads)
        if self._max_fps is not None:
            with self._state_lock:
                now = time.monotonic()
                window_start, count = self._frame_counts.get(session.id, (now, 0))
                if now - window_start >= 1.0:
                    window_start, count = now, 0
                count += 1
                self._frame_counts[session.id] = (window_start, count)
                if count > self._max_fps:
                    return

        if self._pipeline is not None:
            self._pipeline.process_frame(session, frame)

    def _on_pipeline_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Handle speech end from pipeline — fire hooks and transcribe."""
        # Pop the stream state now so a rapid SPEECH_START can't steal it.
        # Without this, a new stream created by _start_stt_stream would
        # overwrite _stt_streams[session.id] before _process_speech_end
        # runs, causing it to grab the wrong (new) stream.
        stream_state = self._stt_streams.pop(session.id, None)
        if stream_state is not None:
            # Flush remaining buffered frames before sending sentinel
            self._flush_stt_buffer(stream_state, session.id)
            try:
                stream_state.queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning("STT stream queue full on sentinel for %s", session.id)

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        self._schedule(
            self._process_speech_end(session, audio, room_id, stream_state),
            name=f"speech_end:{session.id}",
        )

    def _on_pipeline_vad_event(self, session: VoiceSession, vad_event: VADEvent) -> None:
        """Handle VAD events from pipeline — fire corresponding hooks."""
        from roomkit.voice.pipeline.vad.base import VADEventType

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        if vad_event.type == VADEventType.SPEECH_START:
            # Check for barge-in using InterruptionHandler
            with self._state_lock:
                playback = self._playing_sessions.get(session.id)
            if playback:
                # During drain period (send_audio returned, waiting for echo
                # decay), skip barge-in — nothing is actually playing.
                done_ev = self._playback_done_events.get(session.id)
                if done_ev is not None and done_ev.is_set():
                    pass  # Drain period — skip barge-in
                else:
                    decision = self._interruption_handler.evaluate(
                        playback_position_ms=playback.position_ms,
                        speech_duration_ms=0,
                    )
                    if decision.should_interrupt:
                        self._schedule(
                            self._handle_barge_in(session, playback, room_id),
                            name=f"barge_in:{session.id}",
                        )
                    elif decision.is_backchannel:
                        self._schedule(
                            self._fire_backchannel_hook(session, "", room_id),
                            name=f"backchannel:{session.id}",
                        )

            # Start streaming STT if provider supports it
            if self._stt and self._stt.supports_streaming:
                self._start_stt_stream(session, room_id, pre_roll=vad_event.audio_bytes)

            self._schedule(
                self._fire_speech_start_hooks(session, room_id),
                name=f"speech_start:{session.id}",
            )
        elif vad_event.type == VADEventType.SPEECH_END:
            # ON_SPEECH_END hooks are fired by _process_speech_end() to
            # guarantee ordering (ON_SPEECH_END before ON_TRANSCRIPTION).
            # Do NOT fire them here to avoid duplicate invocations.
            pass
        elif vad_event.type == VADEventType.SILENCE:
            self._schedule(
                self._fire_vad_silence_hook(session, int(vad_event.duration_ms or 0), room_id),
                name=f"vad_silence:{session.id}",
            )
        elif vad_event.type == VADEventType.AUDIO_LEVEL:
            self._schedule(
                self._fire_vad_audio_level_hook(
                    session,
                    vad_event.level_db or 0.0,
                    vad_event.confidence is not None and vad_event.confidence > 0.5,
                    room_id,
                ),
                name=f"vad_audio_level:{session.id}",
            )

    def _on_processed_frame_for_level(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Fire ON_INPUT_AUDIO_LEVEL hook, throttled to ~10/sec per session."""
        now = time.monotonic()
        if now - self._last_input_level_at.get(session.id, 0.0) < 0.1:
            return
        self._last_input_level_at[session.id] = now
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return
        room_id, _ = binding_info
        self._schedule(
            self._fire_audio_level_hook(
                session,
                rms_db(frame.data),
                room_id,
                HookTrigger.ON_INPUT_AUDIO_LEVEL,
            ),
            name=f"input_audio_level:{session.id}",
        )

    def _on_audio_played_for_level(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL at real playback pace (PortAudio callback).

        Shares ``_last_output_level_at`` with ``_fire_output_level`` in the
        TTS mixin so they naturally deduplicate.
        """
        now = time.monotonic()
        if now - self._last_output_level_at.get(session.id, 0.0) < 0.1:
            return
        self._last_output_level_at[session.id] = now
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return
        room_id, _ = binding_info
        self._schedule(
            self._fire_audio_level_hook(
                session,
                rms_db(frame.data),
                room_id,
                HookTrigger.ON_OUTPUT_AUDIO_LEVEL,
            ),
            name=f"output_audio_level:{session.id}",
        )

    def _on_pipeline_speaker_change(
        self, session: VoiceSession, result: DiarizationResult
    ) -> None:
        """Handle speaker change from pipeline — fire ON_SPEAKER_CHANGE hook."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info

        self._schedule(
            self._fire_speaker_change_hook(session, result, room_id),
            name=f"speaker_change:{session.id}",
        )

    def _on_pipeline_dtmf(self, session: VoiceSession, dtmf_event: Any) -> None:
        """Handle DTMF event from pipeline — fire ON_DTMF hook."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_dtmf_hook(session, dtmf_event, room_id),
            name=f"dtmf:{session.id}",
        )

    def _on_pipeline_recording_started(self, session: VoiceSession, handle: Any) -> None:
        """Handle recording started from pipeline — fire hook."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_recording_started_hook(session, handle, room_id),
            name=f"recording_started:{session.id}",
        )

    def _on_pipeline_recording_stopped(self, session: VoiceSession, result: Any) -> None:
        """Handle recording stopped from pipeline — fire hook."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_recording_stopped_hook(session, result, room_id),
            name=f"recording_stopped:{session.id}",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _schedule(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        """Schedule *coro* as a fire-and-forget task.

        Works from both the event-loop thread and foreign threads (e.g.
        PortAudio audio callbacks).  On foreign threads we use
        ``call_soon_threadsafe`` to dispatch to the cached event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Foreign thread — dispatch via cached event loop.
            cached = self._event_loop
            if cached is not None and cached.is_running():
                cached.call_soon_threadsafe(self._create_task, coro, name)
            else:
                coro.close()
            return
        # Cache the loop for future cross-thread calls.
        self._event_loop = loop
        self._create_task(coro, name)

    def _create_task(self, coro: Coroutine[Any, Any, Any], name: str) -> None:
        """Create and track an asyncio task (must be called on the event loop thread)."""
        task = asyncio.get_running_loop().create_task(coro, name=name)
        task.add_done_callback(self._task_done)
        self._scheduled_tasks.add(task)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        """Done-callback for scheduled tasks: log exceptions and remove from set."""
        self._scheduled_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unhandled exception in scheduled task %s: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    def set_framework(self, framework: RoomKit) -> None:
        """Set the framework reference for inbound routing.

        Called automatically when the channel is registered with RoomKit.
        """
        self._framework = framework
        # Propagate telemetry to STT, TTS providers and backend
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            if self._stt is not None:
                self._stt._telemetry = telemetry  # type: ignore[attr-defined]
            if self._tts is not None:
                self._tts._telemetry = telemetry  # type: ignore[attr-defined]
            if self._backend is not None:
                self._backend._telemetry = telemetry  # type: ignore[attr-defined]
        # Bridge trace emitter to backend when framework wiring enables it
        self._sync_trace_emitter()

    def on_trace(
        self,
        callback: Any,
        *,
        protocols: list[str] | None = None,
    ) -> None:
        """Register a trace observer and bridge to the backend."""
        super().on_trace(callback, protocols=protocols)
        self._sync_trace_emitter()

    def _sync_trace_emitter(self) -> None:
        """Set or clear the backend trace emitter based on trace_enabled."""
        if self._backend is not None and hasattr(self._backend, "set_trace_emitter"):
            self._backend.set_trace_emitter(
                self.emit_trace if self.trace_enabled else None,
            )

    def resolve_trace_room(self, session_id: str | None) -> str | None:
        """Resolve room_id from voice session bindings."""
        if session_id is None:
            return None
        binding_info = self._session_bindings.get(session_id)
        if binding_info:
            return binding_info[0]
        return None

    def bind_session(self, session: VoiceSession, room_id: str, binding: ChannelBinding) -> None:
        """Bind a voice session to a room for message routing."""
        with self._state_lock:
            self._session_bindings[session.id] = (room_id, binding)
            # Dual-signal: atomically check and clear pending ready flag
            # under the same lock that writes _session_bindings, so
            # _on_session_ready cannot interleave between the two.
            was_ready_pending = session.id in self._session_ready_pending
            self._session_ready_pending.discard(session.id)
        # Start VOICE_SESSION telemetry span early so pipeline activation
        # and subsequent operations appear as children in traces.
        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.noop import NoopTelemetryProvider

        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        span_id = telemetry.start_span(
            SpanKind.VOICE_SESSION,
            "voice.session",
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
            attributes={
                Attr.BACKEND_TYPE: self._backend.name if self._backend else "none",
                Attr.PROVIDER: self._stt.name if self._stt else "none",
                "tts_provider": self._tts.name if self._tts else "none",
            },
        )
        self._voice_session_spans[session.id] = span_id
        # Tell the pipeline about the parent span for speech segment spans
        if self._pipeline is not None:
            self._pipeline.set_parent_span(session.id, span_id)
        # Notify pipeline of session activation
        if self._pipeline is not None:
            self._pipeline.on_session_active(session)
        # Initialize batch buffer for this session
        if self._batch_mode:
            self._batch_audio_buffers[session.id] = bytearray()
        # Start continuous STT if enabled (no local VAD)
        if self._continuous_stt:
            self._start_continuous_stt(session)
        # Emit voice_session_started framework event
        if self._framework:
            self._schedule(
                self._emit_session_started(session, room_id),
                name=f"session_started:{session.id}",
            )
        # Dual-signal: if backend already signalled ready, fire hook now
        if was_ready_pending and self._framework:
            self._schedule(
                self._fire_session_started_hook(session, room_id),
                name=f"session_started:{session.id}",
            )

    async def connect_session(
        self,
        session: Any,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Accept a voice session via process_inbound.

        Delegates to :meth:`bind_session` which handles pipeline
        activation and framework events.
        """
        self.bind_session(session, room_id, binding)

    async def disconnect_session(self, session: Any, room_id: str) -> None:
        """Clean up a voice session on remote disconnect."""
        self.unbind_session(session)
        if self._backend is not None:
            await self._backend.disconnect(session)

    def update_binding(self, room_id: str, binding: ChannelBinding) -> None:
        """Update cached bindings for all sessions in a room.

        Called by the framework after mute/unmute/set_access so the
        audio gate in ``_on_audio_received`` sees the new state.
        """
        with self._state_lock:
            for sid, (rid, _old) in self._session_bindings.items():
                if rid == room_id:
                    self._session_bindings[sid] = (rid, binding)

    def unbind_session(self, session: VoiceSession) -> None:
        """Remove session binding."""
        # Cancel any active streaming STT
        self._cancel_stt_stream(session.id)
        self._session_ready_pending.discard(session.id)
        with self._state_lock:
            binding_info = self._session_bindings.pop(session.id, None)
        # Notify pipeline of session end
        if self._pipeline is not None:
            self._pipeline.on_session_ended(session)
        # Clear pending turns, audio, and interrupt cooldown
        self._pending_turns.pop(session.id, None)
        self._pending_audio.pop(session.id, None)
        self._last_tts_ended_at.pop(session.id, None)
        # Clear per-session audio level timestamps
        self._last_input_level_at.pop(session.id, None)
        self._last_output_level_at.pop(session.id, None)
        # Clear frame rate counter
        self._frame_counts.pop(session.id, None)
        # Clear batch buffers
        self._batch_audio_buffers.pop(session.id, None)
        self._batch_audio_sample_rate.pop(session.id, None)
        # End VOICE_SESSION telemetry span
        voice_spans = getattr(self, "_voice_session_spans", {})
        span_id = voice_spans.pop(session.id, None)
        if span_id:
            from roomkit.telemetry.noop import NoopTelemetryProvider

            telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
            telemetry.end_span(span_id)
            # Flush immediately — VOICE_SESSION is the trace root and long-lived;
            # without this, BatchSpanProcessor may not export it before shutdown.
            telemetry.flush()
        # Emit voice_session_ended framework event
        if binding_info and self._framework:
            room_id, _ = binding_info
            self._schedule(
                self._emit_session_ended(session, room_id),
                name=f"session_ended:{session.id}",
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def provider_name(self) -> str | None:
        return self._backend.name if self._backend is not None else None

    @property
    def backend(self) -> VoiceBackend | None:
        """The voice backend (if configured)."""
        return self._backend

    @property
    def info(self) -> dict[str, Any]:
        return {
            "stt": self._stt.name if self._stt else None,
            "tts": self._tts.name if self._tts else None,
            "backend": self._backend.name if self._backend else None,
            "streaming": self._streaming,
            "pipeline": self._pipeline_config is not None,
            "batch_mode": self._batch_mode,
        }

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether this channel can accept streaming text delivery."""
        return (
            self._tts is not None
            and getattr(self._tts, "supports_streaming_input", False)
            and self._backend is not None
        )

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            media_types=[ChannelMediaType.AUDIO, ChannelMediaType.TEXT],
            supports_audio=True,
            supported_audio_formats=["wav", "mp3", "ogg", "webm"],
            max_audio_duration_seconds=3600,
        )

    def update_voice_map(self, entries: dict[str, str]) -> None:
        """Merge entries into the per-agent voice map.

        Called by :meth:`ConversationPipeline.install` to auto-wire
        voice IDs from :class:`Agent` instances.
        """
        self._voice_map.update(entries)

    # -------------------------------------------------------------------------
    # Barge-in handling
    # -------------------------------------------------------------------------

    async def _handle_barge_in(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        if not self._framework:
            return
        # NOTE: do NOT cancel STT here.  _on_pipeline_vad_event already
        # called _start_stt_stream (which cancels + replaces the old one)
        # before scheduling this barge-in task.  Cancelling again would
        # destroy the *new* stream that's collecting the user's utterance.
        try:
            from roomkit.voice.events import BargeInEvent

            context = await self._framework._build_context(room_id)
            event = BargeInEvent(
                session=session,
                interrupted_text=playback.text,
                audio_position_ms=playback.position_ms,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BARGE_IN,
                event,
                context,
                skip_event_filter=True,
            )
            await self.interrupt(session, reason="barge_in")
        except Exception:
            logger.exception("Error handling barge-in for session %s", session.id)

    async def interrupt(self, session: VoiceSession, *, reason: str = "explicit") -> bool:
        """Interrupt ongoing TTS playback for a session."""
        import time as _time

        with self._state_lock:
            playback = self._playing_sessions.pop(session.id, None)
        if not playback:
            return False

        self._last_tts_ended_at[session.id] = _time.monotonic()

        if self._backend and VoiceCapability.INTERRUPTION in self._backend.capabilities:
            await self._backend.cancel_audio(session)

        # Deactivate AEC — after TTS stops the adaptive filter is stale
        # and will suppress the user's voice.  Reset + bypass so audio
        # passes through cleanly until the next TTS starts.
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            aec = self._pipeline._config.aec
            aec.reset()
            aec.set_active(False)

        if self._framework:
            binding_info = self._session_bindings.get(session.id)
            if binding_info:
                room_id, _ = binding_info
                try:
                    from roomkit.voice.events import TTSCancelledEvent

                    context = await self._framework._build_context(room_id)
                    event = TTSCancelledEvent(
                        session=session,
                        reason=reason,  # type: ignore[arg-type]
                        text=playback.text,
                        audio_position_ms=playback.position_ms,
                    )
                    await self._framework.hook_engine.run_async_hooks(
                        room_id,
                        HookTrigger.ON_TTS_CANCELLED,
                        event,
                        context,
                        skip_event_filter=True,
                    )
                except Exception:
                    logger.exception("Error firing ON_TTS_CANCELLED hook")

        logger.info(
            "TTS interrupted for session %s: reason=%s, position=%dms",
            session.id,
            reason,
            playback.position_ms,
        )
        return True

    async def interrupt_all(self, room_id: str, *, reason: str = "task_delivery") -> int:
        """Interrupt all active TTS playback in a room.

        Returns:
            Number of sessions that were interrupted.
        """
        with self._state_lock:
            session_ids = [
                sid
                for sid, (rid, _) in self._session_bindings.items()
                if rid == room_id and sid in self._playing_sessions
            ]
        count = 0
        for sid in session_ids:
            session = self._backend.get_session(sid) if self._backend else None
            if session and await self.interrupt(session, reason=reason):
                count += 1
        return count

    async def wait_playback_done(self, room_id: str, timeout: float = 15.0) -> None:
        """Wait until active TTS playback finishes for all sessions in *room_id*.

        Returns immediately if no playback is in progress.  Uses per-session
        events that are set when ``send_audio()`` returns (before the echo
        drain delay), so callers don't wait for the 2-second drain window.
        """
        with self._state_lock:
            events = [
                self._playback_done_events[sid]
                for sid, (rid, _) in self._session_bindings.items()
                if rid == room_id
                and sid in self._playing_sessions
                and sid in self._playback_done_events
                and not self._playback_done_events[sid].is_set()
            ]
        if not events:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(e.wait() for e in events)),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "wait_playback_done timed out for room %s after %.1fs",
                room_id,
                timeout,
            )

    def _on_backend_barge_in(self, session: VoiceSession) -> None:
        """Handle barge-in detected by backend."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
            playback = self._playing_sessions.get(session.id)
        if not binding_info or not self._framework:
            return

        room_id, _ = binding_info
        if not playback:
            return

        # During drain period (send_audio returned, waiting for echo
        # decay), skip barge-in — nothing is actually playing.
        done_ev = self._playback_done_events.get(session.id)
        if done_ev is not None and done_ev.is_set():
            return

        self._schedule(
            self._handle_barge_in(session, playback, room_id),
            name=f"backend_barge_in:{session.id}",
        )

    # -------------------------------------------------------------------------
    # Channel interface
    # -------------------------------------------------------------------------

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        from roomkit.models.event import EventSource
        from roomkit.models.event import RoomEvent as RoomEventModel

        return RoomEventModel(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
                external_id=message.external_id,
                provider=self.provider_name,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
            metadata=message.metadata,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        from roomkit.models.channel import ChannelOutput as ChannelOutputModel
        from roomkit.models.event import TextContent

        # Skip system events and internal-visibility events — they carry
        # orchestration metadata (e.g. handoff notifications) that should
        # never be spoken aloud via TTS.
        if event.type == EventType.SYSTEM or event.visibility == "internal":
            return ChannelOutputModel.empty()

        if self._streaming and not self._backend:
            return ChannelOutputModel.empty()

        if self._streaming and self._backend and isinstance(event.content, TextContent):
            await self._deliver_voice(event, binding, context)
            return ChannelOutputModel.empty()

        if not self._streaming and isinstance(event.content, TextContent) and self._tts:
            raise NotImplementedError(
                "VoiceChannel store-and-forward mode requires MediaStore support. "
                "Use streaming=True (default) for real-time voice, or implement "
                "MediaStore for async audio delivery."
            )

        return ChannelOutputModel.empty()

    async def close(self) -> None:
        # 1. Cancel STT streams first (stops feeding audio)
        for sid in list(self._stt_streams):
            self._cancel_stt_stream(sid)
        # 2. Cancel scheduled fire-and-forget tasks and await completion
        for task in self._scheduled_tasks:
            task.cancel()
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        self._scheduled_tasks.clear()
        # 3. Close pipeline (stops audio processing)
        if self._pipeline is not None:
            self._pipeline.close()
        # 4. Close STT/TTS providers
        if self._stt:
            await self._stt.close()
        if self._tts:
            await self._tts.close()
        # 5. Close backend last (transport layer)
        if self._backend:
            await self._backend.close()
        self._session_bindings.clear()
        self._playing_sessions.clear()
        self._batch_audio_buffers.clear()
        self._batch_audio_sample_rate.clear()
        self._frame_counts.clear()
        self._session_ready_pending.clear()

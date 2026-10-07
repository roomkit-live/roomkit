"""Voice channel for real-time audio communication."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit.channels._voice_hooks import VoiceHooksMixin
from roomkit.channels._voice_pipeline import VoicePipelineMixin
from roomkit.channels._voice_recording_hooks import VoiceRecordingHooksMixin
from roomkit.channels._voice_speakers import (
    PipelineSpeakerTally,
    SpeakerAttribution,
    SpeakerTracker,
)
from roomkit.channels._voice_stt import VoiceSTTMixin
from roomkit.channels._voice_tts import VoiceTTSMixin
from roomkit.channels._voice_turn import VoiceTurnMixin
from roomkit.channels._voice_unheard import UnheardTurns
from roomkit.channels.base import Channel, FrameworkAwareChannel
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.voice.base import VoiceCapability
from roomkit.voice.bridge import AudioBridge, AudioBridgeConfig, BridgeFrameFilter
from roomkit.voice.interruption import InterruptionConfig
from roomkit.voice.stt.base import diarizes
from roomkit.voice.tts.context import TTSContextConfig, TTSContextLevel, TTSContextStore
from roomkit.voice.utils import rms_db

if TYPE_CHECKING:
    from concurrent.futures import Future

    from roomkit.core.framework import RoomKit
    from roomkit.models.channel import ChannelBinding, ChannelOutput
    from roomkit.models.context import RoomContext
    from roomkit.models.delivery import InboundMessage
    from roomkit.models.event import RoomEvent
    from roomkit.recorder.base import ChannelRecordingConfig
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization.base import DiarizationResult
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.pipeline.turn.base import TurnEntry
    from roomkit.voice.pipeline.vad.base import VADEvent
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.stt.language import STTLanguageLock
    from roomkit.voice.tts.base import TTSProvider
    from roomkit.voice.tts.context import AssistantTurnRecorder

logger = logging.getLogger("roomkit.voice")


def _refuse_diarizing_stt(
    stt: STTProvider | None, pipeline: AudioPipelineConfig | None, *, batch_mode: bool
) -> None:
    """Refuse an STT that labels speakers unless the channel runs continuous STT.

    Labels compare only within one stream (RFC §12.2.3). Continuous mode keeps
    one stream across turns for such a provider; VAD mode opens a stream per
    utterance and batch mode one per flush, so every turn would restart at
    the first label.
    """
    if stt is None or not diarizes(stt):
        return
    continuous = (
        not batch_mode
        and (pipeline is None or pipeline.vad is None)
        and getattr(stt, "supports_streaming", False) is True
    )
    if not continuous:
        raise ValueError(
            f"{stt.name} labels speakers (supports_diarization=True), and speaker labels "
            "hold within one STT stream: VoiceChannel carries them only in continuous "
            "mode, a streaming STT with no pipeline VAD and no batch_mode (RFC §12.2.3)"
        )


def _refuse_pipeline_speakers(pipeline: AudioPipelineConfig | None, *, batch_mode: bool) -> None:
    """Refuse ``pipeline_speakers`` where it could name nobody (RFC §12.2.3)."""
    if pipeline is None or pipeline.diarization is None:
        raise ValueError(
            "pipeline_speakers needs a DiarizationProvider in the pipeline "
            "(AudioPipelineConfig(diarization=...)): it names speakers from that stage"
        )
    if batch_mode:
        raise ValueError(
            "pipeline_speakers does not apply in batch_mode: a flush may hold several "
            "speakers, and the channel names one per utterance or continuous final"
        )


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

# A kept (diarizing) STT stream is checked against the clock whenever its audio
# queue stays empty this long, and padded with silence when it is behind
# (RFC §12.2.3): a pause shorter than this is never filled.
_KEPT_STREAM_PACE_S = 0.25

# How many speech segments the DISABLED strategy holds while the bot is
# speaking (RFC §12.6 — speech is queued, not discarded). A bounded backlog:
# a caller who talks through a long answer gets their turn, a stuck playback
# cannot grow the queue without limit.
_QUEUED_SPEECH_MAX_SEGMENTS = 8


@dataclass
class _STTStreamState:
    """Track an active streaming STT session."""

    queue: asyncio.Queue[Any]  # Queue[AudioChunk | None]
    task: asyncio.Task[Any] | None = None  # consumer task running transcribe_stream
    frame_buffer: bytearray = field(default_factory=bytearray)
    frame_buffer_rate: int = 16000
    final_text: str | None = None
    partial_text: str | None = None
    final_result: TranscriptionResult | None = None
    partial_result: TranscriptionResult | None = None
    error: bool = False
    cancelled: bool = False
    streams_opened: int = 0  # continuous mode: the next stream's label epoch (RFC §12.2.3)


@dataclass
class TTSPlaybackState:
    """Track ongoing TTS playback for barge-in detection."""

    session_id: str
    text: str
    started_at: datetime = field(default_factory=_utcnow)
    total_duration_ms: int | None = None
    first_audio_at: float | None = None
    """Monotonic time the first audio chunk was handed to the transport."""
    audio_ms: float | None = None
    """Audio handed to the transport so far; None while nothing is measured."""
    measured: bool = False
    """True once a synthesis stream is observed: before its first chunk the
    user has heard nothing, whatever the elapsed time."""
    stopped_at: float | None = None
    """Monotonic time playback was interrupted."""
    context_turn: tuple[AssistantTurnRecorder, str] | None = None
    """The TTS context turn this playback will record, and its speaker."""
    answer_channel_id: str | None = None
    """The intelligence channel whose answer this playback speaks; ``None`` for
    a playback that answers nothing (``say()``, a greeting)."""
    answer_responds_to: str | None = None
    """The event that answer responds to (RFC §8.5)."""
    barge_in_claimed: bool = False
    """Set by the first barge-in on this playback; every later trigger finds
    it taken. The playback leaves ``_playing_sessions`` only once that barge-in
    reaches :meth:`VoiceChannel.interrupt`, and the user is still talking in
    between."""

    @property
    def position_ms(self) -> int:
        """Time since this state was created, synthesis latency included.

        Not what the user heard: hooks, logs and the interruption policy read
        ``played_ms``.
        """
        elapsed = datetime.now(UTC) - self.started_at
        return int(elapsed.total_seconds() * 1000)

    def start_measuring(self) -> None:
        """Mark the playback as measured by its audio, before any chunk left."""
        self.measured = True

    def note_audio(self, chunk: AudioChunk) -> None:
        """Account for an outbound chunk, so ``played_ms`` measures audio.

        The chunk is 16-bit PCM: a VoiceChannel refuses any other before it
        gets here (``_pcm16_only``, RFC section 12.2).
        """
        if not chunk.data:
            return
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()
        duration = len(chunk.data) / (2 * chunk.channels * chunk.sample_rate) * 1000
        self.audio_ms = (self.audio_ms or 0.0) + duration

    @property
    def played_ms(self) -> int:
        """How much audio the user heard: time since the first chunk, capped
        at the audio produced when its duration is known, frozen when
        playback was cut off.

        A measured stream is at 0 until its first chunk goes out. Falls back
        to ``position_ms`` for a state no stream went through.
        """
        if self.first_audio_at is None:
            return 0 if self.measured else self.position_ms
        end = self.stopped_at if self.stopped_at is not None else time.monotonic()
        heard = (end - self.first_audio_at) * 1000
        if self.audio_ms is not None:
            heard = min(heard, self.audio_ms)
        return int(max(0.0, heard))


class VoiceChannel(
    VoiceSTTMixin,
    VoiceTTSMixin,
    VoiceHooksMixin,
    VoiceRecordingHooksMixin,
    VoiceTurnMixin,
    VoicePipelineMixin,
    FrameworkAwareChannel,
    Channel,
):
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

    The STT language can be chosen per session at runtime with
    :meth:`set_stt_language`; it applies from the session's next stream.
    ``stt_language_lock`` installs an
    :class:`~roomkit.voice.stt.language.STTLanguageLock` that starts every
    session detecting (Deepgram ``multi``), pins it to the language the
    speaker uses, and releases it when the results stop fitting.

    A TTS provider whose ``context_level`` is not NONE receives the dialogue
    of the session on every streaming call (RFC §12.2.2): the channel keeps it
    per session, as ``tts_context`` bounds it, and releases it when the
    session is unbound.

    Who spoke (RFC §12.2.3): with a diarizing STT in continuous mode each
    segment becomes its own message carrying ``speaker_label``,
    ``speaker_epoch`` and ``sender_name``. ``pipeline_speakers=True`` does the
    same from the pipeline's ``DiarizationProvider`` when the STT labels
    nothing: each transcript takes the speaker the stage heard the longest
    over it (refused in batch mode, where a flush may hold several). Opt-in,
    because it changes the name every message carries.
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
        bridge: bool | AudioBridgeConfig | None = None,
        recording: ChannelRecordingConfig | None = None,
        close_providers: bool = True,
        stt_language_lock: STTLanguageLock | None = None,
        tts_context: TTSContextConfig | None = None,
        pipeline_speakers: bool = False,
    ) -> None:
        super().__init__(channel_id)
        _refuse_diarizing_stt(stt, pipeline, batch_mode=batch_mode)
        if pipeline_speakers:
            _refuse_pipeline_speakers(pipeline, batch_mode=batch_mode)
        # The pipeline stage's speakers, by audio heard, per session (RFC §12.2.3)
        self._pipeline_speaker_tally = PipelineSpeakerTally() if pipeline_speakers else None
        self._stt = stt
        self._tts = tts
        # The dialogue a context-aware TTS hears (RFC §12.2.2); absent for a
        # provider that consumes none, so nothing is kept on its behalf.
        # A duck-typed provider without the attribute consumes none.
        context_config = tts_context or TTSContextConfig()
        context_level = getattr(tts, "context_level", TTSContextLevel.NONE)
        self._tts_context: TTSContextStore | None = (
            TTSContextStore(context_config, context_level)
            if context_config.enabled and context_level != TTSContextLevel.NONE
            else None
        )
        self._backend = backend
        # When False, close() leaves the injected STT/TTS providers open —
        # the caller owns their lifecycle (e.g. reuses cached models across
        # sessions, or closes them itself to avoid a double-close hang). The
        # backend is always closed by close(). Defaults to True: the channel
        # owns the providers it was given, matching the simple single-use case.
        self._close_providers = close_providers
        self._pipeline_config = pipeline
        self._recording = recording
        self._streaming = streaming
        self._framework: RoomKit | None = None
        # Lock for shared state accessed from both asyncio and audio threads
        self._state_lock = threading.Lock()
        # Map session_id -> (room_id, binding) for routing
        self._session_bindings: dict[str, tuple[str, ChannelBinding]] = {}
        # Room of a session being unbound, for the recording-stopped report
        # the pipeline makes while the binding is already gone
        self._ending_session_rooms: dict[str, str] = {}
        # Transports registered with add_backend(), and the sessions they
        # serve (a session absent here is the primary backend's)
        self._extra_backends: list[VoiceBackend] = []
        self._session_backends: dict[str, VoiceBackend] = {}
        # Track TTS playback for barge-in detection
        self._playing_sessions: dict[str, TTSPlaybackState] = {}
        # Signalled when send_audio() returns for a session (before drain delay)
        self._playback_done_events: dict[str, asyncio.Event] = {}
        # Sessions where speech was suppressed (echo during TTS playback)
        self._suppressed_sessions: set[str] = set()
        # Sessions whose speech is queued until playback ends (RFC §12.6
        # DISABLED) — the speech is not echo, it is simply not allowed to cut in
        self._queueing_sessions: set[str] = set()
        # SEMANTIC segments held during playback while their words are
        # transcribed (session_id -> whether ON_BACKCHANNEL already fired)
        self._held_for_transcript: dict[str, bool] = {}
        # Continuous mode: latest words of the speech burst under way, and
        # the words of it judged a backchannel (SEMANTIC, RFC §12.3.13)
        self._burst_words: dict[str, str] = {}
        self._burst_backchannel: dict[str, str] = {}
        # Speech segments captured while queueing, replayed once TTS finishes,
        # each with its pipeline speaker claim (pipeline_speakers)
        self._queued_speech: dict[
            str, list[tuple[bytes, Future[SpeakerAttribution | None] | None]]
        ] = {}
        # Monotonic timestamp of the current speech onset per session, used to
        # measure sustained speech for the CONFIRMED strategy (RFC §12.6)
        self._speech_started_at: dict[str, float] = {}
        # Armed re-evaluations for CONFIRMED (session_id -> task)
        self._confirm_tasks: dict[str, asyncio.Task[Any]] = {}
        # Track delivered event IDs to prevent duplicate TTS delivery
        self._delivered_tts_events: OrderedDict[str, None] = OrderedDict()
        # The instantiated pipeline engine (if config provided)
        self._pipeline: AudioPipeline | None = None
        # Pending turns for turn detection (session_id -> list of TurnEntry)
        self._pending_turns: dict[str, list[TurnEntry]] = {}
        # Who said the pending turn, when a diarizing STT labelled it (RFC §12.2.3)
        self._pending_turn_speakers: dict[str, SpeakerAttribution] = {}
        # The speaker-change rule per session for a diarizing STT's labels
        self._speaker_trackers: dict[str, SpeakerTracker] = {}
        # Continuous-mode finals of a session are processed one at a time
        self._transcript_locks: dict[str, asyncio.Lock] = {}
        # Pending audio for audio-native turn detectors (session_id -> accumulated PCM)
        self._pending_audio: dict[str, bytearray] = {}
        self._turn_speech_state: dict[str, tuple[bool, float]] = {}
        self._turn_transcripts_due: dict[str, int] = {}
        self._turn_wait_tasks: dict[str, asyncio.Task[None]] = {}
        # The routed turn per session whose response is not heard yet (RFC §12.3.12)
        self._unheard_turns = UnheardTurns()
        # Active streaming STT sessions (session_id -> state)
        self._stt_streams: dict[str, _STTStreamState] = {}
        # STT language chosen per session (session_id -> language); absent
        # means the provider's own configuration. Read when a stream opens.
        self._stt_languages: dict[str, str] = {}
        # Policy that picks the session language from what the speaker uses
        if stt_language_lock is not None:
            if stt is None:
                raise ValueError("stt_language_lock requires an STT provider")
            if not getattr(stt, "supports_language_override", False):
                raise ValueError(
                    f"stt_language_lock requires an STT provider that supports a "
                    f"per-session language; {stt.name} does not"
                )
        self._stt_language_lock = stt_language_lock
        # Continuous STT mode: stream all audio to STT, no local VAD
        self._continuous_stt = False
        # Post-denoiser energy barge-in state (continuous STT mode, per-session)
        self._barge_in_energy_count: dict[str, int] = {}
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
        # Outbound audio taps (e.g. room-level recording, avatar lip-sync)
        self._outbound_audio_taps: list[Callable[..., None]] = []
        # Audio bridge for session-to-session forwarding
        if bridge is True:
            self._bridge: AudioBridge | None = AudioBridge()
        elif isinstance(bridge, AudioBridgeConfig):
            self._bridge = AudioBridge(bridge)
        else:
            self._bridge = None

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
            backend.on_client_disconnected(self._on_backend_disconnected)

    def _setup_pipeline(self, backend: VoiceBackend, config: AudioPipelineConfig) -> None:
        """Create AudioPipeline and wire backend -> pipeline -> callbacks."""
        # Shared infrastructure: pipeline creation, audio reception, AEC reference
        pipeline = self._create_pipeline(config, backend)

        # Continuous STT: no local VAD, stream all audio to STT provider.
        # Batch mode takes priority — audio accumulates for manual flush.
        self._continuous_stt = (
            not self._batch_mode
            and config.vad is None
            and self._stt is not None
            and self._stt.supports_streaming
        )

        if self._pipeline_speaker_tally is not None:
            pipeline.on_processed_frame(self._pipeline_speaker_tally.add_frame)
        # Pipeline events -> VoiceChannel hooks
        if self._continuous_stt:
            pipeline.on_processed_frame(self._on_processed_frame_for_stt)
        elif self._batch_mode and self._stt is not None:
            pipeline.on_processed_frame(self._on_processed_frame_for_batch)
        else:
            pipeline.on_speech_end(self._on_pipeline_speech_end)
            pipeline.on_vad_event(self._on_pipeline_vad_event)
            pipeline.on_speech_frame(self._on_pipeline_speech_frame)
        if config.diarization is not None:
            pipeline.on_speaker_change(self._on_pipeline_speaker_change)
        if config.dtmf is not None:
            pipeline.on_dtmf(self._on_pipeline_dtmf)
        if config.recorder is not None:
            self._wire_recording_hooks(pipeline)

        # Audio bridge: forward processed frames to other sessions
        if self._bridge is not None:
            pipeline.on_processed_frame(self._on_processed_frame_for_bridge)
            self._bridge.set_frame_processor(self._process_bridge_outbound)

        # Audio level hooks:
        # - Input: fires from pipeline processed-frame callback.
        # - Output: fires from _wrap_outbound() in the TTS mixin (all backends),
        #   AND from on_audio_played at real playback pace (backends that support it).
        #   Both share _last_output_level_at so they naturally deduplicate.
        pipeline.on_processed_frame(self._on_processed_frame_for_level)
        if backend.supports_playback_callback:
            backend.on_audio_played(self._on_audio_played_for_level)

        # Barge-in from backend (transport-level)
        if VoiceCapability.BARGE_IN in backend.capabilities:
            backend.on_barge_in(self._on_backend_barge_in)

        # Out-of-band DTMF from backend (e.g. RFC 4733 via RTP)
        if VoiceCapability.DTMF_SIGNALING in backend.capabilities and hasattr(
            backend, "on_dtmf_received"
        ):
            backend.on_dtmf_received(self._on_pipeline_dtmf)  # ty: ignore[call-non-callable]

    # -------------------------------------------------------------------------
    # Session ready / disconnect (backend callbacks)
    # -------------------------------------------------------------------------

    def _on_backend_disconnected(self, session: VoiceSession) -> None:
        """Auto-called when backend signals that a session has disconnected.

        The backend has already cleaned up its own transport state, so we
        call ``unbind_session`` (not ``disconnect_session``) to avoid a
        redundant backend disconnect call.
        """
        self.unbind_session(session)

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

    def _pipeline_on_audio_received(
        self,
        session: VoiceSession,
        frame: AudioFrame,
    ) -> None:
        """Handle raw audio frame from backend — feed into pipeline.

        Extends the mixin's binding gating with VoiceChannel-specific
        audio frame rate limiting.
        """
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if binding_info is not None:
            binding = binding_info[1]
            if binding.access in (Access.READ_ONLY, Access.NONE) or binding.muted:
                return

        # Every session is counted, bound or not: the limit guards the pipeline.
        if self._over_frame_rate(session.id):
            return

        self._pipeline_submit_inbound(session, frame)

    def _over_frame_rate(self, session_id: str) -> bool:
        """Count one frame against the session's per-second budget; whether it is over.

        Thread-safe: frames arrive from audio callback threads.
        """
        if self._max_fps is None:
            return False
        with self._state_lock:
            now = time.monotonic()
            self._forget_expired_frame_windows(now)
            window_start, count = self._frame_counts.get(session_id, (now, 0))
            if now - window_start >= 1.0:
                window_start, count = now, 0
            count += 1
            self._frame_counts[session_id] = (window_start, count)
            return count > self._max_fps

    # When the expired frame windows were last forgotten (never, by default).
    _frame_windows_swept_at = 0.0

    def _forget_expired_frame_windows(self, now: float) -> None:
        """Drop, once a second, the frame windows that expired; the caller holds the lock.

        An expired window counts for nothing (the session's next frame opens a
        new one), so forgetting it changes no decision. It keeps a session that
        stopped sending, unbound or gone, from holding an entry for good.
        """
        if now - self._frame_windows_swept_at < 1.0:
            return
        self._frame_windows_swept_at = now
        expired = [sid for sid, (start, _) in self._frame_counts.items() if now - start >= 1.0]
        for sid in expired:
            del self._frame_counts[sid]

    def _on_pipeline_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Handle speech end from pipeline — fire hooks and transcribe."""
        # Before the segment is judged, so speech seen then is new speech.
        self._note_turn_speech(session.id, speaking=False)
        self._unheard_turns.note_speech_end(session.id)
        # Claimed first: the next utterance's SPEECH_START would start a new
        # count, and a segment queued for after playback keeps its claim.
        # Answered once the stage has seen this closing frame, which the
        # pipeline hands SPEECH_END callbacks before its diarization.
        tally = self._pipeline_speaker_tally
        speaker_claim = tally.claim(session.id) if tally is not None else None
        # If this speech segment was suppressed (echo during TTS), discard it —
        # unless the strategy is DISABLED, which queues it for after playback
        # (RFC §12.6) rather than throwing it away.
        with self._state_lock:
            onset = self._speech_started_at.pop(session.id, None)
            was_suppressed = session.id in self._suppressed_sessions
            self._suppressed_sessions.discard(session.id)
            held = self._held_for_transcript.pop(session.id, None)
            was_queueing = session.id in self._queueing_sessions
            self._queueing_sessions.discard(session.id)
            if was_queueing and audio:
                queue = self._queued_speech.setdefault(session.id, [])
                if len(queue) < _QUEUED_SPEECH_MAX_SEGMENTS:
                    queue.append((audio, speaker_claim))
                else:
                    logger.warning(
                        "Queued speech backlog full for %s — dropping segment",
                        session.id,
                    )
        if was_queueing:
            # A DTMF mark stays for the queued segments (taken at their flush).
            logger.debug("Queued speech during playback for %s", session.id)
            self._release_unheard_turn(session.id)
            return
        # The segment that just ended owns any DTMF heard since the last one.
        dtmf_seen = self._tts_context is not None and self._tts_context.take_dtmf(session.id)
        if was_suppressed:
            if held is False:
                # SEMANTIC holds it for its words and none came before the
                # speech ended: its final transcript decides (RFC §12.3.13).
                self._settle_wordless_held_speech(session, audio, onset, speaker_claim, dtmf_seen)
                return
            if held:
                # Classified as a backchannel: its words go nowhere (RFC §12.6
                # step 5).
                self._cancel_stt_stream(session.id)
            logger.debug("Suppressed echo speech end for %s", session.id)
            self._release_unheard_turn(session.id)
            return

        # Pop the stream state now so a rapid SPEECH_START can't steal it.
        # Without this, a new stream created by _start_stt_stream would
        # overwrite _stt_streams[session.id] before _process_speech_end
        # runs, causing it to grab the wrong (new) stream.
        stream_state = self._end_stt_stream(session.id)

        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if not binding_info or not self._framework:
            self._release_unheard_turn(session.id)
            return

        room_id, _ = binding_info

        dtmf_kwargs = {"dtmf_seen": True} if dtmf_seen else {}
        self._schedule(
            self._process_speech_end(
                session,
                audio,
                room_id,
                stream_state,
                speaker_claim=speaker_claim,
                # Due from now: a turn waiting in silence holds for these words.
                transcript=self._expect_transcript(session.id),
                **dtmf_kwargs,
            ),
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
            # More speech: a turn waiting to be judged over now goes on (RFC §12).
            self._note_turn_speech(session.id, speaking=True)
            if self._pipeline_speaker_tally is not None:
                # An utterance starts: its speaker is counted from here.
                self._pipeline_speaker_tally.reset(session.id)
            # Check for barge-in using InterruptionHandler
            suppress_speech = False
            with self._state_lock:
                playback = self._playing_sessions.get(session.id)
                # Speech onset: the clock the CONFIRMED strategy measures
                # sustained speech against (RFC §12.6).
                self._speech_started_at[session.id] = time.monotonic()
            # A measured playback with no chunk out yet has said nothing; any
            # other one may be audible, and its echo is guarded as usual.
            nothing_played = playback is None or (
                playback.measured and playback.first_audio_at is None
            )
            if nothing_played and self._unheard_turns.hold(session.id):
                # The routed turn's response is not heard yet: it waits, and
                # this speech is a user turn, not a barge-in. Nothing plays,
                # so there is no echo to suppress (RFC §12.3.12).
                logger.debug("Speech resumed before the response was heard: holding it")
                playback = None
            if playback:
                # During drain period (send_audio returned, waiting for echo
                # decay), skip barge-in — nothing is actually playing.
                done_ev = self._playback_done_events.get(session.id)
                if done_ev is not None and done_ev.is_set():
                    # Drain period — TTS finished but echo decay in progress.
                    # Suppress STT to avoid transcribing residual echo.
                    suppress_speech = True
                    with self._state_lock:
                        self._suppressed_sessions.add(session.id)
                else:
                    decision = self._interruption_handler.evaluate(
                        playback_position_ms=playback.played_ms,
                        speech_duration_ms=0,
                        transcript_expected=self._can_transcribe_held_speech(),
                    )
                    if decision.should_interrupt:
                        self._schedule(
                            self._handle_barge_in(session, playback, room_id),
                            name=f"barge_in:{session.id}",
                        )
                    elif decision.is_backchannel:
                        # The bot keeps talking and the acknowledgement is not
                        # a turn: its segment is discarded (RFC §12.6 step 5).
                        suppress_speech = True
                        with self._state_lock:
                            self._suppressed_sessions.add(session.id)
                        self._schedule(
                            self._fire_backchannel_hook(session, "", room_id),
                            name=f"backchannel:{session.id}",
                        )
                    else:
                        # No interruption (yet) — suppress STT so TTS echo
                        # picked up by the mic cannot loop back.
                        suppress_speech = True
                        with self._state_lock:
                            self._suppressed_sessions.add(session.id)
                        if decision.queue_speech:
                            # DISABLED: the speech is not echo to throw away,
                            # it waits its turn (RFC §12.6).
                            with self._state_lock:
                                self._queueing_sessions.add(session.id)
                        elif decision.pending_confirmation:
                            # CONFIRMED: this first look happens at speech
                            # onset, where no strategy with a minimum duration
                            # can ever fire. Take a second look once the speech
                            # has had time to sustain (RFC §12.6).
                            self._arm_barge_in_confirmation(
                                session, room_id, decision.confirm_after_ms
                            )
                            if decision.awaiting_transcript:
                                self._hold_for_transcript(session, room_id, vad_event.audio_bytes)

            if not suppress_speech:
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
            self._note_turn_speech(session.id, speaking=False)
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
        self._fire_level_hook(
            session, frame.data, self._last_input_level_at, HookTrigger.ON_INPUT_AUDIO_LEVEL
        )

    def _fire_level_hook(
        self,
        session: VoiceSession,
        data: bytes,
        last_fired: dict[str, float],
        trigger: HookTrigger,
    ) -> None:
        """Fire an audio level hook for a bound session, at most every 100 ms.

        The binding is read before the throttle's timestamp is written: audio
        still flowing for a session unbound since leaves no entry behind.
        """
        now = time.monotonic()
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
            if binding_info is None or now - last_fired.get(session.id, 0.0) < 0.1:
                return
            last_fired[session.id] = now
        if not self._framework:
            return
        self._schedule(
            self._fire_audio_level_hook(session, rms_db(data), binding_info[0], trigger),
            name=f"{trigger.value.removeprefix('on_')}:{session.id}",
        )

    def _on_audio_played_for_level(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Fire ON_OUTPUT_AUDIO_LEVEL at real playback pace (PortAudio callback).

        Shares ``_last_output_level_at`` with ``_fire_output_level`` in the
        TTS mixin so they naturally deduplicate.
        """
        self._fire_level_hook(
            session, frame.data, self._last_output_level_at, HookTrigger.ON_OUTPUT_AUDIO_LEVEL
        )

    def _on_processed_frame_for_bridge(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Forward processed audio to other sessions via the bridge."""
        if self._bridge is None:
            return
        # Fast path: no framework or no BEFORE_BRIDGE_AUDIO hooks registered
        # → forward right here in the frame callback (on the pipeline's loop,
        # with or without the DSP pool), no task in between.
        if self._framework is None or not self._framework.hook_engine.has_hooks(
            HookTrigger.BEFORE_BRIDGE_AUDIO
        ):
            self._bridge.forward(session, frame)
            return
        # Slow path: schedule on event loop so we can fire sync hooks
        # (which support block/modify) before forwarding.
        self._schedule(
            self._fire_bridge_audio_and_forward(session, frame),
            name=f"bridge_audio:{session.id}",
        )

    async def _fire_bridge_audio_and_forward(
        self, session: VoiceSession, frame: AudioFrame
    ) -> None:
        """Fire BEFORE_BRIDGE_AUDIO hook, then forward via bridge.

        Called on the event loop when BEFORE_BRIDGE_AUDIO hooks are
        registered.  The hook can block the frame (``HookResult.block()``)
        or modify it (``HookResult.modify(event=...)``).
        """
        if self._bridge is None or self._framework is None:
            return
        room_id = self._session_bindings.get(session.id, (None, None))[0]
        if room_id is None:
            return
        try:
            from roomkit.voice.events import BridgeAudioEvent

            with self._voice_span_ctx(session):
                context = await self._framework._build_context(room_id)
                event = BridgeAudioEvent(session=session, frame=frame, room_id=room_id)
                result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.BEFORE_BRIDGE_AUDIO,
                    event,
                    context,
                    skip_event_filter=True,
                )
                if not result.allowed:
                    return
                # A hook that returned a modified event forwards *its* frame
                # (RFC §12.7): returning one and having it dropped is worse
                # than not offering the outcome at all. ``set_bridge_filter``
                # stays the cheaper path for per-frame work — it runs in the
                # audio thread and builds no context — but choosing it is the
                # integrator's call, not something silence should decide.
                if isinstance(result.event, BridgeAudioEvent):
                    frame = result.event.frame
        except Exception:
            logger.exception("Error firing BEFORE_BRIDGE_AUDIO hook")
        self._bridge.forward(session, frame)

    def _process_bridge_outbound(self, target_session: VoiceSession, frame: AudioFrame) -> Any:
        """Process bridged audio through the outbound pipeline for a target.

        Called by AudioBridge for each target before sending.  Runs the
        outbound pipeline (postprocessors, recorder tap, AEC reference,
        resampler) so that recording captures both sides and AEC stays
        accurate.
        """
        from roomkit.voice.audio_frame import AudioFrame as _AudioFrame
        from roomkit.voice.base import AudioChunk

        outbound_frame = _AudioFrame(
            data=frame.data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
        )
        if self._pipeline is not None:
            self._pipeline.set_aec_active(target_session.id, True, source="bridge")
            outbound_frame = self._pipeline.process_outbound(target_session, outbound_frame)
        return AudioChunk(
            data=outbound_frame.data,
            sample_rate=outbound_frame.sample_rate,
            channels=outbound_frame.channels,
        )

    def _resolve_session_backend(self, session: VoiceSession) -> VoiceBackend | None:
        """Return the backend for a session (bridge-aware).

        A session bound from a transport registered with :meth:`add_backend`
        (or bridged with an explicit backend) is served by that transport.
        Falls back to the channel's default ``_backend``.
        """
        if self._bridge is not None:
            bridge_backend = self._bridge.get_session_backend(session.id)
            if bridge_backend is not None:
                return bridge_backend
        return self._session_backends.get(session.id, self._backend)

    def add_backend(self, backend: VoiceBackend) -> None:
        """Serve sessions from another transport on this channel (RFC §12.7.3).

        A bridged room often mixes transports — phone callers on SIP beside
        browser participants on WebRTC. The added backend's inbound audio
        enters this channel's pipeline, and its session-ready and
        client-disconnected signals drive the channel's session lifecycle,
        as for the backend given at construction. Everything addressed to one
        of its sessions — bridged audio, TTS, assistant transcriptions, the
        playback cancel of an interruption — goes out through it.

        The pipeline is built once, for the primary backend's capabilities,
        so the channel must have been constructed with a backend. A session
        is matched to its transport by ``kit.join(..., backend=)``, or else
        by asking each added backend whether it holds the session.
        Adding a backend already served does nothing. Closing the channel
        closes the added backends too.

        Raises:
            RuntimeError: The channel has no primary backend or pipeline.
        """
        if self._backend is None or self._pipeline is None:
            raise RuntimeError(
                "add_backend() needs a channel constructed with a backend: "
                "the pipeline every transport's audio goes through is built for it"
            )
        if backend is self._backend or backend in self._extra_backends:
            return
        self._extra_backends.append(backend)
        self._pipeline_unsubscribers.append(
            backend.on_audio_received(self._pipeline_on_audio_received)
        )
        backend.on_session_ready(self._on_session_ready)
        backend.on_client_disconnected(self._on_backend_disconnected)

    def _added_backend_holding(self, session: VoiceSession) -> VoiceBackend | None:
        """The added transport that holds *session*, if one says it does."""
        for backend in self._extra_backends:
            if backend.get_session(session.id) is not None:
                return backend
        return None

    async def _broadcast_bridge_transcription(
        self, source_session: VoiceSession, text: str, room_id: str
    ) -> None:
        """Send a transcription to all OTHER bridged sessions in the room.

        In bridge mode, when one participant speaks, all other participants
        should see the transcription attributed to the speaker.
        """
        if self._bridge is None:
            return
        speaker = (
            source_session.metadata.get("caller_display_name")
            or source_session.metadata.get("caller_user")
            or source_session.participant_id
            or source_session.id[:8]
        )
        for session, backend in self._bridge.get_bridged_sessions(room_id):
            if session.id == source_session.id:
                continue
            try:
                await backend.send_transcription(session, text, speaker)
            except Exception:
                logger.debug("Failed to send bridge transcription to %s", session.id)

    def set_bridge_filter(self, fn: BridgeFrameFilter | None) -> None:
        """Set a synchronous filter for bridged audio frames.

        The filter runs in the audio callback thread before each frame
        is forwarded.  It receives ``(source_session, frame)`` and
        returns the frame (possibly modified) or ``None`` to drop it.

        This is the synchronous equivalent of ``BEFORE_BRIDGE_AUDIO``
        — use it for fast operations like per-session muting or gain.

        Args:
            fn: Filter function, or ``None`` to remove.
        """
        if self._bridge is not None:
            self._bridge.set_frame_filter(fn)

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
        if not binding_info:
            return  # unbound since: its TTS context is released, nothing to note
        redaction = self._pipeline_config.dtmf_redaction if self._pipeline_config else None
        if self._tts_context is not None and redaction is not None and redaction.enabled:
            # In-band tones carry the digits: the turn keeps no audio (RFC §17.6).
            self._tts_context.note_dtmf(session.id)
        if not self._framework:
            return

        room_id, _ = binding_info
        self._schedule(
            self._fire_dtmf_hook(session, dtmf_event, room_id),
            name=f"dtmf:{session.id}",
        )

    def _recording_room(self, session: VoiceSession) -> str | None:
        """The room a session's recording reports to, its binding's or, while
        the session is being unbound, the one it is leaving."""
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        return binding_info[0] if binding_info else self._ending_session_rooms.get(session.id)

    def _recording_span(self, session: VoiceSession) -> str | None:
        return self._voice_session_spans.get(session.id)

    def _schedule_recording_hook(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        self._schedule(coro, name=name)

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
                self._stt._telemetry = telemetry  # ty: ignore[unresolved-attribute]
            if self._tts is not None:
                self._tts._telemetry = telemetry  # ty: ignore[unresolved-attribute]
            if self._backend is not None:
                self._backend._telemetry = telemetry  # ty: ignore[unresolved-attribute]
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

    def bind_session(
        self,
        session: VoiceSession,
        room_id: str,
        binding: ChannelBinding,
        *,
        backend: VoiceBackend | None = None,
    ) -> None:
        """Bind a voice session to a room for message routing.

        Args:
            session: The voice session to bind.
            room_id: Target room ID.
            binding: Channel binding descriptor.
            backend: Override backend for the bridge.  When bridging
                sessions from different transports (e.g. SIP + WebRTC),
                pass the session's own backend so the bridge sends audio
                through the correct transport.
        """
        if self._tts_context is not None:
            self._tts_context.open(session.id)
        with self._state_lock:
            self._session_bindings[session.id] = (room_id, binding)
            # Dual-signal: atomically check and clear pending ready flag
            # under the same lock that writes _session_bindings, so
            # _on_session_ready cannot interleave between the two.
            was_ready_pending = session.id in self._session_ready_pending
            self._session_ready_pending.discard(session.id)
            # The lock decides the opening language before the first stream
            # (continuous mode opens it right below).
            if self._stt_language_lock is not None:
                self._stt_languages[session.id] = self._stt_language_lock.language_for(session.id)
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
        # Notify pipeline of session activation, under the session's span
        self._pipeline_session_active(session, parent_span=span_id)
        # Register session with audio bridge
        bridge_backend = backend or self._added_backend_holding(session) or self._backend
        if bridge_backend is not None and bridge_backend is not self._backend:
            self._session_backends[session.id] = bridge_backend
        if self._bridge is not None and bridge_backend is not None:
            self._bridge.add_session(session, room_id, bridge_backend)
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

    def add_media_tap(self, callback: Callable[[VoiceSession, AudioFrame], None]) -> None:
        """Register a tap on processed inbound audio frames (for room recording).

        Delegates to the pipeline's ``on_processed_frame`` callback list.
        """
        if self._pipeline is not None:
            self._pipeline.on_processed_frame(callback)

    def add_outbound_media_tap(self, callback: Callable[[VoiceSession, bytes, int], None]) -> None:
        """Register a tap on outbound TTS audio (for room recording).

        The callback receives ``(session, pcm_data, sample_rate)`` for
        every outbound chunk after pipeline processing.
        """
        self._outbound_audio_taps.append(callback)

    def unbind_session(self, session: VoiceSession) -> None:
        """Remove session binding."""
        # Cancel any active streaming STT
        self._cancel_stt_stream(session.id)
        with self._state_lock:
            self._session_ready_pending.discard(session.id)
            self._forget_speech_state(session.id)
            binding_info = self._session_bindings.pop(session.id, None)
            self._session_backends.pop(session.id, None)
        if binding_info is None:
            return  # Already unbound — prevent double pipeline/telemetry calls
        # Unregister from audio bridge
        if self._bridge is not None:
            self._bridge.remove_session(session.id)
            room_id, _ = binding_info
            if self._bridge.get_participant_count(room_id) < 2 and self._pipeline is not None:
                with self._state_lock:
                    remaining_session_ids = [
                        sid
                        for sid, (bound_room_id, _) in self._session_bindings.items()
                        if bound_room_id == room_id
                    ]
                for remaining_session_id in remaining_session_ids:
                    self._pipeline.set_aec_active(
                        remaining_session_id,
                        False,
                        source="bridge",
                    )
        # Notify pipeline of session end. It stops the session's recording
        # and reports it synchronously from inside this call, after the
        # binding is gone: the room rides along so ON_RECORDING_STOPPED
        # still knows where to fire.
        self._ending_session_rooms[session.id] = binding_info[0]
        try:
            self._pipeline_session_ended(session)
        finally:
            self._ending_session_rooms.pop(session.id, None)
        self._release_tts_context(session.id)
        # Clear pending turns, audio, and interrupt cooldown
        self._pending_turns.pop(session.id, None)
        self._pending_turn_speakers.pop(session.id, None)
        self._speaker_trackers.pop(session.id, None)
        self._transcript_locks.pop(session.id, None)
        if self._pipeline_speaker_tally is not None:
            self._pipeline_speaker_tally.reset(session.id)
        self._pending_audio.pop(session.id, None)
        self._cancel_turn_wait(session.id)
        self._unheard_turns.release(session.id)
        self._last_tts_ended_at.pop(session.id, None)
        # Clear the session's STT language and what the lock knew about it
        with self._state_lock:
            self._stt_languages.pop(session.id, None)
        if self._stt_language_lock is not None:
            self._stt_language_lock.forget(session.id)
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
        if self._framework:
            room_id, _ = binding_info
            self._schedule(
                self._emit_session_ended(session, room_id),
                name=f"session_ended:{session.id}",
            )

    def _forget_speech_state(self, session_id: str) -> None:
        """Drop what the VAD handlers keep about a session's speech; the caller holds the lock.

        A session unbound mid-utterance saw its SPEECH_START and never its
        SPEECH_END, which is what would have cleared most of this.
        """
        self._held_for_transcript.pop(session_id, None)
        self._burst_words.pop(session_id, None)
        self._burst_backchannel.pop(session_id, None)
        self._speech_started_at.pop(session_id, None)
        self._barge_in_energy_count.pop(session_id, None)
        self._suppressed_sessions.discard(session_id)
        self._queueing_sessions.discard(session_id)
        self._queued_speech.pop(session_id, None)

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
    # Per-session STT language
    # -------------------------------------------------------------------------

    def set_stt_language(self, session: VoiceSession, language: str | None) -> None:
        """Choose the STT language for one session, from its next stream on.

        ``None`` returns the session to the provider's configured language.
        A streaming STT fixes its language when the stream opens, so the
        choice lands on the next stream:

        - **VAD mode** — the next utterance. A stream that is open stays
          open; restarting it would cut the utterance in progress in two.
        - **Continuous mode** — right away: the current cycle is ended and
          the loop reconnects with the new language. Audio arriving in the
          gap is kept, as on every reconnect.
        - **Batch mode** — the next :meth:`flush_stt`.

        The typical caller is an ``ON_TRANSCRIPTION`` hook reading
        ``event.language`` from a detecting stream (Deepgram ``multi``) and
        pinning the session to what it heard;
        :class:`~roomkit.voice.stt.language.STTLanguageLock` packages that
        loop.

        Raises:
            RuntimeError: No STT provider, or one whose
                ``supports_language_override`` is false.
        """
        if self._stt is None:
            raise RuntimeError("set_stt_language() requires an STT provider")
        if not getattr(self._stt, "supports_language_override", False):
            raise RuntimeError(
                f"{self._stt.name} does not support a per-session language "
                "(supports_language_override is false)"
            )
        if not self._store_stt_language(session.id, language):
            return
        if self._continuous_stt:
            self._restart_continuous_stt_cycle(session.id)

    def get_stt_language(self, session: VoiceSession) -> str | None:
        """The STT language chosen for a session, ``None`` for the provider's default."""
        with self._state_lock:
            return self._stt_languages.get(session.id)

    # -------------------------------------------------------------------------
    # Outbound DTMF
    # -------------------------------------------------------------------------

    _VALID_DTMF_DIGITS = frozenset("0123456789*#ABCD")

    def send_dtmf(self, session: VoiceSession, digit: str, duration_ms: int = 160) -> None:
        """Send a DTMF digit to the remote party via the voice backend.

        The digit is sent as an RFC 4733 telephone-event (out-of-band).
        Requires a backend with ``DTMF_SIGNALING`` capability (SIP, RTP).

        Args:
            session: The active voice session.
            digit: DTMF digit ('0'-'9', '*', '#', 'A'-'D').
            duration_ms: Tone duration in milliseconds (default 160).

        Raises:
            RuntimeError: If no backend is configured or session is ended.
            ValueError: If *digit* or *duration_ms* is invalid.
        """
        if self._backend is None:
            raise RuntimeError("No voice backend configured")
        if digit not in self._VALID_DTMF_DIGITS:
            raise ValueError(f"Invalid DTMF digit {digit!r}. Must be one of 0-9, *, #, A-D.")
        if not 1 <= duration_ms <= 10000:
            raise ValueError(f"duration_ms must be between 1 and 10000, got {duration_ms}")
        from roomkit.voice.base import VoiceSessionState

        if session.state == VoiceSessionState.ENDED:
            raise RuntimeError(f"Cannot send DTMF on ended session {session.id}")
        self._backend.send_dtmf(session, digit, duration_ms)

    # -------------------------------------------------------------------------
    # Barge-in handling
    # -------------------------------------------------------------------------

    async def _flush_queued_speech(self, session_id: str) -> None:
        """Process speech the DISABLED strategy held during playback (RFC §12.6).

        Called once the bot's audio has drained. Segments are replayed in the
        order they were spoken, through the ordinary speech-end path, so the
        caller's turn arrives late rather than not at all.
        """
        with self._state_lock:
            queued = self._queued_speech.pop(session_id, [])
            binding_info = self._session_bindings.get(session_id)
        if not queued or binding_info is None or not self._framework:
            return
        session = self._backend.get_session(session_id) if self._backend else None
        if session is None:
            return
        room_id, _ = binding_info
        # Logged at info like the barge-in confirmation: this is the observable
        # proof that speech held during playback was replayed rather than
        # dropped (RFC §12.6), and it fires once per playback at most.
        logger.info(
            "Flushing %d queued speech segment(s) for %s (playback finished)",
            len(queued),
            session_id,
        )
        # Any DTMF heard while speech was held goes with every held segment.
        dtmf_seen = self._tts_context is not None and self._tts_context.take_dtmf(session_id)
        dtmf_kwargs = {"dtmf_seen": True} if dtmf_seen else {}
        for audio, speaker_claim in queued:
            await self._process_speech_end(
                session, audio, room_id, None, speaker_claim=speaker_claim, **dtmf_kwargs
            )

    def _arm_barge_in_confirmation(
        self, session: VoiceSession, room_id: str, delay_ms: int, *, from_vad: bool = True
    ) -> None:
        """Schedule the second look a duration-based strategy needs (RFC §12.6).

        ``InterruptionHandler.evaluate`` is first called at speech onset, where
        the sustained duration is 0 by construction — CONFIRMED (and SEMANTIC
        falling back to it) can never fire from that call alone. One task per
        session re-evaluates once the speech has had ``delay_ms`` to sustain.

        ``from_vad`` marks the pipeline-VAD path, which suppressed the segment
        while waiting and therefore owns un-suppressing it on confirmation. A
        transport that detected the speech itself keeps its own capture.
        """
        self._cancel_barge_in_confirmation(session.id)
        self._schedule(
            self._confirm_barge_in(
                session, room_id, delay_ms, onset=time.monotonic(), from_vad=from_vad
            ),
            name=f"barge_in_confirm:{session.id}",
        )

    def _cancel_barge_in_confirmation(self, session_id: str) -> None:
        with self._state_lock:
            existing = self._confirm_tasks.pop(session_id, None)
        if existing is not None:
            existing.cancel()

    def _hold_for_transcript(
        self, session: VoiceSession, room_id: str, pre_roll: bytes | None
    ) -> None:
        """Transcribe a held SEMANTIC segment so its words can be classified.

        The segment stays suppressed: nothing reaches hooks or the AI unless a
        partial transcript (or the duration-only second look) says it cuts in.
        Without a streaming STT there are no words to wait for, and the second
        look armed alongside decides on duration alone (CONFIRMED fallback).
        """
        if not self._can_transcribe_held_speech():
            return
        with self._state_lock:
            self._held_for_transcript[session.id] = False
        self._start_stt_stream(session, room_id, pre_roll=pre_roll)

    def _can_transcribe_held_speech(self) -> bool:
        """A VAD segment held during playback can be streamed to the STT."""
        return self._stt is not None and self._stt.supports_streaming and not self._continuous_stt

    def _is_held_for_transcript(self, session_id: str) -> bool:
        with self._state_lock:
            return session_id in self._held_for_transcript

    def _barge_in_words(self, session_id: str) -> tuple[str, bool]:
        """The words heard so far of the speech being judged, and whether a
        streaming STT is producing them (RFC §12.3.13)."""
        if self._continuous_stt:
            with self._state_lock:
                return self._burst_words.get(session_id, ""), True
        if self._is_held_for_transcript(session_id):
            return self._held_transcript(session_id), True
        return "", False

    def _held_transcript(self, session_id: str) -> str:
        """Latest words of a held segment, empty when none arrived yet."""
        with self._state_lock:
            if session_id not in self._held_for_transcript:
                return ""
        state = self._stt_streams.get(session_id)
        if state is None:
            return ""
        return state.partial_text or state.final_text or ""

    def _on_held_transcript(self, session: VoiceSession, room_id: str, text: str) -> None:
        """Classify the words of a held SEMANTIC segment as they arrive."""
        with self._state_lock:
            if session.id not in self._held_for_transcript:
                return
            playback = self._playing_sessions.get(session.id)
            onset = self._speech_started_at.get(session.id)
        if playback is None:
            return
        done_ev = self._playback_done_events.get(session.id)
        if done_ev is not None and done_ev.is_set():
            return
        duration_ms = int((time.monotonic() - onset) * 1000) if onset is not None else 0
        decision = self._interruption_handler.evaluate(
            playback_position_ms=playback.played_ms,
            speech_duration_ms=duration_ms,
            speech_text=text,
        )
        if decision.should_interrupt:
            with self._state_lock:
                # One partial cuts in; the ones behind it find nothing held.
                if self._held_for_transcript.pop(session.id, None) is None:
                    return
                # Now, not in the scheduled task: a speech end landing in
                # between must see the user's turn, not echo to discard.
                self._suppressed_sessions.discard(session.id)
            self._cancel_barge_in_confirmation(session.id)
            logger.info("Barge-in on held transcript %r (session %s)", text, session.id)
            self._schedule(
                self._cut_in(session, playback, room_id, restart_stt=False),
                name=f"barge_in:{session.id}",
            )
        elif decision.is_backchannel:
            # Acknowledged: the timer must not cut it for running long. Later
            # words are still classified, so "uh-huh... wait" can cut in.
            self._cancel_barge_in_confirmation(session.id)
            self._note_backchannel(session, text, room_id)

    def _settle_wordless_held_speech(
        self,
        session: VoiceSession,
        audio: bytes,
        onset: float | None,
        speaker_claim: Future[SpeakerAttribution | None] | None,
        dtmf_seen: bool,
    ) -> None:
        """Have a held segment that ended before its first word judged on its final words."""
        stream_state = self._end_stt_stream(session.id)
        with self._state_lock:
            binding_info = self._session_bindings.get(session.id)
        if stream_state is None or not binding_info or not self._framework:
            self._release_unheard_turn(session.id)
            return
        duration_ms = int((time.monotonic() - onset) * 1000) if onset is not None else 0
        self._schedule(
            self._judge_held_speech(
                session,
                audio,
                binding_info[0],
                stream_state,
                speech_duration_ms=duration_ms,
                speaker_claim=speaker_claim,
                dtmf_seen=dtmf_seen,
                transcript=self._expect_transcript(session.id),
            ),
            name=f"held_speech_end:{session.id}",
        )

    def _note_backchannel(self, session: VoiceSession, text: str, room_id: str) -> None:
        """Fire ON_BACKCHANNEL, once per held segment."""
        with self._state_lock:
            if self._held_for_transcript.get(session.id):
                return
            if session.id in self._held_for_transcript:
                self._held_for_transcript[session.id] = True
        self._schedule(
            self._fire_backchannel_hook(session, text, room_id),
            name=f"backchannel:{session.id}",
        )

    async def _cut_in(
        self,
        session: VoiceSession,
        playback: TTSPlaybackState,
        room_id: str,
        *,
        restart_stt: bool,
    ) -> None:
        """Let a held VAD segment through as the user's turn, then barge in.

        ``restart_stt`` is False when the segment is already being transcribed
        (SEMANTIC held it for its words, pre-roll included).
        """
        with self._state_lock:
            self._suppressed_sessions.discard(session.id)
            self._held_for_transcript.pop(session.id, None)
        if restart_stt and self._stt is not None and self._stt.supports_streaming:
            self._start_stt_stream(session, room_id)
        await self._fire_speech_start_hooks(session, room_id)
        await self._handle_barge_in(session, playback, room_id)

    def _playback_to_confirm(self, session_id: str, *, from_vad: bool) -> TTSPlaybackState | None:
        """The playback a pending interruption may still cut, or None."""
        with self._state_lock:
            playback = self._playing_sessions.get(session_id)
            still_speaking = session_id in self._speech_started_at
            still_suppressed = session_id in self._suppressed_sessions
        # The bot finished on its own, or something else already resolved
        # the turn: there is nothing left to interrupt.
        if playback is None:
            return None
        done_ev = self._playback_done_events.get(session_id)
        if done_ev is not None and done_ev.is_set():
            return None
        # The speech stopped before it could confirm: a blip, or echo the
        # suppression correctly swallowed.
        if from_vad and not (still_speaking and still_suppressed):
            return None
        return playback

    async def _confirm_barge_in(
        self,
        session: VoiceSession,
        room_id: str,
        delay_ms: int,
        *,
        onset: float,
        from_vad: bool,
    ) -> None:
        """Re-evaluate a pending interruption once the speech has sustained."""
        task = asyncio.current_task()
        if task is not None:
            with self._state_lock:
                self._confirm_tasks[session.id] = task
        try:
            while True:
                await asyncio.sleep(max(delay_ms, 0) / 1000.0)
                playback = self._playback_to_confirm(session.id, from_vad=from_vad)
                if playback is None:
                    return
                words, transcript_expected = self._barge_in_words(session.id)
                speech_duration_ms = int((time.monotonic() - onset) * 1000)
                decision = self._interruption_handler.evaluate(
                    playback_position_ms=playback.played_ms,
                    speech_duration_ms=speech_duration_ms,
                    speech_text=words,
                    transcript_expected=transcript_expected,
                )
                if decision.is_backchannel:
                    if from_vad:
                        self._note_backchannel(session, words, room_id)
                    elif self._continuous_stt:
                        self._report_burst_backchannel(session, words, room_id)
                if decision.should_interrupt:
                    break
                if not decision.pending_confirmation:
                    return
                # Still waiting for the words of a transcribed segment.
                delay_ms = decision.confirm_after_ms

            logger.info(
                "Barge-in confirmed after %dms of sustained speech (session %s)",
                speech_duration_ms,
                session.id,
            )
            if from_vad:
                # Confirmed: the segment is the user talking, not echo. Let it
                # through and capture it, unless SEMANTIC already does.
                with self._state_lock:
                    held = session.id in self._held_for_transcript
                await self._cut_in(session, playback, room_id, restart_stt=not held)
            else:
                await self._handle_barge_in(session, playback, room_id)
        except asyncio.CancelledError:
            raise
        finally:
            with self._state_lock:
                if self._confirm_tasks.get(session.id) is task:
                    self._confirm_tasks.pop(session.id, None)

    def _claim_barge_in(self, playback: TTSPlaybackState) -> bool:
        """Take *playback* for one barge-in; False when one already owns it."""
        with self._state_lock:
            if playback.barge_in_claimed:
                return False
            playback.barge_in_claimed = True
            return True

    async def _handle_barge_in(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        if not self._framework:
            return
        # One interruption per playback. Building the context and running the
        # hooks takes as long as the store does, and until interrupt() pops the
        # playback every trigger path still sees it playing: without the claim
        # the energy check re-fires every 100 ms of continued speech, and two
        # paths deciding on the same speech both fire. Claimed before any
        # await, so the tasks scheduled behind this one return here.
        if not self._claim_barge_in(playback):
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
                audio_position_ms=playback.played_ms,
            )
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_BARGE_IN,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.exception("Error firing ON_BARGE_IN for session %s", session.id)
        # The playback is claimed: it is cut even when the hooks could not
        # run, or nothing would ever interrupt it again.
        try:
            await self.interrupt(session, reason="barge_in")
        except Exception:
            logger.exception("Error interrupting playback for session %s", session.id)

    async def _store_interrupted_utterance(
        self, session: VoiceSession, playback: TTSPlaybackState, room_id: str
    ) -> None:
        """Record that the bot was cut off mid-utterance (RFC §12.3.13 step 2).

        The timeline otherwise says the bot said the whole thing, because the
        AI's response event is written when the text is produced, not when it
        is heard. What the room actually heard is a prefix of it, and nothing
        recorded that.

        The full text is stored, not a truncated guess: ``played_ms`` is the
        honest measure of how far playback got, and cutting the string at the
        same proportion would invent a word boundary that TTS timing does not
        guarantee. ``played_percentage`` follows only where the utterance's
        total duration is known.
        """
        if self._framework is None or not playback.text:
            return
        from roomkit.models.event import TextContent

        played_ms = playback.played_ms
        metadata: dict[str, Any] = {
            "interrupted": True,
            "played_ms": played_ms,
            "voice_session_id": session.id,
        }
        if playback.total_duration_ms:
            metadata["played_percentage"] = round(
                min(100.0, 100.0 * played_ms / playback.total_duration_ms), 1
            )
        # The agent's words, not the listener's (RMK-533): no participant id,
        # and the answer it cut named, so the AI channel can mark it as cut.
        if playback.answer_channel_id is not None:
            metadata["answer_channel_id"] = playback.answer_channel_id
            metadata["answer_responds_to"] = playback.answer_responds_to
        try:
            await self._framework.send_event(
                room_id,
                self.channel_id,
                TextContent(body=playback.text),
                metadata=metadata,
                visibility=Visibility.INTERNAL,
            )
        except Exception:
            logger.exception(
                "Could not record the interrupted utterance for session %s", session.id
            )

    async def interrupt(self, session: VoiceSession, *, reason: str = "explicit") -> bool:
        """Interrupt ongoing TTS playback for a session."""
        import time as _time

        with self._state_lock:
            playback = self._playing_sessions.pop(session.id, None)
        if not playback:
            return False

        self._last_tts_ended_at[session.id] = _time.monotonic()
        # send_audio() already returned: the utterance went out whole and only
        # the echo-decay window was left, so nothing is cut off here.
        done_ev = self._playback_done_events.get(session.id)
        drained = done_ev is not None and done_ev.is_set()

        config = self._interruption_handler.config
        # ``flush_partial_tts`` decides what happens to audio already handed to
        # the backend (RFC §12.3.13 step 1). Left true, the buffer is dropped
        # and the bot stops mid-word; set false, the current utterance is
        # allowed to finish while the user's speech is processed alongside it.
        if not drained:
            # The cut the timeline and the TTS context both record: one
            # played_ms, taken here (RFC §12.3.13 steps 2 and 3).
            playback.stopped_at = _time.monotonic()
            self._end_tts_turn(playback)
        if (
            config.flush_partial_tts
            and self._backend
            and VoiceCapability.INTERRUPTION in self._backend.capabilities
        ):
            await self._session_output_backend(session).cancel_audio(session)

        # Bypass AEC after TTS stops so user audio passes unchanged.  Keep the
        # converged hardware echo path for the next playback turn.
        if self._pipeline is not None and self._pipeline._config.aec is not None:
            self._pipeline.set_aec_active(session.id, False)

        if self._framework:
            binding_info = self._session_bindings.get(session.id)
            if binding_info:
                room_id, _ = binding_info
                if config.keep_partial_transcript and not drained:
                    await self._store_interrupted_utterance(session, playback, room_id)
                try:
                    from roomkit.voice.events import TTSCancelledEvent

                    context = await self._framework._build_context(room_id)
                    event = TTSCancelledEvent(
                        session=session,
                        reason=reason,  # ty: ignore[invalid-argument-type]
                        text=playback.text,
                        audio_position_ms=playback.played_ms,
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
            playback.played_ms,
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
        """Handle barge-in detected by the transport.

        A backend that detects speech itself still answers to the configured
        interruption policy (RFC §12.6): the strategy decides, whichever layer
        noticed the speech. ``DISABLED`` therefore holds against a backend with
        native detection, as it does against the pipeline's own VAD.
        """
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

        # The transport reports detected speech, not a decision. It carries no
        # duration, so a duration-based strategy gets its second look the same
        # way the VAD path does.
        words, transcript_expected = self._barge_in_words(session.id)
        decision = self._interruption_handler.evaluate(
            playback_position_ms=playback.played_ms,
            speech_duration_ms=0,
            speech_text=words,
            transcript_expected=transcript_expected,
        )
        if not decision.should_interrupt:
            if decision.is_backchannel and self._continuous_stt:
                self._report_burst_backchannel(session, words, room_id)
            if decision.pending_confirmation:
                self._arm_barge_in_confirmation(
                    session, room_id, decision.confirm_after_ms, from_vad=False
                )
            else:
                logger.debug("Backend barge-in ignored for %s: %s", session.id, decision.reason)
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
        if event.type == EventType.SYSTEM or event.visibility == Visibility.INTERNAL:
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
        # 0. No more frames, and those in flight finish first: their callbacks
        #    run against live state and anything they open is swept below.
        await self._pipeline_quiesce()
        # 1. Cancel STT streams first (stops feeding audio)
        for sid in list(self._stt_streams):
            self._cancel_stt_stream(sid)
        # 2. Cancel scheduled fire-and-forget tasks and await completion
        for task in self._scheduled_tasks:
            task.cancel()
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        self._scheduled_tasks.clear()
        # 3. Close the pipeline (its DSP pool was drained at step 0).
        if self._pipeline is not None:
            self._pipeline.close()
        # 3b. Close audio bridge
        if self._bridge is not None:
            self._bridge.close()
        # 4. Drop every TTS context, then close STT/TTS providers (unless the
        #    caller owns their lifecycle)
        if self._tts_context is not None:
            for session_id in set(self._tts_context.sessions()) | set(self._session_bindings):
                self._release_tts_context(session_id)
        if self._close_providers:
            if self._stt:
                await self._stt.close()
            if self._tts:
                await self._tts.close()
        # 5. Close backends last (transport layer)
        if self._backend:
            await self._backend.close()
        for added in self._extra_backends:
            await added.close()
        self._extra_backends.clear()
        self._session_backends.clear()
        self._session_bindings.clear()
        self._playing_sessions.clear()
        self._batch_audio_buffers.clear()
        self._batch_audio_sample_rate.clear()
        self._frame_counts.clear()
        self._session_ready_pending.clear()
        self._delivered_tts_events.clear()

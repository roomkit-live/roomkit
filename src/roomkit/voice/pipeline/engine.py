"""Audio pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.voice.base import VoiceCapability
from roomkit.voice.pipeline._telemetry import _PipelineTelemetry, active_stage_names
from roomkit.voice.pipeline.aec.base import AECProvider
from roomkit.voice.pipeline.vad.base import VADEventType

if TYPE_CHECKING:
    from types import TracebackType

    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.debug_taps import DebugTapSession
    from roomkit.voice.pipeline.diarization.base import DiarizationResult
    from roomkit.voice.pipeline.dtmf.base import DTMFEvent
    from roomkit.voice.pipeline.recorder.base import RecordingHandle, RecordingResult
    from roomkit.voice.pipeline.resampler.base import ResamplerProvider
    from roomkit.voice.pipeline.vad.base import VADEvent

logger = logging.getLogger("roomkit.voice.pipeline")


def _maybe_schedule(result: object) -> None:
    """Schedule a coroutine if the callback returned one."""
    if asyncio.coroutines.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(result)
            task.add_done_callback(log_task_exception)
        except RuntimeError:
            # No running event loop — log and close the coroutine to avoid warning
            logger.warning("Async callback returned outside event loop; dropping")
            result.close()


# Callback type aliases
SpeechEndPipelineCallback = Callable[["VoiceSession", bytes], Any]
SpeechFramePipelineCallback = Callable[["VoiceSession", "AudioFrame"], Any]
ProcessedFrameCallback = Callable[["VoiceSession", "AudioFrame"], Any]
VADEventCallback = Callable[["VoiceSession", "VADEvent"], Any]
SpeakerChangeCallback = Callable[["VoiceSession", "DiarizationResult"], Any]
DTMFCallback = Callable[["VoiceSession", "DTMFEvent"], Any]
RecordingStartedCallback = Callable[["VoiceSession", "RecordingHandle"], Any]
RecordingStoppedCallback = Callable[["VoiceSession", "RecordingResult"], Any]


class _StageEnvelope:
    """What one inbound stage runs inside: a failure it survives, and a clock.

    The pipeline runs on the media path and there is no caller to unwind to, so
    a stage that raises is logged and the frame carries on as the previous stage
    left it — the assignment inside the block simply never happens.  Its time
    still counts: a stage that fails slowly is worth seeing.

    Only a frame inside a speech segment is timed, because that segment's span
    is where the timings are reported and there is nowhere else to put them.

    This is deliberately a plain object rather than a ``@contextmanager``
    generator, which measured 2.3x its cost per stage for the same call site.
    """

    __slots__ = ("_stage", "_started", "_stream", "_telemetry", "_timed")

    def __init__(self, telemetry: _PipelineTelemetry, stream: str, stage: str) -> None:
        self._telemetry = telemetry
        self._stream = stream
        self._stage = stage
        self._timed = telemetry.in_segment(stream)
        self._started = time.perf_counter_ns() if self._timed else 0

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self._timed:
            self._telemetry.add_stage_time(
                self._stream, self._stage, time.perf_counter_ns() - self._started
            )
        if exc is None:
            return False
        logger.error("Pipeline stage '%s' error", self._stage, exc_info=exc)
        return isinstance(exc, Exception)


@dataclass
class InboundResult:
    """What the inbound stages produced for one frame.

    Returned by :meth:`AudioPipeline.process_inbound_stream`, the pull-style
    entry point.  A caller that owns its own lane reads the result rather than
    registering callbacks, because the callback contract is written in terms of
    a VoiceSession and a lane has none.
    """

    frame: AudioFrame
    """The frame as the last stage left it."""

    vad_event: VADEvent | None
    """What the VAD said about this frame, if a VAD is configured."""


class AudioPipeline:
    """Orchestrates audio frame processing through pipeline stages.

    Inbound processing order:
        [Resampler] -> [Recorder tap] -> [DTMF] -> [AEC] -> [AGC] ->
        [Denoiser] -> [VAD] -> [Diarization]

    Outbound processing order:
        [PostProcessors] -> [Recorder tap] -> AEC.feed_reference -> [Resampler]

    AEC and AGC stages are skipped when the backend declares
    NATIVE_AEC / NATIVE_AGC capabilities.
    """

    def __init__(
        self,
        config: AudioPipelineConfig,
        *,
        backend_capabilities: VoiceCapability = VoiceCapability.NONE,
        backend_feeds_aec_reference: bool = False,
    ) -> None:
        self._config = config
        self._agc = config.agc
        if self._agc is None and config.agc_config is not None:
            from roomkit.voice.pipeline.agc.simple import SimpleAGCProvider

            self._agc = SimpleAGCProvider(config.agc_config)
        self._backend_capabilities = backend_capabilities
        self._backend_feeds_aec_ref = backend_feeds_aec_reference
        self._speech_end_callbacks: list[SpeechEndPipelineCallback] = []
        self._speech_frame_callbacks: list[SpeechFramePipelineCallback] = []
        self._processed_frame_callbacks: list[ProcessedFrameCallback] = []
        self._vad_event_callbacks: list[VADEventCallback] = []
        self._in_speech_sessions: set[str] = set()
        # Streams handed to the stages — the keys reset() must release.
        self._stage_streams: set[str] = set()
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []
        self._dtmf_callbacks: list[DTMFCallback] = []
        self._recording_started_callbacks: list[RecordingStartedCallback] = []
        self._recording_stopped_callbacks: list[RecordingStoppedCallback] = []
        self._last_speaker_id: dict[str, str | None] = {}
        # Format that actually reaches AEC after inbound normalization, keyed
        # by stream. A channel can carry sessions with different native
        # formats, and the pre-resampler transport format is not what AEC sees.
        self._aec_capture_formats: dict[str, tuple[int, int, int]] = {}
        self._aec_capture_formats_lock = threading.Lock()
        # A stream may have more than one concurrent playback source (for
        # example TTS layered onto a human-to-human bridge).  Keep the AEC
        # active until the last source stops; resetting it when only TTS ends
        # would destroy the bridge's still-live adaptive filter.
        self._aec_active_sources: dict[str, set[str]] = {}
        self._aec_active_sources_lock = threading.Lock()
        # Lazy resampler for AEC reference (created on first mismatch)
        self._aec_resampler: ResamplerProvider | None = None
        self._aec_resampler_lock = threading.Lock()
        # Active recording handle (per session, keyed by session_id)
        self._recording_handles: dict[str, RecordingHandle] = {}
        # Per-session lock for process_outbound — serializes concurrent
        # outbound calls (bridge forwarding + TTS) for the same target
        # session to protect the recorder handle and other per-session state.
        self._outbound_locks: dict[str, threading.Lock] = {}
        # Debug tap sessions (per session, keyed by session_id)
        self._debug_tap_sessions: dict[str, DebugTapSession] = {}
        # Whether playback-time AEC reference is wired (suppresses
        # generation-time feeding in process_outbound).
        self._playback_aec_wired = False
        # Separate resampler for playback AEC path (may run on audio thread)
        self._playback_aec_resampler: ResamplerProvider | None = None
        self._playback_aec_resampler_lock = threading.Lock()
        # Resolve effective resampler (auto-default when contract is set)
        self._resampler: ResamplerProvider | None
        if config.resampler is not None:
            self._resampler = config.resampler
        elif config.contract is not None:
            self._resampler = _create_default_resampler()
        else:
            self._resampler = None
        # Spans, stage timings and counters — instrumentation, no audio.  The
        # stage list is fixed once the config is: nothing here reads it again.
        self._telemetry = _PipelineTelemetry(
            config.telemetry,
            active_stage_names(
                config,
                resampling=self._resampler is not None and config.contract is not None,
                backend_capabilities=backend_capabilities,
            ),
        )

    # -----------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------

    def on_speech_end(self, callback: SpeechEndPipelineCallback) -> None:
        """Register callback for when VAD detects speech end."""
        self._speech_end_callbacks.append(callback)

    def on_speech_frame(self, callback: SpeechFramePipelineCallback) -> None:
        """Register callback for processed audio frames during speech."""
        self._speech_frame_callbacks.append(callback)

    def on_processed_frame(self, callback: ProcessedFrameCallback) -> None:
        """Register callback for every processed inbound frame.

        Fires after all pipeline stages (AEC, denoiser, VAD, etc.) for
        every frame, regardless of speech state.  Used by continuous STT
        streaming when no local VAD is configured.
        """
        self._processed_frame_callbacks.append(callback)

    def on_vad_event(self, callback: VADEventCallback) -> None:
        """Register callback for all VAD events."""
        self._vad_event_callbacks.append(callback)

    def on_speaker_change(self, callback: SpeakerChangeCallback) -> None:
        """Register callback for speaker change detection."""
        self._speaker_change_callbacks.append(callback)

    def on_dtmf(self, callback: DTMFCallback) -> None:
        """Register callback for DTMF tone detection."""
        self._dtmf_callbacks.append(callback)

    def on_recording_started(self, callback: RecordingStartedCallback) -> None:
        """Register callback for recording start."""
        self._recording_started_callbacks.append(callback)

    def on_recording_stopped(self, callback: RecordingStoppedCallback) -> None:
        """Register callback for recording stop."""
        self._recording_stopped_callbacks.append(callback)

    # -----------------------------------------------------------------
    # Telemetry helpers
    # -----------------------------------------------------------------

    def set_parent_span(self, session_id: str, span_id: str) -> None:
        """Set the parent span (VOICE_SESSION) for pipeline spans."""
        self._telemetry.set_parent_span(session_id, span_id)

    # -----------------------------------------------------------------
    # Inbound processing
    # -----------------------------------------------------------------

    def process_frame(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Process a single inbound audio frame through the pipeline.

        Backwards-compatible alias for process_inbound().
        """
        self.process_inbound(session, frame)

    def _debug_tap(self, session_id: str, stage: str, frame: AudioFrame) -> None:
        """Write a frame to the debug tap for the given stage (if active)."""
        dt = self._debug_tap_sessions.get(session_id)
        if dt is not None:
            dt.tap(stage, frame)

    def _fanout(
        self,
        callbacks: Sequence[Callable[..., Any]],
        subject: VoiceSession | None,
        payload: Any,
        label: str,
    ) -> None:
        """Notify every listener, letting none of them stop the others.

        ``subject`` is ``None`` for a stream-keyed caller, which reads the
        result rather than being called back: the callbacks are typed on a
        VoiceSession, so there is nothing honest to hand them.

        One listener raising must not cost the rest their notification, nor
        interrupt the frame that is still being processed — the pipeline runs
        on the media path and there is no caller to unwind to.
        """
        if subject is None:
            return
        for callback in callbacks:
            try:
                _maybe_schedule(callback(subject, payload))
            except Exception:
                logger.exception("%s callback error", label)

    def _timed_stage(self, stream: str, stage: str) -> _StageEnvelope:
        """Wrap one inbound stage — see :class:`_StageEnvelope`.

        Every stage wears this envelope, which is why the block under it can be
        the stage itself: the canonical order of §12.3 reads as a list of stages
        rather than as the plumbing around them.
        """
        return _StageEnvelope(self._telemetry, stream, stage)

    def process_inbound(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Process a single inbound audio frame through the pipeline.

        Order: [Resampler] -> [Recorder tap] -> [DTMF] -> [AEC] -> [AGC] ->
               [Denoiser] -> [VAD] -> [Diarization]
        """
        self._run_inbound(session.id, session, frame)

    def process_inbound_stream(self, stream: str, frame: AudioFrame) -> InboundResult:
        """Process a frame for a stream that has no VoiceSession behind it.

        Same stages in the same order as :meth:`process_inbound` — there is one
        implementation of the ordering and this shares it — but the result is
        returned instead of fanned out to the registered callbacks, which are
        typed on a VoiceSession.

        This is what a conference lane calls.  The alternative is to fabricate
        a VoiceSession per track, which keying the stages on a stream identity
        exists to make unnecessary.

        Args:
            stream: Identity of the audio stream — the key the stages hold
                their state under, and the one ``release_stream`` frees.
            frame: The frame to process.

        Returns:
            The processed frame and the VAD event it produced, if any.
        """
        return self._run_inbound(stream, None, frame)

    def release_stream(self, stream: str) -> None:
        """Release everything a stream held, in the engine and in the stages.

        A lane calls this when its track goes away.  Stage state is keyed by
        stream and some of it is native memory, so a stream that is never
        released leaks for as long as the pipeline lives.
        """
        self._cleanup_session_state(stream)

    def _run_inbound(
        self, stream: str, subject: VoiceSession | None, frame: AudioFrame
    ) -> InboundResult:
        """Run the inbound stages for one frame.

        ``subject`` is what the registered callbacks are invoked with.  It is
        ``None`` for a stream-keyed caller, which pulls the result instead, and
        the callback fanout is skipped rather than handed something that is not
        a session.
        """
        current_frame = frame

        # Remember which streams the stages hold state for, so the blanket
        # reset() can release every one of them.  Stage state is keyed by
        # stream now, and the engine is the only thing that knows the keys.
        self._stage_streams.add(stream)

        self._telemetry.count_frame(stream)

        # Stage 0: Inbound resampler (transport → internal format)
        if self._resampler is not None and self._config.contract is not None:
            int_fmt = self._config.contract.internal_format
            current_frame.metadata["original_sample_rate"] = current_frame.sample_rate
            current_frame.metadata["original_channels"] = current_frame.channels
            with self._timed_stage(stream, "resampler"):
                current_frame = self._resampler.resample(
                    current_frame,
                    int_fmt.sample_rate,
                    int_fmt.channels,
                    int_fmt.sample_width,
                    stream,
                )

        # AEC reference and capture MUST share the exact PCM format. Record the
        # post-resampler frame rather than the transport frame that entered the
        # method, and keep it per stream.
        if (
            self._config.aec is not None
            and VoiceCapability.NATIVE_AEC not in self._backend_capabilities
        ):
            capture_format = (
                current_frame.sample_rate,
                current_frame.channels,
                current_frame.sample_width,
            )
            with self._aec_capture_formats_lock:
                self._aec_capture_formats[stream] = capture_format

        # Stage 1: Recorder inbound tap
        handle = self._recording_handles.get(stream)
        if handle is not None and self._config.recorder is not None:
            from roomkit.voice.pipeline.recorder.base import RecordingMode

            rec_mode = (
                self._config.recording_config.mode
                if self._config.recording_config is not None
                else RecordingMode.BOTH
            )
            if rec_mode != RecordingMode.OUTBOUND_ONLY:
                try:
                    self._config.recorder.tap_inbound(handle, current_frame)
                except Exception:
                    logger.exception("Recorder inbound tap error")

        # Debug tap: raw (after resampler, before processing)
        self._debug_tap(stream, "raw", current_frame)

        # Stage 1.5: DTMF detection (before AEC/denoiser to preserve tones)
        if self._config.dtmf is not None:
            with self._timed_stage(stream, "dtmf"):
                dtmf_event = self._config.dtmf.process(current_frame, stream)
                if dtmf_event is not None:
                    current_frame.metadata["dtmf"] = {
                        "digit": dtmf_event.digit,
                        "duration_ms": dtmf_event.duration_ms,
                    }
                    self._fanout(self._dtmf_callbacks, subject, dtmf_event, "DTMF")

        # Stage 2: AEC (skip if backend has NATIVE_AEC)
        if (
            self._config.aec is not None
            and VoiceCapability.NATIVE_AEC not in self._backend_capabilities
        ):
            with self._timed_stage(stream, "aec"):
                current_frame = self._config.aec.process(current_frame, stream)
                current_frame.metadata["aec"] = self._config.aec.name

        # Debug tap: post_aec
        self._debug_tap(stream, "post_aec", current_frame)

        # Stage 3: AGC (skip if backend has NATIVE_AGC)
        if self._agc is not None and VoiceCapability.NATIVE_AGC not in self._backend_capabilities:
            with self._timed_stage(stream, "agc"):
                current_frame = self._agc.process(current_frame, stream)
                current_frame.metadata["agc"] = self._agc.name

        # Debug tap: post_agc
        self._debug_tap(stream, "post_agc", current_frame)

        # Stage 4: Denoiser
        if self._config.denoiser is not None:
            with self._timed_stage(stream, "denoiser"):
                current_frame = self._config.denoiser.process(current_frame, stream)
                current_frame.metadata["denoiser"] = self._config.denoiser.name

        # Debug tap: post_denoiser
        self._debug_tap(stream, "post_denoiser", current_frame)

        # Stage 5: VAD
        vad_event: VADEvent | None = None
        if self._config.vad is not None:
            with self._timed_stage(stream, "vad"):
                vad_event = self._config.vad.process(current_frame, stream)

        if vad_event is not None:
            current_frame.metadata["vad"] = {
                "type": vad_event.type,
                "confidence": vad_event.confidence,
            }

            # Track per-stream speech state for speech_frame callbacks
            if vad_event.type == VADEventType.SPEECH_START:
                self._in_speech_sessions.add(stream)
                self._telemetry.start_segment(stream)
            elif vad_event.type == VADEventType.SPEECH_END:
                self._in_speech_sessions.discard(stream)
                self._telemetry.end_segment(stream)

            # Fire VAD event callbacks
            self._fanout(self._vad_event_callbacks, subject, vad_event, "VAD event")

            # Fire speech_end callbacks with accumulated audio
            if vad_event.type == VADEventType.SPEECH_END and vad_event.audio_bytes is not None:
                # Debug tap: post_vad_speech (accumulated speech segment)
                dt = self._debug_tap_sessions.get(stream)
                if dt is not None:
                    dt.tap_vad_speech(
                        vad_event.audio_bytes,
                        sample_rate=current_frame.sample_rate,
                        channels=current_frame.channels,
                        sample_width=current_frame.sample_width,
                    )
                self._fanout(
                    self._speech_end_callbacks, subject, vad_event.audio_bytes, "Speech end"
                )

        # Fire speech_frame callbacks for processed frames during speech.
        # Includes the SPEECH_START frame, excludes the SPEECH_END frame.
        if stream in self._in_speech_sessions:
            self._fanout(self._speech_frame_callbacks, subject, current_frame, "Speech frame")

        # Bridge VAD state into flat metadata keys for diarization.
        # vad_is_speech is True for all frames during speech (including the
        # SPEECH_START frame).  vad_speech_end marks the boundary frame.
        if stream in self._in_speech_sessions:
            current_frame.metadata["vad_is_speech"] = True
        if vad_event is not None and vad_event.type == VADEventType.SPEECH_END:
            current_frame.metadata["vad_speech_end"] = True

        # Stage 6: Diarization
        if self._config.diarization is not None:
            with self._timed_stage(stream, "diarization"):
                diarization_result = self._config.diarization.process(current_frame, stream)
                if diarization_result is not None:
                    current_frame.metadata["diarization"] = {
                        "speaker_id": diarization_result.speaker_id,
                        "confidence": diarization_result.confidence,
                    }
                    if diarization_result.speaker_id != self._last_speaker_id.get(stream):
                        self._last_speaker_id[stream] = diarization_result.speaker_id
                        self._fanout(
                            self._speaker_change_callbacks,
                            subject,
                            diarization_result,
                            "Speaker change",
                        )

        # Fire processed_frame callbacks for every frame (regardless of speech).
        self._fanout(self._processed_frame_callbacks, subject, current_frame, "Processed frame")

        self._telemetry.record_frame(stream, len(frame.data))

        return InboundResult(frame=current_frame, vad_event=vad_event)

    # -----------------------------------------------------------------
    # Outbound processing
    # -----------------------------------------------------------------

    def process_outbound(self, session: VoiceSession, frame: AudioFrame) -> AudioFrame:
        """Process a single outbound audio frame through the pipeline.

        Thread-safe per session: concurrent calls for the same session
        (e.g. bridge forwarding + TTS) are serialized via a per-session
        lock to protect the recorder handle and other per-session state.

        Order: [PostProcessors] -> [Recorder tap] -> AEC.feed_reference ->
               [Resampler]
        """
        lock = self._outbound_locks.get(session.id)
        if lock is not None:
            lock.acquire()
        try:
            return self._process_outbound_unlocked(session, frame)
        finally:
            if lock is not None:
                lock.release()

    def _process_outbound_unlocked(self, session: VoiceSession, frame: AudioFrame) -> AudioFrame:
        """Internal outbound processing (caller holds per-session lock)."""
        current_frame = frame
        self._stage_streams.add(session.id)

        # Debug tap: outbound_raw (before postprocessors)
        self._debug_tap(session.id, "outbound_raw", current_frame)

        # Stage 1: PostProcessors
        for pp in self._config.postprocessors:
            try:
                current_frame = pp.process(current_frame, session.id)
            except Exception:
                logger.exception("PostProcessor '%s' error", pp.name)

        # Debug tap: outbound_final (after postprocessors)
        self._debug_tap(session.id, "outbound_final", current_frame)

        # Stage 2: Recorder outbound tap
        handle = self._recording_handles.get(session.id)
        if handle is not None and self._config.recorder is not None:
            from roomkit.voice.pipeline.recorder.base import RecordingMode

            rec_mode = (
                self._config.recording_config.mode
                if self._config.recording_config is not None
                else RecordingMode.BOTH
            )
            if rec_mode != RecordingMode.INBOUND_ONLY:
                try:
                    self._config.recorder.tap_outbound(handle, current_frame)
                except Exception:
                    logger.exception("Recorder outbound tap error")

        # Stage 3: Feed AEC reference (so it can model echo)
        # Skipped when the backend feeds reference at the transport level
        # (time-aligned with actual speaker output), when the backend
        # has NATIVE_AEC, or when playback-time feeding is wired via
        # feed_aec_reference().  The reference must match the inbound
        # sample rate — resample if the outbound frame is at a different rate.
        if (
            self._config.aec is not None
            and VoiceCapability.NATIVE_AEC not in self._backend_capabilities
            and not self._backend_feeds_aec_ref
            and not self._playback_aec_wired
        ):
            try:
                ref_frame = self._normalize_aec_reference(
                    current_frame,
                    session.id,
                    playback=False,
                )
                self._config.aec.feed_reference(ref_frame, session.id)
            except Exception:
                logger.exception("AEC feed_reference error")

        # Stage 4: Outbound resampler (internal → transport format)
        if self._resampler is not None and self._config.contract is not None:
            out_fmt = self._config.contract.transport_outbound_format
            try:
                current_frame = self._resampler.resample(
                    current_frame,
                    out_fmt.sample_rate,
                    out_fmt.channels,
                    out_fmt.sample_width,
                    session.id,
                )
            except Exception:
                logger.exception("Outbound resampler error")

        return current_frame

    # -----------------------------------------------------------------
    # External AEC reference (playback-time aligned)
    # -----------------------------------------------------------------

    def enable_playback_aec_feed(self) -> None:
        """Mark that AEC reference is fed at playback time.

        When called, ``process_outbound()`` skips its own
        ``aec.feed_reference()`` to avoid double-feeding with
        misaligned timing.
        """
        self._playback_aec_wired = True

    def feed_aec_reference(self, frame: AudioFrame, stream: str) -> None:
        """Feed an AEC reference frame directly (from speaker output).

        Called by the backend's speaker callback at playback time so
        the AEC has time-aligned reference for echo cancellation.

        Args:
            frame: The audio frame the speaker is playing.
            stream: Identity of the stream this playback belongs to — the
                session whose canceller should model this echo.

        Thread-safety: may be called from the audio I/O thread.  Uses
        a separate resampler instance from ``process_outbound`` to
        avoid thread-safety issues.
        """
        if (
            self._config.aec is None
            or VoiceCapability.NATIVE_AEC in self._backend_capabilities
            or self._backend_feeds_aec_ref
        ):
            return
        try:
            ref_frame = self._normalize_aec_reference(frame, stream, playback=True)
            self._config.aec.feed_reference(ref_frame, stream)
        except Exception:
            logger.exception("AEC feed_reference error (playback)")

    def _normalize_aec_reference(
        self,
        frame: AudioFrame,
        stream: str,
        *,
        playback: bool,
    ) -> AudioFrame:
        """Convert a reference to the exact PCM format AEC sees on capture."""
        with self._aec_capture_formats_lock:
            target = self._aec_capture_formats.get(stream)
        if target is None and self._config.contract is not None:
            internal = self._config.contract.internal_format
            target = (internal.sample_rate, internal.channels, internal.sample_width)
        if target is None:
            return frame

        target_rate, target_channels, target_width = target
        if (
            frame.sample_rate == target_rate
            and frame.channels == target_channels
            and frame.sample_width == target_width
        ):
            return frame

        lock = self._playback_aec_resampler_lock if playback else self._aec_resampler_lock
        with lock:
            if playback:
                if self._playback_aec_resampler is None:
                    self._playback_aec_resampler = _create_default_resampler()
                resampler = self._playback_aec_resampler
            else:
                if self._aec_resampler is None:
                    self._aec_resampler = _create_default_resampler()
                resampler = self._aec_resampler
            return resampler.resample(
                frame,
                target_rate,
                target_channels,
                target_width,
                stream,
            )

    def set_aec_active(
        self,
        stream: str,
        active: bool,
        *,
        source: str = "playback",
    ) -> None:
        """Track one playback source and update per-stream AEC activity.

        Multiple sources can play into the same session concurrently.  AEC is
        bypassed only after the final source stops.  Its converged adaptive
        filter is preserved for the next playback turn; session teardown owns
        the destructive reset.
        """
        aec = self._config.aec
        if aec is None or VoiceCapability.NATIVE_AEC in self._backend_capabilities:
            return
        with self._aec_active_sources_lock:
            was_globally_active = any(self._aec_active_sources.values())
            sources = self._aec_active_sources.setdefault(stream, set())
            previous_sources = set(sources)
            was_active = bool(sources)
            if active:
                sources.add(source)
            else:
                sources.discard(source)
            is_active = bool(sources)
            if not is_active:
                self._aec_active_sources.pop(stream, None)
            is_globally_active = any(self._aec_active_sources.values())

            # Repeated audio frames from one source must not repeatedly touch
            # provider state on the realtime audio thread.
            stream_state_changed = was_active != is_active
            supports_stream_activity = (
                type(aec).set_stream_active is not AECProvider.set_stream_active
            )
            provider_state_changed = (
                stream_state_changed
                if supports_stream_activity
                else was_globally_active != is_globally_active
            )
            if not provider_state_changed:
                return
            try:
                if supports_stream_activity:
                    aec.set_stream_active(stream, is_active)
                else:
                    aec.set_active(is_globally_active)
            except Exception:
                # Roll the logical edge back so a repeated lifecycle signal
                # retries a transient provider failure instead of looking like
                # a no-op while provider and engine disagree.
                if previous_sources:
                    self._aec_active_sources[stream] = previous_sources
                else:
                    self._aec_active_sources.pop(stream, None)
                logger.exception("AEC activation error for stream %s", stream)

    # -----------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------

    def _cleanup_session_state(self, session_id: str) -> None:
        """Release everything this session held, in the engine and in the stages.

        The stages keep their own per-stream state now, so leaving them alone
        would accumulate one speaker's worth of buffers for every session the
        room ever had.
        """
        self._release_stage_streams(session_id)
        self._in_speech_sessions.discard(session_id)
        self._telemetry.release(session_id)
        self._last_speaker_id.pop(session_id, None)
        self._outbound_locks.pop(session_id, None)
        handle = self._recording_handles.pop(session_id, None)
        if handle is not None and self._config.recorder is not None:
            try:
                self._config.recorder.stop(handle)
            except Exception:
                logger.exception("Failed to stop stale recording for %s", session_id)
        dt = self._debug_tap_sessions.pop(session_id, None)
        if dt is not None:
            try:
                dt.close()
            except Exception:
                logger.exception("Failed to close stale debug taps for %s", session_id)

    def on_session_active(self, session: VoiceSession) -> None:
        """Called when a voice session becomes active.

        Cleans up stale state for this session and starts recording if configured.
        """
        self._cleanup_session_state(session.id)
        self._outbound_locks[session.id] = threading.Lock()

        # Start debug taps if configured
        if self._config.debug_taps is not None and self._config.debug_taps.output_dir:
            from roomkit.voice.pipeline.debug_taps import DebugTapSession

            self._debug_tap_sessions[session.id] = DebugTapSession(
                self._config.debug_taps, session.id
            )

        # Start recording if configured
        if self._config.recorder is not None and self._config.recording_config is not None:
            try:
                handle = self._config.recorder.start(session, self._config.recording_config)
                self._recording_handles[session.id] = handle
                for cb in self._recording_started_callbacks:
                    try:
                        result = cb(session, handle)
                        _maybe_schedule(result)
                    except Exception:
                        logger.exception("Recording started callback error")
            except Exception:
                logger.exception("Failed to start recording for session %s", session.id)

    def _release_stage_streams(self, session_id: str) -> None:
        """Drop one stream's state in every stage.

        Stage state is keyed by stream and some of it is native — a
        SpeexEchoState, a DenoiseState, a WebRTC AudioProcessing. Leaving it
        behind when a speaker leaves does not fail a test, it leaks C memory
        for every speaker a long-running room ever had.

        The resamplers are released with the rest: a stateful one holds a
        delay line of the speaker's own audio, so leaving it behind both
        leaks and — if the stream key is ever reused, as a track id can be —
        opens the next stream with the previous one's samples.
        """
        self._release_aec_activity(session_id)

        stages = (
            self._resampler,
            self._aec_resampler,
            self._playback_aec_resampler,
            self._config.vad,
            self._config.denoiser,
            self._config.aec,
            self._agc,
            self._config.dtmf,
            self._config.diarization,
            *self._config.postprocessors,
        )
        for stage in stages:
            if stage is None:
                continue
            try:
                stage.reset(session_id)
            except Exception:
                # Best effort: one stage failing to release must not strand
                # the others, which is the leak this method exists to prevent.
                logger.exception("Stage '%s' reset error for session %s", stage.name, session_id)
        with self._aec_capture_formats_lock:
            self._aec_capture_formats.pop(session_id, None)
        self._stage_streams.discard(session_id)

    def _release_aec_activity(self, session_id: str) -> None:
        """Deactivate AEC when a stream disappears without an explicit stop.

        A transport can end while TTS or a bridge is still marked active.  In
        that path no matching ``set_aec_active(..., False)`` arrives, so the
        lifecycle cleanup must update the provider before dropping bookkeeping.
        This is especially important for legacy providers whose bypass state is
        global: the final departing stream must turn that provider off.
        """
        aec = self._config.aec
        with self._aec_active_sources_lock:
            was_globally_active = any(self._aec_active_sources.values())
            sources = self._aec_active_sources.pop(session_id, None)
            is_globally_active = any(self._aec_active_sources.values())

        if not sources or aec is None or VoiceCapability.NATIVE_AEC in self._backend_capabilities:
            return

        supports_stream_activity = type(aec).set_stream_active is not AECProvider.set_stream_active
        try:
            if supports_stream_activity:
                aec.set_stream_active(session_id, False)
            elif was_globally_active != is_globally_active:
                aec.set_active(is_globally_active)
        except Exception:
            logger.exception("AEC deactivation error for stream %s", session_id)

    def on_session_ended(self, session: VoiceSession) -> None:
        """Called when a voice session ends.

        Releases the stages' state for this stream, then stops recording and
        debug taps if active.
        """
        self._release_stage_streams(session.id)
        self._in_speech_sessions.discard(session.id)
        # Closes an active segment span too — the session may end mid-speech.
        self._telemetry.release(session.id)
        self._last_speaker_id.pop(session.id, None)
        self._outbound_locks.pop(session.id, None)

        # Close debug taps
        dt = self._debug_tap_sessions.pop(session.id, None)
        if dt is not None:
            try:
                dt.close()
            except Exception:
                logger.exception("Failed to close debug taps for %s", session.id)

        handle = self._recording_handles.pop(session.id, None)
        if handle is not None and self._config.recorder is not None:
            try:
                recording_result = self._config.recorder.stop(handle)
                for cb in self._recording_stopped_callbacks:
                    try:
                        result = cb(session, recording_result)
                        _maybe_schedule(result)
                    except Exception:
                        logger.exception("Recording stopped callback error")
            except Exception:
                logger.exception("Failed to stop recording for session %s", session.id)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def reset(self) -> None:
        """Reset all pipeline stage state."""
        self._in_speech_sessions.clear()
        self._telemetry.clear()
        self._outbound_locks.clear()
        with self._aec_capture_formats_lock:
            self._aec_capture_formats.clear()
        if self._aec_resampler is not None:
            self._aec_resampler.reset()
        if self._playback_aec_resampler is not None:
            self._playback_aec_resampler.reset()
        if self._resampler is not None:
            self._resampler.reset()
        # Stage state is keyed by stream, so a blanket reset releases every
        # stream the stages were given rather than one anonymous slot.
        with self._aec_active_sources_lock:
            active_streams = set(self._aec_active_sources)
        for stream in sorted(self._stage_streams | active_streams):
            self._release_stage_streams(stream)
        self._last_speaker_id.clear()
        # Stop active recordings before clearing handles
        for handle in self._recording_handles.values():
            if self._config.recorder is not None:
                try:
                    self._config.recorder.stop(handle)
                except Exception:
                    logger.exception("Failed to stop recording during reset")
        self._recording_handles.clear()
        # Close debug taps from any previous session
        for session_id, dt in self._debug_tap_sessions.items():
            try:
                dt.close()
            except Exception:
                logger.exception("Failed to close debug taps during reset for %s", session_id)
        self._debug_tap_sessions.clear()

    def close(self) -> None:
        """Release all pipeline resources.

        Every provider is closed whatever became of the ones before it — the
        providers are independent, and stopping at the first failure left
        every provider after it open for good. What failed is raised together,
        as an ``ExceptionGroup``, once everything has been asked to close.

        Raises:
            ExceptionGroup: if any provider's ``close()`` raised.
        """
        self._outbound_locks.clear()
        with self._aec_active_sources_lock:
            self._aec_active_sources.clear()
        # Stop active recordings before closing providers
        for handle in self._recording_handles.values():
            if self._config.recorder is not None:
                try:
                    self._config.recorder.stop(handle)
                except Exception:
                    logger.exception("Failed to stop recording during close")
        self._recording_handles.clear()
        failures: list[Exception] = []

        def _close(provider: Any) -> None:
            if provider is None:
                return
            try:
                provider.close()
            except Exception as exc:
                failures.append(exc)
                logger.exception(
                    "Failed to close %s during pipeline close", type(provider).__name__
                )

        # Close debug taps
        for dt in self._debug_tap_sessions.values():
            _close(dt)
        self._debug_tap_sessions.clear()
        _close(self._resampler)
        _close(self._aec_resampler)
        _close(self._playback_aec_resampler)
        _close(self._config.vad)
        _close(self._config.denoiser)
        _close(self._config.diarization)
        _close(self._config.aec)
        _close(self._agc)
        _close(self._config.dtmf)
        _close(self._config.recorder)
        for pp in self._config.postprocessors:
            _close(pp)
        _close(self._config.turn_detector)
        _close(self._config.backchannel_detector)
        if failures:
            raise ExceptionGroup(
                f"closing the audio pipeline failed for {len(failures)} provider(s)", failures
            )


def _create_default_resampler() -> ResamplerProvider:
    """Return the best available resampler: NumPy if installed, else pure Python."""
    try:
        from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider

        return NumpyResamplerProvider()
    except ImportError:
        from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

        logger.warning(
            "NumPy is not installed: the audio pipeline falls back to the "
            "pure-Python linear resampler, which is an order of magnitude "
            "slower per frame (~200 us vs ~15 us measured for 20 ms @ 16 kHz) "
            "and runs on every frame of every realtime session. Install numpy "
            "(any roomkit voice extra ships it) for production voice."
        )
        return LinearResamplerProvider()

"""Audio pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import VoiceCapability
from roomkit.voice.pipeline.vad.base import VADEventType

if TYPE_CHECKING:
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
            loop.create_task(result)
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
        self._backend_capabilities = backend_capabilities
        self._backend_feeds_aec_ref = backend_feeds_aec_reference
        self._speech_end_callbacks: list[SpeechEndPipelineCallback] = []
        self._speech_frame_callbacks: list[SpeechFramePipelineCallback] = []
        self._processed_frame_callbacks: list[ProcessedFrameCallback] = []
        self._vad_event_callbacks: list[VADEventCallback] = []
        self._in_speech_sessions: set[str] = set()
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []
        self._dtmf_callbacks: list[DTMFCallback] = []
        self._recording_started_callbacks: list[RecordingStartedCallback] = []
        self._recording_stopped_callbacks: list[RecordingStoppedCallback] = []
        self._last_speaker_id: str | None = None
        # Inbound sample rate — tracked from first frame, used to resample
        # AEC reference (outbound) to match inbound processing rate.
        self._inbound_sample_rate: int | None = None
        # Lazy resampler for AEC reference (created on first mismatch)
        self._aec_resampler: ResamplerProvider | None = None
        # Active recording handle (per session, keyed by session_id)
        self._recording_handles: dict[str, RecordingHandle] = {}
        # Debug tap sessions (per session, keyed by session_id)
        self._debug_tap_sessions: dict[str, DebugTapSession] = {}
        # Whether playback-time AEC reference is wired (suppresses
        # generation-time feeding in process_outbound).
        self._playback_aec_wired = False
        # Separate resampler for playback AEC path (may run on audio thread)
        self._playback_aec_resampler: ResamplerProvider | None = None
        # Telemetry counters (lightweight, emitted periodically)
        self._telemetry = config.telemetry
        self._frame_count: int = 0
        self._bytes_processed: int = 0
        self._metric_interval: int = 500  # emit metrics every N frames
        # Resolve effective resampler (auto-default when contract is set)
        self._resampler: ResamplerProvider | None
        if config.resampler is not None:
            self._resampler = config.resampler
        elif config.contract is not None:
            from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider

            self._resampler = LinearResamplerProvider()
        else:
            self._resampler = None

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

    def process_inbound(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Process a single inbound audio frame through the pipeline.

        Order: [Resampler] -> [Recorder tap] -> [DTMF] -> [AEC] -> [AGC] ->
               [Denoiser] -> [VAD] -> [Diarization]
        """
        current_frame = frame

        # Track inbound sample rate for AEC reference resampling
        if self._inbound_sample_rate is None:
            self._inbound_sample_rate = frame.sample_rate

        # Stage 0: Inbound resampler (transport → internal format)
        if self._resampler is not None and self._config.contract is not None:
            int_fmt = self._config.contract.internal_format
            current_frame.metadata["original_sample_rate"] = current_frame.sample_rate
            current_frame.metadata["original_channels"] = current_frame.channels
            try:
                current_frame = self._resampler.resample(
                    current_frame,
                    int_fmt.sample_rate,
                    int_fmt.channels,
                    int_fmt.sample_width,
                )
            except Exception:
                logger.exception("Inbound resampler error")

        # Stage 1: Recorder inbound tap
        handle = self._recording_handles.get(session.id)
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
        self._debug_tap(session.id, "raw", current_frame)

        # Stage 1.5: DTMF detection (before AEC/denoiser to preserve tones)
        if self._config.dtmf is not None:
            try:
                dtmf_event = self._config.dtmf.process(current_frame)
                if dtmf_event is not None:
                    current_frame.metadata["dtmf"] = {
                        "digit": dtmf_event.digit,
                        "duration_ms": dtmf_event.duration_ms,
                    }
                    for cb in self._dtmf_callbacks:
                        try:
                            result = cb(session, dtmf_event)
                            _maybe_schedule(result)
                        except Exception:
                            logger.exception("DTMF callback error")
            except Exception:
                logger.exception("DTMF detection error")

        # Stage 2: AEC (skip if backend has NATIVE_AEC)
        if (
            self._config.aec is not None
            and VoiceCapability.NATIVE_AEC not in self._backend_capabilities
        ):
            try:
                current_frame = self._config.aec.process(current_frame)
                current_frame.metadata["aec"] = self._config.aec.name
            except Exception:
                logger.exception("AEC error")

        # Debug tap: post_aec
        self._debug_tap(session.id, "post_aec", current_frame)

        # Stage 3: AGC (skip if backend has NATIVE_AGC)
        if (
            self._config.agc is not None
            and VoiceCapability.NATIVE_AGC not in self._backend_capabilities
        ):
            try:
                current_frame = self._config.agc.process(current_frame)
                current_frame.metadata["agc"] = self._config.agc.name
            except Exception:
                logger.exception("AGC error")

        # Debug tap: post_agc
        self._debug_tap(session.id, "post_agc", current_frame)

        # Stage 4: Denoiser
        if self._config.denoiser is not None:
            try:
                current_frame = self._config.denoiser.process(current_frame)
                current_frame.metadata["denoiser"] = self._config.denoiser.name
            except Exception:
                logger.exception("Denoiser error")

        # Debug tap: post_denoiser
        self._debug_tap(session.id, "post_denoiser", current_frame)

        # Stage 5: VAD
        vad_event: VADEvent | None = None
        if self._config.vad is not None:
            try:
                vad_event = self._config.vad.process(current_frame)
            except Exception:
                logger.exception("VAD error")

        if vad_event is not None:
            current_frame.metadata["vad"] = {
                "type": vad_event.type,
                "confidence": vad_event.confidence,
            }

            # Track per-session speech state for speech_frame callbacks
            if vad_event.type == VADEventType.SPEECH_START:
                self._in_speech_sessions.add(session.id)
            elif vad_event.type == VADEventType.SPEECH_END:
                self._in_speech_sessions.discard(session.id)

            # Fire VAD event callbacks
            for vad_cb in self._vad_event_callbacks:
                try:
                    result = vad_cb(session, vad_event)
                    _maybe_schedule(result)
                except Exception:
                    logger.exception("VAD event callback error")

            # Fire speech_end callbacks with accumulated audio
            if vad_event.type == VADEventType.SPEECH_END and vad_event.audio_bytes is not None:
                # Debug tap: post_vad_speech (accumulated speech segment)
                dt = self._debug_tap_sessions.get(session.id)
                if dt is not None:
                    dt.tap_vad_speech(
                        vad_event.audio_bytes,
                        sample_rate=current_frame.sample_rate,
                        channels=current_frame.channels,
                        sample_width=current_frame.sample_width,
                    )
                for se_cb in self._speech_end_callbacks:
                    try:
                        result = se_cb(session, vad_event.audio_bytes)
                        _maybe_schedule(result)
                    except Exception:
                        logger.exception("Speech end callback error")

        # Fire speech_frame callbacks for processed frames during speech.
        # Includes the SPEECH_START frame, excludes the SPEECH_END frame.
        if session.id in self._in_speech_sessions and self._speech_frame_callbacks:
            for sf_cb in self._speech_frame_callbacks:
                try:
                    result = sf_cb(session, current_frame)
                    _maybe_schedule(result)
                except Exception:
                    logger.exception("Speech frame callback error")

        # Bridge VAD state into flat metadata keys for diarization.
        # vad_is_speech is True for all frames during speech (including the
        # SPEECH_START frame).  vad_speech_end marks the boundary frame.
        if session.id in self._in_speech_sessions:
            current_frame.metadata["vad_is_speech"] = True
        if vad_event is not None and vad_event.type == VADEventType.SPEECH_END:
            current_frame.metadata["vad_speech_end"] = True

        # Stage 6: Diarization
        if self._config.diarization is not None:
            try:
                diarization_result = self._config.diarization.process(current_frame)
                if diarization_result is not None:
                    current_frame.metadata["diarization"] = {
                        "speaker_id": diarization_result.speaker_id,
                        "confidence": diarization_result.confidence,
                    }
                    if diarization_result.speaker_id != self._last_speaker_id:
                        self._last_speaker_id = diarization_result.speaker_id
                        for sc_cb in self._speaker_change_callbacks:
                            try:
                                result = sc_cb(session, diarization_result)
                                _maybe_schedule(result)
                            except Exception:
                                logger.exception("Speaker change callback error")
            except Exception:
                logger.exception("Diarization error")

        # Fire processed_frame callbacks for every frame (regardless of speech).
        if self._processed_frame_callbacks:
            for pf_cb in self._processed_frame_callbacks:
                try:
                    result = pf_cb(session, current_frame)
                    _maybe_schedule(result)
                except Exception:
                    logger.exception("Processed frame callback error")

        # Telemetry metrics (lightweight, periodic)
        self._frame_count += 1
        self._bytes_processed += len(frame.data)
        if self._telemetry is not None and self._frame_count % self._metric_interval == 0:
            self._telemetry.record_metric(
                "roomkit.pipeline.frame_count",
                float(self._frame_count),
                attributes={"session_id": session.id},
            )
            self._telemetry.record_metric(
                "roomkit.pipeline.bytes_processed",
                float(self._bytes_processed),
                unit="bytes",
                attributes={"session_id": session.id},
            )

    # -----------------------------------------------------------------
    # Outbound processing
    # -----------------------------------------------------------------

    def process_outbound(self, session: VoiceSession, frame: AudioFrame) -> AudioFrame:
        """Process a single outbound audio frame through the pipeline.

        Order: [PostProcessors] -> [Recorder tap] -> AEC.feed_reference ->
               [Resampler]
        """
        current_frame = frame

        # Debug tap: outbound_raw (before postprocessors)
        self._debug_tap(session.id, "outbound_raw", current_frame)

        # Stage 1: PostProcessors
        for pp in self._config.postprocessors:
            try:
                current_frame = pp.process(current_frame)
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
                ref_frame = current_frame
                target_rate = self._inbound_sample_rate
                if target_rate and ref_frame.sample_rate != target_rate:
                    if self._aec_resampler is None:
                        from roomkit.voice.pipeline.resampler.linear import (
                            LinearResamplerProvider,
                        )

                        self._aec_resampler = LinearResamplerProvider()
                    ref_frame = self._aec_resampler.resample(
                        ref_frame,
                        target_rate,
                        ref_frame.channels,
                        ref_frame.sample_width,
                    )
                self._config.aec.feed_reference(ref_frame)
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

    def feed_aec_reference(self, frame: AudioFrame) -> None:
        """Feed an AEC reference frame directly (from speaker output).

        Called by the backend's speaker callback at playback time so
        the AEC has time-aligned reference for echo cancellation.

        Thread-safety: may be called from the audio I/O thread.  Uses
        a separate resampler instance from ``process_outbound`` to
        avoid thread-safety issues.
        """
        if self._config.aec is None:
            return
        try:
            ref_frame = frame
            target_rate = self._inbound_sample_rate
            if target_rate and ref_frame.sample_rate != target_rate:
                if self._playback_aec_resampler is None:
                    from roomkit.voice.pipeline.resampler.linear import (
                        LinearResamplerProvider,
                    )

                    self._playback_aec_resampler = LinearResamplerProvider()
                ref_frame = self._playback_aec_resampler.resample(
                    ref_frame,
                    target_rate,
                    ref_frame.channels,
                    ref_frame.sample_width,
                )
            self._config.aec.feed_reference(ref_frame)
        except Exception:
            logger.exception("AEC feed_reference error (playback)")

    # -----------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------

    def on_session_active(self, session: VoiceSession) -> None:
        """Called when a voice session becomes active.

        Resets all pipeline stages and starts recording if configured.
        """
        self.reset()

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

    def on_session_ended(self, session: VoiceSession) -> None:
        """Called when a voice session ends.

        Stops recording and debug taps if active.
        """
        self._in_speech_sessions.discard(session.id)

        # Close debug taps
        dt = self._debug_tap_sessions.pop(session.id, None)
        if dt is not None:
            dt.close()

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
        self._inbound_sample_rate = None
        if self._aec_resampler is not None:
            self._aec_resampler.reset()
        if self._playback_aec_resampler is not None:
            self._playback_aec_resampler.reset()
        if self._resampler is not None:
            self._resampler.reset()
        if self._config.vad is not None:
            self._config.vad.reset()
        if self._config.diarization is not None:
            self._config.diarization.reset()
        if self._config.denoiser is not None:
            self._config.denoiser.reset()
        if self._config.aec is not None:
            self._config.aec.reset()
        if self._config.agc is not None:
            self._config.agc.reset()
        if self._config.dtmf is not None:
            self._config.dtmf.reset()
        for pp in self._config.postprocessors:
            pp.reset()
        self._last_speaker_id = None
        self._recording_handles.clear()
        # Close debug taps from any previous session
        for dt in self._debug_tap_sessions.values():
            dt.close()
        self._debug_tap_sessions.clear()

    def close(self) -> None:
        """Release all pipeline resources."""
        if self._resampler is not None:
            self._resampler.close()
        if self._aec_resampler is not None:
            self._aec_resampler.close()
        if self._playback_aec_resampler is not None:
            self._playback_aec_resampler.close()
        if self._config.vad is not None:
            self._config.vad.close()
        if self._config.denoiser is not None:
            self._config.denoiser.close()
        if self._config.diarization is not None:
            self._config.diarization.close()
        if self._config.aec is not None:
            self._config.aec.close()
        if self._config.agc is not None:
            self._config.agc.close()
        if self._config.dtmf is not None:
            self._config.dtmf.close()
        if self._config.recorder is not None:
            self._config.recorder.close()
        for pp in self._config.postprocessors:
            pp.close()
        if self._config.turn_detector is not None:
            self._config.turn_detector.close()
        if self._config.backchannel_detector is not None:
            self._config.backchannel_detector.close()

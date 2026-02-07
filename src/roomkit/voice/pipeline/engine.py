"""Audio pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import VoiceCapability
from roomkit.voice.pipeline.vad_provider import VADEventType

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization_provider import DiarizationResult
    from roomkit.voice.pipeline.dtmf_detector import DTMFEvent
    from roomkit.voice.pipeline.recorder import RecordingHandle, RecordingResult
    from roomkit.voice.pipeline.vad_provider import VADEvent

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
VADEventCallback = Callable[["VoiceSession", "VADEvent"], Any]
SpeakerChangeCallback = Callable[["VoiceSession", "DiarizationResult"], Any]
DTMFCallback = Callable[["VoiceSession", "DTMFEvent"], Any]
RecordingStartedCallback = Callable[["VoiceSession", "RecordingHandle"], Any]
RecordingStoppedCallback = Callable[["VoiceSession", "RecordingResult"], Any]


class AudioPipeline:
    """Orchestrates audio frame processing through pipeline stages.

    Inbound processing order:
        [Resampler] -> [Recorder tap] -> [AEC] -> [AGC] -> [Denoiser] ->
        [VAD] -> [Diarization] + [DTMF parallel]

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
    ) -> None:
        self._config = config
        self._backend_capabilities = backend_capabilities
        self._speech_end_callbacks: list[SpeechEndPipelineCallback] = []
        self._vad_event_callbacks: list[VADEventCallback] = []
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []
        self._dtmf_callbacks: list[DTMFCallback] = []
        self._recording_started_callbacks: list[RecordingStartedCallback] = []
        self._recording_stopped_callbacks: list[RecordingStoppedCallback] = []
        self._last_speaker_id: str | None = None
        # Active recording handle (per session, keyed by session_id)
        self._recording_handles: dict[str, RecordingHandle] = {}

    # -----------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------

    def on_speech_end(self, callback: SpeechEndPipelineCallback) -> None:
        """Register callback for when VAD detects speech end."""
        self._speech_end_callbacks.append(callback)

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

    def process_inbound(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Process a single inbound audio frame through the pipeline.

        Order: [Resampler] -> [Recorder tap] -> [AEC] -> [AGC] ->
               [Denoiser] -> [VAD] -> [Diarization] + [DTMF parallel]
        """
        current_frame = frame

        # Stage 1: Recorder inbound tap
        handle = self._recording_handles.get(session.id)
        if handle is not None and self._config.recorder is not None:
            try:
                self._config.recorder.tap_inbound(handle, current_frame)
            except Exception:
                logger.exception("Recorder inbound tap error")

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

        # Stage 4: Denoiser
        if self._config.denoiser is not None:
            try:
                current_frame = self._config.denoiser.process(current_frame)
                current_frame.metadata["denoiser"] = self._config.denoiser.name
            except Exception:
                logger.exception("Denoiser error")

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

            # Fire VAD event callbacks
            for cb in self._vad_event_callbacks:
                try:
                    result = cb(session, vad_event)
                    _maybe_schedule(result)
                except Exception:
                    logger.exception("VAD event callback error")

            # Fire speech_end callbacks with accumulated audio
            if vad_event.type == VADEventType.SPEECH_END and vad_event.audio_bytes is not None:
                for cb in self._speech_end_callbacks:
                    try:
                        result = cb(session, vad_event.audio_bytes)
                        _maybe_schedule(result)
                    except Exception:
                        logger.exception("Speech end callback error")

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
                        for cb in self._speaker_change_callbacks:
                            try:
                                result = cb(session, diarization_result)
                                _maybe_schedule(result)
                            except Exception:
                                logger.exception("Speaker change callback error")
            except Exception:
                logger.exception("Diarization error")

        # Stage 7: DTMF detection (parallel — independent of main chain)
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

    # -----------------------------------------------------------------
    # Outbound processing
    # -----------------------------------------------------------------

    def process_outbound(self, session: VoiceSession, frame: AudioFrame) -> AudioFrame:
        """Process a single outbound audio frame through the pipeline.

        Order: [PostProcessors] -> [Recorder tap] -> AEC.feed_reference ->
               [Resampler]
        """
        current_frame = frame

        # Stage 1: PostProcessors
        for pp in self._config.postprocessors:
            try:
                current_frame = pp.process(current_frame)
            except Exception:
                logger.exception("PostProcessor '%s' error", pp.name)

        # Stage 2: Recorder outbound tap
        handle = self._recording_handles.get(session.id)
        if handle is not None and self._config.recorder is not None:
            try:
                self._config.recorder.tap_outbound(handle, current_frame)
            except Exception:
                logger.exception("Recorder outbound tap error")

        # Stage 3: Feed AEC reference (so it can model echo)
        if (
            self._config.aec is not None
            and VoiceCapability.NATIVE_AEC not in self._backend_capabilities
        ):
            try:
                self._config.aec.feed_reference(current_frame)
            except Exception:
                logger.exception("AEC feed_reference error")

        return current_frame

    # -----------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------

    def on_session_active(self, session: VoiceSession) -> None:
        """Called when a voice session becomes active.

        Resets all pipeline stages and starts recording if configured.
        """
        self.reset()

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

        Stops recording if active.
        """
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

    def close(self) -> None:
        """Release all pipeline resources."""
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

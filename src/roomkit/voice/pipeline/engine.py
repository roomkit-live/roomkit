"""Audio pipeline engine — frame processing orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.vad_provider import VADEventType

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.pipeline.diarization_provider import DiarizationResult
    from roomkit.voice.pipeline.vad_provider import VADEvent

logger = logging.getLogger("roomkit.voice.pipeline")

# Callback type aliases
SpeechEndPipelineCallback = Callable[["VoiceSession", bytes], Any]
VADEventCallback = Callable[["VoiceSession", "VADEvent"], Any]
SpeakerChangeCallback = Callable[["VoiceSession", "DiarizationResult"], Any]


class AudioPipeline:
    """Orchestrates audio frame processing through pipeline stages.

    Processing order: denoiser -> VAD -> diarization.
    Fires registered callbacks based on pipeline events.
    """

    def __init__(self, config: AudioPipelineConfig) -> None:
        self._config = config
        self._speech_end_callbacks: list[SpeechEndPipelineCallback] = []
        self._vad_event_callbacks: list[VADEventCallback] = []
        self._speaker_change_callbacks: list[SpeakerChangeCallback] = []
        self._last_speaker_id: str | None = None

    def on_speech_end(self, callback: SpeechEndPipelineCallback) -> None:
        """Register callback for when VAD detects speech end.

        Args:
            callback: Function called with (session, audio_bytes).
        """
        self._speech_end_callbacks.append(callback)

    def on_vad_event(self, callback: VADEventCallback) -> None:
        """Register callback for all VAD events.

        Args:
            callback: Function called with (session, vad_event).
        """
        self._vad_event_callbacks.append(callback)

    def on_speaker_change(self, callback: SpeakerChangeCallback) -> None:
        """Register callback for speaker change detection.

        Args:
            callback: Function called with (session, diarization_result).
        """
        self._speaker_change_callbacks.append(callback)

    def process_frame(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Process a single audio frame through the pipeline.

        Order: denoiser -> VAD -> diarization.
        Fires callbacks as events are produced.

        Args:
            session: The voice session this frame belongs to.
            frame: The audio frame to process.
        """
        current_frame = frame

        # Stage 1: Denoiser (optional)
        if self._config.denoiser is not None:
            try:
                current_frame = self._config.denoiser.process(current_frame)
                current_frame.metadata["denoiser"] = self._config.denoiser.name
            except Exception:
                logger.exception("Denoiser error")

        # Stage 2: VAD (optional)
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
                    if hasattr(result, "__await__"):
                        # Caller is responsible for running in async context
                        pass
                except Exception:
                    logger.exception("VAD event callback error")

            # Fire speech_end callbacks with accumulated audio
            if vad_event.type == VADEventType.SPEECH_END and vad_event.audio_bytes is not None:
                for cb in self._speech_end_callbacks:
                    try:
                        result = cb(session, vad_event.audio_bytes)
                        if hasattr(result, "__await__"):
                            pass
                    except Exception:
                        logger.exception("Speech end callback error")

        # Stage 3: Diarization (optional)
        if self._config.diarization is not None:
            try:
                diarization_result = self._config.diarization.process(current_frame)
                if diarization_result is not None:
                    current_frame.metadata["diarization"] = {
                        "speaker_id": diarization_result.speaker_id,
                        "confidence": diarization_result.confidence,
                    }

                    # Fire speaker change callbacks if speaker changed
                    if diarization_result.speaker_id != self._last_speaker_id:
                        self._last_speaker_id = diarization_result.speaker_id
                        for cb in self._speaker_change_callbacks:
                            try:
                                result = cb(session, diarization_result)
                                if hasattr(result, "__await__"):
                                    pass
                            except Exception:
                                logger.exception("Speaker change callback error")
            except Exception:
                logger.exception("Diarization error")

    def reset(self) -> None:
        """Reset all pipeline stage state."""
        if self._config.vad is not None:
            self._config.vad.reset()
        if self._config.diarization is not None:
            self._config.diarization.reset()
        self._last_speaker_id = None

    def close(self) -> None:
        """Release all pipeline resources."""
        if self._config.vad is not None:
            self._config.vad.close()
        if self._config.denoiser is not None:
            self._config.denoiser.close()
        if self._config.diarization is not None:
            self._config.diarization.close()
        for pp in self._config.postprocessors:
            pp.close()

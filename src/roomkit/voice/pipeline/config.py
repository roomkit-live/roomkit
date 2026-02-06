"""Audio pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
    from roomkit.voice.pipeline.diarization_provider import DiarizationProvider
    from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
    from roomkit.voice.pipeline.vad_provider import VADConfig, VADProvider


@dataclass
class AudioPipelineConfig:
    """Configuration for the audio processing pipeline.

    All stages are optional.  At least one provider should be set for the
    pipeline to be useful.

    Typical combinations:

    - **VoiceChannel** (STT path): ``vad`` (+ optional denoiser/diarization)
    - **RealtimeVoiceChannel** (speech-to-speech): ``denoiser`` and/or
      ``diarization`` (VAD not needed — the AI provider handles turn detection)
    """

    vad: VADProvider | None = None
    """Optional Voice Activity Detection provider."""

    denoiser: DenoiserProvider | None = None
    """Optional denoiser applied before VAD."""

    diarization: DiarizationProvider | None = None
    """Optional speaker diarization applied after VAD."""

    postprocessors: list[AudioPostProcessor] = field(default_factory=list)
    """Optional postprocessors (interface only, deferred)."""

    vad_config: VADConfig | None = None
    """Optional VAD-specific configuration override."""

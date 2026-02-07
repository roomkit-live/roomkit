"""Audio pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.interruption import InterruptionConfig
    from roomkit.voice.pipeline.aec_provider import AECProvider
    from roomkit.voice.pipeline.agc_provider import AGCConfig, AGCProvider
    from roomkit.voice.pipeline.backchannel_detector import BackchannelDetector
    from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
    from roomkit.voice.pipeline.diarization_provider import DiarizationProvider
    from roomkit.voice.pipeline.dtmf_detector import DTMFDetector
    from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
    from roomkit.voice.pipeline.recorder import AudioRecorder, RecordingConfig
    from roomkit.voice.pipeline.turn_detector import TurnDetector
    from roomkit.voice.pipeline.vad_provider import VADConfig, VADProvider


@dataclass
class AudioFormat:
    """Describes an audio format."""

    sample_rate: int = 16000
    """Sample rate in Hz."""

    channels: int = 1
    """Number of audio channels."""

    sample_width: int = 2
    """Bytes per sample (2 = 16-bit PCM)."""


@dataclass
class AudioPipelineContract:
    """Declares expected input/output formats for the pipeline."""

    input_format: AudioFormat = field(default_factory=AudioFormat)
    """Expected format of audio entering the pipeline."""

    output_format: AudioFormat = field(default_factory=AudioFormat)
    """Format of audio leaving the pipeline."""


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
    """Optional postprocessors applied on the outbound path."""

    vad_config: VADConfig | None = None
    """Optional VAD-specific configuration override."""

    # --- Additional stages (all optional, backwards compatible) ---

    aec: AECProvider | None = None
    """Optional Acoustic Echo Cancellation provider."""

    agc: AGCProvider | None = None
    """Optional Automatic Gain Control provider."""

    agc_config: AGCConfig | None = None
    """Optional AGC-specific configuration override."""

    dtmf: DTMFDetector | None = None
    """Optional DTMF tone detector (runs in parallel with main chain)."""

    turn_detector: TurnDetector | None = None
    """Optional post-STT turn completion detector."""

    backchannel_detector: BackchannelDetector | None = None
    """Optional backchannel detector for semantic interruption strategy."""

    recorder: AudioRecorder | None = None
    """Optional audio recorder."""

    recording_config: RecordingConfig | None = None
    """Optional recording configuration."""

    interruption: InterruptionConfig | None = None
    """Optional interruption (barge-in) configuration."""

    contract: AudioPipelineContract | None = None
    """Optional input/output format contract."""

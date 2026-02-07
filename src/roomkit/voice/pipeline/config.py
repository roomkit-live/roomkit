"""Audio pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.interruption import InterruptionConfig
    from roomkit.voice.pipeline.aec.base import AECProvider
    from roomkit.voice.pipeline.agc.base import AGCConfig, AGCProvider
    from roomkit.voice.pipeline.backchannel.base import BackchannelDetector
    from roomkit.voice.pipeline.debug_taps import PipelineDebugTaps
    from roomkit.voice.pipeline.denoiser.base import DenoiserProvider
    from roomkit.voice.pipeline.diarization.base import DiarizationProvider
    from roomkit.voice.pipeline.dtmf.base import DTMFDetector
    from roomkit.voice.pipeline.postprocessor.base import AudioPostProcessor
    from roomkit.voice.pipeline.recorder.base import AudioRecorder, RecordingConfig
    from roomkit.voice.pipeline.resampler.base import ResamplerProvider
    from roomkit.voice.pipeline.turn.base import TurnDetector
    from roomkit.voice.pipeline.vad.base import VADConfig, VADProvider


@dataclass
class AudioFormat:
    """Describes an audio format."""

    sample_rate: int = 16000
    """Sample rate in Hz."""

    channels: int = 1
    """Number of audio channels."""

    sample_width: int = 2
    """Bytes per sample (2 = 16-bit PCM)."""

    codec: str = "pcm_s16le"
    """Codec identifier (e.g. ``pcm_s16le``, ``opus``)."""


@dataclass
class AudioPipelineContract:
    """Declares expected input/output/internal formats for the pipeline.

    The 3-format model separates:
    - **transport_inbound_format**: format of audio arriving from the backend
    - **transport_outbound_format**: format of audio sent back to the backend
    - **internal_format**: format used inside the pipeline (after inbound
      resampling, before outbound resampling)
    """

    transport_inbound_format: AudioFormat = field(default_factory=AudioFormat)
    """Format of audio arriving from the transport/backend."""

    transport_outbound_format: AudioFormat = field(default_factory=AudioFormat)
    """Format of audio sent back to the transport/backend."""

    internal_format: AudioFormat = field(default_factory=AudioFormat)
    """Format used internally by pipeline stages."""

    @property
    def input_format(self) -> AudioFormat:
        """Backwards-compatible alias for ``transport_inbound_format``."""
        return self.transport_inbound_format

    @property
    def output_format(self) -> AudioFormat:
        """Backwards-compatible alias for ``transport_outbound_format``."""
        return self.transport_outbound_format


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

    resampler: ResamplerProvider | None = None
    """Optional resampler provider for format conversion."""

    contract: AudioPipelineContract | None = None
    """Optional input/output format contract."""

    debug_taps: PipelineDebugTaps | None = None
    """Optional diagnostic audio capture at pipeline stage boundaries."""

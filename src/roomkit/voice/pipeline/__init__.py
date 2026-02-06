"""Audio processing pipeline for voice (RFC §12.3)."""

from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
from roomkit.voice.pipeline.diarization_provider import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.mock import (
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockVADProvider,
)
from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
from roomkit.voice.pipeline.vad_provider import VADConfig, VADEvent, VADEventType, VADProvider

__all__ = [
    # Config
    "AudioPipelineConfig",
    # Engine
    "AudioPipeline",
    # Provider ABCs
    "DenoiserProvider",
    "DiarizationProvider",
    "VADProvider",
    "AudioPostProcessor",
    # Data types
    "DiarizationResult",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    # Mocks
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockVADProvider",
]

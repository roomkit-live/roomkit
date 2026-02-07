"""Audio processing pipeline for voice (RFC §12.3)."""

from roomkit.voice.pipeline.aec_provider import AECProvider
from roomkit.voice.pipeline.speex_aec import SpeexAECProvider
from roomkit.voice.pipeline.agc_provider import AGCConfig, AGCProvider
from roomkit.voice.pipeline.backchannel_detector import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
)
from roomkit.voice.pipeline.config import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    ResamplerConfig,
)
from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
from roomkit.voice.pipeline.diarization_provider import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.dtmf_detector import DTMFDetector, DTMFEvent
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.mock import (
    MockAECProvider,
    MockAGCProvider,
    MockAudioRecorder,
    MockBackchannelDetector,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockDTMFDetector,
    MockTurnDetector,
    MockVADProvider,
)
from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
from roomkit.voice.pipeline.recorder import (
    AudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
)
from roomkit.voice.pipeline.turn_detector import (
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
)
from roomkit.voice.pipeline.vad_provider import VADConfig, VADEvent, VADEventType, VADProvider

__all__ = [
    # Config
    "AudioFormat",
    "AudioPipelineConfig",
    "AudioPipelineContract",
    "ResamplerConfig",
    # Engine
    "AudioPipeline",
    # Provider ABCs
    "AECProvider",
    "SpeexAECProvider",
    "AGCProvider",
    "AudioPostProcessor",
    "AudioRecorder",
    "BackchannelDetector",
    "DenoiserProvider",
    "DiarizationProvider",
    "DTMFDetector",
    "TurnDetector",
    "VADProvider",
    # Data types
    "AGCConfig",
    "BackchannelContext",
    "BackchannelDecision",
    "DiarizationResult",
    "DTMFEvent",
    "RecordingChannelMode",
    "RecordingConfig",
    "RecordingHandle",
    "RecordingMode",
    "RecordingResult",
    "RecordingTrigger",
    "TurnContext",
    "TurnDecision",
    "TurnEntry",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    # Mocks
    "MockAECProvider",
    "MockAGCProvider",
    "MockAudioRecorder",
    "MockBackchannelDetector",
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockDTMFDetector",
    "MockTurnDetector",
    "MockVADProvider",
]

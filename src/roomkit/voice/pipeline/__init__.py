"""Audio processing pipeline for voice (RFC §12.3)."""

from roomkit.voice.pipeline.aec import AECProvider, MockAECProvider, SpeexAECProvider
from roomkit.voice.pipeline.agc import AGCConfig, AGCProvider, MockAGCProvider
from roomkit.voice.pipeline.backchannel import (
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
    MockBackchannelDetector,
)
from roomkit.voice.pipeline.config import (
    AudioFormat,
    AudioPipelineConfig,
    AudioPipelineContract,
    ResamplerConfig,
)
from roomkit.voice.pipeline.denoiser import (
    DenoiserProvider,
    MockDenoiserProvider,
    RNNoiseDenoiserProvider,
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)
from roomkit.voice.pipeline.diarization import (
    DiarizationProvider,
    DiarizationResult,
    MockDiarizationProvider,
)
from roomkit.voice.pipeline.dtmf import DTMFDetector, DTMFEvent, MockDTMFDetector
from roomkit.voice.pipeline.engine import AudioPipeline
from roomkit.voice.pipeline.postprocessor import AudioPostProcessor
from roomkit.voice.pipeline.recorder import (
    AudioRecorder,
    MockAudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
    WavFileRecorder,
)
from roomkit.voice.pipeline.turn import (
    MockTurnDetector,
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
)
from roomkit.voice.pipeline.vad import (
    EnergyVADProvider,
    MockVADProvider,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
)

__all__ = [
    # Config
    "AudioFormat",
    "AudioPipelineConfig",
    "AudioPipelineContract",
    "ResamplerConfig",
    # Engine
    "AudioPipeline",
    # Provider ABCs + implementations
    "AECProvider",
    "SpeexAECProvider",
    "EnergyVADProvider",
    "SherpaOnnxVADConfig",
    "SherpaOnnxVADProvider",
    "AGCProvider",
    "AudioPostProcessor",
    "AudioRecorder",
    "WavFileRecorder",
    "BackchannelDetector",
    "DenoiserProvider",
    "RNNoiseDenoiserProvider",
    "SherpaOnnxDenoiserConfig",
    "SherpaOnnxDenoiserProvider",
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

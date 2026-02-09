"""Voice support for RoomKit (STT, TTS, streaming audio, audio pipeline)."""

from __future__ import annotations

from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import AudioReceivedCallback, VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    TranscriptionResult,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)
from roomkit.voice.events import (
    BackchannelEvent,
    BargeInEvent,
    DTMFDetectedEvent,
    PartialTranscriptionEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    SpeakerChangeEvent,
    TTSCancelledEvent,
    TurnCompleteEvent,
    TurnIncompleteEvent,
    VADAudioLevelEvent,
    VADSilenceEvent,
)
from roomkit.voice.interruption import (
    InterruptionConfig,
    InterruptionDecision,
    InterruptionHandler,
    InterruptionStrategy,
)
from roomkit.voice.pipeline import (
    AECProvider,
    AGCConfig,
    AGCProvider,
    AudioFormat,
    AudioPipeline,
    AudioPipelineConfig,
    AudioPipelineContract,
    AudioPostProcessor,
    AudioRecorder,
    BackchannelContext,
    BackchannelDecision,
    BackchannelDetector,
    DenoiserProvider,
    DiarizationProvider,
    DiarizationResult,
    DTMFDetector,
    DTMFEvent,
    EnergyVADProvider,
    LinearResamplerProvider,
    MockAECProvider,
    MockAGCProvider,
    MockAudioRecorder,
    MockBackchannelDetector,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockDTMFDetector,
    MockResamplerProvider,
    MockTurnDetector,
    MockVADProvider,
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
    ResamplerProvider,
    RNNoiseDenoiserProvider,
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
    SherpaOnnxVADConfig,
    SherpaOnnxVADProvider,
    SpeexAECProvider,
    TurnContext,
    TurnDecision,
    TurnDetector,
    TurnEntry,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
    WavFileRecorder,
)
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.tts.base import TTSProvider

__all__ = [
    # Base types
    "AudioChunk",
    "AudioFrame",
    "TranscriptionResult",
    "VoiceBackend",
    "VoiceCapability",
    "VoiceSession",
    "VoiceSessionState",
    # Callback types
    "AudioReceivedCallback",
    "BargeInCallback",
    # Event types
    "BackchannelEvent",
    "BargeInEvent",
    "DTMFDetectedEvent",
    "PartialTranscriptionEvent",
    "RecordingStartedEvent",
    "RecordingStoppedEvent",
    "SpeakerChangeEvent",
    "TTSCancelledEvent",
    "TurnCompleteEvent",
    "TurnIncompleteEvent",
    "VADAudioLevelEvent",
    "VADSilenceEvent",
    # Pipeline config
    "AudioFormat",
    "AudioPipeline",
    "AudioPipelineConfig",
    "AudioPipelineContract",
    "ResamplerProvider",
    "LinearResamplerProvider",
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
    # Interruption
    "InterruptionConfig",
    "InterruptionDecision",
    "InterruptionHandler",
    "InterruptionStrategy",
    # Pipeline mocks
    "MockAECProvider",
    "MockAGCProvider",
    "MockAudioRecorder",
    "MockBackchannelDetector",
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockDTMFDetector",
    "MockResamplerProvider",
    "MockTurnDetector",
    "MockVADProvider",
    # Providers
    "STTProvider",
    "TTSProvider",
]

# Optional providers (lazy imports to avoid requiring dependencies)


def get_deepgram_provider() -> type:
    """Get DeepgramSTTProvider class (requires httpx, websockets)."""
    from roomkit.voice.stt.deepgram import DeepgramSTTProvider

    return DeepgramSTTProvider


def get_deepgram_config() -> type:
    """Get DeepgramConfig class."""
    from roomkit.voice.stt.deepgram import DeepgramConfig

    return DeepgramConfig


def get_elevenlabs_provider() -> type:
    """Get ElevenLabsTTSProvider class (requires httpx, websockets)."""
    from roomkit.voice.tts.elevenlabs import ElevenLabsTTSProvider

    return ElevenLabsTTSProvider


def get_elevenlabs_config() -> type:
    """Get ElevenLabsConfig class."""
    from roomkit.voice.tts.elevenlabs import ElevenLabsConfig

    return ElevenLabsConfig


def get_local_audio_backend() -> type:
    """Get LocalAudioBackend class (requires sounddevice, numpy)."""
    from roomkit.voice.backends.local import LocalAudioBackend

    return LocalAudioBackend


def get_rtp_backend() -> type:
    """Get RTPVoiceBackend class (requires aiortp)."""
    from roomkit.voice.backends.rtp import RTPVoiceBackend

    return RTPVoiceBackend


def get_fastrtc_backend() -> type:
    """Get FastRTCVoiceBackend class (requires fastrtc, numpy)."""
    from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend

    return FastRTCVoiceBackend


def get_mount_fastrtc_voice() -> Any:
    """Get mount_fastrtc_voice function (requires fastrtc, numpy)."""
    from roomkit.voice.backends.fastrtc import mount_fastrtc_voice

    return mount_fastrtc_voice


def get_sherpa_onnx_stt_provider() -> type:
    """Get SherpaOnnxSTTProvider class (requires sherpa-onnx)."""
    from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTProvider

    return SherpaOnnxSTTProvider


def get_sherpa_onnx_stt_config() -> type:
    """Get SherpaOnnxSTTConfig class."""
    from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig

    return SherpaOnnxSTTConfig


def get_sherpa_onnx_tts_provider() -> type:
    """Get SherpaOnnxTTSProvider class (requires sherpa-onnx)."""
    from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSProvider

    return SherpaOnnxTTSProvider


def get_sherpa_onnx_tts_config() -> type:
    """Get SherpaOnnxTTSConfig class."""
    from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig

    return SherpaOnnxTTSConfig


def get_qwen3_tts_provider() -> type:
    """Get Qwen3TTSProvider class (requires qwen-tts)."""
    from roomkit.voice.tts.qwen3 import Qwen3TTSProvider

    return Qwen3TTSProvider


def get_qwen3_tts_config() -> type:
    """Get Qwen3TTSConfig class."""
    from roomkit.voice.tts.qwen3 import Qwen3TTSConfig

    return Qwen3TTSConfig


def get_qwen3_voice_clone_config() -> type:
    """Get VoiceCloneConfig class for Qwen3-TTS."""
    from roomkit.voice.tts.qwen3 import VoiceCloneConfig

    return VoiceCloneConfig


def get_openai_realtime_provider() -> type:
    """Get OpenAIRealtimeProvider class (requires openai, websockets)."""
    from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

    return OpenAIRealtimeProvider


def get_gemini_live_provider() -> type:
    """Get GeminiLiveProvider class (requires google-genai)."""
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider


def get_websocket_realtime_transport() -> type:
    """Get WebSocketRealtimeTransport class (requires websockets)."""
    from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport

    return WebSocketRealtimeTransport


def get_local_audio_transport() -> type:
    """Get LocalAudioTransport class (requires sounddevice, numpy)."""
    from roomkit.voice.realtime.local_transport import LocalAudioTransport

    return LocalAudioTransport


def get_fastrtc_realtime_transport() -> type:
    """Get FastRTCRealtimeTransport class (requires fastrtc, numpy)."""
    from roomkit.voice.realtime.fastrtc_transport import FastRTCRealtimeTransport

    return FastRTCRealtimeTransport


def get_rnnoise_denoiser_provider() -> type:
    """Get RNNoiseDenoiserProvider class (requires librnnoise system library)."""
    from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider

    return RNNoiseDenoiserProvider


def get_sherpa_onnx_denoiser_provider() -> type:
    """Get SherpaOnnxDenoiserProvider class (requires sherpa-onnx)."""
    from roomkit.voice.pipeline.denoiser.sherpa_onnx import SherpaOnnxDenoiserProvider

    return SherpaOnnxDenoiserProvider


def get_sherpa_onnx_denoiser_config() -> type:
    """Get SherpaOnnxDenoiserConfig class."""
    from roomkit.voice.pipeline.denoiser.sherpa_onnx import SherpaOnnxDenoiserConfig

    return SherpaOnnxDenoiserConfig


def get_sherpa_onnx_vad_provider() -> type:
    """Get SherpaOnnxVADProvider class (requires sherpa-onnx)."""
    from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADProvider

    return SherpaOnnxVADProvider


def get_sherpa_onnx_vad_config() -> type:
    """Get SherpaOnnxVADConfig class."""
    from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig

    return SherpaOnnxVADConfig


def get_speex_aec_provider() -> type:
    """Get SpeexAECProvider class (requires libspeexdsp system library)."""
    from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

    return SpeexAECProvider


def get_mount_fastrtc_realtime() -> Any:
    """Get mount_fastrtc_realtime function (requires fastrtc, numpy)."""
    from roomkit.voice.realtime.fastrtc_transport import mount_fastrtc_realtime

    return mount_fastrtc_realtime

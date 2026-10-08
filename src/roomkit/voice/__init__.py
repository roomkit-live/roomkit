"""Voice support for RoomKit (STT, TTS, streaming audio, audio pipeline)."""

from __future__ import annotations

from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import (
    AudioReceivedCallback,
    PlaybackErrors,
    SessionReadyCallback,
    VoiceBackend,
)
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    SpeakerSegment,
    TranscriptionResult,
    VoiceCapability,
    VoiceSession,
    VoiceSessionState,
)
from roomkit.voice.bridge import (
    AudioBridge,
    AudioBridgeConfig,
    BridgeFrameFilter,
    BridgeFrameProcessor,
)
from roomkit.voice.capture import (
    AudioCaptureSource,
    CaptureFrameCallback,
    CaptureMark,
    CaptureSubscription,
    LocalMicSource,
    MockCaptureSource,
)
from roomkit.voice.events import (
    BackchannelEvent,
    BargeInEvent,
    BridgeAudioEvent,
    DTMFDetectedEvent,
    PartialTranscriptionEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    SpeakerChangeEvent,
    TranscriptionEvent,
    TTSCancelledEvent,
    TurnCompleteEvent,
    TurnIncompleteEvent,
    VADAudioLevelEvent,
    VADSilenceEvent,
)
from roomkit.voice.inbound import parse_voice_session
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
    MixerProvider,
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
    PhraseBackchannelDetector,
    PythonMixerProvider,
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
    SimpleAGCProvider,
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
    WebRTCNoiseSuppressorProvider,
)
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.language import STTLanguageLock
from roomkit.voice.testing import ScenarioVoiceBackend, VoiceTrace
from roomkit.voice.tts.base import TTSProvider
from roomkit.voice.tts.context import (
    ConversationTurn,
    TTSContext,
    TTSContextConfig,
    TTSContextLevel,
)
from roomkit.voice.tts.filters import (
    StripBrackets,
    StripEmoji,
    StripInternalTags,
    TTSFilterChain,
    TTSStreamFilter,
)
from roomkit.voice.tts.library import CustomVoice, VoiceConsentError, VoiceLibrary
from roomkit.voice.voices import DialogueTurn, VoiceInfo, filter_voices

__all__ = [
    # Bridge
    "AudioBridge",
    "AudioBridgeConfig",
    "BridgeFrameFilter",
    "BridgeFrameProcessor",
    # Capture sources
    "AudioCaptureSource",
    "CaptureFrameCallback",
    "CaptureMark",
    "CaptureSubscription",
    "LocalMicSource",
    "MockCaptureSource",
    # Base types
    "AudioChunk",
    "AudioFrame",
    "SpeakerSegment",
    "TranscriptionResult",
    "VoiceBackend",
    "PlaybackErrors",
    "VoiceCapability",
    "VoiceSession",
    "VoiceSessionState",
    # Callback types
    "AudioReceivedCallback",
    "BargeInCallback",
    "SessionReadyCallback",
    # Event types
    "BackchannelEvent",
    "BargeInEvent",
    "BridgeAudioEvent",
    "DTMFDetectedEvent",
    "PartialTranscriptionEvent",
    "TranscriptionEvent",
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
    "SimpleAGCProvider",
    "AudioPostProcessor",
    "AudioRecorder",
    "WavFileRecorder",
    "BackchannelDetector",
    "DenoiserProvider",
    "RNNoiseDenoiserProvider",
    "SherpaOnnxDenoiserConfig",
    "SherpaOnnxDenoiserProvider",
    "WebRTCNoiseSuppressorProvider",
    "DiarizationProvider",
    "DTMFDetector",
    "MixerProvider",
    "NumpyMixerProvider",
    "PythonMixerProvider",
    "TurnDetector",
    "VADProvider",
    # Test bench
    "ScenarioVoiceBackend",
    "VoiceTrace",
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
    "PhraseBackchannelDetector",
    "MockDenoiserProvider",
    "MockDiarizationProvider",
    "MockDTMFDetector",
    "MockResamplerProvider",
    "MockTurnDetector",
    "MockVADProvider",
    # Inbound helpers
    "parse_voice_session",
    # Providers
    "STTLanguageLock",
    "STTProvider",
    "TTSProvider",
    # Custom voices (RFC §12.2.4)
    "CustomVoice",
    "VoiceConsentError",
    "VoiceLibrary",
    # Voices (RFC §12.2)
    "DialogueTurn",
    "VoiceInfo",
    "filter_voices",
    # TTS conversation context
    "ConversationTurn",
    "TTSContext",
    "TTSContextConfig",
    "TTSContextLevel",
    # TTS filters
    "StripBrackets",
    "StripEmoji",
    "TTSFilterChain",
    "StripInternalTags",
    "TTSStreamFilter",
]

# Lazy imports for optional dependencies


def __getattr__(name: str) -> object:
    if name == "NumpyMixerProvider":
        from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider

        return NumpyMixerProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def get_gemini_transcribe_provider() -> type:
    """Get GeminiTranscribeProvider class (requires google-genai)."""
    from roomkit.voice.stt.gemini_transcribe import GeminiTranscribeProvider

    return GeminiTranscribeProvider


def get_gemini_transcribe_config() -> type:
    """Get GeminiTranscribeConfig class."""
    from roomkit.voice.stt.gemini_transcribe import GeminiTranscribeConfig

    return GeminiTranscribeConfig


def get_gradium_stt_provider() -> type:
    """Get GradiumSTTProvider class (requires gradium)."""
    from roomkit.voice.stt.gradium import GradiumSTTProvider

    return GradiumSTTProvider


def get_gradium_stt_config() -> type:
    """Get GradiumSTTConfig class."""
    from roomkit.voice.stt.gradium import GradiumSTTConfig

    return GradiumSTTConfig


def get_meta_stt_provider() -> type:
    """Get MetaSTTProvider class (requires websockets, httpx)."""
    from roomkit.voice.stt.meta import MetaSTTProvider

    return MetaSTTProvider


def get_meta_stt_config() -> type:
    """Get MetaSTTConfig class."""
    from roomkit.voice.stt.meta import MetaSTTConfig

    return MetaSTTConfig


def get_azure_mai_stt_provider() -> type:
    """Get AzureMAISTTProvider class for MAI-Transcribe (requires websockets)."""
    from roomkit.voice.stt.azure_mai import AzureMAISTTProvider

    return AzureMAISTTProvider


def get_azure_mai_stt_config() -> type:
    """Get AzureMAISTTConfig class."""
    from roomkit.voice.stt.azure_mai import AzureMAISTTConfig

    return AzureMAISTTConfig


def get_gradium_tts_provider() -> type:
    """Get GradiumTTSProvider class (requires gradium)."""
    from roomkit.voice.tts.gradium import GradiumTTSProvider

    return GradiumTTSProvider


def get_gradium_tts_config() -> type:
    """Get GradiumTTSConfig class."""
    from roomkit.voice.tts.gradium import GradiumTTSConfig

    return GradiumTTSConfig


def get_local_audio_backend() -> type:
    """Get LocalAudioBackend class (requires sounddevice, numpy)."""
    from roomkit.voice.backends.local import LocalAudioBackend

    return LocalAudioBackend


def get_rtp_backend() -> type:
    """Get RTPVoiceBackend class (requires aiortp)."""
    from roomkit.voice.backends.rtp import RTPVoiceBackend

    return RTPVoiceBackend


def get_sip_backend() -> type:
    """Get SIPVoiceBackend class (requires aiosipua[rtp])."""
    from roomkit.voice.backends.sip import SIPVoiceBackend

    return SIPVoiceBackend


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


def get_qwen3_asr_provider() -> type:
    """Get Qwen3ASRProvider class (requires qwen-asr)."""
    from roomkit.voice.stt.qwen3 import Qwen3ASRProvider

    return Qwen3ASRProvider


def get_qwen3_asr_config() -> type:
    """Get Qwen3ASRConfig class."""
    from roomkit.voice.stt.qwen3 import Qwen3ASRConfig

    return Qwen3ASRConfig


def get_neutts_provider() -> type:
    """Get NeuTTSProvider class (requires neutts)."""
    from roomkit.voice.tts.neutts import NeuTTSProvider

    return NeuTTSProvider


def get_neutts_config() -> type:
    """Get NeuTTSConfig class."""
    from roomkit.voice.tts.neutts import NeuTTSConfig

    return NeuTTSConfig


def get_neutts_voice_config() -> type:
    """Get NeuTTSVoiceConfig class for NeuTTS."""
    from roomkit.voice.tts.neutts import NeuTTSVoiceConfig

    return NeuTTSVoiceConfig


def get_vui_tts_provider() -> type:
    """Get VuiTTSProvider class (requires vui-tts, Python 3.12)."""
    from roomkit.voice.tts.vui import VuiTTSProvider

    return VuiTTSProvider


def get_vui_tts_config() -> type:
    """Get VuiTTSConfig class."""
    from roomkit.voice.tts.vui import VuiTTSConfig

    return VuiTTSConfig


def get_vui_voice() -> type:
    """Get VuiVoice class for Vui Nano."""
    from roomkit.voice.tts.vui import VuiVoice

    return VuiVoice


def get_fluxions_tts_provider() -> type:
    """Get FluxionsTTSProvider class for Vui hosted by fluxions.ai (requires httpx)."""
    from roomkit.voice.tts.fluxions import FluxionsTTSProvider

    return FluxionsTTSProvider


def get_fluxions_tts_config() -> type:
    """Get FluxionsTTSConfig class."""
    from roomkit.voice.tts.fluxions import FluxionsTTSConfig

    return FluxionsTTSConfig


def get_azure_speech_tts_provider() -> type:
    """Get AzureSpeechTTSProvider class for MAI-Voice and Azure voices (requires httpx)."""
    from roomkit.voice.tts.azure_speech import AzureSpeechTTSProvider

    return AzureSpeechTTSProvider


def get_azure_speech_tts_config() -> type:
    """Get AzureSpeechTTSConfig class."""
    from roomkit.voice.tts.azure_speech import AzureSpeechTTSConfig

    return AzureSpeechTTSConfig


def get_pocket_tts_provider() -> type:
    """Get PocketTTSProvider class (requires pocket-tts)."""
    from roomkit.voice.tts.pocket import PocketTTSProvider

    return PocketTTSProvider


def get_pocket_tts_config() -> type:
    """Get PocketTTSConfig class."""
    from roomkit.voice.tts.pocket import PocketTTSConfig

    return PocketTTSConfig


def get_grok_tts_provider() -> type:
    """Get GrokTTSProvider class (requires httpx, websockets)."""
    from roomkit.voice.tts.grok import GrokTTSProvider

    return GrokTTSProvider


def get_grok_tts_config() -> type:
    """Get GrokTTSConfig class."""
    from roomkit.voice.tts.grok import GrokTTSConfig

    return GrokTTSConfig


def get_gemini_tts_provider() -> type:
    """Get GeminiTTSProvider class (requires google-genai)."""
    from roomkit.voice.tts.gemini import GeminiTTSProvider

    return GeminiTTSProvider


def get_gemini_tts_config() -> type:
    """Get GeminiTTSConfig class."""
    from roomkit.voice.tts.gemini import GeminiTTSConfig

    return GeminiTTSConfig


def get_gemini_voice_library() -> type:
    """Get GeminiVoiceLibrary class (requires google-genai)."""
    from roomkit.voice.tts.gemini_library import GeminiVoiceLibrary

    return GeminiVoiceLibrary


def get_gemini_voice_library_config() -> type:
    """Get GeminiVoiceLibraryConfig class."""
    from roomkit.voice.tts.gemini_library import GeminiVoiceLibraryConfig

    return GeminiVoiceLibraryConfig


def get_openai_realtime_provider() -> type:
    """Get OpenAIRealtimeProvider class (requires openai, websockets)."""
    from roomkit.providers.openai.realtime import OpenAIRealtimeProvider

    return OpenAIRealtimeProvider


def get_xai_realtime_provider() -> type:
    """Get XAIRealtimeProvider class (requires websockets)."""
    from roomkit.providers.xai.realtime import XAIRealtimeProvider

    return XAIRealtimeProvider


def get_xai_realtime_config() -> type:
    """Get XAIRealtimeConfig class."""
    from roomkit.providers.xai.config import XAIRealtimeConfig

    return XAIRealtimeConfig


def get_gemini_live_provider() -> type:
    """Get GeminiLiveProvider class (requires google-genai)."""
    from roomkit.providers.gemini.realtime import GeminiLiveProvider

    return GeminiLiveProvider


def get_websocket_realtime_transport() -> type:
    """Get WebSocketRealtimeTransport class (requires websockets)."""
    from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport

    return WebSocketRealtimeTransport


def get_buzz_huddle_backend() -> type:
    """Get BuzzHuddleBackend class (requires buzzkit — pip install roomkit[buzz])."""
    from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend

    return BuzzHuddleBackend


def get_local_audio_transport() -> type:
    """Get LocalAudioBackend class (requires sounddevice, numpy).

    LocalAudioTransport was merged into LocalAudioBackend — this
    function now returns the unified backend which supports both
    the VoiceChannel (connect/start_listening) and RealtimeVoiceChannel
    (accept) paths.
    """
    from roomkit.voice.backends.local import LocalAudioBackend

    return LocalAudioBackend


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


def get_webtransport_backend() -> type:
    """Get WebTransportBackend class (requires aioquic)."""
    from roomkit.voice.backends.webtransport import WebTransportBackend

    return WebTransportBackend

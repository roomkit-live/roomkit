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
    BargeInEvent,
    PartialTranscriptionEvent,
    SpeakerChangeEvent,
    TTSCancelledEvent,
    VADAudioLevelEvent,
    VADSilenceEvent,
)
from roomkit.voice.pipeline import (
    AudioPipeline,
    AudioPipelineConfig,
    AudioPostProcessor,
    DenoiserProvider,
    DiarizationProvider,
    DiarizationResult,
    MockDenoiserProvider,
    MockDiarizationProvider,
    MockVADProvider,
    VADConfig,
    VADEvent,
    VADEventType,
    VADProvider,
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
    "BargeInEvent",
    "PartialTranscriptionEvent",
    "SpeakerChangeEvent",
    "TTSCancelledEvent",
    "VADAudioLevelEvent",
    "VADSilenceEvent",
    # Pipeline types
    "AudioPipeline",
    "AudioPipelineConfig",
    "AudioPostProcessor",
    "DenoiserProvider",
    "DiarizationProvider",
    "DiarizationResult",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    "VADProvider",
    # Pipeline mocks
    "MockDenoiserProvider",
    "MockDiarizationProvider",
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


def get_fastrtc_realtime_transport() -> type:
    """Get FastRTCRealtimeTransport class (requires fastrtc, numpy)."""
    from roomkit.voice.realtime.fastrtc_transport import FastRTCRealtimeTransport

    return FastRTCRealtimeTransport


def get_mount_fastrtc_realtime() -> Any:
    """Get mount_fastrtc_realtime function (requires fastrtc, numpy)."""
    from roomkit.voice.realtime.fastrtc_transport import mount_fastrtc_realtime

    return mount_fastrtc_realtime

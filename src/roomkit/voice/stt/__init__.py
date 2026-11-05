"""Speech-to-text providers.

Each provider and its config is imported from here. A provider's module loads
on first use, with the vendor SDK it needs; ``import roomkit`` loads none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.voice.stt.azure_mai import AzureMAISTTConfig, AzureMAISTTProvider
    from roomkit.voice.stt.azure_mai_protocol import AzureMAISTTError
    from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
    from roomkit.voice.stt.gemini import GeminiSTTConfig, GeminiSTTProvider
    from roomkit.voice.stt.gemini_transcribe import (
        GeminiTranscribeConfig,
        GeminiTranscribeProvider,
    )
    from roomkit.voice.stt.gemini_transcript import Transcript as GeminiTranscript
    from roomkit.voice.stt.gemini_transcript import TranscriptSegment as GeminiTranscriptSegment
    from roomkit.voice.stt.gemini_transcript import TranscriptWord as GeminiTranscriptWord
    from roomkit.voice.stt.gradium import GradiumSTTConfig, GradiumSTTProvider
    from roomkit.voice.stt.meta import MetaSTTConfig, MetaSTTProvider
    from roomkit.voice.stt.meta_protocol import MetaSTTError
    from roomkit.voice.stt.mock import MockSTTProvider
    from roomkit.voice.stt.qwen3 import Qwen3ASRConfig, Qwen3ASRProvider
    from roomkit.voice.stt.rime import RimeSTTConfig, RimeSTTError, RimeSTTProvider
    from roomkit.voice.stt.sherpa_onnx import SherpaOnnxSTTConfig, SherpaOnnxSTTProvider
    from roomkit.voice.stt.voxtral import (
        VoxtralSTTConfig,
        VoxtralSTTProvider,
        VoxtralTurnDetection,
    )
    from roomkit.voice.stt.voxtral_protocol import VoxtralSTTError
else:
    __getattr__ = lazy_exports(
        __name__,
        {
            "AzureMAISTTConfig": "roomkit.voice.stt.azure_mai",
            "AzureMAISTTError": "roomkit.voice.stt.azure_mai_protocol",
            "AzureMAISTTProvider": "roomkit.voice.stt.azure_mai",
            "DeepgramConfig": "roomkit.voice.stt.deepgram",
            "DeepgramSTTProvider": "roomkit.voice.stt.deepgram",
            "GeminiSTTConfig": "roomkit.voice.stt.gemini",
            "GeminiSTTProvider": "roomkit.voice.stt.gemini",
            "GeminiTranscribeConfig": "roomkit.voice.stt.gemini_transcribe",
            "GeminiTranscribeProvider": "roomkit.voice.stt.gemini_transcribe",
            "GeminiTranscript": "roomkit.voice.stt.gemini_transcript:Transcript",
            "GeminiTranscriptSegment": "roomkit.voice.stt.gemini_transcript:TranscriptSegment",
            "GeminiTranscriptWord": "roomkit.voice.stt.gemini_transcript:TranscriptWord",
            "GradiumSTTConfig": "roomkit.voice.stt.gradium",
            "GradiumSTTProvider": "roomkit.voice.stt.gradium",
            "MetaSTTConfig": "roomkit.voice.stt.meta",
            "MetaSTTError": "roomkit.voice.stt.meta_protocol",
            "MetaSTTProvider": "roomkit.voice.stt.meta",
            "MockSTTProvider": "roomkit.voice.stt.mock",
            "Qwen3ASRConfig": "roomkit.voice.stt.qwen3",
            "Qwen3ASRProvider": "roomkit.voice.stt.qwen3",
            "RimeSTTConfig": "roomkit.voice.stt.rime",
            "RimeSTTError": "roomkit.voice.stt.rime",
            "RimeSTTProvider": "roomkit.voice.stt.rime",
            "SherpaOnnxSTTConfig": "roomkit.voice.stt.sherpa_onnx",
            "SherpaOnnxSTTProvider": "roomkit.voice.stt.sherpa_onnx",
            "VoxtralSTTConfig": "roomkit.voice.stt.voxtral",
            "VoxtralSTTError": "roomkit.voice.stt.voxtral_protocol",
            "VoxtralSTTProvider": "roomkit.voice.stt.voxtral",
            "VoxtralTurnDetection": "roomkit.voice.stt.voxtral",
        },
    )

__all__ = [
    "AzureMAISTTConfig",
    "AzureMAISTTError",
    "AzureMAISTTProvider",
    "DeepgramConfig",
    "DeepgramSTTProvider",
    "GeminiSTTConfig",
    "GeminiSTTProvider",
    "GeminiTranscribeConfig",
    "GeminiTranscribeProvider",
    "GeminiTranscript",
    "GeminiTranscriptSegment",
    "GeminiTranscriptWord",
    "GradiumSTTConfig",
    "GradiumSTTProvider",
    "MetaSTTConfig",
    "MetaSTTError",
    "MetaSTTProvider",
    "MockSTTProvider",
    "Qwen3ASRConfig",
    "Qwen3ASRProvider",
    "RimeSTTConfig",
    "RimeSTTError",
    "RimeSTTProvider",
    "STTProvider",
    "SherpaOnnxSTTConfig",
    "SherpaOnnxSTTProvider",
    "VoxtralSTTConfig",
    "VoxtralSTTError",
    "VoxtralSTTProvider",
    "VoxtralTurnDetection",
]

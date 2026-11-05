"""Text-to-speech providers.

Each provider and its config is imported from here. A provider's module loads
on first use, with the vendor SDK it needs; ``import roomkit`` loads none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
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
    StripTechnicalText,
    TTSFilterChain,
    TTSStreamFilter,
    filtered_stream,
)
from roomkit.voice.tts.sentence_splitter import split_sentences

if TYPE_CHECKING:
    from roomkit.voice.tts.azure_speech import (
        AzureSpeechTTSConfig,
        AzureSpeechTTSError,
        AzureSpeechTTSProvider,
    )
    from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider
    from roomkit.voice.tts.fish_audio import (
        FishAudioTTSConfig,
        FishAudioTTSError,
        FishAudioTTSProvider,
    )
    from roomkit.voice.tts.fluxions import FluxionsTTSConfig, FluxionsTTSProvider
    from roomkit.voice.tts.gemini import GeminiTTSConfig, GeminiTTSProvider
    from roomkit.voice.tts.gemini_library import (
        CONSENT_STATEMENTS,
        GeminiVoiceLibrary,
        GeminiVoiceLibraryConfig,
    )
    from roomkit.voice.tts.gradium import GradiumTTSConfig, GradiumTTSProvider
    from roomkit.voice.tts.grok import GrokTTSConfig, GrokTTSProvider
    from roomkit.voice.tts.mock import MockTTSProvider
    from roomkit.voice.tts.neutts import NeuTTSConfig, NeuTTSProvider, NeuTTSVoiceConfig
    from roomkit.voice.tts.phonon import PhononTTSConfig, PhononTTSProvider
    from roomkit.voice.tts.pocket import PocketTTSConfig, PocketTTSProvider
    from roomkit.voice.tts.qwen3 import Qwen3TTSConfig, Qwen3TTSProvider
    from roomkit.voice.tts.qwen3 import VoiceCloneConfig as Qwen3VoiceCloneConfig
    from roomkit.voice.tts.rime import RimeTTSConfig, RimeTTSError, RimeTTSProvider
    from roomkit.voice.tts.sherpa_onnx import SherpaOnnxTTSConfig, SherpaOnnxTTSProvider
    from roomkit.voice.tts.voxtral import VoxtralTTSConfig, VoxtralTTSError, VoxtralTTSProvider
    from roomkit.voice.tts.vui import VUI_TAGS, VuiTTSConfig, VuiTTSProvider, VuiVoice
else:
    __getattr__ = lazy_exports(
        __name__,
        {
            "AzureSpeechTTSConfig": "roomkit.voice.tts.azure_speech",
            "AzureSpeechTTSError": "roomkit.voice.tts.azure_speech",
            "AzureSpeechTTSProvider": "roomkit.voice.tts.azure_speech",
            "CONSENT_STATEMENTS": "roomkit.voice.tts.gemini_library",
            "ElevenLabsConfig": "roomkit.voice.tts.elevenlabs",
            "ElevenLabsTTSProvider": "roomkit.voice.tts.elevenlabs",
            "FishAudioTTSConfig": "roomkit.voice.tts.fish_audio",
            "FishAudioTTSError": "roomkit.voice.tts.fish_audio",
            "FishAudioTTSProvider": "roomkit.voice.tts.fish_audio",
            "FluxionsTTSConfig": "roomkit.voice.tts.fluxions",
            "FluxionsTTSProvider": "roomkit.voice.tts.fluxions",
            "GeminiTTSConfig": "roomkit.voice.tts.gemini",
            "GeminiTTSProvider": "roomkit.voice.tts.gemini",
            "GeminiVoiceLibrary": "roomkit.voice.tts.gemini_library",
            "GeminiVoiceLibraryConfig": "roomkit.voice.tts.gemini_library",
            "GradiumTTSConfig": "roomkit.voice.tts.gradium",
            "GradiumTTSProvider": "roomkit.voice.tts.gradium",
            "GrokTTSConfig": "roomkit.voice.tts.grok",
            "GrokTTSProvider": "roomkit.voice.tts.grok",
            "MockTTSProvider": "roomkit.voice.tts.mock",
            "NeuTTSConfig": "roomkit.voice.tts.neutts",
            "NeuTTSProvider": "roomkit.voice.tts.neutts",
            "NeuTTSVoiceConfig": "roomkit.voice.tts.neutts",
            "PhononTTSConfig": "roomkit.voice.tts.phonon",
            "PhononTTSProvider": "roomkit.voice.tts.phonon",
            "PocketTTSConfig": "roomkit.voice.tts.pocket",
            "PocketTTSProvider": "roomkit.voice.tts.pocket",
            "Qwen3TTSConfig": "roomkit.voice.tts.qwen3",
            "Qwen3TTSProvider": "roomkit.voice.tts.qwen3",
            "Qwen3VoiceCloneConfig": "roomkit.voice.tts.qwen3:VoiceCloneConfig",
            "RimeTTSConfig": "roomkit.voice.tts.rime",
            "RimeTTSError": "roomkit.voice.tts.rime",
            "RimeTTSProvider": "roomkit.voice.tts.rime",
            "SherpaOnnxTTSConfig": "roomkit.voice.tts.sherpa_onnx",
            "SherpaOnnxTTSProvider": "roomkit.voice.tts.sherpa_onnx",
            "VUI_TAGS": "roomkit.voice.tts.vui",
            "VoxtralTTSConfig": "roomkit.voice.tts.voxtral",
            "VoxtralTTSError": "roomkit.voice.tts.voxtral",
            "VoxtralTTSProvider": "roomkit.voice.tts.voxtral",
            "VuiTTSConfig": "roomkit.voice.tts.vui",
            "VuiTTSProvider": "roomkit.voice.tts.vui",
            "VuiVoice": "roomkit.voice.tts.vui",
        },
    )

__all__ = [
    "AzureSpeechTTSConfig",
    "AzureSpeechTTSError",
    "AzureSpeechTTSProvider",
    "CONSENT_STATEMENTS",
    "ConversationTurn",
    "ElevenLabsConfig",
    "ElevenLabsTTSProvider",
    "FishAudioTTSConfig",
    "FishAudioTTSError",
    "FishAudioTTSProvider",
    "FluxionsTTSConfig",
    "FluxionsTTSProvider",
    "GeminiTTSConfig",
    "GeminiTTSProvider",
    "GeminiVoiceLibrary",
    "GeminiVoiceLibraryConfig",
    "GradiumTTSConfig",
    "GradiumTTSProvider",
    "GrokTTSConfig",
    "GrokTTSProvider",
    "MockTTSProvider",
    "NeuTTSConfig",
    "NeuTTSProvider",
    "NeuTTSVoiceConfig",
    "PhononTTSConfig",
    "PhononTTSProvider",
    "PocketTTSConfig",
    "PocketTTSProvider",
    "Qwen3TTSConfig",
    "Qwen3TTSProvider",
    "Qwen3VoiceCloneConfig",
    "RimeTTSConfig",
    "RimeTTSError",
    "RimeTTSProvider",
    "SherpaOnnxTTSConfig",
    "SherpaOnnxTTSProvider",
    "StripBrackets",
    "StripEmoji",
    "StripInternalTags",
    "StripTechnicalText",
    "TTSContext",
    "TTSContextConfig",
    "TTSContextLevel",
    "TTSFilterChain",
    "TTSProvider",
    "TTSStreamFilter",
    "VUI_TAGS",
    "VoxtralTTSConfig",
    "VoxtralTTSError",
    "VoxtralTTSProvider",
    "VuiTTSConfig",
    "VuiTTSProvider",
    "VuiVoice",
    "filtered_stream",
    "split_sentences",
]

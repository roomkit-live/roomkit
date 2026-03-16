"""Realtime voice support for speech-to-speech AI providers."""

from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.events import (
    RealtimeErrorEvent,
    RealtimeSpeechEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptionEvent,
)
from roomkit.voice.realtime.mock import (
    MockCall,
    MockRealtimeAudioVideoProvider,
    MockRealtimeProvider,
    MockRealtimeTransport,
)
from roomkit.voice.realtime.provider import (
    RealtimeAudioVideoProvider,
    RealtimeVideoCallback,
    RealtimeVoiceProvider,
)

__all__ = [
    # ABCs
    "RealtimeAudioVideoProvider",
    "RealtimeAVBridge",
    "RealtimeVideoCallback",
    "RealtimeVoiceProvider",
    # Events
    "RealtimeErrorEvent",
    "RealtimeSpeechEvent",
    "RealtimeToolCallEvent",
    "RealtimeTranscriptionEvent",
    # Mocks
    "MockCall",
    "MockRealtimeAudioVideoProvider",
    "MockRealtimeProvider",
    "MockRealtimeTransport",
]

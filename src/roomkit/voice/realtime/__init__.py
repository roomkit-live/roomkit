"""Realtime voice support for speech-to-speech AI providers."""

from roomkit.voice.realtime.events import (
    RealtimeErrorEvent,
    RealtimeSpeechEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptionEvent,
)
from roomkit.voice.realtime.mock import MockCall, MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.provider import RealtimeVoiceProvider

__all__ = [
    # ABCs
    "RealtimeVoiceProvider",
    # Events
    "RealtimeErrorEvent",
    "RealtimeSpeechEvent",
    "RealtimeToolCallEvent",
    "RealtimeTranscriptionEvent",
    # Mocks
    "MockCall",
    "MockRealtimeProvider",
    "MockRealtimeTransport",
]

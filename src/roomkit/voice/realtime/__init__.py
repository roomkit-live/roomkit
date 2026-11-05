"""Realtime voice support for speech-to-speech AI providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit._lazy import lazy_exports
from roomkit.voice.realtime.bridge import RealtimeAVBridge
from roomkit.voice.realtime.events import (
    RealtimeDelegationEvent,
    RealtimeErrorEvent,
    RealtimeSpeechEvent,
    RealtimeToolCallEvent,
    RealtimeTranscriptionEvent,
)
from roomkit.voice.realtime.injection import VoiceInjectionResult
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
    VoiceInfo,
)
from roomkit.voice.realtime.reasoning import (
    AgentReasoningBackend,
    AIProviderReasoningBackend,
    ReasoningBackend,
    ReasoningCutShortError,
    ReasoningOutput,
    ReasoningRequest,
    ToolCallResult,
    TranscriptLine,
    render_transcript_request,
)

if TYPE_CHECKING:
    from roomkit.voice.realtime.fastrtc_transport import (
        FastRTCRealtimeTransport,
        mount_fastrtc_realtime,
    )
    from roomkit.voice.realtime.rime import RimeRealtimeConfig, RimeRealtimeProvider
    from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport
    from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport
else:
    __getattr__ = lazy_exports(
        __name__,
        {
            "FastRTCRealtimeTransport": "roomkit.voice.realtime.fastrtc_transport",
            "SIPRealtimeTransport": "roomkit.voice.realtime.sip_transport",
            "RimeRealtimeConfig": "roomkit.voice.realtime.rime",
            "RimeRealtimeProvider": "roomkit.voice.realtime.rime",
            "WebSocketRealtimeTransport": "roomkit.voice.realtime.ws_transport",
            "mount_fastrtc_realtime": "roomkit.voice.realtime.fastrtc_transport",
        },
    )

__all__ = [
    "VoiceInjectionResult",
    # ABCs
    "RealtimeAudioVideoProvider",
    "RealtimeAVBridge",
    "RealtimeVideoCallback",
    "RealtimeVoiceProvider",
    "RimeRealtimeConfig",
    "RimeRealtimeProvider",
    "VoiceInfo",
    # Reasoning delegation (RFC §12.4.1)
    "AgentReasoningBackend",
    "AIProviderReasoningBackend",
    "ReasoningBackend",
    "ReasoningCutShortError",
    "ReasoningOutput",
    "ReasoningRequest",
    "ToolCallResult",
    "TranscriptLine",
    "render_transcript_request",
    # Transports
    "FastRTCRealtimeTransport",
    "SIPRealtimeTransport",
    "WebSocketRealtimeTransport",
    "mount_fastrtc_realtime",
    # Events
    "RealtimeDelegationEvent",
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

"""Conference support for RoomKit (SFU orchestration).

A conference extends a Room with a multi-party real-time media session. An
external SFU routes media between human participants; RoomKit orchestrates the
conference and joins it as a bot participant to provide intelligence. RoomKit
never sits in the media path between human participants.
"""

from __future__ import annotations

from roomkit.conference._mock_faults import ErrorSpec, MockFaults
from roomkit.conference._mock_media import MockDelivery, MockTrackFormat, MockUtterance
from roomkit.conference.base import (
    ActiveSpeakerCallback,
    BotSessionEndedCallback,
    ConferenceBackend,
    ConnectionQualityCallback,
    ParticipantCallback,
    TrackAudioCallback,
    TrackCallback,
    TrackVideoCallback,
)
from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.conference.mock import (
    INJECTABLE_EMISSIONS,
    INJECTABLE_METHODS,
    MockConferenceBackend,
    MockConferenceCall,
)
from roomkit.conference.models import (
    BotSession,
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    ConferenceParticipant,
    ConferenceRecordingConfig,
    ConferenceRecordingMode,
    ConferenceTrack,
    TrackKind,
)

__all__ = [
    "ActiveSpeakerCallback",
    "BotSession",
    "BotSessionEndedCallback",
    "ConferenceBackend",
    "ConferenceAccess",
    "ConferenceCapability",
    "ConferenceGrants",
    "ConferenceInterruptionConfig",
    "ConferenceInterruptionScope",
    "ConferenceParticipant",
    "ConferenceRecordingConfig",
    "ConferenceRecordingMode",
    "ConferenceTrack",
    "ConnectionQualityCallback",
    "INJECTABLE_EMISSIONS",
    "INJECTABLE_METHODS",
    "ErrorSpec",
    "LiveKitConferenceBackend",
    "LiveKitConfig",
    "MockConferenceBackend",
    "MockConferenceCall",
    "MockDelivery",
    "MockFaults",
    "MockTrackFormat",
    "MockUtterance",
    "ParticipantCallback",
    "TrackAudioCallback",
    "TrackCallback",
    "TrackKind",
    "TrackVideoCallback",
]

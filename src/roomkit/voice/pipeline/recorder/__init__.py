"""Audio recorder providers."""

from roomkit.voice.pipeline.recorder.base import (
    AudioRecorder,
    RecordingChannelMode,
    RecordingConfig,
    RecordingHandle,
    RecordingMode,
    RecordingResult,
    RecordingTrigger,
)
from roomkit.voice.pipeline.recorder.mock import MockAudioRecorder

__all__ = [
    "AudioRecorder",
    "MockAudioRecorder",
    "RecordingChannelMode",
    "RecordingConfig",
    "RecordingHandle",
    "RecordingMode",
    "RecordingResult",
    "RecordingTrigger",
]

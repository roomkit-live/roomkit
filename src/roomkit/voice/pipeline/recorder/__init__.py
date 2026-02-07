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
from roomkit.voice.pipeline.recorder.wav import WavFileRecorder

__all__ = [
    "AudioRecorder",
    "MockAudioRecorder",
    "WavFileRecorder",
    "RecordingChannelMode",
    "RecordingConfig",
    "RecordingHandle",
    "RecordingMode",
    "RecordingResult",
    "RecordingTrigger",
]

"""Mock pipeline providers for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.denoiser_provider import DenoiserProvider
from roomkit.voice.pipeline.diarization_provider import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.vad_provider import VADEvent, VADProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class MockVADProvider(VADProvider):
    """Mock VAD provider that returns a preconfigured sequence of events.

    Example:
        from roomkit.voice.pipeline.vad_provider import VADEvent, VADEventType

        events = [
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio"),
        ]
        vad = MockVADProvider(events=events)
    """

    def __init__(self, events: list[VADEvent | None] | None = None) -> None:
        self._events = events or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockVADProvider"

    def process(self, frame: AudioFrame) -> VADEvent | None:
        self.frames.append(frame)
        if self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            return event
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class MockDenoiserProvider(DenoiserProvider):
    """Mock denoiser that passes frames through unchanged.

    Tracks processed frames for test assertions.
    """

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDenoiserProvider"

    def process(self, frame: AudioFrame) -> AudioFrame:
        self.frames.append(frame)
        return frame

    def close(self) -> None:
        self.closed = True


class MockDiarizationProvider(DiarizationProvider):
    """Mock diarization provider that returns a preconfigured sequence of results.

    Example:
        results = [
            DiarizationResult(speaker_id="speaker_0", confidence=0.9, is_new_speaker=True),
            DiarizationResult(speaker_id="speaker_0", confidence=0.95, is_new_speaker=False),
        ]
        diarizer = MockDiarizationProvider(results=results)
    """

    def __init__(self, results: list[DiarizationResult | None] | None = None) -> None:
        self._results = results or []
        self._index = 0
        self.frames: list[AudioFrame] = []
        self.reset_count = 0
        self.closed = False

    @property
    def name(self) -> str:
        return "MockDiarizationProvider"

    def process(self, frame: AudioFrame) -> DiarizationResult | None:
        self.frames.append(frame)
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True

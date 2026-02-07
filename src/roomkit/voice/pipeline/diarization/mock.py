"""Mock diarization provider for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from roomkit.voice.pipeline.diarization.base import DiarizationProvider, DiarizationResult

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


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

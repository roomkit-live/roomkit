"""Speaker diarization providers."""

from roomkit.voice.pipeline.diarization.base import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider

__all__ = ["DiarizationProvider", "DiarizationResult", "MockDiarizationProvider"]

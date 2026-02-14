"""Speaker diarization providers."""

from roomkit.voice.pipeline.diarization.base import DiarizationProvider, DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.pipeline.diarization.sherpa_onnx import (
    SherpaOnnxDiarizationConfig,
    SherpaOnnxDiarizationProvider,
)

__all__ = [
    "DiarizationProvider",
    "DiarizationResult",
    "MockDiarizationProvider",
    "SherpaOnnxDiarizationConfig",
    "SherpaOnnxDiarizationProvider",
]

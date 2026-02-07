"""Audio denoiser providers."""

from roomkit.voice.pipeline.denoiser.base import DenoiserProvider
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider
from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
    SherpaOnnxDenoiserConfig,
    SherpaOnnxDenoiserProvider,
)

__all__ = [
    "DenoiserProvider",
    "MockDenoiserProvider",
    "RNNoiseDenoiserProvider",
    "SherpaOnnxDenoiserConfig",
    "SherpaOnnxDenoiserProvider",
]

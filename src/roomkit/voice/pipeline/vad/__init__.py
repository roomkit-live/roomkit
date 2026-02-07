"""Voice Activity Detection providers."""

from roomkit.voice.pipeline.vad.base import VADConfig, VADEvent, VADEventType, VADProvider
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.pipeline.vad.sherpa_onnx import SherpaOnnxVADConfig, SherpaOnnxVADProvider

__all__ = [
    "EnergyVADProvider",
    "MockVADProvider",
    "SherpaOnnxVADConfig",
    "SherpaOnnxVADProvider",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    "VADProvider",
]

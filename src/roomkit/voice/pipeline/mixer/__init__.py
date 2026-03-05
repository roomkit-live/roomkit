"""Audio mixer providers for PCM frame mixing."""

from roomkit.voice.pipeline.mixer.base import MixerProvider
from roomkit.voice.pipeline.mixer.python import PythonMixerProvider

__all__ = [
    "MixerProvider",
    "NumpyMixerProvider",
    "PythonMixerProvider",
]


def __getattr__(name: str) -> object:
    # Lazy import for NumpyMixerProvider to avoid hard numpy dependency
    if name == "NumpyMixerProvider":
        from roomkit.voice.pipeline.mixer.numpy import NumpyMixerProvider

        return NumpyMixerProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

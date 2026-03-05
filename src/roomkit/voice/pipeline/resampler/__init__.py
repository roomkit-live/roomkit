"""Resampler providers."""

import contextlib

from roomkit.voice.pipeline.resampler.base import ResamplerProvider
from roomkit.voice.pipeline.resampler.linear import LinearResamplerProvider
from roomkit.voice.pipeline.resampler.mock import MockResamplerProvider
from roomkit.voice.pipeline.resampler.sinc import SincResamplerProvider

with contextlib.suppress(ImportError):
    from roomkit.voice.pipeline.resampler.numpy import NumpyResamplerProvider

__all__ = [
    "LinearResamplerProvider",
    "MockResamplerProvider",
    "NumpyResamplerProvider",
    "ResamplerProvider",
    "SincResamplerProvider",
]

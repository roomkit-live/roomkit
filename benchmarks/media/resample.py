"""The artifacts the core's default resampler leaves (RFC §12.3.3, RK-12.3.3-007).

The RFC says implementations SHOULD resample with high quality, to avoid
artifacts that degrade VAD and STT accuracy, and gives no figure: no threshold
is decided before these numbers. The core's default is linear interpolation,
``NumpyResamplerProvider`` (NumPy is in every voice extra), or
the pure-Python ``LinearResamplerProvider``; both are measured, frame by frame as
the pipeline calls them (20 ms frames, one stream), on fixed tones:

- ``thdn_db``: total harmonic distortion plus noise of a 1 kHz tone at -6 dBFS,
  the output's residual after a least-squares fit of the tone, against the tone;
- ``alias_db``: downsampling only, a tone a third of the way from the target's
  Nyquist frequency to the source's, which an ideal resampler removes: the
  output's level against the input's;
- ``image_db``: upsampling only, the output's energy above the source's Nyquist
  frequency against all of it, for the 1 kHz tone;
- ``us_per_frame``: the time one frame's conversion takes.

A vendor's resampler is its provider's to measure.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from roomkit.voice import AudioFrame
from roomkit.voice.pipeline.resampler import (
    LinearResamplerProvider,
    NumpyResamplerProvider,
    ResamplerProvider,
)

PAIRS = (
    (8000, 16000),
    (16000, 8000),
    (24000, 16000),
    (48000, 16000),
    (16000, 48000),
    (44100, 16000),
)
TONE_HZ = 1000.0
AMPLITUDE = 16384.0  # -6 dBFS
SECONDS = 1.0
FRAMES_PER_SECOND = 50


def _providers() -> dict[str, ResamplerProvider]:
    return {"numpy": NumpyResamplerProvider(), "linear": LinearResamplerProvider()}


def _tone(rate: int, hz: float) -> np.ndarray:
    t = np.arange(int(rate * SECONDS)) / rate
    return AMPLITUDE * np.sin(2 * np.pi * hz * t)


def _resample(
    provider: ResamplerProvider, signal: np.ndarray, src: int, dst: int
) -> tuple[np.ndarray, list[float]]:
    """*signal* converted frame by frame, and the microseconds each frame took."""
    pcm = np.rint(signal).astype(np.int16)
    step = src // FRAMES_PER_SECOND
    out: list[np.ndarray] = []
    costs: list[float] = []
    for start in range(0, len(pcm) - step + 1, step):
        frame = AudioFrame(data=pcm[start : start + step].tobytes(), sample_rate=src, channels=1)
        began = time.perf_counter_ns()
        converted = provider.resample(frame, dst, 1, 2, "s")
        costs.append((time.perf_counter_ns() - began) / 1000)
        out.append(np.frombuffer(converted.data, dtype=np.int16).astype(np.float64))
    return np.concatenate(out), costs


def _db(power: float, reference: float) -> float:
    return float(10 * np.log10(max(power, 1e-30) / reference))


def thdn_db(output: np.ndarray, rate: int, hz: float) -> float:
    """The residual after fitting a tone of *hz*, against the fitted tone."""
    t = np.arange(len(output)) / rate
    basis = np.column_stack(
        [np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t), np.ones_like(t)]
    )
    coefficients, *_ = np.linalg.lstsq(basis, output, rcond=None)
    fitted = basis[:, :2] @ coefficients[:2]
    residual = output - basis @ coefficients
    return _db(float(np.mean(residual**2)), float(np.mean(fitted**2)))


def level_db(output: np.ndarray, reference: np.ndarray) -> float:
    return _db(float(np.mean(output**2)), float(np.mean(reference**2)))


def image_db(output: np.ndarray, rate: int, above_hz: float) -> float:
    """The output's energy above *above_hz*, against all of it."""
    spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output)))) ** 2
    hz = np.fft.rfftfreq(len(output), 1 / rate)
    return _db(float(spectrum[hz > above_hz].sum()), float(spectrum.sum()))


def alias_tone_hz(src: int, dst: int) -> float:
    """A tone a third of the way from the target's Nyquist frequency to the
    source's: what a downsampling should remove, its alias inside the band."""
    return dst / 2 + (src - dst) / 6


def _pair(provider: ResamplerProvider, src: int, dst: int) -> dict[str, list[float]]:
    tone = _tone(src, TONE_HZ)
    output, costs = _resample(provider, tone, src, dst)
    figures = {"thdn_db": [thdn_db(output, dst, TONE_HZ)], "us_per_frame": costs}
    if dst < src:
        high = _tone(src, alias_tone_hz(src, dst))
        aliased, _ = _resample(provider, high, src, dst)
        figures["alias_db"] = [level_db(aliased, high)]
    else:
        figures["image_db"] = [image_db(output, dst, src / 2)]
    return figures


def measure() -> Callable[[], dict[str, list[float]]]:
    """One pass: every pair through each default resampler."""

    def run() -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name, provider in _providers().items():
            for src, dst in PAIRS:
                for figure, values in _pair(provider, src, dst).items():
                    out[f"{name} {src}->{dst} {figure}"] = values
        return out

    return run

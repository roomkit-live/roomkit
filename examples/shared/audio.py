"""Shared audio pipeline builders for RoomKit examples.

All builder functions use lazy imports so that missing optional
dependencies only cause a warning rather than crashing the example.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AEC
# ---------------------------------------------------------------------------


def build_aec(
    sample_rate: int,
    block_ms: int = 20,
    *,
    default: str = "webrtc",
    enable_ns: bool = True,
) -> object | None:
    """Build an AEC provider based on the ``AEC`` env var.

    Env: ``AEC=webrtc|speex|1|0`` (default comes from *default* param).

    * ``webrtc`` / ``1`` — WebRTC AEC3 (``pip install aec-audio-processing``)
    * ``speex`` — SpeexDSP (``apt install libspeexdsp1``)
    * ``0`` — disabled

    Returns the provider instance or ``None``.
    """
    aec_mode = os.environ.get("AEC", default).lower()
    if aec_mode == "0":
        logger.info("AEC disabled (AEC=0)")
        return None

    if aec_mode in ("1", "webrtc"):
        try:
            from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

            logger.info("AEC enabled (WebRTC AEC3%s)", " + NS" if enable_ns else "")
            return WebRTCAECProvider(sample_rate=sample_rate, enable_ns=enable_ns)
        except ImportError:
            print("\n  >>> Install AEC: pip install aec-audio-processing <<<\n")
            return None

    if aec_mode == "speex":
        try:
            from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

            frame_size = sample_rate * block_ms // 1000
            logger.info("AEC enabled (Speex)")
            return SpeexAECProvider(
                frame_size=frame_size,
                filter_length=frame_size * 10,
                sample_rate=sample_rate,
            )
        except ImportError:
            print("\n  >>> Install Speex: apt install libspeexdsp1 <<<\n")
            return None

    logger.warning("Unknown AEC mode %r — disabling", aec_mode)
    return None


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


def build_denoiser(sample_rate: int = 16000, *, default: str = "0") -> object | None:
    """Build a denoiser provider based on the ``DENOISE`` env var.

    Env: ``DENOISE=rnnoise|sherpa|1|0`` (default comes from *default* param).
    For ``sherpa``, ``DENOISE_MODEL`` sets the model file (default
    ``gtcrn_simple.onnx``).

    Returns the provider instance or ``None``.
    """
    mode = os.environ.get("DENOISE", default).lower()
    if mode == "0":
        return None

    # "sherpa" or "1" when default is sherpa-flavored
    if mode in ("sherpa", "1") and default in ("sherpa", "1"):
        model = os.environ.get("DENOISE_MODEL", "gtcrn_simple.onnx")
        try:
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                SherpaOnnxDenoiserConfig,
                SherpaOnnxDenoiserProvider,
            )

            logger.info("Denoiser enabled (sherpa-onnx GTCRN, model=%s)", model)
            return SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=model))
        except ImportError:
            logger.warning("sherpa-onnx not available (DENOISE=0 to skip)")
            return None

    # "rnnoise" or "1" when default is rnnoise-flavored
    if mode in ("rnnoise", "1"):
        try:
            from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider

            logger.info("Denoiser enabled (RNNoise)")
            return RNNoiseDenoiserProvider(sample_rate=sample_rate)
        except ImportError:
            logger.warning("RNNoise not available (DENOISE=0 to skip)")
            return None

    # Explicit "sherpa" even if default wasn't sherpa
    if mode == "sherpa":
        model = os.environ.get("DENOISE_MODEL", "gtcrn_simple.onnx")
        try:
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                SherpaOnnxDenoiserConfig,
                SherpaOnnxDenoiserProvider,
            )

            logger.info("Denoiser enabled (sherpa-onnx GTCRN, model=%s)", model)
            return SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig(model=model))
        except ImportError:
            logger.warning("sherpa-onnx not available (DENOISE=0 to skip)")
            return None

    logger.warning("Unknown DENOISE mode %r — disabling", mode)
    return None


# ---------------------------------------------------------------------------
# Debug taps
# ---------------------------------------------------------------------------


def build_debug_taps() -> object | None:
    """Build pipeline debug taps based on the ``DEBUG_AUDIO`` env var.

    Env: ``DEBUG_AUDIO=1|0`` (default ``0``).

    Returns a :class:`PipelineDebugTaps` or ``None``.
    """
    if os.environ.get("DEBUG_AUDIO", "0") != "1":
        return None
    from roomkit.voice.pipeline.debug_taps import PipelineDebugTaps

    logger.info("Debug audio taps enabled → ./debug_audio/")
    return PipelineDebugTaps(output_dir="./debug_audio/", stages=["all"])


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def build_pipeline(
    *,
    aec: object | None = None,
    denoiser: object | None = None,
    debug_taps: object | None = None,
    **kwargs: object,
) -> object | None:
    """Build an :class:`AudioPipelineConfig` if any stage is set.

    Forwards all keyword arguments to the ``AudioPipelineConfig``
    constructor.  Returns ``None`` when every stage is ``None`` / empty.
    """
    all_values = {"aec": aec, "denoiser": denoiser, "debug_taps": debug_taps, **kwargs}
    if not any(v for v in all_values.values()):
        return None
    from roomkit.voice.pipeline.config import AudioPipelineConfig

    return AudioPipelineConfig(**all_values)  # type: ignore[arg-type]

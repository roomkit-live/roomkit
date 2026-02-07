"""Speech enhancement provider using sherpa-onnx GTCRN model.

Uses sherpa-onnx's ``OfflineSpeechDenoiser`` with the GTCRN model for
neural noise suppression.  Unlike RNNoise (which requires a system
library), sherpa-onnx is a pure-Python dependency::

    pip install roomkit[sherpa-onnx]

Usage::

    from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
        SherpaOnnxDenoiserConfig,
        SherpaOnnxDenoiserProvider,
    )

    denoiser = SherpaOnnxDenoiserProvider(
        SherpaOnnxDenoiserConfig(model="gtcrn_simple.onnx")
    )
    config = AudioPipelineConfig(denoiser=denoiser)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


def _pcm_s16le_to_float32(data: bytes) -> list[float]:
    """Convert PCM signed 16-bit little-endian bytes to float32 list in [-1, 1]."""
    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    return [s / 32768.0 for s in samples]


def _float32_to_pcm_s16le(samples: list[float]) -> bytes:
    """Convert float32 list in [-1, 1] to PCM signed 16-bit little-endian bytes."""
    clamped = [max(-1.0, min(1.0, s)) for s in samples]
    int_samples = [int(s * 32767) for s in clamped]
    return struct.pack(f"<{len(int_samples)}h", *int_samples)


@dataclass
class SherpaOnnxDenoiserConfig:
    """Configuration for the sherpa-onnx GTCRN denoiser.

    Attributes:
        model: Path to the ``gtcrn_simple.onnx`` model file.
        num_threads: Number of CPU threads for inference.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
    """

    model: str = ""
    num_threads: int = 1
    provider: str = "cpu"


class SherpaOnnxDenoiserProvider(DenoiserProvider):
    """Denoiser provider using sherpa-onnx GTCRN speech enhancement.

    The denoiser is created lazily on the first call to :meth:`process`.
    sherpa-onnx must be installed (``pip install roomkit[sherpa-onnx]``).

    Parameters:
        config: Provider configuration.
    """

    def __init__(self, config: SherpaOnnxDenoiserConfig) -> None:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sherpa-onnx is required for SherpaOnnxDenoiserProvider. "
                "Install it with: pip install roomkit[sherpa-onnx]"
            ) from exc

        self._config = config
        self._sherpa: Any = __import__("sherpa_onnx")
        self._denoiser: Any = None

    @property
    def name(self) -> str:
        return "SherpaOnnxDenoiser"

    def _ensure_denoiser(self) -> None:
        """Lazily create the sherpa-onnx OfflineSpeechDenoiser."""
        if self._denoiser is not None:
            return

        cfg = self._config
        sherpa = self._sherpa

        gtcrn_config = sherpa.OfflineSpeechDenoiserGtcrnModelConfig(
            model=cfg.model,
        )
        model_config = sherpa.OfflineSpeechDenoiserModelConfig(
            gtcrn=gtcrn_config,
            num_threads=cfg.num_threads,
            provider=cfg.provider,
        )
        denoiser_config = sherpa.OfflineSpeechDenoiserConfig(
            model=model_config,
        )
        self._denoiser = sherpa.OfflineSpeechDenoiser(denoiser_config)
        logger.debug(
            "SherpaOnnxDenoiser: created denoiser model=%s",
            cfg.model,
        )

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Denoise an audio frame using GTCRN speech enhancement.

        Converts int16 PCM to float32, runs the denoiser, and converts
        back.  On error, logs a warning and returns the original frame.
        """
        if self._denoiser is None:
            try:
                self._ensure_denoiser()
            except Exception:
                logger.warning(
                    "SherpaOnnxDenoiser: failed to initialize, passing through",
                    exc_info=True,
                )
                return frame

        try:
            float_samples = _pcm_s16le_to_float32(frame.data)
            result = self._denoiser.run(float_samples)
            out_data = _float32_to_pcm_s16le(result.samples)
        except Exception:
            logger.warning(
                "SherpaOnnxDenoiser: error during processing, passing through",
                exc_info=True,
            )
            return frame

        from roomkit.voice.audio_frame import AudioFrame

        return AudioFrame(
            data=out_data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

    def reset(self) -> None:  # noqa: B027
        """Reset internal state (no-op — GTCRN is stateless per call)."""

    def close(self) -> None:
        """Release the denoiser."""
        self._denoiser = None

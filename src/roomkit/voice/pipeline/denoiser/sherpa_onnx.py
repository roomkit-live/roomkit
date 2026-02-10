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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

if TYPE_CHECKING:
    import numpy as np

    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


def _pcm_s16le_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM signed 16-bit little-endian bytes to float32 array in [-1, 1]."""
    import numpy as np

    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_pcm_s16le(samples: np.ndarray | list[float]) -> bytes:
    """Convert float32 array in [-1, 1] to PCM signed 16-bit little-endian bytes."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    return bytes(np.clip(arr * 32767, -32767, 32767).astype(np.int16).tobytes())


@dataclass
class SherpaOnnxDenoiserConfig:
    """Configuration for the sherpa-onnx GTCRN denoiser.

    Attributes:
        model: Path to the ``gtcrn_simple.onnx`` model file.
        num_threads: Number of CPU threads for inference.
        provider: ONNX execution provider (``"cpu"`` or ``"cuda"``).
        context_frames: Number of preceding frames to include as context
            when denoising.  GTCRN is an offline model — processing tiny
            20 ms frames in isolation produces poor quality and boundary
            artifacts.  A sliding context window (default 3 = 60 ms,
            ~6 STFT frames at 6.25 ms hop) gives the model's recurrent
            layers enough warmup context while keeping inference ~3x
            faster than the previous 200 ms window.
        silence_threshold: RMS energy threshold (in float32 [-1, 1] scale)
            below which ONNX inference is skipped entirely.  Frames below
            this level are replaced with silence.  Set to 0 to disable.
            Default 0.005 ≈ −46 dBFS — well below any speech.
    """

    model: str = ""
    num_threads: int = 1
    provider: str = "cpu"
    context_frames: int = 3
    silence_threshold: float = 0.005


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

        import numpy as np

        self._config = config
        self._sherpa: Any = __import__("sherpa_onnx")
        self._denoiser: Any = None
        self._context: np.ndarray = np.array([], dtype=np.float32)

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

        Uses a sliding context window so the model sees preceding frames
        for temporal context.  Only the current frame's portion of the
        denoised output is returned — no latency is added.
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
            import numpy as np

            float_samples = _pcm_s16le_to_float32(frame.data)
            n_frame = len(float_samples)

            # Append current frame to sliding context buffer
            self._context = np.concatenate([self._context, float_samples])

            # Trim to at most context_frames worth of samples
            max_context = n_frame * max(self._config.context_frames, 1)
            if len(self._context) > max_context:
                self._context = self._context[-max_context:]

            # Skip inference for near-silent frames — output silence directly.
            # The denoiser would suppress this to ~silence anyway; skipping
            # avoids ONNX inference cost entirely at idle.  Context is kept
            # up-to-date so speech onset always has a warm context window.
            threshold = self._config.silence_threshold
            if threshold > 0:
                rms = float(np.sqrt(np.dot(float_samples, float_samples) / n_frame))
                if rms < threshold:
                    from roomkit.voice.audio_frame import AudioFrame

                    return AudioFrame(
                        data=b"\x00" * len(frame.data),
                        sample_rate=frame.sample_rate,
                        channels=frame.channels,
                        sample_width=frame.sample_width,
                        timestamp_ms=frame.timestamp_ms,
                        metadata=dict(frame.metadata),
                    )

            # Pad at the START to GTCRN block size (256 samples).
            # Padding must go at the beginning so the real audio stays at
            # the tail — we extract the last n_frame samples from the output.
            block = 256
            remainder = len(self._context) % block
            if remainder:
                pad = np.zeros(block - remainder, dtype=np.float32)
                to_process = np.concatenate([pad, self._context])
            else:
                to_process = self._context

            result = self._denoiser.run(to_process.tolist(), frame.sample_rate)

            # Extract the last n_frame samples (the current frame's portion)
            denoised = result.samples
            if len(denoised) >= n_frame:
                out_samples = denoised[len(denoised) - n_frame :]
            else:
                # Fallback: model returned fewer samples than expected
                out_samples = denoised

            out_data = _float32_to_pcm_s16le(out_samples)
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

    def reset(self) -> None:
        """Reset sliding context buffer."""
        import numpy as np

        self._context = np.array([], dtype=np.float32)

    def close(self) -> None:
        """Release the denoiser."""
        import numpy as np

        self._denoiser = None
        self._context = np.array([], dtype=np.float32)

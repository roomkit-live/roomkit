"""Speech enhancement provider using ai|coustics Quail models.

Uses the ``aic-sdk`` package (Rust + PyO3) for on-device neural noise
suppression, dereverberation, and speaker isolation (Voice Focus).
Optimized for STT/ASR accuracy with ~2 ms inference per 10 ms frame
and 30 ms algorithmic delay::

    pip install roomkit[aicoustics]

Usage::

    from roomkit.voice.pipeline.denoiser.aicoustics import (
        AICousticsDenoiserConfig,
        AICousticsDenoiserProvider,
    )

    denoiser = AICousticsDenoiserProvider(
        AICousticsDenoiserConfig(model="quail-vf-2.0-l-16khz")
    )
    config = AudioPipelineConfig(denoiser=denoiser)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

if TYPE_CHECKING:
    import numpy as np

    from roomkit.voice.audio_frame import AudioFrame

logger = logging.getLogger("roomkit.voice.pipeline.aicoustics")


def _pcm_s16le_to_float32(data: bytes) -> np.ndarray[Any, Any]:
    """Convert PCM signed 16-bit little-endian bytes to float32 array in [-1, 1]."""
    import numpy as np

    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_pcm_s16le(samples: np.ndarray[Any, Any]) -> bytes:
    """Convert float32 array in [-1, 1] to PCM signed 16-bit little-endian bytes."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    return bytes(np.clip(arr * 32767, -32767, 32767).astype(np.int16).tobytes())


@dataclass
class AICousticsDenoiserConfig:
    """Configuration for the ai|coustics Quail denoiser.

    Attributes:
        model: Model identifier for download (e.g. ``"quail-vf-2.0-l-16khz"``).
        model_dir: Local cache directory for downloaded models.
        license_key: SDK license key.  Defaults to the ``AIC_SDK_LICENSE``
            environment variable if not provided.
        enhancement_level: Enhancement strength from 0.0 (off) to 1.0
            (maximum).  0.8 gives the best WER for voice AI workloads.
        num_channels: Number of audio channels (1 = mono, 2 = stereo).
    """

    model: str = "quail-vf-2.0-l-16khz"
    model_dir: str = "./models"
    license_key: str = ""
    enhancement_level: float = 0.8
    num_channels: int = 1
    _resolved_license_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved_license_key = self.license_key or os.environ.get("AIC_SDK_LICENSE", "")


class AICousticsDenoiserProvider(DenoiserProvider):
    """Denoiser provider using ai|coustics Quail speech enhancement.

    The processor is created lazily on the first call to :meth:`process`.
    ``aic-sdk`` must be installed (``pip install roomkit[aicoustics]``).

    Parameters:
        config: Provider configuration.
    """

    def __init__(self, config: AICousticsDenoiserConfig | None = None) -> None:
        try:
            import aic_sdk  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "aic-sdk is required for AICousticsDenoiserProvider. "
                "Install it with: pip install roomkit[aicoustics]"
            ) from exc

        self._config = config or AICousticsDenoiserConfig()
        self._aic: Any = __import__("aic_sdk")
        self._processor: Any = None
        self._frame_size: int = 0
        self._buffer: bytes = b""
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "aicoustics"

    def _ensure_processor(self) -> None:
        """Lazily download model and create the aic_sdk Processor."""
        if self._processor is not None:
            return

        cfg = self._config
        aic = self._aic

        # Download model to local cache (sync, idempotent).
        model_path = aic.Model.download(cfg.model, cfg.model_dir)
        logger.debug("AICoustics: downloaded model=%s to %s", cfg.model, model_path)

        # Create processor with optimal config.
        processor_config = aic.ProcessorConfig.optimal(
            model_path,
            num_channels=cfg.num_channels,
        )
        self._processor = aic.Processor(model_path, cfg._resolved_license_key, processor_config)
        self._frame_size = processor_config.num_frames

        # Set enhancement level.
        context = self._processor.context()
        context.set_parameter("enhancement_level", cfg.enhancement_level)

        logger.info(
            "AICoustics: created processor model=%s frame_size=%d enhancement=%.2f",
            cfg.model,
            self._frame_size,
            cfg.enhancement_level,
        )

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Denoise an audio frame using Quail speech enhancement.

        Buffers incoming PCM to match the SDK's expected frame size,
        then processes complete chunks.  Any remainder is held for
        the next call.
        """
        with self._lock:
            return self._process_locked(frame)

    def _process_locked(self, frame: AudioFrame) -> AudioFrame:
        if self._processor is None:
            try:
                self._ensure_processor()
            except Exception:
                logger.warning(
                    "AICoustics: failed to initialize, passing through",
                    exc_info=True,
                )
                return frame

        try:
            import numpy as np

            from roomkit.voice.audio_frame import AudioFrame

            # Accumulate raw PCM bytes into the buffer.
            self._buffer += frame.data

            # Each Quail frame is frame_size samples × 2 bytes (int16).
            chunk_bytes = self._frame_size * 2 * self._config.num_channels
            if chunk_bytes <= 0:
                return frame

            processed_parts: list[bytes] = []

            while len(self._buffer) >= chunk_bytes:
                chunk = self._buffer[:chunk_bytes]
                self._buffer = self._buffer[chunk_bytes:]

                float_samples = _pcm_s16le_to_float32(chunk)

                # Quail expects shape (channels, frames).
                samples_2d = float_samples.reshape(self._config.num_channels, self._frame_size)
                result = self._processor.process(samples_2d)

                # Result is (channels, frames) — flatten back.
                out_flat = np.asarray(result, dtype=np.float32).flatten()
                processed_parts.append(_float32_to_pcm_s16le(out_flat))

            if not processed_parts:
                # Not enough data for a full chunk yet — pass through
                # the raw frame so downstream stages (VAD, STT) still
                # receive audio rather than silence.
                return frame

            out_data = b"".join(processed_parts)

            # Ensure output length matches input length.
            if len(out_data) > len(frame.data):
                out_data = out_data[: len(frame.data)]
            elif len(out_data) < len(frame.data):
                out_data += b"\x00" * (len(frame.data) - len(out_data))

            return AudioFrame(
                data=out_data,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                sample_width=frame.sample_width,
                timestamp_ms=frame.timestamp_ms,
                metadata=dict(frame.metadata),
            )
        except Exception:
            logger.warning(
                "AICoustics: error during processing, passing through",
                exc_info=True,
            )
            return frame

    def reset(self) -> None:
        """Reset processor internal state and clear the buffer."""
        with self._lock:
            self._buffer = b""
            if self._processor is not None:
                # Recreate processor to clear internal state.
                try:
                    self._processor = None
                    self._ensure_processor()
                except Exception:
                    logger.warning(
                        "AICoustics: failed to recreate processor on reset",
                        exc_info=True,
                    )

    def close(self) -> None:
        """Release the processor and clear resources."""
        with self._lock:
            self._processor = None
            self._buffer = b""
            self._frame_size = 0

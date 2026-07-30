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
    license_key: str = field(default="", repr=False)
    enhancement_level: float = 0.8
    num_channels: int = 1
    _resolved_license_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved_license_key = self.license_key or os.environ.get("AIC_SDK_LICENSE", "")


@dataclass
class _StreamState:
    """One stream's Quail processor and its chunking buffer.

    The Processor carries the model's recurrent state, so two speakers
    through one processor would each be enhanced against the other's
    residue.  The buffer holds the sub-frame remainder between calls.
    """

    processor: Any = None
    frame_size: int = 0
    buffer: bytes = b""
    lock: threading.Lock = field(default_factory=threading.Lock)


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
        self._streams: dict[str, _StreamState] = {}
        # Guards _streams itself — the per-stream lock guards its contents.
        self._streams_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "aicoustics"

    def _state_for(self, stream: str) -> _StreamState:
        """Get or create this stream's state (without its processor)."""
        with self._streams_lock:
            return self._streams.setdefault(stream, _StreamState())

    def _ensure_processor(self, st: _StreamState) -> None:
        """Lazily download model and create this stream's aic_sdk Processor."""
        if st.processor is not None:
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
        st.processor = aic.Processor(model_path, cfg._resolved_license_key, processor_config)
        st.frame_size = processor_config.num_frames

        # Set enhancement level.
        context = st.processor.context()
        context.set_parameter("enhancement_level", cfg.enhancement_level)

        logger.info(
            "AICoustics: created processor model=%s frame_size=%d enhancement=%.2f",
            cfg.model,
            st.frame_size,
            cfg.enhancement_level,
        )

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Denoise an audio frame using Quail speech enhancement.

        Buffers incoming PCM to match the SDK's expected frame size,
        then processes complete chunks.  Any remainder is held for
        the next call.
        """
        st = self._state_for(stream)
        with st.lock:
            return self._process_locked(frame, st)

    def _process_locked(self, frame: AudioFrame, st: _StreamState) -> AudioFrame:
        if st.processor is None:
            try:
                self._ensure_processor(st)
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
            st.buffer += frame.data

            # Each Quail frame is frame_size samples × 2 bytes (int16).
            chunk_bytes = st.frame_size * 2 * self._config.num_channels
            if chunk_bytes <= 0:
                return frame

            processed_parts: list[bytes] = []

            while len(st.buffer) >= chunk_bytes:
                chunk = st.buffer[:chunk_bytes]
                st.buffer = st.buffer[chunk_bytes:]

                float_samples = _pcm_s16le_to_float32(chunk)

                # Quail expects shape (channels, frames).
                samples_2d = float_samples.reshape(self._config.num_channels, st.frame_size)
                result = st.processor.process(samples_2d)

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

    def reset(self, stream: str) -> None:
        """Drop this stream's processor and its buffer.

        The next frame for this stream builds a fresh processor, which is
        what clears the model's recurrent state.
        """
        with self._streams_lock:
            self._streams.pop(stream, None)

    def close(self) -> None:
        """Release every stream's processor."""
        with self._streams_lock:
            self._streams.clear()

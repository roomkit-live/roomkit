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
        sample_rate: PCM sample rate expected by the selected model.
    """

    model: str = "quail-vf-2.0-l-16khz"
    model_dir: str = "./models"
    license_key: str = field(default="", repr=False)
    enhancement_level: float = 0.8
    num_channels: int = 1
    sample_rate: int = 16000
    _resolved_license_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.enhancement_level <= 1.0:
            raise ValueError("enhancement_level must be between 0 and 1")
        if self.num_channels not in (1, 2):
            raise ValueError("num_channels must be 1 or 2")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
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
    output_buffer: bytes = b""
    chunking: bool = False
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
        self._warned_formats: set[tuple[int, int, int, int]] = set()
        self._closed = False

    @property
    def name(self) -> str:
        return "aicoustics"

    def _state_for(self, stream: str) -> _StreamState | None:
        """Get or create this stream's state (without its processor)."""
        with self._streams_lock:
            if self._closed:
                return None
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
        if not self._matches_format(frame):
            return frame
        st = self._state_for(stream)
        if st is None:
            return frame
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
            from roomkit.voice.audio_frame import AudioFrame

            # Each Quail frame is frame_size samples × 2 bytes (int16).
            chunk_bytes = st.frame_size * 2 * self._config.num_channels
            if chunk_bytes <= 0:
                return frame

            if not st.chunking and not st.buffer and len(frame.data) % chunk_bytes == 0:
                out_data = b"".join(
                    self._process_chunk(st, frame.data[offset : offset + chunk_bytes])
                    for offset in range(0, len(frame.data), chunk_bytes)
                )
            else:
                if not st.chunking:
                    st.chunking = True
                    st.output_buffer += b"\x00" * chunk_bytes
                st.buffer += frame.data
                while len(st.buffer) >= chunk_bytes:
                    chunk = st.buffer[:chunk_bytes]
                    st.buffer = st.buffer[chunk_bytes:]
                    st.output_buffer += self._process_chunk(st, chunk)
                out_data = st.output_buffer[: len(frame.data)]
                st.output_buffer = st.output_buffer[len(frame.data) :]

            metadata = dict(frame.metadata)
            metadata["noise_suppressed"] = True
            return AudioFrame(
                data=out_data,
                sample_rate=frame.sample_rate,
                channels=frame.channels,
                sample_width=frame.sample_width,
                timestamp_ms=frame.timestamp_ms,
                metadata=metadata,
            )
        except Exception:
            st.buffer = b""
            st.output_buffer = b""
            st.chunking = False
            logger.warning(
                "AICoustics: error during processing, passing through",
                exc_info=True,
            )
            return frame

    def _process_chunk(self, st: _StreamState, chunk: bytes) -> bytes:
        """Enhance one exact Quail block. Caller holds ``st.lock``."""
        import numpy as np

        float_samples = _pcm_s16le_to_float32(chunk)
        # Input PCM is frame-interleaved. Quail expects (channels, frames).
        samples_2d = float_samples.reshape(st.frame_size, self._config.num_channels).T
        result = st.processor.process(samples_2d)
        # Convert (channels, frames) back to frame-interleaved PCM.
        out_interleaved = np.asarray(result, dtype=np.float32).T.reshape(-1)
        output = _float32_to_pcm_s16le(out_interleaved)
        if len(output) != len(chunk):
            raise RuntimeError(
                f"AICoustics returned {len(output)} bytes for a {len(chunk)}-byte block"
            )
        return output

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
            self._closed = True
            self._streams.clear()

    def _matches_format(self, frame: AudioFrame) -> bool:
        actual = (frame.sample_rate, frame.channels, frame.sample_width)
        valid = (
            actual == (self._config.sample_rate, self._config.num_channels, 2)
            and len(frame.data) % (2 * self._config.num_channels) == 0
        )
        if valid:
            return True
        key = (*actual, len(frame.data))
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "AICoustics requires %dHz/%dch/2-byte aligned PCM; got "
                "%dHz/%dch/%d-byte with %d bytes; frame ignored",
                self._config.sample_rate,
                self._config.num_channels,
                *actual,
                len(frame.data),
            )
        return False

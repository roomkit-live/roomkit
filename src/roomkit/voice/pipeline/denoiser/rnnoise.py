"""Noise suppression provider using RNNoise (ctypes).

Uses the system ``librnnoise`` library via :mod:`ctypes` — no pip
dependency required.  The library can be installed on most Linux
distributions (``apt install librnnoise0``) and on macOS via Homebrew
(``brew install rnnoise``).

RNNoise operates at **48 kHz** with **480-sample frames** (10 ms) using
**float32** samples in the range [-32768, 32768].  When the pipeline
runs at 16 kHz the provider handles internal resampling (exact 1:3 ratio).

Usage::

    from roomkit.voice.pipeline.denoiser.rnnoise import RNNoiseDenoiserProvider

    denoiser = RNNoiseDenoiserProvider()
    config = AudioPipelineConfig(denoiser=denoiser)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import struct
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

logger = logging.getLogger("roomkit.voice.pipeline.rnnoise")

# ---------------------------------------------------------------------------
# RNNoise C library wrapper
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None

# RNNoise frame size is always 480 samples (10 ms at 48 kHz).
_RNNOISE_FRAME_SIZE = 480


def _load_rnnoise() -> ctypes.CDLL:
    """Load ``librnnoise`` or raise :class:`ImportError`."""
    global _lib  # noqa: PLW0603
    if _lib is not None:
        return _lib

    path = ctypes.util.find_library("rnnoise")

    # find_library only searches the system linker paths.  If the user
    # installed to ~/.local/lib (common when building from source without
    # root), probe well-known prefixes as a fallback.
    if path is None:
        _candidates = [
            os.path.expanduser("~/.local/lib"),
            "/usr/local/lib",
        ]
        _soname = "librnnoise.dylib" if sys.platform == "darwin" else "librnnoise.so"
        for _dir in _candidates:
            _candidate = os.path.join(_dir, _soname)
            if os.path.isfile(_candidate):
                path = _candidate
                break

    if path is None:
        raise ImportError(
            "librnnoise is required for RNNoiseDenoiserProvider. "
            "Install it with your package manager, e.g.: "
            "apt install librnnoise0 (Debian/Ubuntu) or "
            "brew install rnnoise (macOS)."
        )

    _lib = ctypes.CDLL(path)

    # rnnoise_get_frame_size() → int
    _lib.rnnoise_get_frame_size.argtypes = []
    _lib.rnnoise_get_frame_size.restype = ctypes.c_int

    # rnnoise_create(model=NULL) → DenoiseState*
    _lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    _lib.rnnoise_create.restype = ctypes.c_void_p

    # rnnoise_destroy(state)
    _lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    _lib.rnnoise_destroy.restype = None

    # rnnoise_process_frame(state, out_float*, in_float*) → float (VAD prob)
    _lib.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.rnnoise_process_frame.restype = ctypes.c_float

    return _lib


@dataclass
class _StreamState:
    """One stream's RNNoise state and scratch buffers.

    The native DenoiseState carries the model and the filter memory in the same
    object, so a stream cannot share it: two speakers through one state would
    denoise each other's tail.
    """

    # None once destroyed. Cleared under `lock`, so a thread that resolved this
    # state before a concurrent reset() sees the None instead of calling into a
    # freed pointer.
    state: ctypes.c_void_p | None
    in_buf: Any
    out_buf: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


class RNNoiseDenoiserProvider(DenoiserProvider):
    """Denoiser provider backed by RNNoise (Mozilla/Xiph).

    RNNoise is a recurrent neural network that suppresses stationary and
    non-stationary noise in real time.  Internally it operates at 48 kHz
    with 480-sample float32 frames.  When the pipeline delivers 16 kHz
    audio the provider up-samples before processing and down-samples
    afterward (exact 1:3 ratio).

    Args:
        sample_rate: Expected input sample rate.  Must divide evenly
            into 48000 (supports 16000, 24000, and 48000).
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self._streams: dict[str, _StreamState] = {}
        # Guards _streams itself — the per-stream lock guards its contents.
        # Without it, two concurrent first frames for one stream each build a
        # native state and one of them leaks.
        self._streams_lock = threading.Lock()

        # RNNoise operates at 48 kHz.  We support any rate that divides
        # evenly into 48000 (16 kHz → 3×, 24 kHz → 2×, 48 kHz → 1×).
        if 48000 % sample_rate != 0:
            raise ValueError(
                f"RNNoiseDenoiserProvider requires a sample rate that divides "
                f"evenly into 48000, got {sample_rate}"
            )

        self._lib = _load_rnnoise()
        self._sample_rate = sample_rate
        self._resample_factor = 48000 // sample_rate  # 1, 2, or 3

        # Verify the library agrees on frame size.
        frame_size = self._lib.rnnoise_get_frame_size()
        if frame_size != _RNNOISE_FRAME_SIZE:
            raise RuntimeError(
                f"rnnoise_get_frame_size() returned {frame_size}, expected {_RNNOISE_FRAME_SIZE}"
            )

        # Number of int16 samples per input frame at the pipeline rate.
        # e.g. 16 kHz → 160, 24 kHz → 240, 48 kHz → 480
        self._input_frame_samples = _RNNOISE_FRAME_SIZE // self._resample_factor

        self._input_frame_bytes = self._input_frame_samples * 2  # int16

        logger.info(
            "RNNoise init: sample_rate=%d, input_frame_samples=%d",
            sample_rate,
            self._input_frame_samples,
        )

    # ------------------------------------------------------------------
    # DenoiserProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rnnoise"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Denoise an audio frame.

        Accepts frames of any size that is a multiple of the internal
        chunk size (e.g. 240 samples at 24 kHz).  Larger frames are
        split into chunks, each processed by RNNoise independently.
        """
        st = self._state_for(stream)

        pcm = frame.data
        n_samples = len(pcm) // 2
        chunk = self._input_frame_samples

        if n_samples == 0 or n_samples % chunk != 0:
            logger.warning(
                "Frame size %d is not a multiple of %d samples. Passing frame through unchanged.",
                n_samples,
                chunk,
            )
            return frame

        # Unpack int16 PCM.
        samples_i16 = struct.unpack(f"<{n_samples}h", pcm)
        out_i16: list[int] = []
        factor = self._resample_factor

        with st.lock:
            if st.state is None:
                return frame  # reset() destroyed it between the lookup and here

            for offset in range(0, n_samples, chunk):
                # Fill the 480-sample RNNoise input buffer.
                if factor == 1:
                    for i in range(chunk):
                        st.in_buf[i] = float(samples_i16[offset + i])
                else:
                    # Upsample with linear interpolation to avoid
                    # spectral images that corrupt the RNNoise output.
                    for i in range(chunk):
                        cur = float(samples_i16[offset + i])
                        nxt = float(samples_i16[offset + i + 1]) if i + 1 < chunk else cur
                        base = i * factor
                        for j in range(factor):
                            # lerp: cur at j=0, nxt at j=factor
                            st.in_buf[base + j] = cur + (nxt - cur) * j / factor

                self._lib.rnnoise_process_frame(st.state, st.out_buf, st.in_buf)

                # Read the 480-sample output, downsampling back.
                if factor == 1:
                    for i in range(chunk):
                        out_i16.append(max(-32768, min(32767, int(st.out_buf[i]))))
                else:
                    # Downsample by averaging each group of `factor`
                    # samples — acts as a box-car anti-alias filter.
                    for i in range(chunk):
                        base = i * factor
                        avg = 0.0
                        for j in range(factor):
                            avg += st.out_buf[base + j]
                        avg /= factor
                        out_i16.append(max(-32768, min(32767, int(avg))))

        out_data = struct.pack(f"<{n_samples}h", *out_i16)

        return AudioFrame(
            data=out_data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

    def _state_for(self, stream: str) -> _StreamState:
        """Get or create this stream's native state."""
        with self._streams_lock:
            st = self._streams.get(stream)
            if st is None:
                st = _StreamState(
                    state=self._create_state(),
                    in_buf=(ctypes.c_float * _RNNOISE_FRAME_SIZE)(),
                    out_buf=(ctypes.c_float * _RNNOISE_FRAME_SIZE)(),
                )
                self._streams[stream] = st
            return st

    def _destroy(self, st: _StreamState) -> None:
        """Destroy one stream's native state, once, under its own lock.

        Holding the lock is what makes this safe against a denoise already in
        flight on the audio thread: it either finishes first, or finds the
        state cleared and passes the frame through.
        """
        with st.lock:
            if st.state is not None:
                self._lib.rnnoise_destroy(st.state)
                st.state = None

    def reset(self, stream: str) -> None:
        """Destroy this stream's native state and forget it."""
        with self._streams_lock:
            st = self._streams.pop(stream, None)
        if st is not None:
            self._destroy(st)

    def close(self) -> None:
        """Destroy every stream's native state."""
        with self._streams_lock:
            states = list(self._streams.values())
            self._streams.clear()
        for st in states:
            self._destroy(st)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_state(self) -> ctypes.c_void_p:
        state = self._lib.rnnoise_create(None)
        if not state:
            raise RuntimeError("rnnoise_create returned NULL")
        return ctypes.c_void_p(state)

    def __del__(self) -> None:
        self.close()

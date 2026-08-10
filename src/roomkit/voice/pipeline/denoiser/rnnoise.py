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


def _rnnoise_search_dirs() -> list[str]:
    """Directories probed for ``librnnoise`` when the linker cannot find it.

    ``find_library`` only searches the system linker paths, which covers a
    package-manager install and nothing else. A source build commonly lands in
    ``~/.local/lib`` (no root) or ``/usr/local/lib``, and Homebrew's lib
    directory is on neither path — its prefix also moves with the architecture,
    so ``HOMEBREW_PREFIX`` is honoured when the shell exports it.
    """
    dirs = [os.path.expanduser("~/.local/lib"), "/usr/local/lib"]
    brew_prefix = os.environ.get("HOMEBREW_PREFIX")
    if brew_prefix:
        dirs.append(os.path.join(brew_prefix, "lib"))
    dirs.append("/opt/homebrew/lib")
    # An exported HOMEBREW_PREFIX usually IS the default prefix; probing the
    # same directory twice would also print it twice in the error.
    return list(dict.fromkeys(dirs))


def _find_rnnoise() -> str | None:
    """Resolve a loadable ``librnnoise``, or ``None`` if there is none."""
    path: str | None = ctypes.util.find_library("rnnoise")
    if path is not None:
        return path
    soname = "librnnoise.dylib" if sys.platform == "darwin" else "librnnoise.so"
    for directory in _rnnoise_search_dirs():
        candidate = os.path.join(directory, soname)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_rnnoise() -> ctypes.CDLL:
    """Load ``librnnoise`` or raise :class:`ImportError`."""
    global _lib  # noqa: PLW0603
    if _lib is not None:
        return _lib

    path = _find_rnnoise()
    if path is None:
        # The Homebrew package called "rnnoise" is a cask of DAW plugins
        # (VST/LV2/LADSPA) built from a different project. It ships no
        # loadable copy of this C ABI, so recommending it sends macOS users
        # to install 40 MB that can never satisfy this import.
        raise ImportError(
            "librnnoise is required for RNNoiseDenoiserProvider and was not "
            "found on the library search path. Debian/Ubuntu: "
            "apt install librnnoise0. macOS: build xiph/rnnoise from source "
            "(the Homebrew cask named 'rnnoise' is a set of DAW plugins and "
            "does not provide this library). For a denoiser that needs no "
            "system library at all, use SherpaOnnxDenoiserProvider. "
            f"Searched: {', '.join(_rnnoise_search_dirs())}."
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
    input_buffer: bytearray = field(default_factory=bytearray)
    output_buffer: bytearray = field(default_factory=bytearray)
    chunking: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class RNNoiseDenoiserProvider(DenoiserProvider):
    """Denoiser provider backed by RNNoise (Mozilla/Xiph).

    RNNoise is a recurrent neural network that suppresses stationary and
    non-stationary noise in real time.  Internally it operates at 48 kHz
    with 480-sample float32 frames.  When the pipeline delivers 16 kHz
    audio the provider up-samples before processing and down-samples
    afterward (exact 1:3 ratio).

    Args:
        sample_rate: Expected input sample rate. Supports 16000, 24000,
            and 48000 Hz.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self._streams: dict[str, _StreamState] = {}
        # Guards _streams itself — the per-stream lock guards its contents.
        # Without it, two concurrent first frames for one stream each build a
        # native state and one of them leaks.
        self._streams_lock = threading.Lock()

        # These are the rates exercised by RNNoise's exact integer-ratio path.
        if sample_rate not in (16000, 24000, 48000):
            raise ValueError(
                "RNNoiseDenoiserProvider sample_rate must be 16000, 24000, or "
                f"48000, got {sample_rate}"
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
        self._warned_formats: set[tuple[int, int, int, int]] = set()
        self._closed = False

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

        Exact chunks are processed without added delay. Once an irregular
        chunk is observed, a fixed 10 ms output delay preserves byte-for-byte
        timeline continuity while complete native chunks are accumulated.
        """
        if not self._matches_format(frame):
            return frame
        try:
            st = self._state_for(stream)
        except Exception:
            logger.warning("RNNoise initialization failed; frame bypassed", exc_info=True)
            return frame
        if st is None:
            return frame

        try:
            with st.lock:
                if st.state is None:
                    return frame  # reset() destroyed it between the lookup and here
                if (
                    not st.chunking
                    and not st.input_buffer
                    and len(frame.data) % self._input_frame_bytes == 0
                ):
                    out_data = b"".join(
                        self._process_chunk(
                            st,
                            frame.data[offset : offset + self._input_frame_bytes],
                        )
                        for offset in range(0, len(frame.data), self._input_frame_bytes)
                    )
                else:
                    if not st.chunking:
                        st.chunking = True
                        st.output_buffer.extend(b"\x00" * self._input_frame_bytes)
                    st.input_buffer.extend(frame.data)
                    while len(st.input_buffer) >= self._input_frame_bytes:
                        chunk = bytes(st.input_buffer[: self._input_frame_bytes])
                        del st.input_buffer[: self._input_frame_bytes]
                        st.output_buffer.extend(self._process_chunk(st, chunk))
                    out_data = bytes(st.output_buffer[: len(frame.data)])
                    del st.output_buffer[: len(frame.data)]
        except Exception:
            logger.warning("RNNoise processing failed; frame bypassed", exc_info=True)
            with st.lock:
                st.input_buffer.clear()
                st.output_buffer.clear()
                st.chunking = False
            return frame

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

    def _process_chunk(self, st: _StreamState, pcm: bytes) -> bytes:
        """Run one exact native RNNoise block. Caller holds ``st.lock``."""
        samples_i16 = struct.unpack(f"<{self._input_frame_samples}h", pcm)
        factor = self._resample_factor
        chunk = self._input_frame_samples
        if factor == 1:
            for index, sample in enumerate(samples_i16):
                st.in_buf[index] = float(sample)
        else:
            for index, sample in enumerate(samples_i16):
                current = float(sample)
                following = float(samples_i16[index + 1]) if index + 1 < chunk else current
                base = index * factor
                for offset in range(factor):
                    st.in_buf[base + offset] = current + (following - current) * offset / factor

        self._lib.rnnoise_process_frame(st.state, st.out_buf, st.in_buf)
        output: list[int] = []
        if factor == 1:
            for index in range(chunk):
                output.append(max(-32768, min(32767, int(st.out_buf[index]))))
        else:
            for index in range(chunk):
                base = index * factor
                average = sum(st.out_buf[base + offset] for offset in range(factor)) / factor
                output.append(max(-32768, min(32767, int(average))))
        return struct.pack(f"<{chunk}h", *output)

    def _state_for(self, stream: str) -> _StreamState | None:
        """Get or create this stream's native state unless closed."""
        with self._streams_lock:
            if self._closed:
                return None
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
            self._closed = True
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

    def _matches_format(self, frame: AudioFrame) -> bool:
        actual = (frame.sample_rate, frame.channels, frame.sample_width)
        valid = actual == (self._sample_rate, 1, 2) and len(frame.data) % 2 == 0
        if valid:
            return True
        key = (*actual, len(frame.data))
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "RNNoise requires %dHz/1ch/2-byte aligned PCM; got "
                "%dHz/%dch/%d-byte with %d bytes; frame ignored",
                self._sample_rate,
                *actual,
                len(frame.data),
            )
        return False

    def __del__(self) -> None:
        if hasattr(self, "_streams_lock"):
            self.close()

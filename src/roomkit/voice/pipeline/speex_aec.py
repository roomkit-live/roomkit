"""Acoustic Echo Cancellation provider using SpeexDSP (ctypes).

Uses the system ``libspeexdsp`` library via :mod:`ctypes` — no pip
dependency required.  The library ships with most Linux distributions
and can be installed on macOS via Homebrew (``brew install speexdsp``).

Usage::

    from roomkit.voice.pipeline.speex_aec import SpeexAECProvider

    aec = SpeexAECProvider(frame_size=320, filter_length=3200)
    config = AudioPipelineConfig(aec=aec)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import struct

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec_provider import AECProvider

logger = logging.getLogger("roomkit.voice.pipeline.speex_aec")

# ---------------------------------------------------------------------------
# SpeexDSP C library wrapper
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None


def _load_speexdsp() -> ctypes.CDLL:
    """Load ``libspeexdsp`` or raise :class:`ImportError`."""
    global _lib  # noqa: PLW0603
    if _lib is not None:
        return _lib

    path = ctypes.util.find_library("speexdsp")
    if path is None:
        raise ImportError(
            "libspeexdsp is required for SpeexAECProvider. "
            "Install it with your package manager, e.g.: "
            "apt install libspeexdsp1 (Debian/Ubuntu) or "
            "brew install speexdsp (macOS)."
        )

    _lib = ctypes.CDLL(path)

    # Set up function signatures for type safety.
    _lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    _lib.speex_echo_state_init.restype = ctypes.c_void_p

    _lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    _lib.speex_echo_state_destroy.restype = None

    _lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
    _lib.speex_echo_state_reset.restype = None

    # Async (split) API — allows feeding reference and processing
    # capture at different times, which matches our pipeline design.
    _lib.speex_echo_playback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.speex_echo_playback.restype = None

    _lib.speex_echo_capture.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _lib.speex_echo_capture.restype = None

    # Control function for setting sample rate, etc.
    _lib.speex_echo_ctl.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    _lib.speex_echo_ctl.restype = ctypes.c_int

    return _lib


# SpeexDSP echo control constants.
_SPEEX_ECHO_SET_SAMPLING_RATE = 24
_SPEEX_ECHO_GET_SAMPLING_RATE = 25


class SpeexAECProvider(AECProvider):
    """AEC provider backed by SpeexDSP's adaptive echo canceller.

    SpeexDSP uses a *split* (async) API that decouples reference feeding
    from capture processing — a natural fit for the pipeline's separate
    inbound/outbound paths.

    Args:
        frame_size: Number of samples per frame.  Must match the frames
            delivered by the pipeline (e.g. 320 for 20 ms at 16 kHz).
        filter_length: Echo-tail length in samples.  Longer values can
            cancel more reverberation but use more CPU.  A good default
            is 10× the frame size (e.g. 3200 samples = 200 ms at 16 kHz).
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        frame_size: int = 320,
        filter_length: int = 3200,
        sample_rate: int = 16000,
    ) -> None:
        self._lib = _load_speexdsp()
        self._frame_size = frame_size
        self._filter_length = filter_length
        self._sample_rate = sample_rate

        self._state = self._create_state()

    # ------------------------------------------------------------------
    # AECProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "speex_aec"

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Remove echo from a captured (mic) audio frame."""
        if self._state is None:
            return frame

        pcm_in = frame.data
        n_samples = len(pcm_in) // 2  # 16-bit samples

        if n_samples != self._frame_size:
            logger.warning(
                "Frame size mismatch: got %d samples, expected %d. "
                "Passing frame through unchanged.",
                n_samples,
                self._frame_size,
            )
            return frame

        in_buf = (ctypes.c_int16 * n_samples)(*struct.unpack(f"<{n_samples}h", pcm_in))
        out_buf = (ctypes.c_int16 * n_samples)()

        self._lib.speex_echo_capture(self._state, in_buf, out_buf)

        out_bytes = struct.pack(f"<{n_samples}h", *out_buf)
        return AudioFrame(
            data=out_bytes,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

    def feed_reference(self, frame: AudioFrame) -> None:
        """Feed a reference (playback / TTS) frame for echo modelling."""
        if self._state is None:
            return

        pcm = frame.data
        n_samples = len(pcm) // 2

        if n_samples != self._frame_size:
            logger.warning(
                "Reference frame size mismatch: got %d samples, expected %d. Ignoring.",
                n_samples,
                self._frame_size,
            )
            return

        ref_buf = (ctypes.c_int16 * n_samples)(*struct.unpack(f"<{n_samples}h", pcm))
        self._lib.speex_echo_playback(self._state, ref_buf)

    def reset(self) -> None:
        """Reset the adaptive filter state."""
        if self._state is not None:
            self._lib.speex_echo_state_reset(self._state)

    def close(self) -> None:
        """Destroy the SpeexDSP echo state and release resources."""
        if self._state is not None:
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_state(self) -> ctypes.c_void_p:
        state = self._lib.speex_echo_state_init(
            self._frame_size, self._filter_length
        )
        if not state:
            raise RuntimeError("speex_echo_state_init returned NULL")

        # Tell SpeexDSP the sample rate so it can tune its filters.
        sr = ctypes.c_int(self._sample_rate)
        self._lib.speex_echo_ctl(
            state,
            _SPEEX_ECHO_SET_SAMPLING_RATE,
            ctypes.byref(sr),
        )
        return state

    def __del__(self) -> None:
        self.close()

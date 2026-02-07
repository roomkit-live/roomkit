"""Acoustic Echo Cancellation provider using SpeexDSP (ctypes).

Uses the system ``libspeexdsp`` library via :mod:`ctypes` — no pip
dependency required.  The library ships with most Linux distributions
and can be installed on macOS via Homebrew (``brew install speexdsp``).

Usage::

    from roomkit.voice.pipeline.aec.speex import SpeexAECProvider

    aec = SpeexAECProvider(frame_size=320, filter_length=3200)
    config = AudioPipelineConfig(aec=aec)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import os
import threading

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec.base import AECProvider

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

    # Split (asynchronous) API — handles temporal alignment internally
    # via an internal ring buffer.  speex_echo_playback() feeds reference
    # audio (what the speaker is playing), speex_echo_capture() processes
    # mic audio and produces echo-cancelled output.
    _lib.speex_echo_playback.argtypes = [
        ctypes.c_void_p,  # state
        ctypes.c_void_p,  # play (speaker reference)
    ]
    _lib.speex_echo_playback.restype = None

    _lib.speex_echo_capture.argtypes = [
        ctypes.c_void_p,  # state
        ctypes.c_void_p,  # rec (mic input)
        ctypes.c_void_p,  # out (echo-cancelled)
    ]
    _lib.speex_echo_capture.restype = None

    # Synchronous API — kept for tests.
    _lib.speex_echo_cancellation.argtypes = [
        ctypes.c_void_p,  # state
        ctypes.c_void_p,  # rec (mic input)
        ctypes.c_void_p,  # play (speaker reference)
        ctypes.c_void_p,  # out (echo-cancelled)
    ]
    _lib.speex_echo_cancellation.restype = None

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

# Log AEC stats every N process() calls (~1 s at 20 ms frames).
_LOG_INTERVAL = 50


class _StderrSuppressor:
    """Temporarily redirect C-level stderr (fd 2) to ``/dev/null``.

    SpeexDSP prints warnings to C stderr that are expected in our
    usage pattern (e.g. "No playback frame available" during silence).
    This context manager suppresses them without affecting Python's
    logging, which uses its own file object.

    Thread-safe: a lock ensures only one thread has stderr redirected
    at a time.
    """

    def __init__(self) -> None:
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._orig_fd = os.dup(2)
        self._lock = threading.Lock()

    def __enter__(self) -> _StderrSuppressor:
        self._lock.acquire()
        os.dup2(self._devnull_fd, 2)
        return self

    def __exit__(self, *args: object) -> None:
        os.dup2(self._orig_fd, 2)
        self._lock.release()

    def close(self) -> None:
        if self._devnull_fd >= 0:
            os.close(self._devnull_fd)
            self._devnull_fd = -1
        if self._orig_fd >= 0:
            os.close(self._orig_fd)
            self._orig_fd = -1


class SpeexAECProvider(AECProvider):
    """AEC provider backed by SpeexDSP's adaptive echo canceller.

    Uses the split (asynchronous) API — ``speex_echo_playback()`` feeds
    reference audio from the speaker, ``speex_echo_capture()`` processes
    mic audio and returns echo-cancelled output.  The split API maintains
    an internal ring buffer that handles temporal misalignment between
    when reference audio is played and when the echo arrives at the mic,
    which is critical for real hardware with output latency.

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
        self._stderr = _StderrSuppressor()

        # Pre-allocated buffers — reused every call to avoid per-frame
        # heap allocations in the real-time audio path.
        self._in_buf = (ctypes.c_int16 * frame_size)()
        self._out_buf = (ctypes.c_int16 * frame_size)()
        self._ref_buf = (ctypes.c_int16 * frame_size)()
        self._frame_bytes = frame_size * 2  # 2 bytes per int16 sample

        # Lock for SpeexDSP state — protects _state, _in_buf, _out_buf,
        # _ref_buf across the mic and speaker callback threads.
        self._lock = threading.Lock()

        # Track whether speex_echo_playback() has been called since
        # the last speex_echo_capture(), for diagnostics only.
        self._playback_fed = False

        # Diagnostics — counters reset every _LOG_INTERVAL frames.
        self._process_count = 0
        self._ref_hits = 0  # process() had a real reference
        self._ref_misses = 0  # process() used silence
        self._refs_fed = 0  # feed_reference() calls
        self._total_in_energy = 0  # accumulated over _LOG_INTERVAL frames
        self._total_out_energy = 0

        logger.info(
            "SpeexAEC init: frame_size=%d, filter_length=%d (%dms), "
            "sample_rate=%d",
            frame_size,
            filter_length,
            filter_length * 1000 // sample_rate,
            sample_rate,
        )

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

        if len(pcm_in) != self._frame_bytes:
            logger.warning(
                "Frame size mismatch: got %d samples, expected %d. "
                "Passing frame through unchanged.",
                len(pcm_in) // 2,
                self._frame_size,
            )
            return frame

        ctypes.memmove(self._in_buf, pcm_in, self._frame_bytes)

        if self._playback_fed:
            self._ref_hits += 1
        else:
            self._ref_misses += 1
        self._playback_fed = False

        with self._lock:
            # Suppress C stderr — SpeexDSP prints "No playback frame
            # available" when the ring buffer is empty (expected during
            # silence).
            with self._stderr:
                self._lib.speex_echo_capture(
                    self._state, self._in_buf, self._out_buf
                )

            # Accumulate energy over the full interval so the log
            # reflects average behaviour, not a single-frame snapshot.
            in_energy = sum(
                self._in_buf[i] * self._in_buf[i]
                for i in range(self._frame_size)
            )
            out_energy = sum(
                self._out_buf[i] * self._out_buf[i]
                for i in range(self._frame_size)
            )
            self._total_in_energy += in_energy
            self._total_out_energy += out_energy

        self._process_count += 1
        if self._process_count % _LOG_INTERVAL == 0:
            self._log_stats()

        return AudioFrame(
            data=bytes(self._out_buf),
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

    def feed_reference(self, frame: AudioFrame) -> None:
        """Feed a reference (playback / TTS) frame for echo modelling.

        Calls ``speex_echo_playback()`` directly so the internal ring
        buffer tracks the speaker output timing.
        """
        if self._state is None:
            return

        pcm = frame.data
        n_bytes = len(pcm)
        n_samples = n_bytes // 2

        if n_samples != self._frame_size:
            logger.warning(
                "Reference frame size mismatch: got %d samples, expected %d. Ignoring.",
                n_samples,
                self._frame_size,
            )
            return

        ctypes.memmove(self._ref_buf, pcm, n_bytes)
        with self._lock, self._stderr:
            self._lib.speex_echo_playback(self._state, self._ref_buf)
        self._playback_fed = True
        self._refs_fed += 1

    def reset(self) -> None:
        """Reset the adaptive filter state."""
        self._playback_fed = False
        if self._state is not None:
            with self._lock:
                self._lib.speex_echo_state_reset(self._state)

    def close(self) -> None:
        """Destroy the SpeexDSP echo state and release resources."""
        if self._state is not None:
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None
        self._stderr.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_stats(self) -> None:
        """Log periodic AEC diagnostics (averaged over the interval)."""
        n = (self._frame_size * _LOG_INTERVAL) or 1
        in_rms = math.isqrt(self._total_in_energy // n)
        out_rms = math.isqrt(self._total_out_energy // n)

        if in_rms > 0:
            attenuation_db = 20 * math.log10(out_rms / in_rms) if out_rms > 0 else -99
        else:
            attenuation_db = 0.0

        logger.debug(
            "[AEC stats] processed=%d ref_hits=%d ref_misses=%d "
            "refs_fed=%d | "
            "in_rms=%d out_rms=%d attenuation=%.1fdB",
            self._process_count,
            self._ref_hits,
            self._ref_misses,
            self._refs_fed,
            in_rms,
            out_rms,
            attenuation_db,
        )

        # Reset interval counters.
        self._ref_hits = 0
        self._ref_misses = 0
        self._refs_fed = 0
        self._total_in_energy = 0
        self._total_out_energy = 0

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

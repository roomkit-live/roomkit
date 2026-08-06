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
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec.base import AECProvider

logger = logging.getLogger("roomkit.voice.pipeline.speex_aec")

# ---------------------------------------------------------------------------
# SpeexDSP C library wrapper
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL | None = None
_stderr_fd_lock = threading.Lock()


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
        # fd 2 belongs to the process, not one provider instance. Construction,
        # redirection and restoration therefore share one module-level lock.
        with _stderr_fd_lock:
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            self._orig_fd = os.dup(2)

    def __enter__(self) -> _StderrSuppressor:
        _stderr_fd_lock.acquire()
        try:
            os.dup2(self._devnull_fd, 2)
        except BaseException:
            _stderr_fd_lock.release()
            raise
        return self

    def __exit__(self, *args: object) -> None:
        try:
            os.dup2(self._orig_fd, 2)
        finally:
            _stderr_fd_lock.release()

    def close(self) -> None:
        with _stderr_fd_lock:
            if self._devnull_fd >= 0:
                os.close(self._devnull_fd)
                self._devnull_fd = -1
            if self._orig_fd >= 0:
                os.close(self._orig_fd)
                self._orig_fd = -1


@dataclass
class _StreamState:
    """One stream's echo canceller and its scratch buffers.

    The native SpeexEchoState holds the adaptive filter, so two speakers
    cannot share one: the filter converges on whoever spoke last and
    subtracts that estimate from everyone.
    """

    # None once destroyed. Cleared under `lock`, so a thread that resolved this
    # state before a concurrent reset() sees the None instead of calling into a
    # freed pointer.
    state: ctypes.c_void_p | None
    in_buf: Any
    out_buf: Any
    ref_buf: Any
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Whether speex_echo_playback() ran since the last capture — diagnostics.
    playback_fed: bool = False

    # Diagnostics — counters reset every _LOG_INTERVAL frames.
    process_count: int = 0
    ref_hits: int = 0
    ref_misses: int = 0
    refs_fed: int = 0
    total_in_energy: int = 0
    total_out_energy: int = 0


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
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if filter_length <= 0:
            raise ValueError("filter_length must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self._lib = _load_speexdsp()
        self._frame_size = frame_size
        self._filter_length = filter_length
        self._sample_rate = sample_rate
        self._frame_bytes = frame_size * 2  # 2 bytes per int16 sample

        # One echo canceller per stream, created on first use.
        self._streams: dict[str, _StreamState] = {}
        # Guards _streams itself — the per-stream lock guards its contents.
        self._streams_lock = threading.Lock()
        self._closed = False
        self._warned_formats: set[tuple[str, int, int, int]] = set()

        # Process-level file descriptors, not stream state: the suppressor
        # redirects fd 2 for the whole process, so one instance serves every
        # stream and closing it per stream would break the others.
        self._stderr = _StderrSuppressor()

        logger.info(
            "SpeexAEC init: frame_size=%d, filter_length=%d (%dms), sample_rate=%d",
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

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Remove echo from a captured (mic) audio frame."""
        if not self._matches_format(frame, direction="capture"):
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

        st = self._state_for(stream)
        if st is None:
            return frame

        with st.lock:
            if st.state is None:
                return frame  # reset() destroyed it between the lookup and here

            ctypes.memmove(st.in_buf, pcm_in, self._frame_bytes)

            if st.playback_fed:
                st.ref_hits += 1
            else:
                st.ref_misses += 1
            st.playback_fed = False

            # Suppress C stderr — SpeexDSP prints "No playback frame
            # available" when the ring buffer is empty (expected during
            # silence).
            with self._stderr:
                self._lib.speex_echo_capture(st.state, st.in_buf, st.out_buf)

            st.process_count += 1
            should_log = st.process_count % _LOG_INTERVAL == 0
            out_data = bytes(st.out_buf)

        # Energy diagnostics feed a DEBUG-level log, so production pays
        # nothing for them — the per-sample sums cost more than the echo
        # canceller's own numpy-free glue. Computed OUTSIDE the lock from the
        # in/out bytes: feed_reference() blocks on this lock from the
        # playback path, and the diagnostic must not extend its wait.
        # Accumulated over the full interval so the log reflects average
        # behaviour, not a single-frame snapshot.
        if logger.isEnabledFor(logging.DEBUG):
            in_view = memoryview(pcm_in)[: self._frame_bytes].cast("h")
            out_view = memoryview(out_data).cast("h")
            st.total_in_energy += sum(s * s for s in in_view)
            st.total_out_energy += sum(s * s for s in out_view)

        if should_log:
            self._log_stats(stream, st)

        metadata = dict(frame.metadata)
        metadata["echo_cancelled"] = True
        return AudioFrame(
            data=out_data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=metadata,
        )

    def feed_reference(self, frame: AudioFrame, stream: str) -> None:
        """Feed a reference (playback / TTS) frame for echo modelling.

        Calls ``speex_echo_playback()`` directly so the internal ring
        buffer tracks the speaker output timing.
        """
        if not self._matches_format(frame, direction="reference"):
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

        st = self._state_for(stream)
        if st is None:
            return

        with st.lock:
            if st.state is None:
                return  # reset() destroyed it between the lookup and here

            ctypes.memmove(st.ref_buf, pcm, n_bytes)
            with self._stderr:
                self._lib.speex_echo_playback(st.state, st.ref_buf)
            st.playback_fed = True
            st.refs_fed += 1

    def _state_for(self, stream: str) -> _StreamState | None:
        """Get or create this stream's echo canceller, unless closed."""
        with self._streams_lock:
            if self._closed:
                return None
            st = self._streams.get(stream)
            if st is None:
                fs = self._frame_size
                st = _StreamState(
                    state=self._create_state(),
                    in_buf=(ctypes.c_int16 * fs)(),
                    out_buf=(ctypes.c_int16 * fs)(),
                    ref_buf=(ctypes.c_int16 * fs)(),
                )
                self._streams[stream] = st
            return st

    def _matches_format(self, frame: AudioFrame, *, direction: str) -> bool:
        """Reject PCM that the mono int16 Speex state would misinterpret."""
        actual = (frame.sample_rate, frame.channels, frame.sample_width)
        expected = (self._sample_rate, 1, 2)
        if actual == expected:
            return True
        key = (direction, *actual)
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "Speex AEC %s format mismatch: got %dHz/%dch/%d-byte, expected "
                "%dHz/1ch/2-byte; %s frame ignored",
                direction,
                *actual,
                self._sample_rate,
                direction,
            )
        return False

    def _destroy(self, st: _StreamState) -> None:
        """Destroy one stream's native state, once, under its own lock.

        Holding the lock is what makes this safe against a capture already in
        flight on the audio thread: it either finishes first, or finds the
        state cleared and passes the frame through.
        """
        with st.lock:
            if st.state is not None:
                self._lib.speex_echo_state_destroy(st.state)
                st.state = None

    def reset(self, stream: str) -> None:
        """Destroy this stream's echo canceller and forget it."""
        with self._streams_lock:
            st = self._streams.pop(stream, None)
        if st is not None:
            self._destroy(st)

    def close(self) -> None:
        """Destroy every stream's echo canceller and release resources."""
        with self._streams_lock:
            self._closed = True
            states = list(self._streams.values())
            self._streams.clear()
        for st in states:
            self._destroy(st)
        self._stderr.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_stats(self, stream: str, st: _StreamState) -> None:
        """Log periodic AEC diagnostics for one stream (averaged over the interval)."""
        n = (self._frame_size * _LOG_INTERVAL) or 1
        in_rms = math.isqrt(st.total_in_energy // n)
        out_rms = math.isqrt(st.total_out_energy // n)

        if in_rms > 0:
            attenuation_db = 20 * math.log10(out_rms / in_rms) if out_rms > 0 else -99
        else:
            attenuation_db = 0.0

        logger.debug(
            "[AEC stats] stream=%s processed=%d ref_hits=%d ref_misses=%d "
            "refs_fed=%d | "
            "in_rms=%d out_rms=%d attenuation=%.1fdB",
            stream,
            st.process_count,
            st.ref_hits,
            st.ref_misses,
            st.refs_fed,
            in_rms,
            out_rms,
            attenuation_db,
        )

        # Reset interval counters.
        st.ref_hits = 0
        st.ref_misses = 0
        st.refs_fed = 0
        st.total_in_energy = 0
        st.total_out_energy = 0

    def _create_state(self) -> ctypes.c_void_p:
        state = self._lib.speex_echo_state_init(self._frame_size, self._filter_length)
        if not state:
            raise RuntimeError("speex_echo_state_init returned NULL")

        # Tell SpeexDSP the sample rate so it can tune its filters.
        sr = ctypes.c_int(self._sample_rate)
        self._lib.speex_echo_ctl(
            state,
            _SPEEX_ECHO_SET_SAMPLING_RATE,
            ctypes.byref(sr),
        )
        return ctypes.c_void_p(state)

    def __del__(self) -> None:
        # Construction may fail before native resources and locks exist.
        if hasattr(self, "_streams_lock"):
            self.close()

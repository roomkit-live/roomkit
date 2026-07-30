"""Acoustic Echo Cancellation provider using WebRTC AEC3.

Uses the ``aec-audio-processing`` package which wraps the WebRTC audio
processing module — the same AEC3 engine used by Chrome and Android.

Requires the ``aec-audio-processing`` pip dependency::

    pip install aec-audio-processing

Usage::

    from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider

    aec = WebRTCAECProvider(sample_rate=16000)
    config = AudioPipelineConfig(aec=aec)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.aec.base import AECProvider

logger = logging.getLogger("roomkit.voice.pipeline.aec.webrtc")

# WebRTC audio processing requires exactly 10 ms frames.
_WEBRTC_FRAME_MS = 10

# Log AEC stats every N process() calls (~1 s at 10 ms frames).
_LOG_INTERVAL = 100


def _import_webrtc() -> Any:
    """Import AudioProcessor, raising a clear error if missing."""
    try:
        from aec_audio_processing import AudioProcessor

        return AudioProcessor
    except ImportError as exc:
        raise ImportError(
            "aec-audio-processing is required for WebRTCAECProvider. "
            "Install it with: pip install aec-audio-processing"
        ) from exc


def _import_numpy() -> Any:
    """Import numpy, raising a clear error if missing."""
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for WebRTCAECProvider. Install it with: pip install numpy"
        ) from exc


@dataclass
class _StreamState:
    """One stream's WebRTC processor and its chunking buffers.

    AEC3 keeps the adaptive filter, the delay estimate and the double-talk
    detector inside the AudioProcessing object, so two speakers through one
    processor would each be measured against the other's reference.
    """

    ap: Any
    capture_buf: bytearray = field(default_factory=bytearray)
    ref_buf: bytearray = field(default_factory=bytearray)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Diagnostics
    process_count: int = 0
    ref_fed_count: int = 0
    total_in_energy: int = 0
    total_out_energy: int = 0


class WebRTCAECProvider(AECProvider):
    """AEC provider backed by WebRTC AEC3.

    WebRTC AEC3 is significantly more effective than Speex for real-world
    speaker+mic echo cancellation.  It includes nonlinear echo suppression,
    double-talk detection, and comfort noise generation.

    WebRTC requires exactly 10 ms audio frames.  This provider handles
    chunking transparently — callers can pass any frame size and the
    provider will buffer and process in 10 ms increments.

    Args:
        sample_rate: Audio sample rate in Hz (default 16000).
        channels: Number of audio channels (default 1, mono).
        stream_delay_ms: Estimated delay between speaker output and mic
            capture in milliseconds.  Helps the AEC align reference and
            capture for better cancellation.  Default 0.
        enable_ns: Also enable WebRTC noise suppression.  Default False.
        enable_agc: Also enable WebRTC automatic gain control.  Default False.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        stream_delay_ms: int = 0,
        enable_ns: bool = False,
        enable_agc: bool = False,
    ) -> None:
        # Resolved once: streams are created lazily on the audio thread, and
        # re-importing there would put a module lookup in the realtime path.
        self._ap_cls = _import_webrtc()
        self._np = _import_numpy()

        self._sample_rate = sample_rate
        self._channels = channels
        self._enable_ns = enable_ns
        self._enable_agc = enable_agc
        self._stream_delay_ms = stream_delay_ms

        # 10 ms frame: samples and bytes
        self._frame_samples = sample_rate * _WEBRTC_FRAME_MS // 1000
        self._frame_bytes = self._frame_samples * channels * 2  # int16

        # One processor per stream, created on first use.
        self._streams: dict[str, _StreamState] = {}
        # Guards _streams itself — the per-stream lock guards its contents.
        self._streams_lock = threading.Lock()

        # Closed instances refuse new streams rather than resurrecting.
        self._closed = False

        # When True, process() passes audio through without AEC processing.
        # Activated automatically when reference stops (TTS ends) and
        # deactivated when reference resumes (TTS starts).  Avoids the
        # stale adaptive filter suppressing user speech after playback.
        # A channel-wide mode driven by the TTS lifecycle, not adaptive
        # state — process() never mutates it, so it stays off _StreamState.
        self._bypass = True  # Start bypassed — no echo to cancel yet

        logger.info(
            "WebRTC AEC init: sample_rate=%d, channels=%d, "
            "frame=%d samples (%d ms), delay=%d ms, ns=%s, agc=%s",
            sample_rate,
            channels,
            self._frame_samples,
            _WEBRTC_FRAME_MS,
            stream_delay_ms,
            enable_ns,
            enable_agc,
        )

    # ------------------------------------------------------------------
    # AECProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "webrtc_aec3"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Remove echo from a captured (mic) audio frame."""
        if self._bypass:
            return frame  # no active playback — passthrough

        st = self._state_for(stream)
        if st is None:
            return frame  # closed

        pcm_in = frame.data
        output_chunks: list[bytes] = []
        in_processed: list[bytes] = []
        out_processed: list[bytes] = []
        fb = self._frame_bytes

        with st.lock:
            st.capture_buf.extend(pcm_in)
            while len(st.capture_buf) >= fb:
                chunk = bytes(st.capture_buf[:fb])
                del st.capture_buf[:fb]

                result = st.ap.process_stream(chunk)
                output_chunks.append(result)
                in_processed.append(chunk)
                out_processed.append(result)
                st.process_count += 1

        # Energy diagnostics OUTSIDE the lock: the PortAudio speaker callback
        # blocks on this lock in feed_reference(), and the previous per-sample
        # Python loop held it for the whole diagnostic — long enough to delay
        # the realtime audio thread. The lock now covers process_stream only.
        if in_processed:
            np = self._np
            in_s = np.frombuffer(b"".join(in_processed), dtype="<i2").astype(np.int64)
            out_s = np.frombuffer(b"".join(out_processed), dtype="<i2").astype(np.int64)
            st.total_in_energy += int(in_s @ in_s)
            st.total_out_energy += int(out_s @ out_s)

        if st.process_count > 0 and st.process_count % _LOG_INTERVAL == 0:
            self._log_stats(stream, st)

        if not output_chunks:
            return frame

        return AudioFrame(
            data=b"".join(output_chunks),
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=dict(frame.metadata),
        )

    def set_active(self, active: bool) -> None:
        """Enable or disable AEC processing.

        When *active* is ``False``, ``process()`` passes audio through
        without echo cancellation (bypass mode).  Call with ``True``
        when TTS playback starts, and ``False`` when it ends.

        Channel-wide: bypass follows the TTS lifecycle, which is not a
        per-stream property.
        """
        was_bypass = self._bypass
        self._bypass = not active
        if was_bypass != (not active):
            logger.info(
                "AEC %s (streams=%d)", "activated" if active else "bypassed", len(self._streams)
            )

    def feed_reference(self, frame: AudioFrame, stream: str) -> None:
        """Feed a reference (playback / TTS) frame for echo modelling."""
        st = self._state_for(stream)
        if st is None:
            return  # closed

        pcm = frame.data
        fb = self._frame_bytes
        fed_this_call = 0

        with st.lock:
            st.ref_buf.extend(pcm)
            while len(st.ref_buf) >= fb:
                chunk = bytes(st.ref_buf[:fb])
                del st.ref_buf[:fb]
                st.ap.process_reverse_stream(chunk)
                st.ref_fed_count += 1
                fed_this_call += 1
            total_fed = st.ref_fed_count

        if fed_this_call > 0 and (total_fed <= 3 or total_fed % 100 == 0):
            # First feeds at INFO (wiring confirmation); the periodic tick
            # at DEBUG — with a continuous playback-time reference (silence
            # included) it would otherwise log once a second forever.
            level = logging.INFO if total_fed <= 3 else logging.DEBUG
            logger.log(
                level,
                "AEC reference: %d chunks fed to stream=%s (total=%d), bypass=%s",
                fed_this_call,
                stream,
                total_fed,
                self._bypass,
            )

    def _new_processor(self) -> Any:
        """Create and configure a fresh WebRTC AudioProcessing object."""
        ap = self._ap_cls(
            enable_aec=True,
            enable_ns=self._enable_ns,
            enable_agc=self._enable_agc,
        )
        ap.set_stream_format(self._sample_rate, self._channels, self._sample_rate, self._channels)
        ap.set_reverse_stream_format(self._sample_rate, self._channels)
        if self._stream_delay_ms > 0:
            ap.set_stream_delay(self._stream_delay_ms)
        return ap

    def _state_for(self, stream: str) -> _StreamState | None:
        """Get or create this stream's processor, or None once closed."""
        with self._streams_lock:
            if self._closed:
                return None
            st = self._streams.get(stream)
            if st is None:
                st = _StreamState(ap=self._new_processor())
                self._streams[stream] = st
            return st

    def reset(self, stream: str) -> None:
        """Drop this stream's processor, discarding its adaptive filter.

        Critical after barge-in: once TTS stops the old filter is stale and
        will suppress the user's voice for many seconds while it reconverges.
        The next frame builds a fresh processor.
        """
        with self._streams_lock:
            existed = self._streams.pop(stream, None) is not None
        if existed:
            logger.info("WebRTC AEC reset for stream=%s (adaptive filter cleared)", stream)

    def close(self) -> None:
        """Release every stream's processor."""
        with self._streams_lock:
            self._closed = True
            self._streams.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_stats(self, stream: str, st: _StreamState) -> None:
        """Log periodic AEC diagnostics for one stream."""
        n = self._frame_samples * _LOG_INTERVAL or 1
        in_rms = math.isqrt(st.total_in_energy // n)
        out_rms = math.isqrt(st.total_out_energy // n)

        if in_rms > 0:
            attenuation_db = 20 * math.log10(out_rms / in_rms) if out_rms > 0 else -99
        else:
            attenuation_db = 0.0

        logger.info(
            "AEC stats: stream=%s processed=%d refs_fed=%d bypass=%s | "
            "in_rms=%d out_rms=%d attenuation=%.1fdB",
            stream,
            st.process_count,
            st.ref_fed_count,
            self._bypass,
            in_rms,
            out_rms,
            attenuation_db,
        )

        # Reset interval counters.
        st.total_in_energy = 0
        st.total_out_energy = 0

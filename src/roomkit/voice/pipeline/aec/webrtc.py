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
        ap_cls = _import_webrtc()

        self._sample_rate = sample_rate
        self._channels = channels

        # 10 ms frame: samples and bytes
        self._frame_samples = sample_rate * _WEBRTC_FRAME_MS // 1000
        self._frame_bytes = self._frame_samples * channels * 2  # int16

        # Create WebRTC audio processor
        self._ap = ap_cls(
            enable_aec=True,
            enable_ns=enable_ns,
            enable_agc=enable_agc,
        )
        self._ap.set_stream_format(sample_rate, channels, sample_rate, channels)
        self._ap.set_reverse_stream_format(sample_rate, channels)
        if stream_delay_ms > 0:
            self._ap.set_stream_delay(stream_delay_ms)

        # Lock — protects _ap across mic and speaker callback threads.
        self._lock = threading.Lock()

        # Buffers for chunking arbitrary-size frames into 10 ms blocks.
        self._capture_buf = bytearray()
        self._ref_buf = bytearray()

        # Diagnostics
        self._process_count = 0
        self._ref_fed_count = 0
        self._total_in_energy = 0
        self._total_out_energy = 0

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

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Remove echo from a captured (mic) audio frame."""
        pcm_in = frame.data
        self._capture_buf.extend(pcm_in)

        output_chunks: list[bytes] = []
        fb = self._frame_bytes

        with self._lock:
            while len(self._capture_buf) >= fb:
                chunk = bytes(self._capture_buf[:fb])
                del self._capture_buf[:fb]

                result = self._ap.process_stream(chunk)
                output_chunks.append(result)

                self._process_count += 1

                # Energy tracking for diagnostics
                in_energy = sum(
                    int.from_bytes(chunk[i : i + 2], "little", signed=True) ** 2
                    for i in range(0, fb, 2)
                )
                out_energy = sum(
                    int.from_bytes(result[i : i + 2], "little", signed=True) ** 2
                    for i in range(0, fb, 2)
                )
                self._total_in_energy += in_energy
                self._total_out_energy += out_energy

        if self._process_count > 0 and self._process_count % _LOG_INTERVAL == 0:
            self._log_stats()

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

    def feed_reference(self, frame: AudioFrame) -> None:
        """Feed a reference (playback / TTS) frame for echo modelling."""
        pcm = frame.data
        self._ref_buf.extend(pcm)
        fb = self._frame_bytes

        with self._lock:
            while len(self._ref_buf) >= fb:
                chunk = bytes(self._ref_buf[:fb])
                del self._ref_buf[:fb]
                self._ap.process_reverse_stream(chunk)
                self._ref_fed_count += 1

    def reset(self) -> None:
        """Reset internal state."""
        self._capture_buf.clear()
        self._ref_buf.clear()
        self._process_count = 0
        self._ref_fed_count = 0
        self._total_in_energy = 0
        self._total_out_energy = 0

    def close(self) -> None:
        """Release resources."""
        self._ap = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_stats(self) -> None:
        """Log periodic AEC diagnostics."""
        n = self._frame_samples * _LOG_INTERVAL or 1
        in_rms = math.isqrt(self._total_in_energy // n)
        out_rms = math.isqrt(self._total_out_energy // n)

        if in_rms > 0:
            attenuation_db = 20 * math.log10(out_rms / in_rms) if out_rms > 0 else -99
        else:
            attenuation_db = 0.0

        logger.debug(
            "[WebRTC AEC stats] processed=%d refs_fed=%d | "
            "in_rms=%d out_rms=%d attenuation=%.1fdB",
            self._process_count,
            self._ref_fed_count,
            in_rms,
            out_rms,
            attenuation_db,
        )

        # Reset interval counters.
        self._total_in_energy = 0
        self._total_out_energy = 0

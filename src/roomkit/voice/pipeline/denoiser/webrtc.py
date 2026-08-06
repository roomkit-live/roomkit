"""Continuous WebRTC noise suppression without acoustic echo cancellation."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.denoiser.base import DenoiserProvider

logger = logging.getLogger("roomkit.voice.pipeline.denoiser.webrtc")

_FRAME_MS = 10


def _import_processor() -> Any:
    try:
        from aec_audio_processing import AudioProcessor

        return AudioProcessor
    except ImportError as exc:
        raise ImportError(
            "aec-audio-processing is required for WebRTCNoiseSuppressorProvider. "
            "Install it with: pip install roomkit[webrtc-aec]"
        ) from exc


@dataclass
class _StreamState:
    processor: Any
    input_buffer: bytearray = field(default_factory=bytearray)
    output_buffer: bytearray = field(default_factory=bytearray)
    chunking: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class WebRTCNoiseSuppressorProvider(DenoiserProvider):
    """WebRTC noise suppression that runs continuously on the inbound path.

    WebRTC consumes exact 10 ms PCM16 blocks. Arbitrary caller chunk sizes are
    converted to a fixed one-block-delay stream so every input byte has exactly
    one output byte and buffered audio is never emitted twice.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be between 8000 and 192000 Hz")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        self._processor_cls = _import_processor()
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_bytes = sample_rate * _FRAME_MS // 1000 * channels * 2
        self._streams: dict[str, _StreamState] = {}
        self._streams_lock = threading.Lock()
        self._warned_formats: set[tuple[int, int, int, int]] = set()
        self._closed = False

    @property
    def name(self) -> str:
        return "webrtc_ns"

    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        if not self._matches_format(frame):
            return frame
        try:
            state = self._state_for(stream)
        except Exception:
            logger.warning("WebRTC NS initialization failed; frame bypassed", exc_info=True)
            return frame
        if state is None:
            return frame

        try:
            with state.lock:
                output = self._process_locked(state, frame.data)
        except Exception:
            logger.warning("WebRTC NS processing failed; frame bypassed", exc_info=True)
            with state.lock:
                state.input_buffer.clear()
                state.output_buffer.clear()
                state.chunking = False
            return frame

        metadata = dict(frame.metadata)
        metadata["noise_suppressed"] = True
        return AudioFrame(
            data=output,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            sample_width=frame.sample_width,
            timestamp_ms=frame.timestamp_ms,
            metadata=metadata,
        )

    def _process_locked(self, state: _StreamState, data: bytes) -> bytes:
        if not state.chunking and not state.input_buffer and len(data) % self._frame_bytes == 0:
            return b"".join(
                self._process_chunk(
                    state,
                    data[offset : offset + self._frame_bytes],
                )
                for offset in range(0, len(data), self._frame_bytes)
            )

        if not state.chunking:
            state.chunking = True
            state.output_buffer.extend(b"\x00" * self._frame_bytes)
        state.input_buffer.extend(data)
        while len(state.input_buffer) >= self._frame_bytes:
            chunk = bytes(state.input_buffer[: self._frame_bytes])
            del state.input_buffer[: self._frame_bytes]
            state.output_buffer.extend(self._process_chunk(state, chunk))
        output = bytes(state.output_buffer[: len(data)])
        del state.output_buffer[: len(data)]
        return output

    @staticmethod
    def _process_chunk(state: _StreamState, chunk: bytes) -> bytes:
        output = state.processor.process_stream(chunk)
        if len(output) != len(chunk):
            raise RuntimeError(
                f"WebRTC NS returned {len(output)} bytes for a {len(chunk)}-byte block"
            )
        return output

    def reset(self, stream: str) -> None:
        with self._streams_lock:
            self._streams.pop(stream, None)

    def close(self) -> None:
        with self._streams_lock:
            self._closed = True
            self._streams.clear()

    def _state_for(self, stream: str) -> _StreamState | None:
        with self._streams_lock:
            if self._closed:
                return None
            state = self._streams.get(stream)
            if state is None:
                processor = self._processor_cls(
                    enable_aec=False,
                    enable_ns=True,
                    enable_agc=False,
                )
                processor.set_stream_format(
                    self._sample_rate,
                    self._channels,
                    self._sample_rate,
                    self._channels,
                )
                state = _StreamState(processor=processor)
                self._streams[stream] = state
            return state

    def _matches_format(self, frame: AudioFrame) -> bool:
        expected = (self._sample_rate, self._channels, 2)
        actual = (frame.sample_rate, frame.channels, frame.sample_width)
        valid = actual == expected and len(frame.data) % (2 * self._channels) == 0
        if valid:
            return True
        key = (*actual, len(frame.data))
        if key not in self._warned_formats:
            self._warned_formats.add(key)
            logger.warning(
                "WebRTC NS requires %dHz/%dch/2-byte aligned PCM; got "
                "%dHz/%dch/%d-byte with %d bytes; frame ignored",
                *expected[:2],
                *actual,
                len(frame.data),
            )
        return False

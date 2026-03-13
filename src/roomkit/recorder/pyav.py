"""PyAV-based media recorder — muxes audio + video into a single MP4."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from roomkit.recorder.base import (
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    validate_storage_path,
)

logger = logging.getLogger("roomkit.recorder.pyav")


def _import_av() -> Any:
    """Import PyAV, raising a clear error if missing."""
    try:
        import av

        return av
    except ImportError as exc:
        raise ImportError(
            "av (PyAV) is required for PyAVMediaRecorder. Install with: pip install roomkit[video]"
        ) from exc


def _resolve_video_codec(codec: str) -> str:
    """Resolve 'auto' to a concrete video codec name."""
    if codec != "auto":
        return codec
    try:
        av = _import_av()
        av.codec.Codec("h264_nvenc", "w")
        return "h264_nvenc"
    except Exception:
        return "libx264"


@dataclass
class _TrackState:
    """Per-track muxing state."""

    track: RecordingTrack
    stream: Any = None
    frame_count: int = 0
    ready: bool = False


@dataclass
class _RecordingState:
    """Per-recording muxing state."""

    config: MediaRecordingConfig = field(default_factory=MediaRecordingConfig)
    tracks: dict[str, _TrackState] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    container: Any = None
    path: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    encoding_started: bool = False


class PyAVMediaRecorder(MediaRecorder):
    """Muxes audio (AAC) and video (H.264) into a single MP4 via PyAV.

    Data is buffered per-track until every registered track has
    received at least one ``on_data`` call.  At that point the
    container and all streams are created at once — so the MP4 header
    is written with full knowledge of every stream's parameters.

    A/V sync uses a single ``time.monotonic()`` clock: both audio and
    video PTS are derived from wall-clock elapsed time since encoding
    started.  This keeps streams aligned regardless of pipeline
    latency or backend timing differences.

    Thread-safe: each recording has its own lock so video frames from
    capture threads and audio from the event loop can write concurrently.
    """

    @property
    def name(self) -> str:
        return "pyav"

    def __init__(self) -> None:
        self._av = _import_av()
        self._recordings: dict[str, _RecordingState] = {}

    # -- lifecycle -----------------------------------------------------------

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        handle_id = uuid4().hex[:12]
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        storage = config.storage or os.path.join(os.getcwd(), "recordings")
        resolved = validate_storage_path(storage)
        path = os.path.join(resolved, f"room_{handle_id}_{ts}.{config.format}")

        state = _RecordingState(config=config, path=path)
        self._recordings[handle_id] = state

        handle = MediaRecordingHandle(
            id=handle_id,
            room_id="",
            state="recording",
            started_at=state.started_at,
            path=path,
        )
        logger.info("PyAV recording created: %s → %s", handle_id, path)
        return handle

    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        state = self._recordings.pop(handle.id, None)
        if state is None:
            return MediaRecordingResult(id=handle.id)

        result_tracks: list[RecordingTrack] = []
        with state.lock:
            if state.container is not None:
                for ts in state.tracks.values():
                    result_tracks.append(ts.track)
                    if ts.stream is not None and ts.frame_count > 0:
                        try:
                            for packet in ts.stream.encode(None):
                                state.container.mux(packet)
                        except Exception:
                            logger.debug("Flush error for track %s", ts.track.id)
                state.container.close()
                state.container = None

        handle.state = "stopped"
        duration = (datetime.now(UTC) - state.started_at).total_seconds()
        size_bytes = 0
        if os.path.exists(state.path):
            size_bytes = os.path.getsize(state.path)
        return MediaRecordingResult(
            id=handle.id,
            url=state.path,
            duration_seconds=duration,
            tracks=result_tracks,
            format=state.config.format,
            size_bytes=size_bytes,
        )

    # -- track management ----------------------------------------------------

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        state = self._recordings.get(handle.id)
        if state is None:
            return
        with state.lock:
            state.tracks[track.id] = _TrackState(track=track)
        logger.debug("Track registered: %s (%s)", track.id, track.kind)

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        state = self._recordings.get(handle.id)
        if state is None:
            return
        with state.lock:
            ts = state.tracks.pop(track.id, None)
            if ts and ts.stream and state.container and ts.frame_count > 0:
                try:
                    for packet in ts.stream.encode(None):
                        state.container.mux(packet)
                except Exception:
                    logger.debug("Flush error on track removal: %s", track.id)

    # -- data ingestion ------------------------------------------------------

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        state = self._recordings.get(handle.id)
        if state is None:
            return

        with state.lock:
            ts = state.tracks.get(track.id)
            if ts is None:
                return

            if not state.encoding_started:
                ts.ready = True
                if all(t.ready for t in state.tracks.values()):
                    self._start_encoding(state, track, ts, data)
                return

            self._encode_frame(state, ts, track, data)
            ts.frame_count += 1

    # -- internal helpers ----------------------------------------------------

    def _start_encoding(
        self,
        state: _RecordingState,
        trigger_track: RecordingTrack,
        trigger_ts: _TrackState,
        trigger_data: bytes,
    ) -> None:
        """Create container + all streams, encode the triggering frame.

        Only the frame that completed readiness is encoded — all
        earlier data from other tracks is discarded (pre-roll).
        """
        state.container = self._av.open(state.path, mode="w")

        for ts in state.tracks.values():
            ts.stream = self._create_stream(state.container, ts.track, state.config)

        state.encoding_started = True
        logger.debug("Encoding started with %d streams", len(state.tracks))

        # Encode only the triggering frame (the one that made all ready)
        self._encode_frame(state, trigger_ts, trigger_track, trigger_data)
        trigger_ts.frame_count += 1

    def _create_stream(
        self,
        container: Any,
        track: RecordingTrack,
        config: MediaRecordingConfig,
    ) -> Any:
        """Add a stream to the container with known parameters."""
        if track.kind == "video":
            w = track.width or 640
            h = track.height or 480
            codec = _resolve_video_codec(config.video_codec)
            try:
                stream = container.add_stream(codec, rate=config.video_fps)
                stream.pix_fmt = "yuv420p"
                stream.width = w
                stream.height = h
            except Exception:
                if codec != "libx264":
                    logger.info("Codec %s failed, falling back to libx264", codec)
                    stream = container.add_stream("libx264", rate=config.video_fps)
                    stream.pix_fmt = "yuv420p"
                    stream.width = w
                    stream.height = h
                else:
                    raise
            return stream
        # audio
        rate = track.sample_rate or config.audio_sample_rate
        stream = container.add_stream(config.audio_codec, rate=rate)
        stream.layout = "mono"
        return stream

    def _encode_frame(
        self,
        state: _RecordingState,
        ts: _TrackState,
        track: RecordingTrack,
        data: bytes,
    ) -> None:
        """Encode a single frame and mux resulting packets."""
        if track.kind == "video":
            self._write_video(state, ts, data)
        elif track.kind == "audio":
            self._write_audio(state, ts, data)

    def _write_video(self, state: _RecordingState, ts: _TrackState, data: bytes) -> None:
        """Encode and mux a video frame.

        PTS = frame_count — advances by 1 per frame at the stream
        rate.  Immune to event-loop scheduling jitter.
        """
        import numpy as np

        stream = ts.stream
        w, h = stream.width, stream.height
        expected = w * h * 3
        if len(data) < expected:
            data = data + b"\x00" * (expected - len(data))
        arr = np.frombuffer(data[:expected], dtype=np.uint8).reshape(h, w, 3)
        frame = self._av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = ts.frame_count
        for packet in stream.encode(frame):
            state.container.mux(packet)

    def _write_audio(self, state: _RecordingState, ts: _TrackState, data: bytes) -> None:
        """Encode and mux an audio frame.

        PTS = frame_count * samples_per_frame — advances by the
        number of samples per frame at the stream rate.
        """
        import numpy as np

        stream = ts.stream
        samples = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        frame = self._av.audio.frame.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = stream.rate
        frame.pts = ts.frame_count * len(samples[0])
        for packet in stream.encode(frame):
            state.container.mux(packet)

    def close(self) -> None:
        for handle_id in list(self._recordings):
            state = self._recordings.pop(handle_id, None)
            if state is not None and state.container is not None:
                with state.lock:
                    if state.container is not None:
                        state.container.close()
                        state.container = None

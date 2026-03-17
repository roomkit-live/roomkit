"""PyAV-based media recorder — muxes audio + video into a single MP4."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from roomkit.recorder._pyav_mux import (
    ENCODED_VIDEO_CODECS,
    compute_pts,
    create_stream,
    h264_annex_b,
    import_av,
    probe_encoded_dimensions,
    safe_mux,
)
from roomkit.recorder.base import (
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    validate_storage_path,
)

logger = logging.getLogger("roomkit.recorder.pyav")


@dataclass
class _TrackState:
    """Per-track muxing state."""

    track: RecordingTrack
    stream: Any = None
    frame_count: int = 0
    last_pts: int = -1  # monotonic guard for PTS
    video_fps: int = 30  # stream rate for video timestamp→PTS conversion
    decoder: Any = None  # codec context for decoding encoded video
    pending: list[tuple[bytes, float | None]] = field(default_factory=list)
    t0_ms: float = 0.0  # per-track timestamp origin
    mux_error_logged: bool = False  # per-track first-error suppression flag


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
    """Muxes audio (AAC) and video (H.264/VP9) into a single MP4 via PyAV.

    MP4 requires all streams to exist before any data is muxed.
    Streams are created when all registered tracks are "ready":

    - Audio tracks are ready immediately (codec is known at registration).
    - Video tracks become ready when the first frame arrives (the tap
      populates ``track.codec`` from the frame).

    If only some tracks have data, frames are buffered until all tracks
    are ready.  Once ready, all streams are created and buffered frames
    are flushed.

    A/V sync strategy: both audio and video PTS are derived from each
    frame's capture timestamp (set at acquisition time), so playback
    speed matches real time regardless of the configured stream rate
    or actual capture FPS.  Falls back to frame-count PTS when
    timestamps are unavailable.

    Thread-safe: each recording has its own lock so video frames from
    capture threads and audio from the event loop can write concurrently.
    """

    @property
    def name(self) -> str:
        return "pyav"

    def __init__(self) -> None:
        self._av = import_av()
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
                    ts.decoder = None
                frame_counts = {
                    t.id: state.tracks[t.id].frame_count
                    for t in result_tracks
                    if t.id in state.tracks
                }
                logger.info(
                    "Closing container %s (%d tracks, %s frames)",
                    state.path,
                    len(result_tracks),
                    frame_counts,
                )
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
            ts = _TrackState(track=track)
            state.tracks[track.id] = ts

            # If encoding already started, we can't add streams to an MP4
            # container after muxing. Log a warning — to fix, ensure all
            # channels connect before video capture starts, or increase
            # the encoding_delay_seconds on MediaRecordingConfig.
            if state.encoding_started and state.container is not None:
                logger.warning(
                    "Late track %s (%s) added after encoding started — "
                    "audio won't be recorded. Connect voice before video capture.",
                    track.id, track.kind,
                )

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
                ts.pending.append((data, timestamp_ms))
                # Wait for all tracks AND a minimum startup delay so late
                # tracks (e.g. voice connecting after video) can register.
                min_tracks = state.config.min_tracks if hasattr(state.config, "min_tracks") else 1
                if len(state.tracks) >= min_tracks and self._all_tracks_ready(state):
                    self._start_encoding(state)
                return

            # Set per-track t0 on first frame if not set from pending
            if ts.frame_count == 0 and timestamp_ms is not None:
                ts.t0_ms = timestamp_ms
            self._encode_frame(state, ts, track, data, timestamp_ms)
            ts.frame_count += 1

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _track_ready(ts: _TrackState) -> bool:
        if ts.track.kind != "video":
            return True
        if ts.track.codec:
            return True
        if ts.track.width is not None and ts.track.height is not None:
            return True
        return bool(ts.pending)

    def _all_tracks_ready(self, state: _RecordingState) -> bool:
        return all(self._track_ready(ts) for ts in state.tracks.values())

    def _start_encoding(self, state: _RecordingState) -> None:
        # Probe dimensions for encoded video tracks
        for ts in state.tracks.values():
            if (
                ts.track.kind == "video"
                and ts.track.codec in ENCODED_VIDEO_CODECS
                and ts.track.width is None
                and ts.pending
            ):
                dims = probe_encoded_dimensions(
                    self._av,
                    ts.pending,
                    ts.track.codec,
                )
                if dims:
                    ts.track.width, ts.track.height = dims
                    logger.info(
                        "Probed video %s: %s %dx%d",
                        ts.track.id,
                        ts.track.codec,
                        dims[0],
                        dims[1],
                    )

        # Skip video tracks whose dimensions are still unknown
        skipped: set[str] = set()
        for ts in state.tracks.values():
            if (
                ts.track.kind == "video"
                and ts.track.codec in ENCODED_VIDEO_CODECS
                and (ts.track.width is None or ts.track.height is None)
            ):
                skipped.add(ts.track.id)
                logger.warning(
                    "Skipping video track %s — unknown dimensions",
                    ts.track.id,
                )

        # Disable strict interleaving — video encoders (libx264) buffer
        # frames, so audio can advance well ahead before any video packets
        # are produced.  Without this, the MP4 muxer rejects audio with
        # EINVAL once the interleave gap exceeds ~1 second.
        state.container = self._av.open(
            state.path,
            mode="w",
            options={"max_interleave_delta": "0"},
        )
        for ts in state.tracks.values():
            if ts.track.id not in skipped:
                ts.stream = create_stream(
                    state.container,
                    ts.track,
                    state.config,
                )
                if ts.track.kind == "video":
                    ts.video_fps = state.config.video_fps

        state.encoding_started = True

        # Per-track t0: each stream starts at PTS=0 from its own first
        # frame.  This avoids A/V offset caused by startup timing
        # differences (mic starts before camera, audio RTP arrives
        # before video RTP).
        now = time.monotonic() * 1000
        for ts in state.tracks.values():
            first_ts: float | None = None
            for _, ts_ms in ts.pending:
                if ts_ms is not None and (first_ts is None or ts_ms < first_ts):
                    first_ts = ts_ms
            ts.t0_ms = first_ts if first_ts is not None else now
        pending_counts = {ts.track.id: len(ts.pending) for ts in state.tracks.values()}
        logger.info(
            "Encoding started: pending=%s",
            pending_counts,
        )

        # Flush buffered frames
        for ts in state.tracks.values():
            if ts.stream is None:
                ts.pending.clear()
                continue
            for data, ts_ms in ts.pending:
                try:
                    self._encode_frame(state, ts, ts.track, data, ts_ms)
                    ts.frame_count += 1
                except Exception:
                    logger.error(
                        "Flush failed for track %s at frame %d",
                        ts.track.id,
                        ts.frame_count,
                        exc_info=True,
                    )
                    break
            ts.pending.clear()

    def _encode_frame(
        self,
        state: _RecordingState,
        ts: _TrackState,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        if track.kind == "video":
            self._write_video(state, ts, data, timestamp_ms)
        elif track.kind == "audio":
            self._write_audio(state, ts, data, timestamp_ms)

    def _write_video(
        self,
        state: _RecordingState,
        ts: _TrackState,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        if ts.stream is None:
            return
        if ts.track.codec in ENCODED_VIDEO_CODECS:
            self._write_encoded_video(state, ts, data, timestamp_ms)
        else:
            self._write_raw_video(state, ts, data, timestamp_ms)

    def _write_raw_video(
        self,
        state: _RecordingState,
        ts: _TrackState,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        import numpy as np

        w, h = ts.stream.width, ts.stream.height
        expected = w * h * 3
        if len(data) < expected:
            data = data + b"\x00" * (expected - len(data))
        arr = np.frombuffer(data[:expected], dtype=np.uint8).reshape(h, w, 3)
        frame = self._av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = compute_pts(
            timestamp_ms,
            ts.t0_ms,
            ts.video_fps,
            ts.last_pts,
            ts.frame_count,
        )
        ts.last_pts = frame.pts
        safe_mux(
            ts.stream,
            state.container,
            frame,
            ts,
            state.path,
            label="raw_video",
        )

    def _write_encoded_video(
        self,
        state: _RecordingState,
        ts: _TrackState,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        if ts.stream is None:
            return
        if ts.decoder is None:
            ts.decoder = self._av.CodecContext.create(ts.track.codec, "r")

        # Each data blob is a single NAL from RTP depacketization
        raw = h264_annex_b(data) if ts.track.codec == "h264" else data
        try:
            decoded_frames = ts.decoder.decode(self._av.Packet(raw))
        except Exception:
            logger.debug("Decode error for track %s", ts.track.id)
            return

        for decoded_frame in decoded_frames:
            decoded_frame.pts = compute_pts(
                timestamp_ms,
                ts.t0_ms,
                ts.video_fps,
                ts.last_pts,
                ts.frame_count,
            )
            ts.last_pts = decoded_frame.pts
            safe_mux(
                ts.stream,
                state.container,
                decoded_frame,
                ts,
                state.path,
                label=f"encoded_video:{ts.track.codec}",
            )

    def _write_audio(
        self,
        state: _RecordingState,
        ts: _TrackState,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        import numpy as np

        stream = ts.stream
        if stream is None:
            return
        samples = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        frame = self._av.audio.frame.AudioFrame.from_ndarray(
            samples,
            format="s16",
            layout="mono",
        )
        frame.sample_rate = stream.rate
        frame.pts = compute_pts(
            timestamp_ms,
            ts.t0_ms,
            stream.rate,
            ts.last_pts,
            ts.frame_count * len(samples[0]),
        )
        ts.last_pts = frame.pts
        safe_mux(
            stream,
            state.container,
            frame,
            ts,
            state.path,
            label="audio",
        )

    def close(self) -> None:
        for handle_id in list(self._recordings):
            state = self._recordings.pop(handle_id, None)
            if state is not None and state.container is not None:
                with state.lock:
                    if state.container is not None:
                        state.container.close()
                        state.container = None

"""Tests for PyAV-based room-level media recorder."""

from __future__ import annotations

import os
from fractions import Fraction

import pytest

from roomkit.recorder.base import (
    MediaRecordingConfig,
    RecordingTrack,
)

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


def _get_recorder():
    from roomkit.recorder.pyav import PyAVMediaRecorder

    return PyAVMediaRecorder()


class TestPyAVMediaRecorder:
    def test_name(self) -> None:
        recorder = _get_recorder()
        assert recorder.name == "pyav"

    def test_audio_only_recording(self, tmp_path: object) -> None:
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            audio_codec="aac",
            audio_sample_rate=16000,
        )
        handle = recorder.on_recording_start(config)
        assert handle.state == "recording"

        track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="voice-1",
            participant_id="user-1",
            codec="pcm_s16le",
            sample_rate=16000,
        )
        recorder.on_track_added(handle, track)

        # Feed 10 frames of 160 samples (10ms at 16kHz)
        for i in range(10):
            pcm = np.zeros(160, dtype=np.int16).tobytes()
            recorder.on_data(handle, track, pcm, i * 10.0)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert os.path.exists(result.url)
        assert result.format == "mp4"
        assert len(result.tracks) == 1
        assert result.tracks[0].kind == "audio"

    def test_video_only_recording(self, tmp_path: object) -> None:
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_codec="libx264",
        )
        handle = recorder.on_recording_start(config)

        track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="video-1",
            width=64,
            height=48,
        )
        recorder.on_track_added(handle, track)

        # Feed 5 RGB frames
        for i in range(5):
            rgb = np.zeros((48, 64, 3), dtype=np.uint8).tobytes()
            recorder.on_data(handle, track, rgb, i * 66.6)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert os.path.exists(result.url)

    def test_mixed_audio_video(self, tmp_path: object) -> None:
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_codec="libx264",
            audio_codec="aac",
            audio_sample_rate=16000,
        )
        handle = recorder.on_recording_start(config)

        audio_track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="voice-1",
            sample_rate=16000,
        )
        video_track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="video-1",
            width=64,
            height=48,
        )
        recorder.on_track_added(handle, audio_track)
        recorder.on_track_added(handle, video_track)

        # Interleave audio and video
        for i in range(5):
            pcm = np.zeros(160, dtype=np.int16).tobytes()
            recorder.on_data(handle, audio_track, pcm, i * 10.0)
            rgb = np.zeros((48, 64, 3), dtype=np.uint8).tobytes()
            recorder.on_data(handle, video_track, rgb, i * 66.6)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert len(result.tracks) == 2

    def test_track_removal_flushes(self, tmp_path: object) -> None:
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            audio_codec="aac",
        )
        handle = recorder.on_recording_start(config)
        track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="v1",
            sample_rate=16000,
        )
        recorder.on_track_added(handle, track)

        pcm = np.zeros(160, dtype=np.int16).tobytes()
        recorder.on_data(handle, track, pcm, 0.0)

        # Remove track (should flush encoder)
        recorder.on_track_removed(handle, track)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0

    def test_close_cleans_up(self, tmp_path: object) -> None:
        recorder = _get_recorder()
        config = MediaRecordingConfig(storage=str(tmp_path))
        handle = recorder.on_recording_start(config)

        track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="v1",
            sample_rate=16000,
        )
        recorder.on_track_added(handle, track)
        pcm = np.zeros(160, dtype=np.int16).tobytes()
        recorder.on_data(handle, track, pcm, 0.0)

        recorder.close()
        # After close, internal state should be empty
        assert len(recorder._recordings) == 0

    def test_encoded_video_h264(self, tmp_path: object) -> None:
        """Encoded H.264 NAL data is decoded and re-encoded to output."""
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_codec="libx264",
            video_fps=30,
        )
        handle = recorder.on_recording_start(config)

        track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="voice-1",
            codec="h264",
            # No width/height — should be learned from decoded frames
        )
        recorder.on_track_added(handle, track)

        # Generate valid H.264 NALs by encoding a frame with PyAV
        enc = av.CodecContext.create("libx264", "w")
        enc.width = 64
        enc.height = 48
        enc.pix_fmt = "yuv420p"
        enc.time_base = Fraction(1, 30)
        enc.open()

        for i in range(5):
            frame = av.VideoFrame(64, 48, "yuv420p")
            frame.pts = i
            for pkt in enc.encode(frame):
                # Strip Annex B start codes — _write_encoded_video prepends them
                nal_data = bytes(pkt)
                recorder.on_data(handle, track, nal_data, i * 33.3)

        # Flush encoder to get remaining packets
        for pkt in enc.encode(None):
            nal_data = bytes(pkt)
            recorder.on_data(handle, track, nal_data, 5 * 33.3)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert os.path.exists(result.url)

        # Track dimensions should have been learned from decoded frames
        assert track.width == 64
        assert track.height == 48

    def test_encoded_video_codec_populated(self, tmp_path: object) -> None:
        """Track codec is set correctly for encoded video."""
        recorder = _get_recorder()
        config = MediaRecordingConfig(storage=str(tmp_path), video_codec="libx264")
        handle = recorder.on_recording_start(config)

        track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="voice-1",
            codec="h264",
        )
        recorder.on_track_added(handle, track)
        assert track.codec == "h264"

    def test_stop_without_data(self, tmp_path: object) -> None:
        """Stop a recording that never received any data."""
        recorder = _get_recorder()
        config = MediaRecordingConfig(storage=str(tmp_path))
        handle = recorder.on_recording_start(config)
        result = recorder.on_recording_stop(handle)
        assert result.id == handle.id
        # No container was ever opened, so no file on disk
        assert result.size_bytes == 0

    def test_late_audio_after_video_24khz(self, tmp_path: object) -> None:
        """Audio arriving after many video frames (screen-agent pattern).

        Reproduces the scenario where video captures for several seconds
        before the AI voice starts responding.  The AAC encoder's initial
        delay packet (DTS=-1024) must be handled correctly.
        """
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_fps=5,
            audio_codec="aac",
            audio_sample_rate=24000,
        )
        handle = recorder.on_recording_start(config)

        audio_track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="voice-1",
            codec="pcm_s16le",
            sample_rate=24000,
        )
        video_track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="video-1",
            width=64,
            height=48,
        )
        recorder.on_track_added(handle, audio_track)
        recorder.on_track_added(handle, video_track)

        # Video runs for ~4 seconds before audio arrives
        base_ms = 100_000.0
        for i in range(20):
            rgb = np.zeros((48, 64, 3), dtype=np.uint8).tobytes()
            recorder.on_data(handle, video_track, rgb, base_ms + i * 200.0)

        # Audio starts late — first audio triggers encoding start
        for i in range(200):
            ts = base_ms + 4000.0 + i * 20.0
            pcm = np.zeros(480, dtype=np.int16).tobytes()
            recorder.on_data(handle, audio_track, pcm, ts)
            if i % 40 == 0 and i > 0:
                rgb = np.zeros((48, 64, 3), dtype=np.uint8).tobytes()
                recorder.on_data(handle, video_track, rgb, ts)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert len(result.tracks) == 2

    def test_odd_video_dimensions_rounded(self, tmp_path: object) -> None:
        """Odd video dimensions are rounded to even for libx264 compat."""
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_codec="libx264",
        )
        handle = recorder.on_recording_start(config)

        # 675 is odd — would fail libx264 without even-rounding
        track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="video-1",
            width=1080,
            height=675,
        )
        recorder.on_track_added(handle, track)

        for i in range(5):
            # Provide data matching the ORIGINAL odd dimensions;
            # the recorder stream uses the rounded-down even size.
            rgb = np.zeros((674, 1080, 3), dtype=np.uint8).tobytes()
            recorder.on_data(handle, track, rgb, i * 200.0)

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0

    def test_variable_size_audio_frames(self, tmp_path: object) -> None:
        """Variable-size audio frames (realtime voice provider pattern).

        Realtime voice providers send audio chunks of varying sizes.
        The cumulative sample count must track correctly to produce
        monotonic PTS for the AAC encoder's af_queue.
        """
        recorder = _get_recorder()
        config = MediaRecordingConfig(
            storage=str(tmp_path),
            video_fps=5,
            audio_codec="aac",
            audio_sample_rate=24000,
        )
        handle = recorder.on_recording_start(config)

        audio_track = RecordingTrack(
            id="audio:s1",
            kind="audio",
            channel_id="voice-1",
            codec="pcm_s16le",
            sample_rate=24000,
        )
        video_track = RecordingTrack(
            id="video:s1",
            kind="video",
            channel_id="video-1",
            width=64,
            height=48,
        )
        recorder.on_track_added(handle, audio_track)
        recorder.on_track_added(handle, video_track)

        # 1 video frame to start encoding
        rgb = np.zeros((48, 64, 3), dtype=np.uint8).tobytes()
        recorder.on_data(handle, video_track, rgb, 0.0)

        # Variable-size audio: mix of 7200 (300ms), 480 (20ms), 2400 (100ms)
        chunk_sizes = [7200, 7200, 480, 480, 2400, 7200, 480, 7200, 480, 2400]
        ts = 0.0
        for size in chunk_sizes:
            pcm = np.zeros(size, dtype=np.int16).tobytes()
            recorder.on_data(handle, audio_track, pcm, ts)
            ts += size / 24.0  # advance by chunk duration in ms

        result = recorder.on_recording_stop(handle)
        assert result.size_bytes > 0
        assert len(result.tracks) == 2


class TestComputePts:
    def test_monotonic_with_timestamp(self) -> None:
        from roomkit.recorder._pyav_mux import compute_pts

        pts = compute_pts(1000.0, 1000.0, 24000, -1, 0)
        assert pts == 0
        pts = compute_pts(1020.0, 1000.0, 24000, 0, 480)
        assert pts == 480

    def test_monotonic_fallback(self) -> None:
        """Fallback PTS must also enforce monotonicity."""
        from roomkit.recorder._pyav_mux import compute_pts

        # last_pts=5000, fallback_pts=3840 → must return 5001, not 3840
        pts = compute_pts(None, 0.0, 24000, 5000, 3840)
        assert pts == 5001

    def test_jitter_clamped(self) -> None:
        from roomkit.recorder._pyav_mux import compute_pts

        # Timestamp jitter could produce a lower PTS than last_pts
        pts = compute_pts(1019.0, 1000.0, 24000, 480, 480)
        assert pts == 481  # clamped to last_pts + 1

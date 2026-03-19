"""Tests for video recording — VideoRecorder ABC, Mock, and channel integration."""

from __future__ import annotations

import pytest

from roomkit import (
    RoomKit,
    VideoChannel,
)
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.recorder import (
    MockVideoRecorder,
    VideoRecordingConfig,
)
from roomkit.video.video_frame import VideoFrame


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestMockVideoRecorder:
    def test_start_stop(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        config = VideoRecordingConfig(storage="/tmp/test")

        handle = recorder.start(session, config)
        assert handle.state == "recording"
        assert handle.session_id == "s1"
        assert handle.id

        result = recorder.stop(handle)
        assert handle.state == "stopped"
        assert result.frame_count == 0
        assert result.id == handle.id

    def test_tap_frame(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        session = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        handle = recorder.start(session, VideoRecordingConfig())

        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        recorder.tap_frame(handle, frame)
        recorder.tap_frame(handle, frame)

        result = recorder.stop(handle)
        assert result.frame_count == 2

    def test_multiple_recordings(self) -> None:
        from roomkit.video.base import VideoSession

        recorder = MockVideoRecorder()
        s1 = VideoSession(id="s1", room_id="r1", participant_id="u1", channel_id="v1")
        s2 = VideoSession(id="s2", room_id="r1", participant_id="u2", channel_id="v1")

        h1 = recorder.start(s1, VideoRecordingConfig())
        h2 = recorder.start(s2, VideoRecordingConfig())

        assert h1.id != h2.id
        assert len(recorder.recordings) == 2


class TestVideoChannelRecording:
    async def test_recording_starts_on_bind(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Recording should have started
        assert session.id in ch._recording_handles
        assert len(recorder.recordings) == 1

    async def test_recording_stops_on_unbind(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        await kit.disconnect_video(session)

        assert session.id not in ch._recording_handles
        # Recording was stopped
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"

    async def test_frames_tapped_to_recorder(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Send frames
        for i in range(5):
            frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=float(i * 100))
            await backend.simulate_video_received(session, frame)

        rec = list(recorder.recordings.values())[0]
        assert len(rec.frames) == 5

    async def test_no_recorder_no_error(self, kit: RoomKit) -> None:
        """Channel works fine without a recorder."""
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        await backend.simulate_video_received(session, frame)
        await kit.disconnect_video(session)

    async def test_close_stops_recordings(self, kit: RoomKit) -> None:
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            pipeline=VideoPipelineConfig(
                recorder=recorder,
                recording_config=VideoRecordingConfig(),
            ),
        )
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")
        await ch.close()

        assert ch._recording_handles == {}
        rec = list(recorder.recordings.values())[0]
        assert rec.handle.state == "stopped"


class TestOpenCVVideoRecorder:
    def test_import(self) -> None:
        pytest.importorskip("cv2", reason="opencv not installed")
        from roomkit.video.recorder.opencv import OpenCVVideoRecorder

        recorder = OpenCVVideoRecorder()
        assert recorder.name == "OpenCVVideoRecorder"

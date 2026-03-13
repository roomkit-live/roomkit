"""Tests for VideoPipelineConfig and VideoChannel pipeline integration."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    MockVideoBackend,
    RoomKit,
    VideoChannel,
    VideoFrame,
    VideoPipelineConfig,
)
from roomkit.video.recorder import (
    MockVideoRecorder,
    VideoRecordingConfig,
)


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


class TestVideoPipelineConfig:
    def test_defaults(self) -> None:
        config = VideoPipelineConfig()
        assert config.recorder is None
        assert config.recording_config is None

    def test_with_recorder(self) -> None:
        recorder = MockVideoRecorder()
        config = VideoPipelineConfig(
            recorder=recorder,
            recording_config=VideoRecordingConfig(storage="/tmp"),
        )
        assert config.recorder is recorder
        assert config.recording_config is not None
        assert config.recording_config.storage == "/tmp"


class TestVideoChannelPipeline:
    async def test_pipeline_recorder_used(self, kit: RoomKit) -> None:
        """Recorder from pipeline config is used for recording."""
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        pipeline = VideoPipelineConfig(
            recorder=recorder,
            recording_config=VideoRecordingConfig(),
        )
        ch = VideoChannel("video-1", backend=backend, pipeline=pipeline)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Recording should have started via pipeline recorder
        assert session.id in ch._recording_handles
        assert len(recorder.recordings) == 1

    async def test_pipeline_frames_tapped(self, kit: RoomKit) -> None:
        """Frames are tapped to the pipeline recorder."""
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        pipeline = VideoPipelineConfig(
            recorder=recorder,
            recording_config=VideoRecordingConfig(),
        )
        ch = VideoChannel("video-1", backend=backend, pipeline=pipeline)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        for i in range(3):
            frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=float(i * 100))
            await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.05)

        rec = list(recorder.recordings.values())[0]
        assert len(rec.frames) == 3

    async def test_pipeline_without_recorder(self, kit: RoomKit) -> None:
        """Pipeline with no recorder works fine."""
        backend = MockVideoBackend()
        pipeline = VideoPipelineConfig()
        ch = VideoChannel("video-1", backend=backend, pipeline=pipeline)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264")
        await backend.simulate_video_received(session, frame)
        await kit.disconnect_video(session)

    async def test_pipeline_close_stops_recordings(self, kit: RoomKit) -> None:
        """close() stops recordings from pipeline recorder."""
        backend = MockVideoBackend()
        recorder = MockVideoRecorder()
        pipeline = VideoPipelineConfig(
            recorder=recorder,
            recording_config=VideoRecordingConfig(),
        )
        ch = VideoChannel("video-1", backend=backend, pipeline=pipeline)
        kit.register_channel(ch)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")

        await kit.connect_video("r1", "user-1", "video-1")
        await ch.close()

        assert ch._recording_handles == {}

"""Tests for room-level media recording (recorder package)."""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    MockMediaRecorder,
    RecordingTrack,
    RoomKit,
    RoomRecorderBinding,
    VideoChannel,
    VoiceChannel,
)
from roomkit.recorder._room_recorder_manager import RoomRecorderManager
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline.config import AudioPipelineConfig

# ---------------------------------------------------------------------------
# Unit tests for MockMediaRecorder
# ---------------------------------------------------------------------------


class TestMockMediaRecorder:
    def test_recording_lifecycle(self) -> None:
        recorder = MockMediaRecorder()
        config = MediaRecordingConfig(storage="/tmp/test")
        handle = recorder.on_recording_start(config)

        assert handle.state == "recording"
        assert len(recorder.handles) == 1

        track = RecordingTrack(id="audio:s1", kind="audio", channel_id="voice-1")
        recorder.on_track_added(handle, track)
        assert len(recorder.tracks) == 1

        recorder.on_data(handle, track, b"\x00" * 320, 0.0)
        recorder.on_data(handle, track, b"\x00" * 320, 20.0)
        assert len(recorder.chunks) == 2

        recorder.on_track_removed(handle, track)
        assert len(recorder.tracks) == 0

        result = recorder.on_recording_stop(handle)
        assert handle.state == "stopped"
        assert result.size_bytes == 640
        assert len(result.tracks) == 0  # track was removed before stop

    def test_close(self) -> None:
        recorder = MockMediaRecorder()
        assert not recorder.closed
        recorder.close()
        assert recorder.closed


# ---------------------------------------------------------------------------
# Unit tests for RoomRecorderManager
# ---------------------------------------------------------------------------


class TestRoomRecorderManager:
    def test_register_and_stop(self) -> None:
        mgr = RoomRecorderManager()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder, config=MediaRecordingConfig(), name="test"
        )
        mgr.register("room-1", [binding])
        assert mgr.has_recorders("room-1")

        track = RecordingTrack(id="a:1", kind="audio", channel_id="v1")
        mgr.on_track_added("room-1", track)
        mgr.on_data("room-1", track, b"\x00" * 100, 0.0)

        results = mgr.stop_room("room-1")
        assert len(results) == 1
        assert results[0].size_bytes == 100
        assert not mgr.has_recorders("room-1")

    def test_disabled_binding_skipped(self) -> None:
        mgr = RoomRecorderManager()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder, config=MediaRecordingConfig(), enabled=False
        )
        mgr.register("room-1", [binding])
        assert not mgr.has_recorders("room-1")

    def test_close_stops_all(self) -> None:
        mgr = RoomRecorderManager()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())
        mgr.register("room-1", [binding])
        mgr.register("room-2", [binding])
        mgr.close()
        assert not mgr.has_recorders("room-1")
        assert not mgr.has_recorders("room-2")
        assert recorder.closed

    def test_fan_out_to_multiple_recorders(self) -> None:
        mgr = RoomRecorderManager()
        r1 = MockMediaRecorder()
        r2 = MockMediaRecorder()
        mgr.register(
            "room-1",
            [
                RoomRecorderBinding(recorder=r1, config=MediaRecordingConfig()),
                RoomRecorderBinding(recorder=r2, config=MediaRecordingConfig()),
            ],
        )
        track = RecordingTrack(id="a:1", kind="audio", channel_id="v1")
        mgr.on_track_added("room-1", track)
        mgr.on_data("room-1", track, b"\x00" * 50, 0.0)

        assert len(r1.chunks) == 1
        assert len(r2.chunks) == 1


# ---------------------------------------------------------------------------
# Integration tests: video channel → room recorder
# ---------------------------------------------------------------------------


class TestVideoRoomRecording:
    async def test_video_frames_reach_recorder(self) -> None:
        recorder = MockMediaRecorder()
        backend = MockVideoBackend()
        ch = VideoChannel(
            "video-1",
            backend=backend,
            recording=ChannelRecordingConfig(video=True),
        )
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room(
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())]
        )
        await kit.attach_channel(room.id, "video-1")

        session = await kit.connect_video(room.id, "user-1", "video-1")

        # Simulate video frames
        frame = VideoFrame(data=b"\xff" * 100, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        frame2 = VideoFrame(data=b"\xfe" * 100, codec="h264", timestamp_ms=33.3)
        await backend.simulate_video_received(session, frame2)

        assert len(recorder.tracks) == 1
        assert recorder.tracks[0].kind == "video"
        assert len(recorder.chunks) == 2

        await kit.disconnect_video(session)
        await kit.close()

    async def test_no_recording_without_channel_config(self) -> None:
        """VideoChannel without recording config should not feed room recorder."""
        recorder = MockMediaRecorder()
        backend = MockVideoBackend()
        ch = VideoChannel("video-1", backend=backend)  # no recording=...
        kit = RoomKit()
        kit.register_channel(ch)
        room = await kit.create_room(
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())]
        )
        await kit.attach_channel(room.id, "video-1")
        session = await kit.connect_video(room.id, "user-1", "video-1")

        frame = VideoFrame(data=b"\xff" * 50, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)

        assert len(recorder.chunks) == 0
        await kit.disconnect_video(session)
        await kit.close()


# ---------------------------------------------------------------------------
# Integration tests: voice channel → room recorder
# ---------------------------------------------------------------------------


class TestVoiceRoomRecording:
    async def test_audio_frames_reach_recorder(self) -> None:
        recorder = MockMediaRecorder()
        backend = MockVoiceBackend()
        pipeline = AudioPipelineConfig()
        ch = VoiceChannel(
            "voice-1",
            backend=backend,
            pipeline=pipeline,
            recording=ChannelRecordingConfig(audio=True),
        )
        kit = RoomKit(voice=backend)
        kit.register_channel(ch)
        room = await kit.create_room(
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())]
        )
        await kit.attach_channel(room.id, "voice-1")

        session = await kit.connect_voice(room.id, "user-1", "voice-1")

        # Simulate audio frames — they go through the pipeline
        from roomkit.voice.audio_frame import AudioFrame

        for _i in range(5):
            await backend.simulate_audio_received(session, AudioFrame(data=b"\x00" * 320))
        await asyncio.sleep(0.05)

        assert len(recorder.tracks) == 1
        assert recorder.tracks[0].kind == "audio"
        assert recorder.tracks[0].codec == "pcm_s16le"
        assert len(recorder.chunks) >= 1

        await kit.disconnect_voice(session)
        await kit.close()


# ---------------------------------------------------------------------------
# Integration test: room close stops recorders
# ---------------------------------------------------------------------------


class TestRoomCloseStopsRecording:
    async def test_close_room_stops_recorders(self) -> None:
        recorder = MockMediaRecorder()
        kit = RoomKit()
        room = await kit.create_room(
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())]
        )
        # Verify recording started
        assert len(recorder.handles) == 1
        assert recorder.handles[0].state == "recording"

        await kit.close_room(room.id)

        # Verify recording stopped
        assert len(recorder.results) == 1
        assert recorder.handles[0].state == "stopped"
        await kit.close()

    async def test_framework_close_stops_all(self) -> None:
        recorder = MockMediaRecorder()
        kit = RoomKit()
        await kit.create_room(
            room_id="r1",
            recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())],
        )
        await kit.close()
        assert recorder.closed

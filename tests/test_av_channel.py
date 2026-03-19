"""Tests for AudioVideoChannel — combined audio+video channel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import AudioVideoChannel, ChannelType, RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.video.base import VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


def _make_av_backend() -> MagicMock:
    """Build a mock that quacks like a combined A/V backend."""
    from roomkit.video.backends.base import VideoBackend
    from roomkit.voice.backends.base import VoiceBackend

    backend = MagicMock()
    backend.__class__ = type("MockAVBackend", (VoiceBackend, VideoBackend), {})
    backend.name = "mock-av"
    backend.auto_connect = False
    backend.capabilities = MagicMock(return_value=0)
    backend.feeds_aec_reference = False
    backend.supports_playback_callback = False
    backend.close = AsyncMock()

    # Track registered callbacks
    backend._video_received_cb = None
    backend._video_taps = []

    def on_video_received(cb):
        backend._video_received_cb = cb

    def add_video_tap(cb):
        backend._video_taps.append(cb)

    backend.on_video_received = on_video_received
    backend.add_video_tap = add_video_tap

    # Default: no video session
    backend.get_session.return_value = None
    return backend


def _make_video_session(session_id: str = "session-1") -> VideoSession:
    return VideoSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VideoSessionState.ACTIVE,
    )


def _make_voice_session(session_id: str = "session-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VoiceSessionState.ACTIVE,
    )


def _make_video_frame(timestamp_ms: float = 0.0) -> VideoFrame:
    return VideoFrame(
        data=b"\x00" * 100,
        codec="h264",
        timestamp_ms=timestamp_ms,
        sequence=1,
        keyframe=True,
    )


class TestAudioVideoChannel:
    def test_register_av_channel(self) -> None:
        """AudioVideoChannel registers with correct channel_type."""
        kit = RoomKit()
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        kit.register_channel(av)

        assert av.channel_type == ChannelType.AUDIO_VIDEO

    def test_capabilities_audio_and_video(self) -> None:
        """capabilities() reports both audio and video support."""
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        caps = av.capabilities()
        assert caps.supports_audio is True
        assert caps.supports_video is True

    def test_video_frames_delivered_to_taps(self) -> None:
        """Video frames from the backend are delivered to registered taps."""
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )

        # Bind session so frames are routed
        session = _make_voice_session()
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="av",
            channel_type=ChannelType.AUDIO_VIDEO,
        )
        av.bind_session(session, "room-1", binding)

        # Add a tap
        received_frames: list[VideoFrame] = []
        av.add_video_media_tap(lambda sess, frame: received_frames.append(frame))

        # Simulate a video frame from the backend
        video_session = _make_video_session()
        frame = _make_video_frame()
        # Call the callback that was registered on the backend
        assert backend._video_received_cb is not None
        backend._video_received_cb(video_session, frame)

        assert len(received_frames) == 1
        assert received_frames[0] is frame

    def test_vision_analysis_throttled(self) -> None:
        """Vision analysis is throttled by vision_interval_ms."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "mock-vision"

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
            vision_interval_ms=1000,
        )

        session = _make_voice_session()
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="av",
            channel_type=ChannelType.AUDIO_VIDEO,
        )
        av.bind_session(session, "room-1", binding)

        video_session = _make_video_session()
        cb = backend._video_received_cb
        assert cb is not None

        # First frame at t=0 — should trigger analysis (schedules task)
        cb(video_session, _make_video_frame(timestamp_ms=0))
        # Second frame at t=500ms — within interval, should NOT trigger
        cb(video_session, _make_video_frame(timestamp_ms=500))
        # Third frame at t=1100ms — outside interval, should trigger
        cb(video_session, _make_video_frame(timestamp_ms=1100))

        # Check last_vision_ts was updated for frames 1 and 3
        assert av._last_vision_ts[video_session.id] == pytest.approx(1100)

    def test_info_includes_vision(self) -> None:
        """info property includes vision fields."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "test-vision"

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
        )

        info = av.info
        assert info["vision"] == "test-vision"
        assert "vision_interval_ms" in info

    async def test_close_cleans_video_state(self) -> None:
        """close() clears video-specific state."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "mock-vision"
        mock_vision.close = AsyncMock()

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
        )
        av.add_video_media_tap(lambda s, f: None)
        av._last_vision_results["s1"] = MagicMock()

        await av.close()

        assert len(av._video_media_taps) == 0
        assert len(av._last_vision_results) == 0
        mock_vision.close.assert_called_once()


class TestAVRecordingWiring:
    async def test_bind_wires_audio_and_video(self) -> None:
        """bind_voice_session on AudioVideoChannel wires both audio and video tracks."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = _make_av_backend()
        video_session = _make_video_session()
        backend.get_session.return_value = video_session

        av = AudioVideoChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            recording=ChannelRecordingConfig(audio=True, video=True),
        )
        kit.register_channel(av)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = _make_voice_session()
        await kit.bind_voice_session(session, "room-1", "voice")

        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids
        assert "video:session-1" in track_ids

        # Verify video tap was added on the channel (not the backend)
        assert len(av._video_media_taps) > 0
        # Backend's add_video_tap should NOT be called
        assert len(backend._video_taps) == 0

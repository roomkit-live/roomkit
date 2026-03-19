"""Tests for recording wiring helpers and bind_voice_session."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.video.base import VideoSession, VideoSessionState
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


def _make_mock_sip_video_backend() -> MagicMock:
    """Build a mock that quacks like SIPVideoBackend."""
    from roomkit.video.backends.base import VideoBackend
    from roomkit.voice.backends.base import VoiceBackend

    # No spec so we can add arbitrary attributes
    backend = MagicMock()
    # Make isinstance checks work for both VoiceBackend and VideoBackend
    backend.__class__ = type("MockSIPVideoBackend", (VoiceBackend, VideoBackend), {})
    backend.name = "SIP-AV"
    backend._video_taps = []

    video_session = VideoSession(
        id="session-1",
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VideoSessionState.ACTIVE,
    )
    backend.get_video_session.return_value = video_session

    def add_video_tap(cb):
        backend._video_taps.append(cb)

    backend.add_video_tap = add_video_tap
    return backend


class TestBindVoiceSession:
    async def test_bind_wires_audio_recording(self) -> None:
        """bind_voice_session wires audio recording tap."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = MagicMock()
        backend.name = "mock"
        backend.capabilities = MagicMock(return_value=0)
        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        await kit.bind_voice_session(session, "room-1", "voice")

        # Verify track was added to recorder
        assert len(recorder.tracks) > 0
        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids

    async def test_bind_wires_video_for_combined_backend(self) -> None:
        """bind_voice_session wires video recording for A/V backends."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = _make_mock_sip_video_backend()
        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        await kit.bind_voice_session(session, "room-1", "voice")

        # Verify both audio and video tracks were added
        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids
        assert "video:session-1" in track_ids

        # Verify a video tap was added to the backend
        assert len(backend._video_taps) == 1

    async def test_bind_records_by_default_without_config(self) -> None:
        """Recording is opt-out: no ChannelRecordingConfig = record everything."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = MagicMock()
        backend.name = "mock"
        backend.capabilities = MagicMock(return_value=0)
        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            # No recording= param — should record by default
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        await kit.bind_voice_session(session, "room-1", "voice")

        # Recording should be wired even without ChannelRecordingConfig
        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids

    async def test_bind_opt_out_audio_recording(self) -> None:
        """ChannelRecordingConfig(audio=False) opts out of audio recording."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = MagicMock()
        backend.name = "mock"
        backend.capabilities = MagicMock(return_value=0)
        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            recording=ChannelRecordingConfig(audio=False),
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        await kit.bind_voice_session(session, "room-1", "voice")

        # Recording should NOT be wired — explicitly opted out
        assert len(recorder.tracks) == 0

    async def test_bind_no_recording_without_recorders(self) -> None:
        """bind_voice_session doesn't wire recording without recorders."""
        kit = RoomKit()

        backend = MagicMock()
        backend.name = "mock"
        backend.capabilities = MagicMock(return_value=0)
        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        # Should not raise — just no recording wiring
        await kit.bind_voice_session(session, "room-1", "voice")

    async def test_bind_invalid_channel_raises(self) -> None:
        """bind_voice_session raises for non-VoiceChannel."""
        kit = RoomKit()
        await kit.create_room(room_id="room-1")

        session = VoiceSession(
            id="s1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="nonexistent",
            state=VoiceSessionState.ACTIVE,
        )

        with pytest.raises(Exception, match="not registered"):
            await kit.bind_voice_session(session, "room-1", "nonexistent")

    async def test_disconnect_voice_removes_tracks_but_keeps_recording(self) -> None:
        """disconnect_voice removes audio/video tracks but does not stop recording."""
        from unittest.mock import AsyncMock

        backend = _make_mock_sip_video_backend()
        backend.disconnect = AsyncMock()
        backend.stop_listening = AsyncMock()

        kit = RoomKit(voice=backend)
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        channel = VoiceChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            recording=ChannelRecordingConfig(audio=True, video=True),
        )
        kit.register_channel(channel)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = VoiceSession(
            id="session-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )

        await kit.bind_voice_session(session, "room-1", "voice")

        # Both tracks registered
        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids
        assert "video:session-1" in track_ids

        # Disconnect voice — tracks should be removed
        await kit.disconnect_voice(session)

        # on_track_removed was called (tracks removed from mock recorder)
        assert "audio:session-1" not in [t.id for t in recorder.tracks]
        assert "video:session-1" not in [t.id for t in recorder.tracks]

        # Recording is NOT stopped — stop_room is close_room's job
        assert kit._room_recorder_mgr.has_recorders("room-1")

"""Tests for RealtimeAudioVideoChannel."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import (
    MockRealtimeAudioVideoProvider,
    MockRealtimeTransport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> MockRealtimeAudioVideoProvider:
    return MockRealtimeAudioVideoProvider()


@pytest.fixture
def transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


@pytest.fixture
def channel(
    provider: MockRealtimeAudioVideoProvider,
    transport: MockRealtimeTransport,
) -> RealtimeAudioVideoChannel:
    return RealtimeAudioVideoChannel(
        "rtav-1",
        provider=provider,
        transport=transport,
        system_prompt="You are a helpful avatar.",
    )


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="sess-1",
        room_id="room-1",
        participant_id="part-1",
        channel_id="rtav-1",
        state=VoiceSessionState.CONNECTING,
    )


def _make_video_frame(seq: int = 0) -> VideoFrame:
    """Create a minimal raw RGB video frame for testing."""
    return VideoFrame(
        data=b"\x00" * (320 * 240 * 3),
        codec="raw_rgb24",
        width=320,
        height=240,
        sequence=seq,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChannelType:
    def test_channel_type(self, channel: RealtimeAudioVideoChannel) -> None:
        assert channel.channel_type == ChannelType.REALTIME_AUDIO_VIDEO

    def test_capabilities_include_video(self, channel: RealtimeAudioVideoChannel) -> None:
        caps = channel.capabilities()
        assert ChannelMediaType.VIDEO in caps.media_types
        assert ChannelMediaType.AUDIO in caps.media_types
        assert caps.supports_video is True
        assert caps.supports_audio is True


class TestVideoCallbackWiring:
    async def test_provider_video_routed_to_taps(
        self,
        channel: RealtimeAudioVideoChannel,
        provider: MockRealtimeAudioVideoProvider,
        transport: MockRealtimeTransport,
        session: VoiceSession,
    ) -> None:
        """Video frames from the provider should reach registered taps."""
        received_frames: list[VideoFrame] = []
        channel.add_video_media_tap(lambda s, f: received_frames.append(f))

        # Start a session so the channel has room_id mapping
        await channel.start_session(
            "room-1",
            "part-1",
            "fake-ws",
        )

        # Get the actual session created by start_session
        sessions = list(channel._sessions.values())
        assert len(sessions) == 1
        active_session = sessions[0]

        # Simulate video from provider
        frame = _make_video_frame()
        await provider.simulate_video(active_session, frame)

        assert len(received_frames) == 1
        assert received_frames[0].codec == "raw_rgb24"

        await channel.end_session(active_session)

    async def test_video_dropped_without_session(
        self,
        channel: RealtimeAudioVideoChannel,
        provider: MockRealtimeAudioVideoProvider,
        session: VoiceSession,
    ) -> None:
        """Video frames for unknown sessions should be silently dropped."""
        received_frames: list[VideoFrame] = []
        channel.add_video_media_tap(lambda s, f: received_frames.append(f))

        # No session started — frame should be dropped
        frame = _make_video_frame()
        await provider.simulate_video(session, frame)

        assert len(received_frames) == 0


class TestSessionLifecycle:
    async def test_start_end_session(
        self,
        channel: RealtimeAudioVideoChannel,
        provider: MockRealtimeAudioVideoProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Session start/end should work without errors."""
        session = await channel.start_session(
            "room-1",
            "part-1",
            "fake-ws",
        )
        assert session.state == VoiceSessionState.ACTIVE

        await channel.end_session(session)
        assert session.state == VoiceSessionState.ENDED

    async def test_close_cleans_up(
        self,
        channel: RealtimeAudioVideoChannel,
        provider: MockRealtimeAudioVideoProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Channel close should end all sessions and clean up."""
        session = await channel.start_session(
            "room-1",
            "part-1",
            "fake-ws",
        )

        await channel.close()
        assert session.state == VoiceSessionState.ENDED


class TestInfo:
    def test_info_includes_vision(self, channel: RealtimeAudioVideoChannel) -> None:
        info = channel.info
        assert "vision" in info
        assert info["vision"] is None
        assert info["vision_interval_ms"] == 2000


class TestMockRealtimeAudioVideoProvider:
    async def test_simulate_video(self) -> None:
        """MockRealtimeAudioVideoProvider should fire video callbacks."""
        provider = MockRealtimeAudioVideoProvider()
        received: list[Any] = []
        provider.on_video(lambda s, f: received.append(f))

        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="p1",
            channel_id="c1",
            state=VoiceSessionState.ACTIVE,
        )
        frame = _make_video_frame()
        await provider.simulate_video(session, frame)

        assert len(received) == 1
        assert received[0] is frame

    def test_name(self) -> None:
        provider = MockRealtimeAudioVideoProvider()
        assert provider.name == "MockRealtimeAudioVideoProvider"

    async def test_inherits_mock_realtime(self) -> None:
        """Should inherit all MockRealtimeProvider functionality."""
        provider = MockRealtimeAudioVideoProvider()
        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="p1",
            channel_id="c1",
            state=VoiceSessionState.CONNECTING,
        )
        await provider.connect(session)
        assert session.state == VoiceSessionState.ACTIVE
        assert provider.calls[-1].method == "connect"

        await provider.disconnect(session)
        assert session.state == VoiceSessionState.ENDED

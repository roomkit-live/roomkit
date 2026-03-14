"""Tests for RTPVideoBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.video.base import VideoChunk, VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import VoiceSession, VoiceSessionState


def _mock_aiortp() -> MagicMock:
    """Build a mock aiortp module with RTPSession + VideoRTPSession."""
    mock = MagicMock()

    # Audio RTPSession
    audio_session = MagicMock()
    audio_session.stats = {"packets_sent": 0}
    audio_session.close = AsyncMock()
    mock.RTPSession.create = AsyncMock(return_value=audio_session)

    # Video VideoRTPSession
    video_session = MagicMock()
    video_session.stats = {"ssrc": 12345}
    video_session.close = AsyncMock()
    mock.VideoRTPSession.create = AsyncMock(return_value=video_session)

    return mock


@pytest.fixture
def mock_aiortp():
    return _mock_aiortp()


@pytest.fixture
def backend(mock_aiortp):
    with patch("roomkit.voice.backends.rtp._import_aiortp", return_value=mock_aiortp):
        from roomkit.video.backends.rtp import RTPVideoBackend

        return RTPVideoBackend(
            local_addr=("127.0.0.1", 10000),
            remote_addr=("10.0.0.1", 20000),
            video_local_addr=("127.0.0.1", 10002),
            video_remote_addr=("10.0.0.1", 20002),
        )


class TestRTPVideoBackendConnect:
    async def test_connect_creates_both_sessions(self, backend, mock_aiortp):
        session = await backend.connect("room-1", "user-1", "voice-1")

        assert isinstance(session, VoiceSession)
        assert session.state == VoiceSessionState.ACTIVE

        # Audio RTP session created
        mock_aiortp.RTPSession.create.assert_awaited_once()

        # Video RTP session created
        mock_aiortp.VideoRTPSession.create.assert_awaited_once()
        call_kwargs = mock_aiortp.VideoRTPSession.create.call_args.kwargs
        assert call_kwargs["local_addr"] == ("127.0.0.1", 10002)
        assert call_kwargs["remote_addr"] == ("10.0.0.1", 20002)
        assert call_kwargs["payload_type"] == 96
        assert call_kwargs["clock_rate"] == 90000

    async def test_connect_stores_video_session(self, backend):
        session = await backend.connect("room-1", "user-1", "voice-1")

        video_session = backend.get_video_session(session.id)
        assert video_session is not None
        assert video_session.id == session.id
        assert video_session.room_id == "room-1"
        assert video_session.state == VideoSessionState.ACTIVE

    async def test_connect_requires_video_remote(self, mock_aiortp):
        with patch("roomkit.voice.backends.rtp._import_aiortp", return_value=mock_aiortp):
            from roomkit.video.backends.rtp import RTPVideoBackend

            backend = RTPVideoBackend(
                local_addr=("127.0.0.1", 10000),
                remote_addr=("10.0.0.1", 20000),
                video_local_addr=("127.0.0.1", 10002),
                # No video_remote_addr
            )

        with pytest.raises(ValueError, match="video_remote_addr"):
            await backend.connect("room-1", "user-1", "voice-1")

    async def test_connect_video_remote_from_metadata(self, mock_aiortp):
        with patch("roomkit.voice.backends.rtp._import_aiortp", return_value=mock_aiortp):
            from roomkit.video.backends.rtp import RTPVideoBackend

            backend = RTPVideoBackend(
                local_addr=("127.0.0.1", 10000),
                remote_addr=("10.0.0.1", 20000),
                video_local_addr=("127.0.0.1", 10002),
            )

        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"video_remote_addr": ("10.0.0.99", 25002)},
        )
        assert session is not None

        call_kwargs = mock_aiortp.VideoRTPSession.create.call_args.kwargs
        assert call_kwargs["remote_addr"] == ("10.0.0.99", 25002)


class TestRTPVideoBackendDisconnect:
    async def test_disconnect_closes_both(self, backend, mock_aiortp):
        session = await backend.connect("room-1", "user-1", "voice-1")

        audio_rtp = mock_aiortp.RTPSession.create.return_value
        audio_rtp.close = AsyncMock()
        video_rtp = mock_aiortp.VideoRTPSession.create.return_value

        await backend.disconnect(session)

        video_rtp.close.assert_awaited_once()
        audio_rtp.close.assert_awaited_once()
        assert backend.get_video_session(session.id) is None

    async def test_disconnect_fires_video_callbacks(self, backend):
        disconnected: list[VideoSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)

        assert len(disconnected) == 1
        assert disconnected[0].state == VideoSessionState.ENDED


class TestRTPVideoBackendInbound:
    async def test_inbound_video_callback(self, backend, mock_aiortp):
        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, frame: received.append(frame))

        await backend.connect("room-1", "user-1", "voice-1")

        # Simulate inbound video frame
        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        on_frame = video_rtp.on_frame
        assert on_frame is not None

        # IDR NAL (type 5)
        on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received) == 1
        assert received[0].codec == "h264"
        assert received[0].keyframe is True
        assert received[0].sequence == 0
        assert received[0].timestamp_ms == pytest.approx(1000.0, rel=0.01)

    async def test_inbound_sequence_increments(self, backend, mock_aiortp):
        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, frame: received.append(frame))

        await backend.connect("room-1", "user-1", "voice-1")

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        on_frame = video_rtp.on_frame

        on_frame(b"\x65\x00", 90000, True)
        on_frame(b"\x41\x00", 93600, False)
        on_frame(b"\x41\x01", 97200, False)

        assert [f.sequence for f in received] == [0, 1, 2]

    async def test_no_callback_no_error(self, backend, mock_aiortp):
        await backend.connect("room-1", "user-1", "voice-1")

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        on_frame = video_rtp.on_frame
        on_frame(b"\x65\x00", 90000, True)  # no callback, no error


class TestRTPVideoBackendVideoTap:
    async def test_add_video_tap(self, backend, mock_aiortp):
        """Video taps receive frames alongside the primary callback."""
        received_primary: list[VideoFrame] = []
        received_tap: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received_primary.append(f))
        backend.add_video_tap(lambda _s, f: received_tap.append(f))

        await backend.connect("room-1", "user-1", "voice-1")

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        video_rtp.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_primary) == 1
        assert len(received_tap) == 1
        assert received_tap[0].codec == "h264"

    async def test_tap_works_without_primary_callback(self, backend, mock_aiortp):
        """Video taps fire even when no primary callback is set."""
        received_tap: list[VideoFrame] = []
        backend.add_video_tap(lambda _s, f: received_tap.append(f))

        await backend.connect("room-1", "user-1", "voice-1")

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        video_rtp.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_tap) == 1

    async def test_multiple_taps(self, backend, mock_aiortp):
        """Multiple taps all receive frames."""
        counts = [0, 0]

        def make_tap(idx):
            def tap(_s, _f):
                counts[idx] += 1

            return tap

        backend.add_video_tap(make_tap(0))
        backend.add_video_tap(make_tap(1))

        await backend.connect("room-1", "user-1", "voice-1")

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        video_rtp.on_frame(b"\x65\x00", 90000, True)
        video_rtp.on_frame(b"\x41\x00", 93600, False)

        assert counts == [2, 2]


class TestRTPVideoBackendOutbound:
    async def test_send_video_bytes(self, backend, mock_aiortp):
        session = await backend.connect("room-1", "user-1", "voice-1")
        video_session = backend.get_video_session(session.id)

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value
        await backend.send_video(video_session, b"\x65\x00\x01\x02")

        video_rtp.send_frame.assert_called_once_with([b"\x65\x00\x01\x02"], 0)

    async def test_send_video_chunks(self, backend, mock_aiortp):
        session = await backend.connect("room-1", "user-1", "voice-1")
        video_session = backend.get_video_session(session.id)

        video_rtp = mock_aiortp.VideoRTPSession.create.return_value

        async def chunk_iter():
            yield VideoChunk(data=b"\x65\x00", keyframe=True, timestamp_ms=0)
            yield VideoChunk(data=b"\x41\x00", keyframe=False, timestamp_ms=33)

        await backend.send_video(video_session, chunk_iter())
        assert video_rtp.send_frame.call_count == 2

    async def test_send_video_no_session(self, backend):
        dummy = VideoSession(id="nonexistent", room_id="r", participant_id="p", channel_id="c")
        await backend.send_video(dummy, b"\x00")  # no error


class TestRTPVideoBackendProperties:
    async def test_name(self, backend):
        assert backend.name == "RTP-AV"

    async def test_list_video_sessions(self, backend):
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-2", "user-2", "voice-2")

        assert len(backend.list_video_sessions("room-1")) == 1
        assert len(backend.list_video_sessions("room-2")) == 1
        assert len(backend.list_video_sessions("room-3")) == 0

    async def test_close_cleans_all(self, backend, mock_aiortp):
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-2", "user-2", "voice-2")

        await backend.close()

        assert len(backend.list_video_sessions("room-1")) == 0
        assert len(backend.list_video_sessions("room-2")) == 0

    async def test_session_ready_callback(self, backend):
        ready: list[VideoSession] = []
        backend.on_session_ready(lambda s: ready.append(s))

        await backend.connect("room-1", "user-1", "voice-1")

        assert len(ready) == 1
        assert ready[0].state == VideoSessionState.ACTIVE

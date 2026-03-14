"""Tests for SIPVideoBackend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.video.base import VideoSession, VideoSessionState
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import VoiceSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdp_offer(*, has_video: bool = True) -> MagicMock:
    """Build a mock SDP offer with audio and optionally video."""
    audio_media = MagicMock(media="audio")
    video_media = MagicMock(media="video") if has_video else None

    offer = MagicMock()
    offer.audio = audio_media
    offer.video = video_media
    offer.rtp_address = ("10.0.0.1", 20000)
    offer.video_rtp_address = ("10.0.0.1", 20002) if has_video else None
    offer.media = [audio_media] + ([video_media] if video_media else [])
    return offer


def _mock_aiosipua() -> MagicMock:
    """Build a mock aiosipua module with transport, UAS, UAC."""
    mock = MagicMock()
    mock.UdpSipTransport.return_value = MagicMock()
    mock.SipUAS.return_value = MagicMock(start=AsyncMock())
    mock.SipUAC.return_value = MagicMock()
    return mock


def _mock_rtp_bridge(av_offer) -> MagicMock:
    """Build a mock rtp_bridge module with CallSession + VideoCallSession."""
    mock = MagicMock()

    # Audio CallSession
    audio_cs = MagicMock()
    audio_cs.sdp_answer = MagicMock()
    audio_cs.sdp_answer.media = [MagicMock(media="audio")]
    audio_cs.sdp_answer.version = 0
    audio_cs.sdp_answer.origin = MagicMock(address="10.0.0.5")
    audio_cs.sdp_answer.session_name = "-"
    audio_cs.sdp_answer.connection = MagicMock(address="10.0.0.5")
    audio_cs.sdp_answer.bandwidths = []
    audio_cs.sdp_answer.timing = MagicMock()
    audio_cs.sdp_answer.attributes = {}
    audio_cs.codec_sample_rate = 8000
    audio_cs.clock_rate = 8000
    audio_cs.remote_addr = ("10.0.0.1", 20000)
    audio_cs.start = AsyncMock()
    audio_cs.close = AsyncMock()
    mock.CallSession.return_value = audio_cs

    # Video VideoCallSession
    video_cs = MagicMock()
    video_media = MagicMock(media="video")
    video_cs.sdp_answer = MagicMock()
    video_cs.sdp_answer.video = video_media
    video_cs.remote_addr = ("10.0.0.1", 20002)
    video_cs.start = AsyncMock()
    video_cs.close = AsyncMock()
    mock.VideoCallSession.return_value = video_cs

    return mock


@dataclass
class _FakeCall:
    """Minimal fake IncomingCall for testing."""

    call_id: str = "test-call-1"
    sdp_offer: MagicMock | None = None
    caller: str = "sip:alice@example.com"
    callee: str = "sip:bot@example.com"
    room_id: str | None = "room-1"
    session_id: str | None = "session-1"
    x_headers: dict[str, str] = field(default_factory=dict)
    source_addr: tuple[str, int] = ("10.0.0.1", 5060)
    invite: MagicMock = field(default_factory=lambda: MagicMock(from_addr=None))

    def ringing(self) -> None:
        pass

    def accept(self, sdp_answer: object) -> None:
        self._accepted_sdp = sdp_answer

    def reject(self, code: int, reason: str) -> None:
        self._rejected = (code, reason)


@pytest.fixture
def av_offer():
    return _make_sdp_offer(has_video=True)


@pytest.fixture
def audio_only_offer():
    return _make_sdp_offer(has_video=False)


@pytest.fixture
def mock_aiosipua():
    return _mock_aiosipua()


@pytest.fixture
def mock_rtp_bridge(av_offer):
    return _mock_rtp_bridge(av_offer)


@pytest.fixture
def backend(mock_aiosipua, mock_rtp_bridge):
    with (
        patch("roomkit.voice.backends.sip._import_aiosipua", return_value=mock_aiosipua),
        patch(
            "roomkit.voice.backends.sip._import_rtp_bridge",
            return_value=mock_rtp_bridge,
        ),
    ):
        from roomkit.video.backends.sip import SIPVideoBackend

        return SIPVideoBackend(
            local_sip_addr=("127.0.0.1", 5060),
            local_rtp_ip="10.0.0.5",
            rtp_port_start=10000,
            rtp_port_end=10100,
        )


class TestSIPVideoBackendInbound:
    async def test_av_invite_creates_both_sessions(self, backend, mock_rtp_bridge, av_offer):
        """A/V INVITE creates CallSession + VideoCallSession."""
        call = _FakeCall(sdp_offer=av_offer)

        await backend._handle_invite(call)

        # Audio session created
        mock_rtp_bridge.CallSession.assert_called_once()
        mock_rtp_bridge.CallSession.return_value.start.assert_awaited_once()

        # Video session created
        mock_rtp_bridge.VideoCallSession.assert_called_once()
        mock_rtp_bridge.VideoCallSession.return_value.start.assert_awaited_once()

    async def test_av_invite_stores_video_session(self, backend, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_session = backend.get_video_session("session-1")
        assert video_session is not None
        assert video_session.state == VideoSessionState.ACTIVE

    async def test_av_invite_fires_on_call(self, backend, av_offer):
        called: list[VoiceSession] = []
        backend.on_call(lambda s: called.append(s))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        assert len(called) == 1
        assert called[0].metadata.get("has_video") is True

    async def test_audio_only_invite_no_video(self, backend, mock_rtp_bridge, audio_only_offer):
        """Audio-only INVITE delegates to parent (no VideoCallSession)."""
        call = _FakeCall(sdp_offer=audio_only_offer)

        await backend._handle_invite(call)

        mock_rtp_bridge.CallSession.assert_called_once()
        mock_rtp_bridge.VideoCallSession.assert_not_called()

    async def test_video_negotiation_failure_falls_back(self, backend, mock_rtp_bridge, av_offer):
        """If video negotiation fails, call proceeds audio-only."""
        mock_rtp_bridge.VideoCallSession.side_effect = Exception("no codec match")

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        # Audio still works
        mock_rtp_bridge.CallSession.return_value.start.assert_awaited_once()

        # No video session
        assert backend.get_video_session("session-1") is None


class TestSIPVideoBackendVideoCallbacks:
    async def test_inbound_video_frame(self, backend, mock_rtp_bridge, av_offer):
        received: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        # Simulate inbound video frame from VideoCallSession
        video_cs = mock_rtp_bridge.VideoCallSession.return_value
        on_frame = video_cs.on_frame
        assert on_frame is not None

        on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received) == 1
        assert received[0].codec == "h264"
        assert received[0].keyframe is True
        assert received[0].sequence == 0

    async def test_video_disconnect_callback(self, backend, mock_rtp_bridge, av_offer):
        disconnected: list[VideoSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        session_id = "session-1"
        backend._cleanup_session(session_id)

        # Let the close task run
        await asyncio.sleep(0)

        assert len(disconnected) == 1
        assert disconnected[0].state == VideoSessionState.ENDED


class TestSIPVideoBackendSendVideo:
    async def test_send_video_bytes(self, backend, mock_rtp_bridge, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_session = backend.get_video_session("session-1")
        video_cs = mock_rtp_bridge.VideoCallSession.return_value

        await backend.send_video(video_session, b"\x65\x00\x01")
        video_cs.send_frame.assert_called_once_with([b"\x65\x00\x01"], 0)

    async def test_send_video_no_session(self, backend):
        dummy = VideoSession(id="nonexistent", room_id="r", participant_id="p", channel_id="c")
        await backend.send_video(dummy, b"\x00")  # no error


class TestSIPVideoBackendProperties:
    def test_name(self, backend):
        assert backend.name == "SIP-AV"

    async def test_list_video_sessions(self, backend, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        sessions = backend.list_video_sessions("room-1")
        assert len(sessions) == 1

    async def test_disconnect_closes_video(self, backend, mock_rtp_bridge, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        state = backend._session_states.get("session-1")
        assert state is not None

        await backend.disconnect(state.session)

        # Video close runs as a fire-and-forget task via _cleanup_session —
        # yield to the event loop so the task completes.
        await asyncio.sleep(0)

        video_cs = mock_rtp_bridge.VideoCallSession.return_value
        video_cs.close.assert_awaited_once()


class TestSIPVideoBackendVideoTap:
    async def test_add_video_tap(self, backend, mock_rtp_bridge, av_offer):
        """Video taps receive frames alongside the primary callback."""
        received_primary: list[VideoFrame] = []
        received_tap: list[VideoFrame] = []
        backend.on_video_received(lambda _s, f: received_primary.append(f))
        backend.add_video_tap(lambda _s, f: received_tap.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_rtp_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_primary) == 1
        assert len(received_tap) == 1
        assert received_tap[0].codec == "h264"

    async def test_tap_works_without_primary_callback(self, backend, mock_rtp_bridge, av_offer):
        """Video taps fire even when no primary callback is set."""
        received_tap: list[VideoFrame] = []
        backend.add_video_tap(lambda _s, f: received_tap.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_rtp_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_tap) == 1

    async def test_multiple_taps(self, backend, mock_rtp_bridge, av_offer):
        """Multiple taps all receive frames."""
        counts = [0, 0]

        def make_tap(idx):
            def tap(_s, _f):
                counts[idx] += 1

            return tap

        backend.add_video_tap(make_tap(0))
        backend.add_video_tap(make_tap(1))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_rtp_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00", 90000, True)
        video_cs.on_frame(b"\x41\x00", 93600, False)

        assert counts == [2, 2]


class TestSIPVideoBackendReinvite:
    async def test_reinvite_uses_combined_answer(self, backend, mock_rtp_bridge, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        # Simulate re-INVITE
        reinvite_call = _FakeCall(
            call_id="test-call-1",
            sdp_offer=av_offer,
        )
        backend._handle_reinvite(reinvite_call)

        # Should have accepted with some SDP (combined answer)
        assert hasattr(reinvite_call, "_accepted_sdp")

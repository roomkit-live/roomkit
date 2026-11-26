"""Tests for SIPVideoBackend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.video.base import VideoChunk, VideoSession, VideoSessionState
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
    """Build a mock rtp_bridge module with CallSession."""
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

    return mock


def _mock_video_bridge() -> MagicMock:
    """Build a mock video_bridge module with VideoCallSession."""
    mock = MagicMock()

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
    dialog: MagicMock = field(default_factory=MagicMock)

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
def mock_video_bridge():
    return _mock_video_bridge()


@pytest.fixture
def backend(mock_aiosipua, mock_rtp_bridge, mock_video_bridge):
    with (
        patch("roomkit.voice.backends.sip.import_aiosipua", return_value=mock_aiosipua),
        patch(
            "roomkit.voice.backends.sip.import_rtp_bridge",
            return_value=mock_rtp_bridge,
        ),
        patch(
            "roomkit.video.backends.sip._import_video_bridge",
            return_value=mock_video_bridge,
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
    async def test_av_invite_creates_both_sessions(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        """A/V INVITE creates CallSession + VideoCallSession."""
        call = _FakeCall(sdp_offer=av_offer)

        await backend._handle_invite(call)

        # Audio session created
        mock_rtp_bridge.CallSession.assert_called_once()
        mock_rtp_bridge.CallSession.return_value.start.assert_awaited_once()

        # Video session created
        mock_video_bridge.VideoCallSession.assert_called_once()
        mock_video_bridge.VideoCallSession.return_value.start.assert_awaited_once()

    async def test_av_invite_stores_video_session(self, backend, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_session = backend.video.get_session("session-1")
        assert video_session is not None
        assert video_session.state == VideoSessionState.ACTIVE

    async def test_av_invite_fires_on_call(self, backend, av_offer):
        called: list[VoiceSession] = []
        backend.on_call(lambda s: called.append(s))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        assert len(called) == 1
        assert called[0].metadata.get("has_video") is True

    async def test_audio_only_invite_no_video(
        self, backend, mock_rtp_bridge, mock_video_bridge, audio_only_offer
    ):
        """Audio-only INVITE delegates to parent (no VideoCallSession)."""
        call = _FakeCall(sdp_offer=audio_only_offer)

        await backend._handle_invite(call)

        mock_rtp_bridge.CallSession.assert_called_once()
        mock_video_bridge.VideoCallSession.assert_not_called()

    async def test_video_negotiation_failure_falls_back(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        """If video negotiation fails, call proceeds audio-only."""
        mock_video_bridge.VideoCallSession.side_effect = Exception("no codec match")

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        # Audio still works
        mock_rtp_bridge.CallSession.return_value.start.assert_awaited_once()

        # No video session
        assert backend.video.get_session("session-1") is None


class TestSIPVideoBackendInviteAuthorization:
    """An ``m=video`` line must not buy a caller past the INVITE gate.

    The A/V path is a separate dispatch branch from the audio-only one, so it
    has to run the digest challenge and the invite filter itself.
    """

    async def test_av_invite_is_challenged_when_auth_configured(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        backend._auth_users = {"alice": "pass123"}
        backend._transport = MagicMock()
        call = _FakeCall(sdp_offer=av_offer)
        call.invite.get_header.return_value = None

        await backend._handle_invite(call)

        # Challenged, and nothing committed before the caller authenticates.
        backend._transport.send_reply.assert_called_once()
        mock_rtp_bridge.CallSession.assert_not_called()
        mock_video_bridge.VideoCallSession.assert_not_called()
        assert backend._allocated_ports == set()

    async def test_av_invite_runs_invite_filter(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        backend._invite_filter = lambda _call: (403, "Forbidden")
        call = _FakeCall(sdp_offer=av_offer)

        await backend._handle_invite(call)

        assert call._rejected == (403, "Forbidden")
        mock_rtp_bridge.CallSession.assert_not_called()
        mock_video_bridge.VideoCallSession.assert_not_called()
        assert backend._allocated_ports == set()

    async def test_authenticated_av_invite_proceeds(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        """The gate must not break the happy path once auth passes."""
        backend._auth_users = {"alice": "pass123"}
        backend._transport = MagicMock()
        call = _FakeCall(sdp_offer=av_offer)
        with patch.object(backend, "_validate_invite_auth", return_value=True):
            await backend._handle_invite(call)

        mock_rtp_bridge.CallSession.assert_called_once()
        mock_video_bridge.VideoCallSession.assert_called_once()


class TestSIPVideoBackendVideoCallbacks:
    async def test_inbound_video_frame(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        received: list[VideoFrame] = []
        backend.video.on_video_received(lambda _s, f: received.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        # Simulate inbound video frame from VideoCallSession
        video_cs = mock_video_bridge.VideoCallSession.return_value
        on_frame = video_cs.on_frame
        assert on_frame is not None

        on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received) == 1
        assert received[0].codec == "h264"
        assert received[0].keyframe is True
        assert received[0].sequence == 0

    async def test_video_disconnect_callback(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        disconnected: list[VideoSession] = []
        backend.video.on_client_disconnected(lambda s: disconnected.append(s))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        session_id = "session-1"
        backend._cleanup_session(session_id)

        # Let the close task run
        await asyncio.sleep(0)

        assert len(disconnected) == 1
        assert disconnected[0].state == VideoSessionState.ENDED


class TestSIPVideoBackendSendVideo:
    async def test_send_video_bytes(self, backend, mock_rtp_bridge, mock_video_bridge, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_session = backend.video.get_session("session-1")
        video_cs = mock_video_bridge.VideoCallSession.return_value

        await backend.video.send_video(video_session, b"\x65\x00\x01")
        video_cs.send_frame.assert_called_once_with([b"\x65\x00\x01"], 0)

    async def test_send_video_no_session(self, backend):
        dummy = VideoSession(id="nonexistent", room_id="r", participant_id="p", channel_id="c")
        await backend.video.send_video(dummy, b"\x00")  # no error


class TestSIPVideoBackendProperties:
    def test_name(self, backend):
        assert backend.name == "SIP-AV"

    async def test_list_video_sessions(self, backend, av_offer):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        sessions = backend.video.list_sessions("room-1")
        assert len(sessions) == 1

    async def test_disconnect_closes_video(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        state = backend._session_states.get("session-1")
        assert state is not None

        await backend.disconnect(state.session)

        # Video close runs as a fire-and-forget task via _cleanup_session —
        # yield to the event loop so the task completes.
        await asyncio.sleep(0)

        video_cs = mock_video_bridge.VideoCallSession.return_value
        video_cs.close.assert_awaited_once()


class TestSIPVideoBackendVideoTap:
    async def test_add_video_tap(self, backend, mock_rtp_bridge, mock_video_bridge, av_offer):
        """Video taps receive frames alongside the primary callback."""
        received_primary: list[VideoFrame] = []
        received_tap: list[VideoFrame] = []
        backend.video.on_video_received(lambda _s, f: received_primary.append(f))
        backend.video.add_video_tap(lambda _s, f: received_tap.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_video_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_primary) == 1
        assert len(received_tap) == 1
        assert received_tap[0].codec == "h264"

    async def test_tap_works_without_primary_callback(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        """Video taps fire even when no primary callback is set."""
        received_tap: list[VideoFrame] = []
        backend.video.add_video_tap(lambda _s, f: received_tap.append(f))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_video_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00\x01", 90000, True)

        assert len(received_tap) == 1

    async def test_multiple_taps(self, backend, mock_rtp_bridge, mock_video_bridge, av_offer):
        """Multiple taps all receive frames."""
        counts = [0, 0]

        def make_tap(idx):
            def tap(_s, _f):
                counts[idx] += 1

            return tap

        backend.video.add_video_tap(make_tap(0))
        backend.video.add_video_tap(make_tap(1))

        call = _FakeCall(sdp_offer=av_offer)
        await backend._handle_invite(call)

        video_cs = mock_video_bridge.VideoCallSession.return_value
        video_cs.on_frame(b"\x65\x00", 90000, True)
        video_cs.on_frame(b"\x41\x00", 93600, False)

        assert counts == [2, 2]


class TestSIPVideoBackendReinvite:
    async def test_reinvite_uses_combined_answer(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
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


class TestSIPVideoBackendAnswerMirrorsTheOffer:
    """RMK-353: the answer has one m-line per offered stream (RFC 3264 §6).

    aiosipua answers audio with the offer's video line refused (port 0); the
    backend appended the negotiated video after it, so a softphone got three
    m-lines for a two-stream offer and saw its video refused.
    """

    async def test_the_negotiated_video_replaces_the_refused_stub(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        audio_answer = mock_rtp_bridge.CallSession.return_value.sdp_answer
        stub = MagicMock(media="video", port=0)
        audio_answer.media = [MagicMock(media="audio"), stub]
        call = _FakeCall(sdp_offer=av_offer)

        await backend._handle_invite(call)

        negotiated = mock_video_bridge.VideoCallSession.return_value.sdp_answer.video
        media = call._accepted_sdp.media
        assert [m.media for m in media] == ["audio", "video"]
        assert media[1] is negotiated

    async def test_an_answer_without_a_stub_gets_the_video_appended(
        self, backend, mock_rtp_bridge, mock_video_bridge, av_offer
    ):
        call = _FakeCall(sdp_offer=av_offer)

        await backend._handle_invite(call)

        negotiated = mock_video_bridge.VideoCallSession.return_value.sdp_answer.video
        assert [m.media for m in call._accepted_sdp.media] == ["audio", "video"]
        assert call._accepted_sdp.media[1] is negotiated


class TestSIPVideoBackendSendsWholeFrames:
    """RMK-353: an Annex B frame, as a VideoChunk or a VideoFrame, is sent as one frame."""

    @pytest.mark.parametrize("sync", [False, True], ids=["send_video", "send_video_sync"])
    async def test_an_annex_b_frame_is_one_frame_of_several_nal_units(
        self, backend, mock_video_bridge, av_offer, sync
    ):
        await backend._handle_invite(_FakeCall(sdp_offer=av_offer))
        video_session = backend.video.get_session("session-1")
        video_cs = mock_video_bridge.VideoCallSession.return_value
        data = b"\x00\x00\x00\x01\x67\x42\x00\x00\x00\x01\x68\xce\x00\x00\x01\x65\x88"

        async def frames():
            yield VideoChunk(data=data, keyframe=True, timestamp_ms=0)

        if sync:
            frame = VideoFrame(data=data, keyframe=True, timestamp_ms=0)
            backend.video.send_video_sync(video_session, frame)
            await asyncio.sleep(0)
        else:
            await backend.video.send_video(video_session, frames())

        video_cs.send_frame.assert_called_once_with(
            [b"\x67\x42", b"\x68\xce", b"\x65\x88"], 0, True
        )

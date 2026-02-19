"""Tests for the SIP voice backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession, VoiceSessionState
from roomkit.voice.pipeline.dtmf.base import DTMFEvent

# ---------------------------------------------------------------------------
# Helpers — mock aiosipua objects
# ---------------------------------------------------------------------------

BASIC_SDP_OFFER = (
    "v=0\r\n"
    "o=- 1234 1234 IN IP4 10.0.0.1\r\n"
    "s=-\r\n"
    "c=IN IP4 10.0.0.1\r\n"
    "t=0 0\r\n"
    "m=audio 20000 RTP/AVP 0 8 101\r\n"
    "a=rtpmap:0 PCMU/8000\r\n"
    "a=rtpmap:8 PCMA/8000\r\n"
    "a=rtpmap:101 telephone-event/8000\r\n"
    "a=fmtp:101 0-16\r\n"
    "a=ptime:20\r\n"
    "a=sendrecv\r\n"
)


def _make_mock_sdp_offer() -> MagicMock:
    """A mock SDP offer with audio at 10.0.0.1:20000."""
    offer = MagicMock()
    offer.rtp_address = ("10.0.0.1", 20000)
    offer.audio = MagicMock()
    offer.audio.codecs = [
        MagicMock(payload_type=0, clock_rate=8000),
        MagicMock(payload_type=8, clock_rate=8000),
    ]
    return offer


def _make_mock_call_session() -> MagicMock:
    """A mock CallSession (aiosipua.rtp_bridge.CallSession)."""
    session = MagicMock()
    session.sdp_answer = MagicMock()
    session.chosen_payload_type = 0
    session.remote_addr = ("10.0.0.1", 20000)
    session.codec_sample_rate = 8000
    session.clock_rate = 8000
    session.on_audio = None
    session.on_dtmf = None
    session.start = AsyncMock()
    session.close = AsyncMock()
    session.send_audio_pcm = MagicMock()
    return session


def _make_mock_incoming_call(
    *,
    call_id: str = "test-call-1",
    caller: str = "sip:alice@example.com",
    callee: str = "sip:bob@example.com",
    sdp_offer: Any = None,
    room_id: str | None = "room-42",
    session_id: str | None = "sess-001",
    x_headers: dict[str, str] | None = None,
) -> MagicMock:
    """A mock IncomingCall."""
    call = MagicMock()
    call.call_id = call_id
    call.caller = caller
    call.callee = callee
    call.sdp_offer = sdp_offer if sdp_offer is not None else _make_mock_sdp_offer()
    call.room_id = room_id
    call.session_id = session_id
    call.x_headers = x_headers or {"X-Room-ID": "room-42", "X-Session-ID": "sess-001"}
    call.source_addr = ("10.0.0.1", 5060)
    call.dialog = MagicMock()
    call.dialog.state = MagicMock()
    call.dialog.state.value = "confirmed"
    call.ringing = MagicMock()
    call.accept = MagicMock()
    call.reject = MagicMock()
    return call


def _make_mock_aiosipua_module() -> MagicMock:
    """Mock the aiosipua module."""
    mod = MagicMock()
    # UdpSipTransport
    mod.UdpSipTransport.return_value = MagicMock()
    mod.UdpSipTransport.return_value.start = AsyncMock()
    mod.UdpSipTransport.return_value.stop = AsyncMock()
    mod.UdpSipTransport.return_value.local_addr = ("0.0.0.0", 5060)
    # SipUAS
    mock_uas = MagicMock()
    mock_uas.start = AsyncMock()
    mock_uas.stop = AsyncMock()
    mock_uas.on_invite = None
    mock_uas.on_bye = None
    mod.SipUAS.return_value = mock_uas
    # SipUAC
    mock_uac = MagicMock()
    mod.SipUAC.return_value = mock_uac
    # DialogState
    mod.DialogState = MagicMock()
    mod.DialogState.CONFIRMED = "confirmed"
    return mod


def _make_mock_rtp_bridge_module(call_session: MagicMock | None = None) -> MagicMock:
    """Mock the aiosipua.rtp_bridge module."""
    mod = MagicMock()
    if call_session is None:
        call_session = _make_mock_call_session()
    mod.CallSession.return_value = call_session
    return mod


async def _chunks_from_bytes(data: bytes, chunk_size: int = 320) -> Any:
    """Yield AudioChunks from a bytes buffer."""
    offset = 0
    while offset < len(data):
        yield AudioChunk(data=data[offset : offset + chunk_size], sample_rate=8000)
        offset += chunk_size
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_call_session() -> MagicMock:
    return _make_mock_call_session()


@pytest.fixture
def mock_aiosipua() -> MagicMock:
    return _make_mock_aiosipua_module()


@pytest.fixture
def mock_rtp_bridge(mock_call_session: MagicMock) -> MagicMock:
    return _make_mock_rtp_bridge_module(mock_call_session)


@pytest.fixture
def backend(mock_aiosipua: MagicMock, mock_rtp_bridge: MagicMock) -> Any:
    """Return a SIPVoiceBackend with mocked aiosipua."""
    with (
        patch("roomkit.voice.backends.sip._import_aiosipua", return_value=mock_aiosipua),
        patch(
            "roomkit.voice.backends.sip._import_rtp_bridge",
            return_value=mock_rtp_bridge,
        ),
    ):
        from roomkit.voice.backends.sip import SIPVoiceBackend

        b = SIPVoiceBackend(
            local_sip_addr=("0.0.0.0", 5060),
            local_rtp_ip="10.0.0.5",
            rtp_port_start=10000,
            rtp_port_end=20000,
        )
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSIPVoiceBackendInit:
    def test_name(self, backend: Any) -> None:
        assert backend.name == "SIP"

    def test_capabilities(self, backend: Any) -> None:
        assert VoiceCapability.DTMF_SIGNALING in backend.capabilities
        assert VoiceCapability.INTERRUPTION in backend.capabilities

    def test_stores_config(self, backend: Any) -> None:
        assert backend._local_sip_addr == ("0.0.0.0", 5060)
        assert backend._local_rtp_ip == "10.0.0.5"
        assert backend._rtp_port_start == 10000
        assert backend._rtp_port_end == 20000
        assert backend._supported_codecs == [9, 0, 8]
        assert backend._dtmf_payload_type == 101

    def test_default_codecs(self, backend: Any) -> None:
        assert backend._supported_codecs == [9, 0, 8]


class TestStart:
    async def test_start_creates_transport_uas_uac(
        self, backend: Any, mock_aiosipua: MagicMock
    ) -> None:
        await backend.start()

        mock_aiosipua.UdpSipTransport.assert_called_once_with(local_addr=("0.0.0.0", 5060))
        mock_aiosipua.SipUAS.assert_called_once()
        mock_aiosipua.SipUAC.assert_called_once()

    async def test_start_registers_callbacks(self, backend: Any, mock_aiosipua: MagicMock) -> None:
        await backend.start()

        uas = mock_aiosipua.SipUAS.return_value
        assert uas.on_invite is not None
        assert uas.on_bye is not None

    async def test_start_calls_uas_start(self, backend: Any, mock_aiosipua: MagicMock) -> None:
        await backend.start()

        uas = mock_aiosipua.SipUAS.return_value
        uas.start.assert_awaited_once()


class TestHandleInvite:
    async def test_invite_creates_session(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()

        await backend._handle_invite(call)

        # Session should be created
        assert len(backend._sessions) == 1
        session = list(backend._sessions.values())[0]
        assert session.state == VoiceSessionState.ACTIVE
        assert session.room_id == "room-42"
        assert session.participant_id == "sess-001"
        assert session.metadata["backend"] == "sip"
        assert session.metadata["call_id"] == "test-call-1"

    async def test_invite_accepts_call(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()

        await backend._handle_invite(call)

        call.ringing.assert_called_once()
        call.accept.assert_called_once_with(mock_call_session.sdp_answer)
        mock_call_session.start.assert_awaited_once()

    async def test_invite_no_sdp_rejects(self, backend: Any) -> None:
        call = _make_mock_incoming_call(sdp_offer=None)
        call.sdp_offer = None

        await backend._handle_invite(call)

        call.reject.assert_called_once_with(488, "Not Acceptable Here")
        assert len(backend._sessions) == 0

    async def test_invite_negotiation_failure_rejects(
        self, backend: Any, mock_rtp_bridge: MagicMock
    ) -> None:
        mock_rtp_bridge.CallSession.side_effect = ValueError("No matching codec")
        call = _make_mock_incoming_call()

        await backend._handle_invite(call)

        call.reject.assert_called_once_with(488, "Not Acceptable Here")
        assert len(backend._sessions) == 0

    async def test_invite_wires_audio_callback(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        received: list[tuple[VoiceSession, AudioFrame]] = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        # Simulate audio from CallSession
        assert mock_call_session.on_audio is not None
        pcm = b"\x00\x01" * 160
        mock_call_session.on_audio(pcm, 0)

        assert len(received) == 1
        sess, frame = received[0]
        assert frame.data == pcm
        assert frame.sample_rate == 8000

    async def test_invite_wires_dtmf_callback(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        received: list[tuple[VoiceSession, DTMFEvent]] = []
        backend.on_dtmf_received(lambda s, e: received.append((s, e)))

        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        # Simulate DTMF from CallSession
        assert mock_call_session.on_dtmf is not None
        mock_call_session.on_dtmf("5", 1280)

        assert len(received) == 1
        sess, event = received[0]
        assert event.digit == "5"
        assert event.duration_ms == 160.0

    async def test_invite_fires_on_call_callback(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        calls: list[VoiceSession] = []
        backend.on_call(lambda s: calls.append(s))

        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        assert len(calls) == 1
        assert calls[0].room_id == "room-42"

    async def test_invite_stores_mappings(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert backend._call_sessions[session.id] is mock_call_session
        assert backend._incoming_calls[session.id] is call
        assert backend._call_to_session["test-call-1"] == session.id
        assert backend._send_timestamps[session.id] == 0
        assert backend._codec_rates[session.id] == 8000
        assert backend._clock_rates[session.id] == 8000


class TestHandleBye:
    async def test_bye_cleans_up_session(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.state == VoiceSessionState.ACTIVE

        # Simulate BYE
        backend._handle_bye(call, MagicMock())

        # Allow async task (close) to run
        await asyncio.sleep(0.01)

        assert session.state == VoiceSessionState.ENDED
        assert len(backend._sessions) == 0
        assert len(backend._call_sessions) == 0
        assert len(backend._incoming_calls) == 0
        assert len(backend._call_to_session) == 0

    async def test_bye_fires_disconnect_callback(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        disconnected: list[VoiceSession] = []
        backend.on_call_disconnected(lambda s: disconnected.append(s))

        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        backend._handle_bye(call, MagicMock())
        await asyncio.sleep(0.01)

        assert len(disconnected) == 1
        assert disconnected[0].state == VoiceSessionState.ENDED

    async def test_bye_unknown_call_id(self, backend: Any) -> None:
        call = _make_mock_incoming_call(call_id="unknown")

        # Should not raise
        backend._handle_bye(call, MagicMock())


class TestConnect:
    async def test_connect_returns_existing_session(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session_id = list(backend._sessions.keys())[0]
        result = await backend.connect(
            "room-42", "user-1", "voice-1", metadata={"session_id": session_id}
        )
        assert result.id == session_id
        assert result.room_id == "room-42"
        assert result.channel_id == "voice-1"

    async def test_connect_no_session_id_raises(self, backend: Any) -> None:
        with pytest.raises(ValueError, match="session_id"):
            await backend.connect("room-1", "user-1", "voice-1")

    async def test_connect_unknown_session_id_raises(self, backend: Any) -> None:
        with pytest.raises(ValueError, match="session_id"):
            await backend.connect(
                "room-1", "user-1", "voice-1", metadata={"session_id": "nonexistent"}
            )


class TestDisconnect:
    async def test_disconnect_sends_bye_and_closes(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        # Set dialog state so disconnect() tries to send BYE
        call.dialog.state = "confirmed"
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        await backend.disconnect(session)

        mock_call_session.close.assert_awaited_once()
        assert session.state == VoiceSessionState.ENDED
        assert len(backend._sessions) == 0

    async def test_disconnect_cleans_state(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        await backend.disconnect(session)

        assert backend.get_session(session.id) is None
        assert session.id not in backend._call_sessions
        assert session.id not in backend._incoming_calls
        assert session.id not in backend._send_timestamps
        assert session.id not in backend._codec_rates
        assert session.id not in backend._clock_rates


class TestSendAudio:
    async def test_send_audio_bytes(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]

        # 20ms frame at 8kHz = 160 samples = 320 bytes
        pcm = b"\x00\x01" * 160
        await backend.send_audio(session, pcm)
        backend.end_of_response(session)
        await asyncio.sleep(0.15)

        mock_call_session.send_audio_pcm.assert_called_once_with(pcm, 0)

    async def test_send_audio_bytes_multiple_frames(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]

        # Two 20ms frames = 640 bytes
        pcm = b"\x00\x01" * 320
        await backend.send_audio(session, pcm)
        backend.end_of_response(session)
        await asyncio.sleep(0.15)

        calls = mock_call_session.send_audio_pcm.call_args_list
        assert len(calls) == 2
        assert calls[0].args == (pcm[:320], 0)
        assert calls[1].args == (pcm[320:], 160)

    async def test_send_audio_stream(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]

        pcm = b"\x00\x01" * 160
        await backend.send_audio(session, _chunks_from_bytes(pcm))

        mock_call_session.send_audio_pcm.assert_called_once_with(pcm, 0)

    async def test_send_audio_no_session(self, backend: Any) -> None:
        session = VoiceSession(
            id="nonexistent",
            room_id="room-1",
            participant_id="user-1",
            channel_id="voice-1",
        )
        # Should not raise
        await backend.send_audio(session, b"\x00\x01" * 160)


class TestCancelAudio:
    async def test_cancel_not_playing(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert await backend.cancel_audio(session) is False

    async def test_is_playing(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert backend.is_playing(session) is False


class TestPortAllocation:
    def test_unique_port_allocation(self, backend: Any) -> None:
        p1 = backend._allocate_rtp_port()
        p2 = backend._allocate_rtp_port()
        p3 = backend._allocate_rtp_port()

        # Ports must be unique even-numbered values in the configured range
        assert len({p1, p2, p3}) == 3
        for p in (p1, p2, p3):
            assert 10000 <= p < 20000
            assert p % 2 == 0

    def test_exhaustion_raises(self, backend: Any) -> None:
        # Exhaust all available ports
        allocated = []
        while backend._available_ports:
            allocated.append(backend._allocate_rtp_port())
        import pytest

        with pytest.raises(RuntimeError, match="No RTP ports available"):
            backend._allocate_rtp_port()

    def test_release_allows_reuse(self, backend: Any) -> None:
        p1 = backend._allocate_rtp_port()
        backend._release_rtp_port(p1)
        # After release, the port should be available again
        assert p1 in backend._available_ports
        assert p1 not in backend._allocated_ports


class TestSessionQueries:
    async def test_get_session(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        sessions = backend.list_sessions("room-42")
        assert len(sessions) == 1

        sessions = backend.list_sessions("other-room")
        assert len(sessions) == 0


class TestClose:
    async def test_close_disconnects_all(
        self,
        backend: Any,
        mock_call_session: MagicMock,
        mock_rtp_bridge: MagicMock,
        mock_aiosipua: MagicMock,
    ) -> None:
        await backend.start()

        call = _make_mock_incoming_call()
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        await backend.close()

        assert session.state == VoiceSessionState.ENDED
        assert len(backend._sessions) == 0

        uas = mock_aiosipua.SipUAS.return_value
        uas.stop.assert_awaited_once()


class TestXHeaderRouting:
    async def test_room_id_from_x_header(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call(room_id="custom-room")
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.room_id == "custom-room"

    async def test_room_id_fallback_to_call_id(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call(room_id=None, call_id="fallback-call")
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.room_id == "fallback-call"

    async def test_participant_id_from_session_header(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call(session_id="participant-99")
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.participant_id == "participant-99"

    async def test_participant_id_fallback_to_caller(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call(session_id=None, caller="sip:alice@example.com")
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.participant_id == "sip:alice@example.com"

    async def test_x_headers_stored_in_metadata(
        self, backend: Any, mock_call_session: MagicMock, mock_rtp_bridge: MagicMock
    ) -> None:
        call = _make_mock_incoming_call(
            x_headers={"X-Room-ID": "r1", "X-Tenant": "t1", "X-Language": "fr"}
        )
        await backend._handle_invite(call)

        session = list(backend._sessions.values())[0]
        assert session.metadata["x_headers"]["X-Language"] == "fr"


class TestCallbackRegistrations:
    def test_on_audio_received(self, backend: Any) -> None:
        cb = MagicMock()
        backend.on_audio_received(cb)
        assert backend._audio_received_callback is cb

    def test_on_barge_in(self, backend: Any) -> None:
        cb = MagicMock()
        backend.on_barge_in(cb)
        assert cb in backend._barge_in_callbacks

    def test_on_dtmf_received(self, backend: Any) -> None:
        cb = MagicMock()
        backend.on_dtmf_received(cb)
        assert cb in backend._dtmf_callbacks

    def test_on_call(self, backend: Any) -> None:
        cb = MagicMock()
        backend.on_call(cb)
        assert backend._on_call_callback is cb

    def test_on_call_disconnected(self, backend: Any) -> None:
        cb = MagicMock()
        backend.on_call_disconnected(cb)
        assert backend._on_disconnect_callback is cb


class TestMultipleCalls:
    async def test_two_concurrent_calls(self, backend: Any, mock_rtp_bridge: MagicMock) -> None:
        session1 = _make_mock_call_session()
        session2 = _make_mock_call_session()
        mock_rtp_bridge.CallSession.side_effect = [session1, session2]

        call1 = _make_mock_incoming_call(call_id="call-1", room_id="room-a", session_id="s1")
        call2 = _make_mock_incoming_call(call_id="call-2", room_id="room-b", session_id="s2")

        await backend._handle_invite(call1)
        await backend._handle_invite(call2)

        assert len(backend._sessions) == 2
        rooms = {s.room_id for s in backend._sessions.values()}
        assert rooms == {"room-a", "room-b"}

        assert backend.list_sessions("room-a") != []
        assert backend.list_sessions("room-b") != []

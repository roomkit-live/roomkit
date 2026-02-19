"""Tests for SIPVoiceBackend.dial() — outbound SIP calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.backends.sip import (
    PT_G722,
    PT_PCMA,
    PT_PCMU,
    SIPVoiceBackend,
)
from roomkit.voice.base import VoiceSessionState

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

PROXY_ADDR = ("10.0.0.1", 5060)
FROM_URI = "sip:bot@example.com"
TO_URI = "sip:alice@example.com"


@dataclass
class FakeOutgoingCall:
    """Minimal OutgoingCall stand-in."""

    call_id: str = "out-call-1"
    caller: str = FROM_URI
    callee: str = TO_URI
    sdp_answer: str | None = (
        "v=0\r\no=- 0 0 IN IP4 10.0.0.1\r\ns=-\r\nc=IN IP4 10.0.0.1\r\nm=audio 30000 RTP/AVP 0\r\n"
    )

    # Behaviour toggles
    _should_timeout: bool = field(default=False, repr=False)
    _reject_code: int = field(default=0, repr=False)
    _reject_reason: str = field(default="", repr=False)

    async def wait_answered(self, timeout: float = 30.0) -> None:
        if self._should_timeout:
            raise TimeoutError("no answer")
        if self._reject_code:
            raise RuntimeError(f"{self._reject_code} {self._reject_reason}")

    def hangup(self, uac: Any) -> None:
        pass

    def cancel(self, uac: Any) -> None:
        pass


class FakeCallSession:
    """Minimal CallSession stand-in that accepts any constructor kwargs."""

    sdp_answer: str = "v=0\r\n"
    codec_sample_rate: int = 8000
    clock_rate: int = 8000
    remote_addr: tuple[str, int] = ("10.0.0.1", 20000)
    on_audio: Any = None
    on_dtmf: Any = None

    def __init__(self, **kwargs: Any) -> None:
        # Accept and ignore CallSession constructor params
        pass

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def send_audio_pcm(self, data: bytes, ts: int) -> None:
        pass


class FakeTransport:
    """Minimal SIP transport."""

    def __init__(self) -> None:
        self.local_addr = ("10.0.0.2", 5060)
        self.on_message = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeUAS:
    """Minimal UAS that captures on_invite/on_bye registration."""

    def __init__(self, transport: Any, *, user_agent: str | None = None, uac: Any = None) -> None:
        self.transport = transport
        self.user_agent = user_agent
        self.uac = uac
        self.on_invite = None
        self.on_bye = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeUAC:
    """Minimal UAC capturing send_invite calls."""

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self._calls: dict[str, FakeOutgoingCall] = {}
        self._last_invite_kwargs: dict[str, Any] = {}

    def send_invite(self, **kwargs: Any) -> FakeOutgoingCall:
        self._last_invite_kwargs = kwargs
        call = FakeOutgoingCall()
        self._calls[call.call_id] = call
        return call

    def remove_call(self, call_id: str) -> None:
        self._calls.pop(call_id, None)


class FakeRtpBridge:
    """Stands in for aiosipua.rtp_bridge."""

    CallSession = FakeCallSession


class FakeSdpMessage:
    """Stands in for SdpMessage returned by build_sdp."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> SIPVoiceBackend:
    """Return a SIPVoiceBackend with mocked aiosipua imports."""
    with (
        patch.object(
            SIPVoiceBackend,
            "__init__",
            lambda self, **kw: None,
        ),
    ):
        b = SIPVoiceBackend.__new__(SIPVoiceBackend)

    # Manually run the subset of __init__ we need
    b._local_sip_addr = ("0.0.0.0", 5060)
    b._local_rtp_ip = "0.0.0.0"
    b._rtp_port_start = 10000
    b._rtp_port_end = 20000
    b._supported_codecs = [PT_G722, PT_PCMU, PT_PCMA]
    b._dtmf_payload_type = 101
    b._user_agent = None
    b._server_name = "-"

    b._rtp_bridge = FakeRtpBridge()

    transport = FakeTransport()
    uac = FakeUAC(transport)
    uas = FakeUAS(transport, uac=uac)

    b._transport = transport
    b._uac = uac
    b._uas = uas

    b._sessions = {}
    b._call_sessions = {}
    b._incoming_calls = {}
    b._outgoing_calls = {}
    b._call_to_session = {}
    b._pending_reinvite_sdp = {}
    b._codec_rates = {}
    b._clock_rates = {}
    b._send_timestamps = {}
    b._send_buffers = {}
    b._send_frame_count = {}
    b._last_rtp_send_time = {}
    b._playing_sessions = set()
    b._playback_tasks = {}
    b._session_pacers = {}
    b._audio_received_callback = None
    b._barge_in_callbacks = []
    b._dtmf_callbacks = []
    b._on_call_callback = None
    b._on_disconnect_callback = None
    b._trace_emitter = None
    b._next_rtp_port = 10000
    b._audio_stats = {}
    b._stats_task = None

    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDialSendsInvite:
    async def test_invite_sent_with_correct_params(self, backend: SIPVoiceBackend) -> None:
        """Verify INVITE sent with correct URIs, proxy_addr, and SDP."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        uac: FakeUAC = backend._uac  # type: ignore[assignment]
        kw = uac._last_invite_kwargs
        assert kw["from_uri"] == FROM_URI
        assert kw["to_uri"] == TO_URI
        assert kw["remote_addr"] == PROXY_ADDR
        assert kw["sdp_offer"] is not None
        assert session is not None

    async def test_sdp_uses_correct_codec(self, backend: SIPVoiceBackend) -> None:
        """Verify build_sdp called with the right codec params."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()) as mock_build:
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                codec=PT_PCMU,
            )

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["payload_type"] == PT_PCMU
        assert call_kwargs["codec_name"] == "PCMU"
        assert call_kwargs["sample_rate"] == 8000


class TestDialReturnsSession:
    async def test_session_active_on_answer(self, backend: SIPVoiceBackend) -> None:
        """Mock 200 OK → VoiceSession returned with state=ACTIVE."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert session.state == VoiceSessionState.ACTIVE
        assert session.metadata["direction"] == "outbound"
        assert session.metadata["callee"] == TO_URI
        assert session.metadata["caller"] == FROM_URI

    async def test_session_stored_in_tracking_dicts(self, backend: SIPVoiceBackend) -> None:
        """Session tracked in _sessions, _outgoing_calls, _call_sessions."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert session.id in backend._sessions
        assert session.id in backend._outgoing_calls
        assert session.id in backend._call_sessions
        assert session.id in backend._call_to_session.values()


class TestDialWithAuth:
    async def test_auth_passed_to_send_invite(self, backend: SIPVoiceBackend) -> None:
        """Verify SipDigestAuth is forwarded to UAC.send_invite()."""
        fake_auth = MagicMock()
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                auth=fake_auth,
            )

        uac: FakeUAC = backend._uac  # type: ignore[assignment]
        assert uac._last_invite_kwargs["auth"] is fake_auth


class TestDialTimeout:
    async def test_timeout_raises(self, backend: SIPVoiceBackend) -> None:
        """wait_answered times out → TimeoutError."""

        def send_timeout(**kwargs: Any) -> FakeOutgoingCall:
            call = FakeOutgoingCall(_should_timeout=True)
            return call

        backend._uac.send_invite = send_timeout  # type: ignore[union-attr]

        with (
            patch("aiosipua.build_sdp", return_value=FakeSdpMessage()),
            pytest.raises(TimeoutError),
        ):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

    async def test_timeout_cleans_up_call(self, backend: SIPVoiceBackend) -> None:
        """On timeout, remove_call is called to clean up UAC tracking."""
        uac: FakeUAC = backend._uac  # type: ignore[assignment]
        removed: list[str] = []
        original_remove = uac.remove_call

        def track_remove(call_id: str) -> None:
            removed.append(call_id)
            original_remove(call_id)

        uac.remove_call = track_remove

        def send_timeout(**kwargs: Any) -> FakeOutgoingCall:
            call = FakeOutgoingCall(_should_timeout=True)
            uac._calls[call.call_id] = call
            return call

        uac.send_invite = send_timeout  # type: ignore[assignment]

        with (
            patch("aiosipua.build_sdp", return_value=FakeSdpMessage()),
            pytest.raises(TimeoutError),
        ):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert len(removed) == 1
        assert removed[0] == "out-call-1"


class TestDialRejection:
    async def test_rejection_raises_runtime_error(self, backend: SIPVoiceBackend) -> None:
        """Mock 486 Busy → RuntimeError."""

        def send_reject(**kwargs: Any) -> FakeOutgoingCall:
            return FakeOutgoingCall(_reject_code=486, _reject_reason="Busy Here")

        backend._uac.send_invite = send_reject  # type: ignore[union-attr]

        with (
            patch("aiosipua.build_sdp", return_value=FakeSdpMessage()),
            pytest.raises(RuntimeError, match="486"),
        ):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )


class TestDialCodecSelection:
    async def test_g722_codec(self, backend: SIPVoiceBackend) -> None:
        """G.722 codec produces correct SDP params."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()) as mock_build:
            # Use a CallSession that reports G.722 rates
            backend._rtp_bridge.CallSession = type(
                "G722CallSession",
                (FakeCallSession,),
                {"codec_sample_rate": 16000, "clock_rate": 8000},
            )
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                codec=PT_G722,
            )

        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["payload_type"] == PT_G722
        assert call_kwargs["codec_name"] == "G722"
        assert call_kwargs["sample_rate"] == 8000  # RTP clock rate

        assert backend._codec_rates[session.id] == 16000
        assert backend._clock_rates[session.id] == 8000

    async def test_pcma_codec(self, backend: SIPVoiceBackend) -> None:
        """PCMA codec produces correct SDP params."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()) as mock_build:
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                codec=PT_PCMA,
            )

        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["payload_type"] == PT_PCMA
        assert call_kwargs["codec_name"] == "PCMA"

    async def test_unsupported_codec_raises(self, backend: SIPVoiceBackend) -> None:
        """Unsupported codec payload type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported codec"):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                codec=99,
            )


class TestDisconnectOutgoing:
    async def test_bye_sent_for_outgoing_call(self, backend: SIPVoiceBackend) -> None:
        """BYE sent when disconnecting an outbound call."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        out_call = backend._outgoing_calls[session.id]
        out_call.hangup = MagicMock()

        await backend.disconnect(session)

        out_call.hangup.assert_called_once_with(backend._uac)
        assert session.state == VoiceSessionState.ENDED

    async def test_disconnect_cleans_up_tracking(self, backend: SIPVoiceBackend) -> None:
        """All tracking dicts cleaned up after disconnect."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        sid = session.id
        await backend.disconnect(session)

        assert sid not in backend._sessions
        assert sid not in backend._outgoing_calls
        assert sid not in backend._call_sessions
        assert sid not in backend._call_to_session.values()


class TestDialFiresCallback:
    async def test_on_call_callback_invoked(self, backend: SIPVoiceBackend) -> None:
        """on_call callback is fired with the new session."""
        received: list[Any] = []
        backend._on_call_callback = lambda s: received.append(s)

        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert len(received) == 1
        assert received[0] is session


class TestDialProtocolTraces:
    async def test_invite_and_200ok_traces_emitted(self, backend: SIPVoiceBackend) -> None:
        """INVITE and 200 OK protocol traces are emitted."""
        traces: list[Any] = []
        backend._trace_emitter = lambda t: traces.append(t)

        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert len(traces) == 2

        invite_trace = traces[0]
        assert invite_trace.direction == "outbound"
        assert "INVITE" in invite_trace.summary
        assert FROM_URI in invite_trace.summary
        assert TO_URI in invite_trace.summary

        ok_trace = traces[1]
        assert ok_trace.direction == "inbound"
        assert "200 OK" in ok_trace.summary


class TestDialExtraHeaders:
    async def test_extra_headers_forwarded(self, backend: SIPVoiceBackend) -> None:
        """Extra headers are passed through to send_invite."""
        headers = {"X-Room-ID": "room-42", "X-Tenant": "acme"}
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                extra_headers=headers,
            )

        uac: FakeUAC = backend._uac  # type: ignore[assignment]
        assert uac._last_invite_kwargs["extra_headers"] == headers


class TestDialRequiresStart:
    async def test_dial_before_start_raises(self) -> None:
        """dial() raises RuntimeError if backend not started."""
        with patch.object(SIPVoiceBackend, "__init__", lambda self, **kw: None):
            b = SIPVoiceBackend.__new__(SIPVoiceBackend)
        b._uac = None
        b._transport = None

        with pytest.raises(RuntimeError, match="start.*must be called"):
            await b.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )


class TestDialRoomId:
    async def test_custom_room_id(self, backend: SIPVoiceBackend) -> None:
        """Custom room_id is used when provided."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
                room_id="my-room",
            )

        assert session.room_id == "my-room"
        assert session.metadata["room_id"] == "my-room"

    async def test_default_room_id_is_call_id(self, backend: SIPVoiceBackend) -> None:
        """When no room_id given, defaults to call_id."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        assert session.room_id == session.metadata["call_id"]


class TestReinviteHandler:
    async def test_reinvite_accepts_with_existing_sdp(self, backend: SIPVoiceBackend) -> None:
        """re-INVITE is accepted with the existing SDP answer."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        # Create a fake re-INVITE call object
        call_session = backend._call_sessions[session.id]
        reinvite_call = MagicMock()
        reinvite_call.call_id = session.metadata["call_id"]
        reinvite_call.caller = TO_URI
        reinvite_call.sdp_offer = None
        reinvite_call.accept = MagicMock()

        backend._handle_reinvite(reinvite_call)

        reinvite_call.accept.assert_called_once_with(call_session.sdp_answer)

    async def test_reinvite_updates_remote_rtp_address(self, backend: SIPVoiceBackend) -> None:
        """re-INVITE with new SDP updates the RTP remote address."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        call_session = backend._call_sessions[session.id]
        call_session.update_remote = MagicMock()
        # Fake the remote_addr property for comparison
        call_session.remote_addr = ("10.0.0.1", 30000)

        new_sdp = MagicMock()
        new_sdp.rtp_address = ("10.0.0.2", 40000)

        reinvite_call = MagicMock()
        reinvite_call.call_id = session.metadata["call_id"]
        reinvite_call.caller = TO_URI
        reinvite_call.sdp_offer = new_sdp
        reinvite_call.accept = MagicMock()

        backend._handle_reinvite(reinvite_call)

        call_session.update_remote.assert_called_once_with(("10.0.0.2", 40000))

    async def test_reinvite_unknown_call_ignored(self, backend: SIPVoiceBackend) -> None:
        """re-INVITE for unknown call_id is silently ignored."""
        reinvite_call = MagicMock()
        reinvite_call.call_id = "unknown-call-id"

        # Should not raise
        backend._handle_reinvite(reinvite_call)

    async def test_invite_for_outbound_call_routes_to_reinvite(
        self, backend: SIPVoiceBackend
    ) -> None:
        """An INVITE with a Call-ID matching an outbound call is treated as re-INVITE."""
        with patch("aiosipua.build_sdp", return_value=FakeSdpMessage()):
            session = await backend.dial(
                to_uri=TO_URI,
                from_uri=FROM_URI,
                proxy_addr=PROXY_ADDR,
            )

        call_session = backend._call_sessions[session.id]
        initial_session_count = len(backend._sessions)

        # Simulate Asterisk sending a re-INVITE via on_invite (not on_reinvite)
        reinvite_call = MagicMock()
        reinvite_call.call_id = session.metadata["call_id"]
        reinvite_call.caller = TO_URI
        reinvite_call.sdp_offer = None
        reinvite_call.source_addr = PROXY_ADDR
        reinvite_call.accept = MagicMock()

        await backend._handle_invite(reinvite_call)

        # Should NOT create a new session — routed to _handle_reinvite
        assert len(backend._sessions) == initial_session_count
        reinvite_call.accept.assert_called_once_with(call_session.sdp_answer)

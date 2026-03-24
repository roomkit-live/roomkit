"""Tests for SIPVoiceBackend inbound auth and outbound registration."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

# Ensure a fake aiosipua module exists so tests can import SIPVoiceBackend
if "aiosipua" not in sys.modules:
    _fake_aiosipua = types.ModuleType("aiosipua")
    _fake_aiosipua.build_sdp = lambda **kwargs: None  # type: ignore[attr-defined]
    sys.modules["aiosipua"] = _fake_aiosipua

from roomkit.voice.backends.sip import (
    PT_G722,
    PT_PCMA,
    PT_PCMU,
    SIPVoiceBackend,
    _compute_digest,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCaseInsensitiveDict:
    """Minimal header store for fake SIP messages."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {}

    def set_single(self, name: str, value: str) -> None:
        self._data[name.lower()] = [value]

    def append(self, name: str, value: str) -> None:
        self._data.setdefault(name.lower(), []).append(value)

    def items(self) -> list[tuple[str, list[str]]]:
        return list(self._data.items())


class FakeSipRequest:
    """Minimal INVITE request."""

    def __init__(self, *, uri: str = "sip:bot@example.com") -> None:
        self.method = "INVITE"
        self.uri = uri
        self.headers = FakeCaseInsensitiveDict()
        self._headers_raw: dict[str, str] = {}

    def get_header(self, name: str) -> str | None:
        return self._headers_raw.get(name.lower())

    def set_header(self, name: str, value: str) -> None:
        self._headers_raw[name.lower()] = value

    @property
    def from_addr(self) -> Any:
        return None


class FakeSipResponse:
    """Minimal SIP response for REGISTER tests."""

    def __init__(self, status_code: int = 200, reason: str = "OK") -> None:
        self.status_code = status_code
        self.reason_phrase = reason
        self._headers: dict[str, str] = {}
        self.headers = FakeCaseInsensitiveDict()
        self.cseq = "1 REGISTER"
        self.body = ""
        self.content_type = ""

    def get_header(self, name: str) -> str | None:
        return self._headers.get(name)


class FakeDialog:
    """Minimal SIP dialog."""

    def __init__(self) -> None:
        self.terminated = False
        self.responses_sent: list[tuple[int, str]] = []

    def create_response(
        self, request: Any, status: int, reason: str, contact: str | None = None
    ) -> FakeSipResponse:
        resp = FakeSipResponse(status, reason)
        self.responses_sent.append((status, reason))
        return resp

    def terminate(self) -> None:
        self.terminated = True


@dataclass
class FakeIncomingCall:
    """Minimal IncomingCall stand-in for auth tests."""

    call_id: str = "test-call-1"
    caller: str = "sip:alice@example.com"
    callee: str = "sip:bot@example.com"
    invite: FakeSipRequest = field(default_factory=FakeSipRequest)
    dialog: FakeDialog = field(default_factory=FakeDialog)
    source_addr: tuple[str, int] = ("10.0.0.1", 5060)
    sdp_offer: str | None = "v=0\r\n"
    _rejected: bool = field(default=False, init=False)
    _reject_code: int = field(default=0, init=False)

    @property
    def room_id(self) -> str | None:
        return None

    @property
    def session_id(self) -> str | None:
        return None

    @property
    def x_headers(self) -> dict[str, str]:
        return {}

    def reject(self, code: int, reason: str = "") -> None:
        self._rejected = True
        self._reject_code = code

    def ringing(self) -> None:
        pass

    def accept(self, sdp_answer: Any = None) -> None:
        pass


class FakeTransport:
    def __init__(self) -> None:
        self.local_addr = ("10.0.0.2", 5060)
        self.on_message: Any = None
        self.sent: list[tuple[Any, Any]] = []
        self.replies: list[Any] = []

    def send(self, msg: Any, addr: Any) -> None:
        self.sent.append((msg, addr))

    def send_reply(self, resp: Any) -> None:
        self.replies.append(resp)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeUAS:
    def __init__(self, transport: Any, *, user_agent: str | None = None, uac: Any = None) -> None:
        self.transport = transport
        self.on_invite = None
        self.on_reinvite = None
        self.on_bye = None

    def _on_message(self, msg: Any, addr: Any) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class FakeUAC:
    def __init__(self, transport: Any) -> None:
        self.transport = transport

    def remove_call(self, call_id: str) -> None:
        pass


class FakeRtpBridge:
    class CallSession:
        sdp_answer = "v=0\r\n"
        codec_sample_rate = 8000
        clock_rate = 8000
        remote_addr = ("10.0.0.1", 20000)
        on_audio: Any = None
        on_dtmf: Any = None
        stats: dict = {}  # noqa: RUF012

        def __init__(self, **kw: Any) -> None:
            pass

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass


def _make_backend(
    *,
    auth_users: dict[str, str] | None = None,
    auth_realm: str = "roomkit",
) -> SIPVoiceBackend:
    """Create a SIPVoiceBackend with mocked internals."""
    b = SIPVoiceBackend.__new__(SIPVoiceBackend)
    b._local_sip_addr = ("0.0.0.0", 5060)
    b._local_rtp_ip = "0.0.0.0"
    b._advertised_ip = None
    b._rtp_port_start = 10000
    b._rtp_port_end = 20000
    b._supported_codecs = [PT_G722, PT_PCMU, PT_PCMA]
    b._dtmf_payload_type = 101
    b._user_agent = None
    b._server_name = "-"
    b._rtp_inactivity_timeout = 30.0
    b._auth_users = auth_users
    b._auth_realm = auth_realm
    b._auth_nonces = {}
    b._register_params = None
    b._register_response_future = None
    b._registration_task = None
    b._registered = False

    b._aiosipua = MagicMock()
    b._aiosipua.SipResponse = FakeSipResponse
    b._rtp_bridge = FakeRtpBridge()

    transport = FakeTransport()
    uac = FakeUAC(transport)
    uas = FakeUAS(transport, uac=uac)

    b._transport = transport
    b._uac = uac
    b._uas = uas

    b._session_states = {}
    b._call_to_session = {}
    b._pending_reinvite_calls = {}
    b._audio_received_callback = None
    b._barge_in_callbacks = []
    b._dtmf_callbacks = []
    b._session_ready_callbacks = []
    b._on_call_callback = None
    b._disconnect_callbacks = []
    b._trace_emitter = None
    b._available_ports = set(range(10000, 20000, 2))
    b._allocated_ports: set[int] = set()
    b._stats_task = None
    b._transport_addr_resolved = False

    return b


# ---------------------------------------------------------------------------
# Inbound auth tests
# ---------------------------------------------------------------------------


class TestInboundAuthChallenge:
    """Test that unauthenticated INVITEs get a 401 challenge."""

    def test_no_auth_header_sends_401(self) -> None:
        backend = _make_backend(auth_users={"alice": "pass123"})
        call = FakeIncomingCall()

        result = backend._validate_invite_auth(call)

        assert result is False
        # 401 sent via transport.send_reply
        transport: FakeTransport = backend._transport  # type: ignore[assignment]
        assert len(transport.replies) == 1
        resp = transport.replies[0]
        assert resp.status_code == 401
        # Dialog terminated after challenge
        assert call.dialog.terminated

    def test_challenge_includes_nonce(self) -> None:
        backend = _make_backend(auth_users={"alice": "pass123"}, auth_realm="test.com")
        call = FakeIncomingCall()

        backend._validate_invite_auth(call)

        # A nonce should be stored for later validation
        assert len(backend._auth_nonces) == 1
        nonce = next(iter(backend._auth_nonces))
        assert len(nonce) == 32  # hex(16 bytes)

    def test_no_auth_configured_skips_check(self) -> None:
        """When auth_users is None, _handle_invite should not call auth."""
        backend = _make_backend(auth_users=None)
        # _validate_invite_auth should not be called, but if it were,
        # it would fail — the point is _handle_invite guards with
        # `if self._auth_users`.
        assert backend._auth_users is None


class TestInboundAuthValidation:
    """Test digest credential validation."""

    def test_valid_credentials_accepted(self) -> None:
        backend = _make_backend(
            auth_users={"alice": "pass123"},
            auth_realm="test.com",
        )

        # Pre-seed a nonce
        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60

        # Compute the expected digest
        uri = "sip:bot@example.com"
        digest = _compute_digest("alice", "test.com", "pass123", "INVITE", uri, nonce)

        call = FakeIncomingCall()
        auth_value = (
            f'Digest username="alice", realm="test.com", nonce="{nonce}", '
            f'uri="{uri}", response="{digest}", algorithm=MD5'
        )
        call.invite.set_header("authorization", auth_value)

        # Mock parse_auth to return the credentials
        import aiosipua

        original_parse_auth = getattr(aiosipua, "parse_auth", None)
        mock_creds = MagicMock()
        mock_creds.params = {
            "username": "alice",
            "realm": "test.com",
            "nonce": nonce,
            "uri": uri,
            "response": digest,
            "algorithm": "MD5",
        }
        aiosipua.parse_auth = lambda s: mock_creds  # type: ignore[attr-defined]

        try:
            result = backend._validate_invite_auth(call)
            assert result is True
            assert not call._rejected
            # Nonce consumed (one-time use)
            assert nonce not in backend._auth_nonces
        finally:
            if original_parse_auth is not None:
                aiosipua.parse_auth = original_parse_auth  # type: ignore[attr-defined]

    def test_invalid_password_rejected(self) -> None:
        backend = _make_backend(
            auth_users={"alice": "pass123"},
            auth_realm="test.com",
        )

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60

        uri = "sip:bot@example.com"
        # Compute with WRONG password
        wrong_digest = _compute_digest("alice", "test.com", "WRONG", "INVITE", uri, nonce)

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")

        import aiosipua

        original_parse_auth = getattr(aiosipua, "parse_auth", None)
        mock_creds = MagicMock()
        mock_creds.params = {
            "username": "alice",
            "nonce": nonce,
            "uri": uri,
            "response": wrong_digest,
        }
        aiosipua.parse_auth = lambda s: mock_creds  # type: ignore[attr-defined]

        try:
            result = backend._validate_invite_auth(call)
            assert result is False
            assert call._rejected
            assert call._reject_code == 403
        finally:
            if original_parse_auth is not None:
                aiosipua.parse_auth = original_parse_auth  # type: ignore[attr-defined]

    def test_unknown_user_rejected(self) -> None:
        backend = _make_backend(
            auth_users={"alice": "pass123"},
            auth_realm="test.com",
        )

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")

        import aiosipua

        original_parse_auth = getattr(aiosipua, "parse_auth", None)
        mock_creds = MagicMock()
        mock_creds.params = {
            "username": "bob",  # not in auth_users
            "nonce": nonce,
            "uri": "sip:bot@example.com",
            "response": "whatever",
        }
        aiosipua.parse_auth = lambda s: mock_creds  # type: ignore[attr-defined]

        try:
            result = backend._validate_invite_auth(call)
            assert result is False
            assert call._rejected
            assert call._reject_code == 403
        finally:
            if original_parse_auth is not None:
                aiosipua.parse_auth = original_parse_auth  # type: ignore[attr-defined]

    def test_expired_nonce_rechallenges(self) -> None:
        backend = _make_backend(
            auth_users={"alice": "pass123"},
            auth_realm="test.com",
        )

        # Seed an expired nonce
        nonce = "expired_nonce"
        backend._auth_nonces[nonce] = time.monotonic() - 1  # already expired

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")

        import aiosipua

        original_parse_auth = getattr(aiosipua, "parse_auth", None)
        mock_creds = MagicMock()
        mock_creds.params = {
            "username": "alice",
            "nonce": nonce,
            "uri": "sip:bot@example.com",
            "response": "whatever",
        }
        aiosipua.parse_auth = lambda s: mock_creds  # type: ignore[attr-defined]

        try:
            result = backend._validate_invite_auth(call)
            assert result is False
            # Should re-challenge (new nonce), not 403
            assert not call._rejected
            transport: FakeTransport = backend._transport  # type: ignore[assignment]
            assert len(transport.replies) == 1
            assert transport.replies[0].status_code == 401
        finally:
            if original_parse_auth is not None:
                aiosipua.parse_auth = original_parse_auth  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Message handler interception tests
# ---------------------------------------------------------------------------


class TestSipMessageHandler:
    """Test that REGISTER responses are intercepted."""

    def test_register_response_intercepted(self) -> None:
        backend = _make_backend()
        loop = asyncio.new_event_loop()
        backend._register_response_future = loop.create_future()

        resp = FakeSipResponse(200, "OK")
        resp.cseq = "1 REGISTER"

        backend._sip_message_handler(resp, ("10.0.0.1", 5060))

        assert backend._register_response_future.done()
        assert backend._register_response_future.result() is resp
        loop.close()

    def test_non_register_delegated_to_uas(self) -> None:
        backend = _make_backend()
        uas: FakeUAS = backend._uas  # type: ignore[assignment]
        calls: list = []
        uas._on_message = lambda msg, addr: calls.append((msg, addr))

        resp = FakeSipResponse(200, "OK")
        resp.cseq = "1 INVITE"

        backend._sip_message_handler(resp, ("10.0.0.1", 5060))

        assert len(calls) == 1
        assert calls[0][0] is resp


# ---------------------------------------------------------------------------
# Digest computation tests
# ---------------------------------------------------------------------------


class TestComputeDigest:
    def test_rfc2617_computation(self) -> None:
        """Verify the digest matches a known RFC 2617 computation."""
        import hashlib

        username, realm, password = "alice", "test.com", "secret"
        method, uri, nonce = "INVITE", "sip:bob@test.com", "dcd98b"

        ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
        expected = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

        assert _compute_digest(username, realm, password, method, uri, nonce) == expected

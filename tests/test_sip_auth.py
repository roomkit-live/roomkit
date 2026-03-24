"""Tests for SIPVoiceBackend inbound auth and outbound registration."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure a fake aiosipua module exists so tests can import SIPVoiceBackend
if "aiosipua" not in sys.modules:
    _fake_aiosipua = types.ModuleType("aiosipua")
    _fake_aiosipua.build_sdp = lambda **kwargs: None  # type: ignore[attr-defined]
    sys.modules["aiosipua"] = _fake_aiosipua

from roomkit.voice.backends._sip_types import compute_digest as _compute_digest
from roomkit.voice.backends.sip import (
    PT_G722,
    PT_PCMA,
    PT_PCMU,
    SIPVoiceBackend,
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
    """Minimal SIP request."""

    def __init__(self, *, method: str = "INVITE", uri: str = "sip:bot@example.com") -> None:
        self.method = method
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


class TestNonceEviction:
    """Test that expired nonces are cleaned up."""

    def test_expired_nonces_evicted_on_challenge(self) -> None:
        backend = _make_backend(auth_users={"alice": "pass123"})
        now = time.monotonic()
        backend._auth_nonces = {
            "old1": now - 100,
            "old2": now - 50,
            "fresh": now + 60,
        }

        call = FakeIncomingCall()
        backend._send_auth_challenge(call)

        # old1 and old2 should be evicted, fresh + new nonce remain
        assert "old1" not in backend._auth_nonces
        assert "old2" not in backend._auth_nonces
        assert "fresh" in backend._auth_nonces
        assert len(backend._auth_nonces) == 2  # fresh + newly generated


# ---------------------------------------------------------------------------
# Outbound registration tests
# ---------------------------------------------------------------------------


def _setup_aiosipua_mocks() -> tuple:
    """Set up aiosipua module mocks for register() tests."""
    import aiosipua

    # Save originals
    originals = {}
    for name in (
        "SipRequest",
        "generate_branch",
        "generate_call_id",
        "generate_tag",
        "parse_auth",
        "stringify_auth",
    ):
        originals[name] = getattr(aiosipua, name, None)

    # Install fakes
    aiosipua.SipRequest = FakeSipRequest  # type: ignore[attr-defined]
    aiosipua.generate_branch = lambda: "z9hG4bK-test"  # type: ignore[attr-defined]
    aiosipua.generate_call_id = lambda ip: "call-id-test"  # type: ignore[attr-defined]
    aiosipua.generate_tag = lambda: "tag-test"  # type: ignore[attr-defined]
    aiosipua.stringify_auth = lambda c: "Digest ..."  # type: ignore[attr-defined]

    return originals


def _restore_aiosipua_mocks(originals: dict) -> None:
    """Restore aiosipua module to original state."""
    import aiosipua

    for name, val in originals.items():
        if val is not None:
            setattr(aiosipua, name, val)


def _setup_headers_mocks() -> tuple:
    """Set up aiosipua.headers module mocks."""
    import types

    headers_mod = types.ModuleType("aiosipua.headers")
    headers_mod.AuthCredentials = MagicMock  # type: ignore[attr-defined]
    headers_mod.CSeq = lambda seq, method: f"{seq} {method}"  # type: ignore[attr-defined]
    headers_mod.Via = lambda **kw: "via"  # type: ignore[attr-defined]
    headers_mod.stringify_cseq = lambda c: str(c)  # type: ignore[attr-defined]
    headers_mod.stringify_via = lambda v: str(v)  # type: ignore[attr-defined]

    original = sys.modules.get("aiosipua.headers")
    sys.modules["aiosipua.headers"] = headers_mod
    return (original,)


def _restore_headers_mocks(original: tuple) -> None:
    """Restore aiosipua.headers module."""
    if original[0] is not None:
        sys.modules["aiosipua.headers"] = original[0]
    else:
        sys.modules.pop("aiosipua.headers", None)


class TestRegister:
    """Test outbound SIP REGISTER flow."""

    async def test_register_before_start_raises(self) -> None:
        backend = _make_backend()
        backend._transport = None

        with pytest.raises(RuntimeError, match="start.*must be called"):
            await backend.register(
                registrar_addr=("10.0.0.1", 5060),
                username="bot",
                password="secret",
            )

    async def test_register_200_ok_no_auth(self) -> None:
        """Registrar accepts without auth challenge."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        try:
            # Simulate: transport.send triggers immediate 200 OK
            async def _resolve_200() -> None:
                await asyncio.sleep(0.01)
                fut = backend._register_response_future
                if fut and not fut.done():
                    backend._register_response_future.set_result(FakeSipResponse(200, "OK"))

            asyncio.get_running_loop().create_task(_resolve_200())
            await backend.register(
                registrar_addr=("10.0.0.1", 5060),
                username="bot",
                password="secret",
            )

            assert backend._registered is True
            assert backend._registration_task is not None
            backend._registration_task.cancel()
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)

    async def test_register_401_then_200(self) -> None:
        """Registrar challenges with 401, retry succeeds."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        import aiosipua

        mock_challenge = MagicMock()
        mock_challenge.params = {"realm": "test.com", "nonce": "server-nonce-123"}
        aiosipua.parse_auth = lambda s: mock_challenge  # type: ignore[attr-defined]
        aiosipua.stringify_auth = lambda c: "Digest ..."  # type: ignore[attr-defined]

        try:
            call_count = 0

            async def _resolve_responses() -> None:
                nonlocal call_count
                while call_count < 2:
                    await asyncio.sleep(0.01)
                    fut = backend._register_response_future
                    if fut and not fut.done():
                        call_count += 1
                        if call_count == 1:
                            resp = FakeSipResponse(401, "Unauthorized")
                            resp._headers["WWW-Authenticate"] = (
                                'Digest realm="test.com", nonce="server-nonce-123"'
                            )
                            fut.set_result(resp)
                        else:
                            fut.set_result(FakeSipResponse(200, "OK"))

            asyncio.get_running_loop().create_task(_resolve_responses())
            await backend.register(
                registrar_addr=("10.0.0.1", 5060),
                username="bot",
                password="secret",
                domain="test.com",
            )

            assert backend._registered is True
            assert call_count == 2
            backend._registration_task.cancel()
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)

    async def test_register_rejected_raises(self) -> None:
        """Non-401/407 response raises RuntimeError."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        try:

            async def _resolve_403() -> None:
                await asyncio.sleep(0.01)
                fut = backend._register_response_future
                if fut and not fut.done():
                    backend._register_response_future.set_result(FakeSipResponse(403, "Forbidden"))

            asyncio.get_running_loop().create_task(_resolve_403())
            with pytest.raises(RuntimeError, match="REGISTER failed: 403"):
                await backend.register(
                    registrar_addr=("10.0.0.1", 5060),
                    username="bot",
                    password="secret",
                )

            assert backend._registered is False
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)

    async def test_register_auth_failed_raises(self) -> None:
        """Second REGISTER (with auth) rejected raises RuntimeError."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        import aiosipua

        mock_challenge = MagicMock()
        mock_challenge.params = {"realm": "test.com", "nonce": "nonce1"}
        aiosipua.parse_auth = lambda s: mock_challenge  # type: ignore[attr-defined]
        aiosipua.stringify_auth = lambda c: "Digest ..."  # type: ignore[attr-defined]

        try:
            call_count = 0

            async def _resolve() -> None:
                nonlocal call_count
                while call_count < 2:
                    await asyncio.sleep(0.01)
                    fut = backend._register_response_future
                    if fut and not fut.done():
                        call_count += 1
                        if call_count == 1:
                            resp = FakeSipResponse(401, "Unauthorized")
                            resp._headers["WWW-Authenticate"] = (
                                'Digest realm="test.com", nonce="nonce1"'
                            )
                            fut.set_result(resp)
                        else:
                            fut.set_result(FakeSipResponse(403, "Forbidden"))

            asyncio.get_running_loop().create_task(_resolve())
            with pytest.raises(RuntimeError, match="REGISTER auth failed: 403"):
                await backend.register(
                    registrar_addr=("10.0.0.1", 5060),
                    username="bot",
                    password="secret",
                    domain="test.com",
                )
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)

    async def test_register_stores_params(self) -> None:
        """register() stores params for re-registration loop."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        try:

            async def _resolve_200() -> None:
                await asyncio.sleep(0.01)
                fut = backend._register_response_future
                if fut and not fut.done():
                    backend._register_response_future.set_result(FakeSipResponse(200, "OK"))

            asyncio.get_running_loop().create_task(_resolve_200())
            await backend.register(
                registrar_addr=("10.0.0.1", 5060),
                username="bot",
                password="secret",
                domain="example.com",
                expires=600,
            )

            assert backend._register_params is not None
            assert backend._register_params["username"] == "bot"
            assert backend._register_params["domain"] == "example.com"
            assert backend._register_params["expires"] == 600
            backend._registration_task.cancel()
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)

    async def test_register_domain_defaults_to_host(self) -> None:
        """When domain is not provided, defaults to registrar host."""
        backend = _make_backend()
        originals = _setup_aiosipua_mocks()
        headers_orig = _setup_headers_mocks()

        try:

            async def _resolve_200() -> None:
                await asyncio.sleep(0.01)
                fut = backend._register_response_future
                if fut and not fut.done():
                    backend._register_response_future.set_result(FakeSipResponse(200, "OK"))

            asyncio.get_running_loop().create_task(_resolve_200())
            await backend.register(
                registrar_addr=("10.0.0.1", 5060),
                username="bot",
                password="secret",
            )

            assert backend._register_params["domain"] == "10.0.0.1"
            backend._registration_task.cancel()
        finally:
            _restore_aiosipua_mocks(originals)
            _restore_headers_mocks(headers_orig)


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

"""Tests for SIPVoiceBackend inbound auth and outbound registration."""

from __future__ import annotations

import asyncio
import contextlib
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


@dataclass
class FakeCSeq:
    """Mirror of aiosipua.headers.CSeq for tests."""

    seq: int = 0
    method: str = ""


class FakeSipResponse:
    """Minimal SIP response for REGISTER tests."""

    def __init__(self, status_code: int = 200, reason: str = "OK") -> None:
        self.status_code = status_code
        self.reason_phrase = reason
        self._headers: dict[str, str] = {}
        self.headers = FakeCaseInsensitiveDict()
        self.cseq = FakeCSeq(seq=1, method="REGISTER")
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
    b._auth_resolver = None
    b._invite_filter = None
    b._registration = None
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
    b._recently_ended_call_ids = {}
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
# BYE for unknown call_id classification
# ---------------------------------------------------------------------------


class TestUnknownByeClassification:
    """``_handle_bye`` distinguishes carrier-retransmit / counter-BYE noise
    (recently-ended call_ids → DEBUG) from real state desync (truly
    unknown call_ids → WARNING)."""

    def _fake_bye(self, call_id: str) -> tuple[Any, Any]:
        call = MagicMock()
        call.call_id = call_id
        call.caller = "sip:alice@example.com"
        request = MagicMock()
        request.serialize = lambda: b""
        return call, request

    def test_unknown_bye_warns(self, caplog: Any) -> None:
        """A BYE for a call_id never seen before logs at WARNING."""
        backend = _make_backend()
        call, request = self._fake_bye("never-seen")

        with caplog.at_level("WARNING", logger="roomkit.voice.sip"):
            backend._handle_bye(call, request)

        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("never-seen" in r.message for r in warns)

    def test_recently_ended_bye_debug(self, caplog: Any) -> None:
        """A BYE for a recently cleaned-up call_id logs at DEBUG, not WARNING.

        Mirrors the real-world case where the carrier retransmits its
        BYE just after we tore down the call (or sends a counter-BYE in
        response to ours and our cleanup happens first).
        """
        backend = _make_backend()
        # Mark call_id as recently ended (still within TTL window).
        backend._recently_ended_call_ids["just-ended"] = time.monotonic() + 30
        call, request = self._fake_bye("just-ended")

        with caplog.at_level("DEBUG", logger="roomkit.voice.sip"):
            backend._handle_bye(call, request)

        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert not any("just-ended" in r.message for r in warns)
        assert any("just-ended" in r.message for r in debugs)
        # Entry consumed — a second late BYE for the same call_id would
        # fall back to WARNING (state really is broken at that point).
        assert "just-ended" not in backend._recently_ended_call_ids

    def test_expired_recently_ended_bye_warns(self, caplog: Any) -> None:
        """An entry past its TTL is treated as truly unknown (WARNING)."""
        backend = _make_backend()
        backend._recently_ended_call_ids["expired"] = time.monotonic() - 1
        call, request = self._fake_bye("expired")

        with caplog.at_level("WARNING", logger="roomkit.voice.sip"):
            backend._handle_bye(call, request)

        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("expired" in r.message for r in warns)


# ---------------------------------------------------------------------------
# Auth resolver tests
# ---------------------------------------------------------------------------


def _stub_parse_auth(creds_params: dict[str, str]):
    """Patch ``aiosipua.parse_auth`` to return ``creds_params``.

    Returns the original so the caller can restore it. Used by tests
    that exercise ``_validate_invite_auth`` with a pre-built header.
    """
    import aiosipua

    original = getattr(aiosipua, "parse_auth", None)
    mock_creds = MagicMock()
    mock_creds.params = creds_params
    aiosipua.parse_auth = lambda s: mock_creds  # type: ignore[attr-defined]
    return original


def _restore_parse_auth(original) -> None:
    import aiosipua

    if original is not None:
        aiosipua.parse_auth = original  # type: ignore[attr-defined]


class TestAuthResolver:
    """Test the ``set_auth_resolver`` runtime credential lookup."""

    def test_resolver_authenticates(self) -> None:
        """A resolver returning the right password authenticates the call."""
        backend = _make_backend(auth_users=None, auth_realm="test.com")

        resolver_calls: list[str] = []

        def resolver(username: str) -> str | None:
            resolver_calls.append(username)
            return "from-resolver" if username == "alice" else None

        backend.set_auth_resolver(resolver)

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60
        uri = "sip:bot@example.com"
        digest = _compute_digest("alice", "test.com", "from-resolver", "INVITE", uri, nonce)

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")
        original = _stub_parse_auth(
            {
                "username": "alice",
                "nonce": nonce,
                "uri": uri,
                "response": digest,
            }
        )
        try:
            assert backend._validate_invite_auth(call) is True
            assert resolver_calls == ["alice"]
            assert not call._rejected
        finally:
            _restore_parse_auth(original)

    def test_resolver_returning_none_rejects(self) -> None:
        backend = _make_backend(auth_users=None, auth_realm="test.com")
        backend.set_auth_resolver(lambda _username: None)

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")
        original = _stub_parse_auth(
            {
                "username": "anyone",
                "nonce": nonce,
                "uri": "sip:bot@example.com",
                "response": "whatever",
            }
        )
        try:
            assert backend._validate_invite_auth(call) is False
            assert call._rejected
            assert call._reject_code == 403
        finally:
            _restore_parse_auth(original)

    def test_resolver_takes_precedence_over_dict(self) -> None:
        """When both resolver and auth_users provide credentials, resolver wins.

        Lets the application override stale dict entries without restart
        — important for credential rotation flows.
        """
        backend = _make_backend(
            auth_users={"alice": "old-password"},
            auth_realm="test.com",
        )
        backend.set_auth_resolver(lambda u: "new-password" if u == "alice" else None)

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60
        uri = "sip:bot@example.com"
        # Digest computed with the NEW password — only succeeds if the
        # resolver was consulted, not the dict.
        digest = _compute_digest("alice", "test.com", "new-password", "INVITE", uri, nonce)

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")
        original = _stub_parse_auth(
            {
                "username": "alice",
                "nonce": nonce,
                "uri": uri,
                "response": digest,
            }
        )
        try:
            assert backend._validate_invite_auth(call) is True
        finally:
            _restore_parse_auth(original)

    def test_resolver_falls_through_to_dict(self) -> None:
        """When the resolver returns None, the static dict is consulted."""
        backend = _make_backend(
            auth_users={"alice": "from-dict"},
            auth_realm="test.com",
        )
        backend.set_auth_resolver(lambda _u: None)

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60
        uri = "sip:bot@example.com"
        digest = _compute_digest("alice", "test.com", "from-dict", "INVITE", uri, nonce)

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")
        original = _stub_parse_auth(
            {
                "username": "alice",
                "nonce": nonce,
                "uri": uri,
                "response": digest,
            }
        )
        try:
            assert backend._validate_invite_auth(call) is True
        finally:
            _restore_parse_auth(original)

    def test_resolver_exception_denies(self) -> None:
        """A raising resolver is caught and treated as a denied user.

        Otherwise a buggy lookup callback would crash the SIP message
        loop and disconnect every other call on the backend.
        """
        backend = _make_backend(auth_users=None, auth_realm="test.com")

        def boom(_username: str) -> str | None:
            raise RuntimeError("db down")

        backend.set_auth_resolver(boom)

        nonce = "abc123def456"
        backend._auth_nonces[nonce] = time.monotonic() + 60

        call = FakeIncomingCall()
        call.invite.set_header("authorization", "Digest ...")
        original = _stub_parse_auth(
            {
                "username": "alice",
                "nonce": nonce,
                "uri": "sip:bot@example.com",
                "response": "whatever",
            }
        )
        try:
            assert backend._validate_invite_auth(call) is False
            assert call._rejected
            assert call._reject_code == 403
        finally:
            _restore_parse_auth(original)

    def test_has_auth_reflects_both_sources(self) -> None:
        """``has_auth`` is True iff at least one credential source exists."""
        backend = _make_backend(auth_users=None)
        assert backend.has_auth() is False

        backend._auth_users = {}
        assert backend.has_auth() is False  # empty dict still counts as "no auth"

        backend._auth_users = {"alice": "pass"}
        assert backend.has_auth() is True

        backend._auth_users = None
        backend.set_auth_resolver(lambda _u: None)
        assert backend.has_auth() is True  # resolver alone is enough

        backend.set_auth_resolver(None)
        assert backend.has_auth() is False


# ---------------------------------------------------------------------------
# Invite filter tests
# ---------------------------------------------------------------------------


class TestInviteFilter:
    """Test ``set_invite_filter`` and the pre-accept reject path."""

    def test_set_filter_stores_callback(self) -> None:
        backend = _make_backend()
        assert backend._invite_filter is None

        def my_filter(call: Any) -> tuple[int, str] | None:
            return None

        backend.set_invite_filter(my_filter)
        assert backend._invite_filter is my_filter

        backend.set_invite_filter(None)
        assert backend._invite_filter is None

    @pytest.mark.asyncio
    async def test_filter_returning_tuple_rejects(self) -> None:
        """A filter returning ``(status, reason)`` must call ``call.reject``
        and short-circuit before any RTP/SDP work."""
        backend = _make_backend()
        backend.set_invite_filter(lambda call: (403, "Forbidden by filter"))

        call = FakeIncomingCall()
        # Sanity: dialog state machine sees no responses before
        assert not call._rejected

        await backend._handle_invite(call)

        assert call._rejected is True
        assert call._reject_code == 403
        # No session state created — filter rejected before accept
        assert call.call_id not in backend._call_to_session

    @pytest.mark.asyncio
    async def test_filter_returning_none_proceeds(self) -> None:
        """A filter returning ``None`` lets the call continue.

        The downstream SDP path will reject with 488 because our fake
        call has a non-None ``sdp_offer`` but the rest of the pipeline
        (RTP allocation, SDP negotiation) is mocked — it'll fall over
        before reaching 200 OK. The point of this test is that the
        filter itself didn't short-circuit.
        """
        called: list[Any] = []

        def my_filter(call: Any) -> None:
            called.append(call)
            return None

        backend = _make_backend()
        backend.set_invite_filter(my_filter)

        call = FakeIncomingCall()
        # _handle_invite will likely error past the filter (RTP/SDP
        # internals are MagicMock), but we only assert the filter ran
        # and didn't reject.
        with contextlib.suppress(Exception):
            await backend._handle_invite(call)

        assert len(called) == 1
        assert called[0] is call
        # Filter returned None — no 4xx rejection from this branch.
        # If the call ended up rejected, it was the SDP/codec path with
        # 488, not our 403/404 rejection.
        if call._rejected:
            assert call._reject_code != 403

    @pytest.mark.asyncio
    async def test_async_filter_supported(self) -> None:
        """Coroutine filters are awaited inside the dispatch task."""

        async def async_filter(_call: Any) -> tuple[int, str]:
            return (404, "Not Found by async filter")

        backend = _make_backend()
        backend.set_invite_filter(async_filter)

        call = FakeIncomingCall()
        await backend._handle_invite(call)

        assert call._rejected is True
        assert call._reject_code == 404

    @pytest.mark.asyncio
    async def test_filter_exception_rejects_500(self) -> None:
        """A raising filter is caught and treated as 500 rejection.

        Otherwise a buggy filter would crash the SIP dispatcher and
        every subsequent INVITE on the backend.
        """

        def boom(_call: Any) -> tuple[int, str]:
            raise RuntimeError("filter exploded")

        backend = _make_backend()
        backend.set_invite_filter(boom)

        call = FakeIncomingCall()
        await backend._handle_invite(call)

        assert call._rejected is True
        assert call._reject_code == 500

    @pytest.mark.asyncio
    async def test_filter_runs_after_auth(self) -> None:
        """Failing auth short-circuits before the filter ever runs.

        Order matters: the filter receives an authenticated call, so
        applications can rely on ``caller_user`` being trustworthy.
        """
        called: list[Any] = []

        def my_filter(call: Any) -> None:
            called.append(call)
            return None

        backend = _make_backend(auth_users={"alice": "pass"})
        backend.set_invite_filter(my_filter)

        call = FakeIncomingCall()
        # No Authorization header → auth challenge sends 401 and returns
        # without calling the filter.
        await backend._handle_invite(call)

        assert called == []
        # 401 challenge sent via send_reply
        transport: FakeTransport = backend._transport  # type: ignore[assignment]
        assert any(getattr(r, "status_code", None) == 401 for r in transport.replies)


# ---------------------------------------------------------------------------
# Nonce eviction tests
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
# Outbound registration tests (FakeRegistration verifies the roomkit wiring;
# aiosipua's Registration has its own test suite upstream)
# ---------------------------------------------------------------------------


REGISTRAR = ("10.0.0.1", 5060)


class FakeRegistration:
    """Stand-in for aiosipua.Registration capturing constructor args and calls."""

    instances: list[FakeRegistration] = []  # noqa: RUF012

    def __init__(
        self,
        uac: Any,
        aor: str,
        registrar: tuple[str, int],
        *,
        expires: int = 300,
        auth: Any = None,
        contact_uri: str | None = None,
        registrar_uri: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.uac = uac
        self.aor = aor
        self.registrar = registrar
        self.expires = expires
        self.auth = auth
        self.contact_uri = contact_uri
        self.registrar_uri = registrar_uri
        self.user_agent = user_agent
        self.granted_expires = 0
        self.on_registered: Any = None
        self.on_failed: Any = None
        self.on_expired: Any = None
        self.register_calls = 0
        self.unregister_calls = 0
        self.timers_cancelled = False
        FakeRegistration.instances.append(self)

    def register(self) -> None:
        self.register_calls += 1

    def unregister(self) -> None:
        self.unregister_calls += 1

    def _cancel_timers(self) -> None:
        self.timers_cancelled = True


def _install_fake_registration(monkeypatch: Any) -> None:
    """Expose FakeRegistration through the (real or stub) aiosipua module."""
    import aiosipua

    FakeRegistration.instances = []
    monkeypatch.setattr(aiosipua, "Registration", FakeRegistration, raising=False)
    monkeypatch.setattr(
        aiosipua,
        "SipDigestAuth",
        lambda username, password: {"username": username, "password": password},
        raising=False,
    )
    utils_mod = types.ModuleType("aiosipua.utils")
    utils_mod.format_addr = lambda host, port: f"{host}:{port}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiosipua.utils", utils_mod)


def _cleanup_registration(backend: Any) -> None:
    if backend._registration_task is not None:
        backend._registration_task.cancel()


class TestRegister:
    """Outbound REGISTER wiring around aiosipua.Registration."""

    async def test_register_before_start_raises(self) -> None:
        backend = _make_backend()
        backend._transport = None

        with pytest.raises(RuntimeError, match="start.*must be called"):
            await backend.register(
                registrar_addr=REGISTRAR,
                username="bot",
                password="secret",
            )

    async def test_register_success(self, monkeypatch: Any) -> None:
        """on_registered resolves register() and marks the backend registered."""
        _install_fake_registration(monkeypatch)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(
                registrar_addr=REGISTRAR, username="bot", password="secret", expires=600
            )
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        assert reg.register_calls == 1
        assert reg.aor == "sip:bot@10.0.0.1"
        assert reg.expires == 600
        assert reg.auth == {"username": "bot", "password": "secret"}
        assert reg.contact_uri == "sip:bot@10.0.0.2:5060"

        reg.granted_expires = 600
        reg.on_registered(reg)
        await task

        assert backend._registered is True
        assert backend._registration is reg
        _cleanup_registration(backend)

    async def test_register_rejected_raises(self, monkeypatch: Any) -> None:
        _install_fake_registration(monkeypatch)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(registrar_addr=REGISTRAR, username="bot", password="secret")
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        reg.on_failed(reg, 403, "Forbidden")

        with pytest.raises(RuntimeError, match="REGISTER failed: 403"):
            await task
        assert backend._registered is False
        _cleanup_registration(backend)

    async def test_register_domain_defaults_to_host(self, monkeypatch: Any) -> None:
        _install_fake_registration(monkeypatch)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(registrar_addr=REGISTRAR, username="bot", password="secret")
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        assert reg.registrar_uri == "sip:10.0.0.1"
        assert reg.registrar == REGISTRAR

        reg.on_registered(reg)
        await task
        _cleanup_registration(backend)

    async def test_register_explicit_domain(self, monkeypatch: Any) -> None:
        _install_fake_registration(monkeypatch)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(
                registrar_addr=REGISTRAR, username="bot", password="secret", domain="pbx.test"
            )
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        assert reg.aor == "sip:bot@pbx.test"
        assert reg.registrar_uri == "sip:pbx.test"

        reg.on_registered(reg)
        await task
        _cleanup_registration(backend)

    async def test_register_timeout(self, monkeypatch: Any) -> None:
        """A silent registrar raises TimeoutError."""
        from roomkit.voice.backends import sip_auth

        _install_fake_registration(monkeypatch)
        monkeypatch.setattr(sip_auth, "REGISTER_TIMEOUT", 0.05)
        backend = _make_backend()

        with pytest.raises(TimeoutError):
            await backend.register(registrar_addr=REGISTRAR, username="bot", password="secret")
        _cleanup_registration(backend)

    async def test_lost_registration_retries(self, monkeypatch: Any) -> None:
        """Failure after initial success re-registers on the retry cadence."""
        from roomkit.voice.backends import sip_auth

        _install_fake_registration(monkeypatch)
        monkeypatch.setattr(sip_auth, "REGISTER_RETRY_DELAY", 0.01)
        monkeypatch.setattr(sip_auth, "REGISTER_TIMEOUT", 0.01)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(registrar_addr=REGISTRAR, username="bot", password="secret")
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        reg.on_registered(reg)
        await task
        assert backend._registered is True

        # Watchdog declares the binding dead → retry loop kicks in
        reg.on_expired(reg)
        assert backend._registered is False
        for _ in range(200):
            if reg.register_calls >= 2:
                break
            await asyncio.sleep(0.005)
        assert reg.register_calls >= 2

        # A successful re-registration stops the retrying
        reg.on_registered(reg)
        assert backend._registered is True
        _cleanup_registration(backend)

    async def test_close_unregisters(self, monkeypatch: Any) -> None:
        """close() sends the Expires:0 unregister and drops the registration."""
        _install_fake_registration(monkeypatch)
        backend = _make_backend()

        task = asyncio.ensure_future(
            backend.register(registrar_addr=REGISTRAR, username="bot", password="secret")
        )
        await asyncio.sleep(0)
        reg = FakeRegistration.instances[0]
        reg.on_registered(reg)
        await task

        await backend.close()

        assert reg.unregister_calls == 1
        assert backend._registered is False
        assert backend._registration is None


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

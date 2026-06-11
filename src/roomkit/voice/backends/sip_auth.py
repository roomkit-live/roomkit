"""SIP inbound authentication and outbound registration mixin."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from roomkit.voice.backends._sip_types import NONCE_TTL, compute_digest, logger, resolve_local_ip

# Outbound registration timing: how long register() waits for the first
# outcome, and the cadence of retries after a later failure or expiry
REGISTER_TIMEOUT = 5.0
REGISTER_RETRY_DELAY = 30.0

__all__ = ["AuthResolver", "InviteFilter", "InviteFilterDecision", "SIPAuthMixin"]


# Callback consulted per authentication attempt. Receives the username
# from the digest credentials and returns the matching password, or
# ``None`` to deny. Synchronous so it can run inside the SIP message
# loop — back the resolver with an in-memory cache when the source of
# truth is a database.
AuthResolver = Callable[[str], str | None]


# Pre-accept rejection decision. ``None`` means accept (proceed to SDP +
# 200 OK); a ``(status_code, reason_phrase)`` tuple means reject the
# INVITE with that 4xx/5xx response *before* the dialog is confirmed —
# the carrier never sees a 200 OK. Use 403 / 404 / 488 for the common
# "no route", "no permission", "no codec" cases respectively.
InviteFilterDecision = tuple[int, str] | None


# Pre-accept hook that the application installs via
# :meth:`SIPVoiceBackend.set_invite_filter`. Called inside
# ``_handle_invite`` after the optional auth challenge but before the
# RTP/SDP/200-OK path. Both sync and async callables are supported —
# the dispatcher picks the right one with ``iscoroutinefunction``.
InviteFilter = Callable[[Any], InviteFilterDecision | Awaitable[InviteFilterDecision]]


@runtime_checkable
class SIPAuthHost(Protocol):
    """Contract: capabilities a host class must provide for SIPAuthMixin.

    Attributes provided by the host's ``__init__``:
        _aiosipua: The aiosipua module (lazy-imported, no type stubs).
        _transport: The SIP UDP transport.
        _uas: The SIP User Agent Server.
        _user_agent: User-Agent header value.
        _auth_users: Username-to-password map for inbound auth.
        _auth_realm: Digest authentication realm.
        _auth_nonces: Active nonce-to-expiry map.
        _auth_resolver: Optional callback-based credential lookup.
        _uac: The SIP User Agent Client (routes REGISTER responses).
        _registration: The aiosipua Registration, when registered.
        _registration_task: Background registration retry task.
        _registered: Whether currently registered with registrar.
        _local_rtp_ip: Local IP for RTP binding.
        _advertised_ip: IP to advertise in SDP/Contact headers.
    """

    _aiosipua: Any
    _transport: Any
    _uas: Any
    _uac: Any
    _user_agent: str | None
    _auth_users: dict[str, str] | None
    _auth_realm: str
    _auth_nonces: dict[str, float]
    _auth_resolver: AuthResolver | None
    _invite_filter: InviteFilter | None
    _registration: Any
    _registration_task: asyncio.Task[None] | None
    _registered: bool
    _local_rtp_ip: str
    _advertised_ip: str | None


class SIPAuthMixin:
    """Mixin providing inbound digest auth and outbound REGISTER for SIPVoiceBackend.

    Host contract: :class:`SIPAuthHost`.
    """

    _aiosipua: Any
    _transport: Any
    _uas: Any
    _uac: Any
    _user_agent: str | None
    _auth_users: dict[str, str] | None
    _auth_realm: str
    _auth_nonces: dict[str, float]
    _auth_resolver: AuthResolver | None
    _invite_filter: InviteFilter | None
    _registration: Any
    _registration_task: asyncio.Task[None] | None
    _registered: bool
    _local_rtp_ip: str
    _advertised_ip: str | None

    # -------------------------------------------------------------------------
    # Public API: runtime credential resolver
    # -------------------------------------------------------------------------

    def set_invite_filter(self, invite_filter: InviteFilter | None) -> None:
        """Install a pre-accept hook that runs before SDP / 200 OK.

        The filter receives the ``IncomingCall`` after any auth challenge
        has succeeded and returns ``None`` to accept (proceed to SDP
        negotiation and 200 OK) or ``(status, reason)`` to reject the
        INVITE with that 4xx/5xx response. The carrier never sees a
        200 OK on a rejection, so CDRs log the call as rejected rather
        than briefly answered.

        Driving use case: application-layer routing decisions ("DID not
        provisioned", "tenant not authorized", "outside business hours")
        that need DB access but should not result in an answered-then-
        dropped call. Call ``set_invite_filter(None)`` to remove.

        The callback may be sync or async. Async filters run inside the
        SIP dispatch task and so should keep DB / network calls fast —
        the dialog is half-set-up while the filter runs.
        """
        self._invite_filter = invite_filter

    def set_auth_resolver(self, resolver: AuthResolver | None) -> None:
        """Install a callback that looks up the password for a username.

        Called for every authenticated INVITE.  Returning ``None`` denies
        the user (the backend sends 403).  Takes precedence over the
        ``auth_users`` dict passed at construction.

        Pass ``None`` to remove a previously installed resolver and fall
        back to ``auth_users`` (or disable auth entirely if both are
        absent).

        Use this when credentials live outside the process — a database,
        secret manager, or per-tenant config — so they can be added or
        revoked without restarting the backend.  The resolver runs
        synchronously inside the SIP message loop, so back it with an
        in-memory cache when the source of truth is remote.
        """
        self._auth_resolver = resolver

    def has_auth(self) -> bool:
        """True when at least one credential source is configured.

        Use this to gate ``_handle_invite``'s auth check — both an
        ``auth_users`` dict and a resolver count as "auth enabled".
        """
        return bool(self._auth_users) or self._auth_resolver is not None

    # -------------------------------------------------------------------------
    # Inbound authentication (RFC 2617 digest)
    # -------------------------------------------------------------------------

    def _validate_invite_auth(self, call: Any) -> bool:
        """Check digest auth on an incoming INVITE.

        Returns ``True`` if the caller is authenticated and the call
        should proceed.  Returns ``False`` if a 401 challenge or 403
        rejection was sent.
        """
        auth_header = call.invite.get_header("Authorization")
        if auth_header is None:
            self._send_auth_challenge(call)
            return False

        from aiosipua import parse_auth

        creds = parse_auth(auth_header)
        username = creds.params.get("username", "")
        nonce = creds.params.get("nonce", "")
        response = creds.params.get("response", "")
        uri = creds.params.get("uri", "")

        # Validate nonce was issued by us and hasn't expired
        nonce_expiry = self._auth_nonces.pop(nonce, None)
        if nonce_expiry is None or nonce_expiry < time.monotonic():
            self._send_auth_challenge(call)
            return False

        # Validate credentials. Resolver wins when set so the application
        # can override the static dict (or skip the dict entirely for
        # multi-tenant / database-backed deployments).
        password: str | None = None
        if self._auth_resolver is not None:
            try:
                password = self._auth_resolver(username)
            except Exception:
                logger.exception("SIP auth resolver raised for username=%s", username)
                password = None
        if password is None and self._auth_users:
            password = self._auth_users.get(username)
        if password is None:
            call.reject(403, "Forbidden")
            return False

        expected = compute_digest(username, self._auth_realm, password, "INVITE", uri, nonce)
        if response != expected:
            call.reject(403, "Forbidden")
            return False

        logger.info("SIP INVITE authenticated: user=%s, call_id=%s", username, call.call_id)
        return True

    def _send_auth_challenge(self, call: Any) -> None:
        """Send 401 Unauthorized with a WWW-Authenticate digest challenge."""
        now = time.monotonic()

        # Evict expired nonces to prevent unbounded memory growth
        self._auth_nonces = {n: exp for n, exp in self._auth_nonces.items() if exp > now}

        nonce = secrets.token_hex(16)
        self._auth_nonces[nonce] = now + NONCE_TTL

        challenge = f'Digest realm="{self._auth_realm}", nonce="{nonce}", algorithm=MD5'

        # Build 401 response with the challenge header
        contact: str | None = None
        if self._transport is not None:
            addr = self._transport.local_addr
            contact = f"<sip:{addr[0]}:{addr[1]}>"

        resp = call.dialog.create_response(call.invite, 401, "Unauthorized", contact=contact)
        resp.headers.set_single("WWW-Authenticate", challenge)
        if self._user_agent:
            resp.headers.set_single("User-Agent", self._user_agent)
        self._transport.send_reply(resp)
        call.dialog.terminate()

        logger.debug(
            "SIP 401 challenge sent for call_id=%s (realm=%s)",
            call.call_id,
            self._auth_realm,
        )

    # -------------------------------------------------------------------------
    # Outbound registration (RFC 3261 §10)
    # -------------------------------------------------------------------------

    async def register(
        self,
        registrar_addr: tuple[str, int],
        *,
        username: str,
        password: str,
        domain: str | None = None,
        expires: int = 300,
    ) -> None:
        """Register with a SIP registrar using digest authentication.

        Delegates to :class:`aiosipua.Registration`: the binding refreshes
        itself before expiry, 423 Min-Expires is honoured, and challenges
        are answered per RFC 7616 (qop, MD5 and SHA-256).  If the
        registration later fails or expires, the backend retries every
        30 s until it sticks or :meth:`close` is called.

        Args:
            registrar_addr: ``(host, port)`` of the SIP registrar/PBX.
            username: SIP username or extension number.
            password: Password for digest authentication.
            domain: SIP domain (defaults to the registrar host).
            expires: Registration TTL in seconds (default 300).

        Raises:
            RuntimeError: If the backend is not started or registration
                is rejected.
            TimeoutError: If the registrar does not respond within 5 s.
        """
        if self._transport is None or self._uac is None:
            raise RuntimeError("SIPVoiceBackend.start() must be called before register()")

        # Re-registering replaces the previous binding: stop its timers and
        # any retry loop so they don't keep firing for the old registrar
        if self._registration is not None:
            self._registration._cancel_timers()
        if self._registration_task is not None:
            self._registration_task.cancel()
            self._registration_task = None

        from aiosipua import Registration, SipDigestAuth
        from aiosipua.utils import format_addr

        reg_domain = domain or registrar_addr[0]
        local_ip, local_port = self._transport.local_addr
        if local_ip in ("0.0.0.0", ""):  # nosec B104
            local_ip = resolve_local_ip(self._local_rtp_ip, registrar_addr)
        signaling_ip = self._advertised_ip or local_ip

        registration = Registration(
            self._uac,
            f"sip:{username}@{reg_domain}",
            registrar_addr,
            expires=expires,
            auth=SipDigestAuth(username, password),
            contact_uri=f"sip:{username}@{format_addr(signaling_ip, local_port)}",
            registrar_uri=f"sip:{reg_domain}",
            user_agent=self._user_agent,
        )
        self._registration = registration

        outcome: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def _on_registered(reg: Any) -> None:
            self._registered = True
            if not outcome.done():
                outcome.set_result(None)
            logger.info(
                "Registered as %s@%s (expires=%ds)", username, reg_domain, reg.granted_expires
            )

        def _on_failed(reg: Any, status: int, reason: str) -> None:
            if not outcome.done():
                outcome.set_exception(RuntimeError(f"REGISTER failed: {status} {reason}"))
                return
            self._on_registration_lost(f"{status} {reason}")

        def _on_expired(reg: Any) -> None:
            self._on_registration_lost("expired without refresh")

        registration.on_registered = _on_registered
        registration.on_failed = _on_failed
        registration.on_expired = _on_expired

        registration.register()
        await asyncio.wait_for(outcome, timeout=REGISTER_TIMEOUT)

    def _on_registration_lost(self, why: str) -> None:
        """The binding failed or expired after initial success — retry forever."""
        self._registered = False
        logger.warning(
            "SIP registration lost (%s) — retrying every %.0fs", why, REGISTER_RETRY_DELAY
        )
        if self._registration is None:
            return
        task = self._registration_task
        if task is not None and not task.done():
            return
        self._registration_task = asyncio.get_running_loop().create_task(
            self._registration_retry_loop(), name="sip_registration_retry"
        )

    async def _registration_retry_loop(self) -> None:
        """Re-attempt registration until it sticks or the backend closes."""
        while not self._registered and self._registration is not None:
            await asyncio.sleep(REGISTER_RETRY_DELAY)
            registration = self._registration
            if registration is None:
                return
            try:
                registration.register()
            except Exception:
                # Transient send failure (transport racing shutdown, network
                # blip) must not kill the retry loop — that is its one job
                logger.exception("SIP re-registration attempt failed — will retry")
                continue
            # Outcome arrives via callbacks; give the registrar a beat
            await asyncio.sleep(REGISTER_TIMEOUT)

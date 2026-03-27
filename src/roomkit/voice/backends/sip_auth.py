"""SIP inbound authentication and outbound registration mixin."""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any, Protocol, runtime_checkable

from roomkit.voice.backends._sip_types import NONCE_TTL, compute_digest, logger, resolve_local_ip

__all__ = ["SIPAuthMixin"]


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
        _register_params: Outbound registration parameters.
        _register_response_future: Pending REGISTER response future.
        _registration_task: Background registration renewal task.
        _registered: Whether currently registered with registrar.
        _local_rtp_ip: Local IP for RTP binding.
        _advertised_ip: IP to advertise in SDP/Contact headers.
    """

    _aiosipua: Any
    _transport: Any
    _uas: Any
    _user_agent: str | None
    _auth_users: dict[str, str] | None
    _auth_realm: str
    _auth_nonces: dict[str, float]
    _register_params: dict[str, Any] | None
    _register_response_future: asyncio.Future[Any] | None
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
    _user_agent: str | None
    _auth_users: dict[str, str] | None
    _auth_realm: str
    _auth_nonces: dict[str, float]
    _register_params: dict[str, Any] | None
    _register_response_future: asyncio.Future[Any] | None
    _registration_task: asyncio.Task[None] | None
    _registered: bool
    _local_rtp_ip: str
    _advertised_ip: str | None

    # -------------------------------------------------------------------------
    # SIP message dispatch (wraps UAS handler for REGISTER interception)
    # -------------------------------------------------------------------------

    def _sip_message_handler(self, msg: Any, addr: tuple[str, int]) -> None:
        """Dispatch SIP messages, intercepting REGISTER responses."""
        sip_response_cls = self._aiosipua.SipResponse
        if isinstance(msg, sip_response_cls) and self._register_response_future is not None:
            cseq_str = msg.cseq or ""
            if "REGISTER" in cseq_str and not self._register_response_future.done():
                self._register_response_future.set_result(msg)
                return
        # Delegate everything else to the UAS
        self._uas._on_message(msg, addr)

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

        # Validate credentials
        password = self._auth_users.get(username) if self._auth_users else None
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

        Sends a REGISTER request, handles the 401 challenge automatically,
        and starts a background task that re-registers before expiry.
        Call :meth:`close` to unregister and stop the renewal loop.

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
        if self._transport is None:
            raise RuntimeError("SIPVoiceBackend.start() must be called before register()")

        self._register_params = {
            "registrar_addr": registrar_addr,
            "username": username,
            "password": password,
            "domain": domain or registrar_addr[0],
            "expires": expires,
        }

        await self._do_register(expires=expires)

        self._registration_task = asyncio.get_running_loop().create_task(
            self._registration_loop(), name="sip_registration"
        )

    async def _do_register(self, *, expires: int) -> None:
        """Execute one REGISTER transaction with digest auth retry."""
        from aiosipua import (
            SipRequest,
            generate_branch,
            generate_call_id,
            generate_tag,
            parse_auth,
            stringify_auth,
        )
        from aiosipua.headers import AuthCredentials, CSeq, Via, stringify_cseq, stringify_via

        params = self._register_params
        if params is None:
            raise RuntimeError("No registration parameters configured")

        registrar_addr: tuple[str, int] = params["registrar_addr"]
        username: str = params["username"]
        password: str = params["password"]
        domain: str = params["domain"]

        local_ip, local_port = self._transport.local_addr
        if local_ip in ("0.0.0.0", ""):  # nosec B104
            local_ip = resolve_local_ip(self._local_rtp_ip, registrar_addr)
        signaling_ip = self._advertised_ip or local_ip

        from_uri = f"sip:{username}@{domain}"
        request_uri = f"sip:{domain}"
        contact_uri = f"sip:{username}@{signaling_ip}:{local_port}"
        call_id = generate_call_id(signaling_ip)
        local_tag = generate_tag()

        def _build(cseq_num: int, auth_header: tuple[str, str] | None = None) -> Any:
            branch = generate_branch()
            reg = SipRequest(method="REGISTER", uri=request_uri)
            via = Via(
                transport="UDP", host=signaling_ip, port=local_port, params={"branch": branch}
            )
            reg.headers.append("Via", stringify_via(via))
            reg.headers.set_single("From", f"<{from_uri}>;tag={local_tag}")
            reg.headers.set_single("To", f"<{from_uri}>")
            reg.headers.set_single("Call-ID", call_id)
            reg.headers.set_single("CSeq", stringify_cseq(CSeq(seq=cseq_num, method="REGISTER")))
            reg.headers.set_single("Contact", f"<{contact_uri}>;expires={expires}")
            reg.headers.set_single("Max-Forwards", "70")
            reg.headers.set_single("Expires", str(expires))
            if self._user_agent:
                reg.headers.set_single("User-Agent", self._user_agent)
            if auth_header is not None:
                reg.headers.set_single(auth_header[0], auth_header[1])
            return reg

        async def _send_and_wait(request: Any) -> Any:
            loop = asyncio.get_running_loop()
            self._register_response_future = loop.create_future()
            self._transport.send(request, registrar_addr)
            try:
                return await asyncio.wait_for(self._register_response_future, timeout=5.0)
            finally:
                self._register_response_future = None

        # First REGISTER (no auth)
        resp = await _send_and_wait(_build(1))

        if resp.status_code == 200:
            self._registered = True
            logger.info("Registered as %s@%s (no auth required)", username, domain)
            return

        if resp.status_code not in (401, 407):
            raise RuntimeError(f"REGISTER failed: {resp.status_code} {resp.reason_phrase}")

        # Handle 401/407 challenge
        hdr = "WWW-Authenticate" if resp.status_code == 401 else "Proxy-Authenticate"
        challenge_str = resp.get_header(hdr)
        if not challenge_str:
            raise RuntimeError(f"No {hdr} in {resp.status_code} response")

        challenge = parse_auth(challenge_str)
        realm = challenge.params.get("realm", "")
        nonce = challenge.params.get("nonce", "")
        if not nonce:
            raise RuntimeError("Missing nonce in REGISTER auth challenge")

        digest = compute_digest(username, realm, password, "REGISTER", request_uri, nonce)
        credentials = AuthCredentials(
            scheme="Digest",
            params={
                "username": username,
                "realm": realm,
                "nonce": nonce,
                "uri": request_uri,
                "response": digest,
                "algorithm": "MD5",
            },
        )
        auth_hdr = "Authorization" if resp.status_code == 401 else "Proxy-Authorization"

        # Second REGISTER (with auth)
        resp2 = await _send_and_wait(_build(2, (auth_hdr, stringify_auth(credentials))))

        if resp2.status_code != 200:
            raise RuntimeError(f"REGISTER auth failed: {resp2.status_code} {resp2.reason_phrase}")

        self._registered = True
        logger.info("Registered as %s@%s (expires=%ds)", username, domain, expires)

    async def _registration_loop(self) -> None:
        """Re-register periodically before the current registration expires."""
        params = self._register_params
        if params is None:
            return
        expires: int = params["expires"]
        while True:
            await asyncio.sleep(expires * 0.8)
            try:
                await self._do_register(expires=expires)
            except Exception:
                logger.exception("SIP re-registration failed — retrying in 30s")
                await asyncio.sleep(30)

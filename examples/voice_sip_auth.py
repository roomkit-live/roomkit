#!/usr/bin/env python3
"""SIP voice backend with registrar authentication.

Demonstrates how to register a RoomKit voice agent as a SIP extension
on a PBX (Asterisk, FreeSWITCH, Kamailio) using digest authentication,
then receive and handle incoming calls.

This is the typical "bot registers as extension 6001 on the PBX" pattern:
the bot sends a REGISTER request, handles the 401 challenge with digest
credentials, and then receives INVITEs routed to that extension.

Requirements:
    pip install roomkit[sip]

Usage:
    SIP_REGISTRAR_HOST=10.0.0.1 \
    SIP_USERNAME=6001 \
    SIP_PASSWORD=secret123 \
    python examples/voice_sip_auth.py

    # Environment variables:
    #   SIP_REGISTRAR_HOST  — PBX/registrar IP       (REQUIRED)
    #   SIP_REGISTRAR_PORT  — registrar port          (default: 5060)
    #   SIP_USERNAME        — extension/username       (REQUIRED)
    #   SIP_PASSWORD        — extension password       (REQUIRED)
    #   SIP_DOMAIN          — SIP domain               (default: registrar host)
    #   SIP_LOCAL_PORT      — local SIP listen port    (default: 5060)
    #   SIP_REGISTER_EXPIRY — registration TTL seconds (default: 300)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import socket
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_auth")

from roomkit import RoomKit, VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.models.trace import ProtocolTrace
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

REGISTRAR_HOST = os.environ.get("SIP_REGISTRAR_HOST", "")
REGISTRAR_PORT = int(os.environ.get("SIP_REGISTRAR_PORT", "5060"))
USERNAME = os.environ.get("SIP_USERNAME", "")
PASSWORD = os.environ.get("SIP_PASSWORD", "")
DOMAIN = os.environ.get("SIP_DOMAIN", "") or REGISTRAR_HOST
LOCAL_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5060"))
REGISTER_EXPIRY = int(os.environ.get("SIP_REGISTER_EXPIRY", "300"))

RTP_PORT_START = 10000
RTP_PORT_END = 20000

# Simple in-memory call log
call_log: list[dict] = []


# ---------------------------------------------------------------------------
# SIP REGISTER helper (RFC 3261 §10)
# ---------------------------------------------------------------------------


def _resolve_local_ip(remote_host: str) -> str:
    """Determine which local IP is routable to the registrar."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((remote_host, 1))
        return s.getsockname()[0]


def _compute_digest(
    username: str,
    realm: str,
    password: str,
    method: str,
    uri: str,
    nonce: str,
) -> str:
    """Compute RFC 2617 MD5 digest response."""
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()  # noqa: S324
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()  # noqa: S324
    return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # noqa: S324


async def sip_register(
    transport: object,
    *,
    registrar_addr: tuple[str, int],
    local_addr: tuple[str, int],
    username: str,
    password: str,
    domain: str,
    expires: int = 300,
) -> int:
    """Send SIP REGISTER with digest auth and return granted expiry.

    Sends an initial REGISTER, handles the 401 challenge by computing
    digest credentials, then re-sends with Authorization.

    Returns:
        Granted expiry in seconds from the registrar's 200 OK.

    Raises:
        RuntimeError: If registration fails (non-200/401 response).
        TimeoutError: If no response within 5 seconds.
    """
    from aiosipua import (
        SipRequest,
        SipResponse,
        generate_branch,
        generate_call_id,
        generate_tag,
        parse_auth,
        stringify_auth,
    )
    from aiosipua.headers import AuthCredentials, CSeq, Via, stringify_cseq, stringify_via

    local_ip, local_port = local_addr[0], local_addr[1]
    if local_ip in ("0.0.0.0", ""):
        local_ip = _resolve_local_ip(registrar_addr[0])

    from_uri = f"sip:{username}@{domain}"
    request_uri = f"sip:{domain}"
    contact_uri = f"sip:{username}@{local_ip}:{local_port}"
    call_id = generate_call_id(local_ip)
    local_tag = generate_tag()
    cseq = 1

    # Capture responses via a temporary interceptor
    response_event: asyncio.Event = asyncio.Event()
    captured_response: list[SipResponse] = []
    original_on_message = transport.on_message  # type: ignore[attr-defined]

    def _intercept(msg: object, addr: tuple[str, int]) -> None:
        if isinstance(msg, SipResponse) and msg.cseq and "REGISTER" in (msg.cseq or ""):
            captured_response.clear()
            captured_response.append(msg)
            response_event.set()
        elif original_on_message is not None:
            original_on_message(msg, addr)

    def _build_register(
        cseq_num: int,
        auth_header: tuple[str, str] | None = None,
    ) -> SipRequest:
        branch = generate_branch()
        reg = SipRequest(method="REGISTER", uri=request_uri)
        via = Via(transport="UDP", host=local_ip, port=local_port, params={"branch": branch})
        reg.headers.append("Via", stringify_via(via))
        reg.headers.set_single("From", f"<{from_uri}>;tag={local_tag}")
        reg.headers.set_single("To", f"<{from_uri}>")
        reg.headers.set_single("Call-ID", call_id)
        reg.headers.set_single("CSeq", stringify_cseq(CSeq(seq=cseq_num, method="REGISTER")))
        reg.headers.set_single("Contact", f"<{contact_uri}>;expires={expires}")
        reg.headers.set_single("Max-Forwards", "70")
        reg.headers.set_single("Expires", str(expires))
        reg.headers.set_single("User-Agent", "RoomKit-SIP/1.0")
        if auth_header is not None:
            reg.headers.set_single(auth_header[0], auth_header[1])
        return reg

    try:
        transport.on_message = _intercept  # type: ignore[attr-defined]

        # --- First REGISTER (no auth) ---
        reg1 = _build_register(cseq)
        transport.send(reg1, registrar_addr)  # type: ignore[attr-defined]
        logger.info("REGISTER sent to %s:%d (no auth)", *registrar_addr)

        try:
            await asyncio.wait_for(response_event.wait(), timeout=5.0)
        except TimeoutError:
            raise TimeoutError("No response to REGISTER within 5s") from None

        resp = captured_response[0]

        if resp.status_code == 200:
            logger.info("Registered (no auth required)")
            return expires

        if resp.status_code not in (401, 407):
            raise RuntimeError(f"REGISTER failed: {resp.status_code} {resp.reason_phrase}")

        # --- Handle auth challenge ---
        header_name = "WWW-Authenticate" if resp.status_code == 401 else "Proxy-Authenticate"
        challenge_str = resp.get_header(header_name)
        if not challenge_str:
            raise RuntimeError(f"No {header_name} header in {resp.status_code} response")

        challenge = parse_auth(challenge_str)
        realm = challenge.params.get("realm", "")
        nonce = challenge.params.get("nonce", "")
        if not nonce:
            raise RuntimeError("Missing nonce in auth challenge")

        digest_response = _compute_digest(
            username, realm, password, "REGISTER", request_uri, nonce
        )
        credentials = AuthCredentials(
            scheme="Digest",
            params={
                "username": username,
                "realm": realm,
                "nonce": nonce,
                "uri": request_uri,
                "response": digest_response,
                "algorithm": "MD5",
            },
        )
        auth_header_name = "Authorization" if resp.status_code == 401 else "Proxy-Authorization"

        # --- Second REGISTER (with auth) ---
        cseq += 1
        response_event.clear()
        reg2 = _build_register(cseq, (auth_header_name, stringify_auth(credentials)))
        transport.send(reg2, registrar_addr)  # type: ignore[attr-defined]
        logger.info("REGISTER sent with %s (realm=%s)", auth_header_name, realm)

        try:
            await asyncio.wait_for(response_event.wait(), timeout=5.0)
        except TimeoutError:
            raise TimeoutError("No response to authenticated REGISTER within 5s") from None

        resp2 = captured_response[0]
        if resp2.status_code != 200:
            raise RuntimeError(f"REGISTER auth failed: {resp2.status_code} {resp2.reason_phrase}")

        logger.info("Registered successfully as %s@%s (expires=%ds)", username, domain, expires)
        return expires

    finally:
        transport.on_message = original_on_message  # type: ignore[attr-defined]


async def _registration_loop(
    transport: object,
    *,
    registrar_addr: tuple[str, int],
    local_addr: tuple[str, int],
    username: str,
    password: str,
    domain: str,
    expires: int,
) -> None:
    """Re-register periodically before the registration expires."""
    while True:
        # Re-register at 80% of the expiry interval
        await asyncio.sleep(expires * 0.8)
        try:
            await sip_register(
                transport,
                registrar_addr=registrar_addr,
                local_addr=local_addr,
                username=username,
                password=password,
                domain=domain,
                expires=expires,
            )
        except Exception:
            logger.exception("Re-registration failed — will retry in 30s")
            await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not REGISTRAR_HOST or not USERNAME or not PASSWORD:
        print(  # noqa: T201
            "Set SIP_REGISTRAR_HOST, SIP_USERNAME, and SIP_PASSWORD to run this example."
        )
        return

    kit = RoomKit()

    # -- SIP backend --
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", LOCAL_PORT),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        rtp_port_start=RTP_PORT_START,
        rtp_port_end=RTP_PORT_END,
    )

    # -- Voice channel (no STT/TTS in this example) --
    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    # -------------------------------------------------------------------
    # Protocol trace
    # -------------------------------------------------------------------

    voice.on_trace(
        lambda t: logger.info("[TRACE] %s %s: %s", t.direction, t.protocol, t.summary),
        protocols=["sip"],
    )

    # -------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def log_and_gate(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log every inbound event. Block outside business hours."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller", "unknown")
            hour = datetime.now(UTC).hour

            logger.info(
                "BEFORE_BROADCAST — new voice session from %s (hour=%d UTC)",
                caller,
                hour,
            )

            if not (8 <= hour < 20):
                logger.warning("Rejecting call outside business hours (hour=%d)", hour)
                return HookResult.block("outside_business_hours")

        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: object, ctx: RoomContext) -> HookResult:
        """Log what the caller says."""
        logger.info("ON_TRANSCRIPTION: %s", event.text)  # type: ignore[attr-defined]
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_PROTOCOL_TRACE)
    async def on_trace(trace: ProtocolTrace, ctx: RoomContext) -> None:
        """Room-level protocol trace."""
        logger.info(
            "ON_PROTOCOL_TRACE [room=%s] %s %s: %s",
            ctx.room.id,
            trace.direction,
            trace.protocol,
            trace.summary,
        )

    @kit.hook(HookTrigger.AFTER_BROADCAST)
    async def track_calls(event: RoomEvent, ctx: RoomContext) -> None:
        """Record every voice session for observability."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            entry = {
                "session_id": event.content.data.get("session_id"),
                "caller": event.content.data.get("caller"),
                "room_id": ctx.room.id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            call_log.append(entry)
            logger.info("AFTER_BROADCAST — call logged: %s", entry)

    # -------------------------------------------------------------------
    # Room setup
    # -------------------------------------------------------------------

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")

    # -------------------------------------------------------------------
    # Call handlers
    # -------------------------------------------------------------------

    @backend.on_call
    async def handle_call(session):
        """Called when an inbound SIP INVITE is accepted."""
        caller = session.metadata.get("caller")
        logger.info("Incoming call — session=%s caller=%s", session.id, caller)
        await kit.join("support", "voice", session=session)

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        """Called when the remote party hangs up (BYE)."""
        logger.info("Call ended — session=%s", session.id)
        await kit.leave(session)

    # -------------------------------------------------------------------
    # Start backend, then register with the PBX
    # -------------------------------------------------------------------

    await backend.start()

    registrar_addr = (REGISTRAR_HOST, REGISTRAR_PORT)
    local_addr = backend._local_sip_addr

    await sip_register(
        backend._transport,
        registrar_addr=registrar_addr,
        local_addr=local_addr,
        username=USERNAME,
        password=PASSWORD,
        domain=DOMAIN,
        expires=REGISTER_EXPIRY,
    )

    # Keep registration alive in the background
    reg_task = asyncio.create_task(
        _registration_loop(
            backend._transport,
            registrar_addr=registrar_addr,
            local_addr=local_addr,
            username=USERNAME,
            password=PASSWORD,
            domain=DOMAIN,
            expires=REGISTER_EXPIRY,
        ),
        name="sip_registration",
    )

    logger.info(
        "SIP agent ready — registered as %s@%s, listening on port %d, RTP %d-%d",
        USERNAME,
        DOMAIN,
        LOCAL_PORT,
        RTP_PORT_START,
        RTP_PORT_END,
    )

    # Run forever
    try:
        await asyncio.Event().wait()
    finally:
        reg_task.cancel()
        await backend.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""SIP voice backend with authentication and registrar support.

Demonstrates two SIP authentication features built into SIPVoiceBackend:

1. **Inbound auth** — incoming INVITEs are challenged with 401 digest
   authentication.  Only callers whose credentials match ``auth_users``
   are accepted.

2. **Outbound registration** — the backend registers as a SIP extension
   on a PBX (Asterisk, FreeSWITCH, Kamailio) via ``backend.register()``,
   handling the 401 challenge and periodic re-registration automatically.

Requirements:
    pip install roomkit[sip]

Usage:
    # Inbound auth only (no PBX registration):
    python examples/voice_sip_auth.py

    # With PBX registration:
    SIP_REGISTRAR_HOST=10.0.0.1 \
    SIP_USERNAME=6001 \
    SIP_PASSWORD=secret123 \
    python examples/voice_sip_auth.py

    # Environment variables (all optional):
    #   SIP_REGISTRAR_HOST  — PBX/registrar IP (enables registration)
    #   SIP_REGISTRAR_PORT  — registrar port          (default: 5060)
    #   SIP_USERNAME        — extension/username
    #   SIP_PASSWORD        — extension password
    #   SIP_DOMAIN          — SIP domain               (default: registrar host)
    #   SIP_LOCAL_PORT      — local SIP listen port    (default: 5060)
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_auth")

from roomkit import RoomKit, VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.models.hook import HookResult
from roomkit.voice.backends.sip import SIPVoiceBackend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_PORT = int(os.environ.get("SIP_LOCAL_PORT", "5060"))
REGISTRAR_HOST = os.environ.get("SIP_REGISTRAR_HOST", "")
REGISTRAR_PORT = int(os.environ.get("SIP_REGISTRAR_PORT", "5060"))
USERNAME = os.environ.get("SIP_USERNAME", "")
PASSWORD = os.environ.get("SIP_PASSWORD", "")
DOMAIN = os.environ.get("SIP_DOMAIN", "") or REGISTRAR_HOST

# Credentials accepted for inbound calls (username → password)
INBOUND_USERS = {"agent": "s3cret", "test": "test123"}


async def main() -> None:
    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # -- SIP backend with inbound digest auth --
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", LOCAL_PORT),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
        auth_users=INBOUND_USERS,
        auth_realm="mycompany.com",
    )

    voice = VoiceChannel("voice", backend=backend)
    kit.register_channel(voice)

    # -------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def log_calls(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Log new voice sessions. Block outside business hours."""
        if isinstance(event.content, SystemContent) and event.content.code == "session_started":
            caller = event.content.data.get("caller", "unknown")
            hour = datetime.now(UTC).hour
            logger.info("New call from %s (hour=%d UTC)", caller, hour)
            if not (8 <= hour < 20):
                return HookResult.block("outside_business_hours")
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: object, ctx: RoomContext) -> HookResult:
        logger.info("TRANSCRIPTION: %s", event.text)  # type: ignore[attr-defined]
        return HookResult.allow()

    # -------------------------------------------------------------------
    # Room + call handlers
    # -------------------------------------------------------------------

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "voice")

    @backend.on_call
    async def handle_call(session):
        logger.info(
            "Incoming call — session=%s caller=%s", session.id, session.metadata.get("caller")
        )
        await kit.join("support", "voice", session=session)

    @backend.on_call_disconnected
    async def handle_disconnect(session):
        logger.info("Call ended — session=%s", session.id)
        await kit.leave(session)

    # -------------------------------------------------------------------
    # Start + optional PBX registration
    # -------------------------------------------------------------------

    await backend.start()

    if REGISTRAR_HOST and USERNAME and PASSWORD:
        await backend.register(
            registrar_addr=(REGISTRAR_HOST, REGISTRAR_PORT),
            username=USERNAME,
            password=PASSWORD,
            domain=DOMAIN or None,
        )
        logger.info("Registered as %s@%s", USERNAME, DOMAIN)

    logger.info("SIP agent ready — listening on port %d", LOCAL_PORT)

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

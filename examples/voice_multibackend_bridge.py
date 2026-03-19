#!/usr/bin/env python3
"""Multi-transport audio bridge: SIP + WebRTC + WebSocket in one room.

Bridges participants from different transports into a single conference
room with N-party mixing.  SIP phones, WebRTC browsers, and WebSocket
clients all hear each other with live Deepgram transcription.  When the
last participant leaves, Claude generates a meeting summary.

Architecture::

    SIP phone  ──► SIP backend ──────┐
    Browser    ──► FastRTC (WebRTC) ──┼──► AudioBridge (N-party mix)
    Browser    ──► FastRTC (WS)    ──┘        │
                                              ├──► STT (Deepgram) ──► transcript
                                              └──► all other participants

Web UI: open examples/voice_agent_ui.html in a browser, set:
  - Server URL: http://localhost:8000
  - WS endpoint: /voice
  - Transport: WebSocket or WebRTC

Requirements:
    pip install roomkit[sip,fastrtc]

Run with:
    DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/voice_multibackend_bridge.py

Environment variables:
    DEEPGRAM_API_KEY   -- Deepgram API key (required)
    ANTHROPIC_API_KEY  -- Anthropic API key (required)
    STT_LANGUAGE       -- Language code for STT (default: multi = auto-detect)
    CLAUDE_MODEL       -- Claude model ID (default: claude-sonnet-4-20250514)
    SIP_LISTEN_ADDR    -- SIP listen IP   (default: 0.0.0.0)
    SIP_LISTEN_PORT    -- SIP listen port (default: 5060)
    RTP_IP             -- RTP bind IP     (default: 0.0.0.0)
    HTTP_PORT          -- HTTP port for FastRTC + UI (default: 8000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_multibackend_bridge")

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, mount_fastrtc_voice
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.bridge import AudioBridgeConfig
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "multi")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8000"))

ROOM_ID = "bridge-room"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

kit = RoomKit()
transcript: list[tuple[str, str]] = []


def _participant_name(session: Any) -> str:
    """Human-readable name from session metadata."""
    meta = session.metadata
    transport = meta.get("transport", "unknown")
    # SIP: use caller display name
    name = (
        meta.get("caller_display_name")
        or meta.get("caller_user")
        or session.participant_id
        or session.id[:8]
    )
    return f"{name} ({transport})"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

sip_backend = SIPVoiceBackend(
    local_sip_addr=(SIP_LISTEN_ADDR, SIP_LISTEN_PORT),  # nosec B104
    local_rtp_ip=RTP_IP,  # nosec B104
)

fastrtc_backend = FastRTCVoiceBackend(
    input_sample_rate=16000,
    output_sample_rate=16000,
)

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

stt = DeepgramSTTProvider(
    config=DeepgramConfig(
        api_key=DEEPGRAM_API_KEY,
        model="nova-3",
        language=STT_LANGUAGE,
        punctuate=True,
        smart_format=True,
        endpointing=300,
    )
)

# ---------------------------------------------------------------------------
# Voice channel: FastRTC backend + bridge (N-party mix) + STT
# ---------------------------------------------------------------------------

bridge_config = AudioBridgeConfig(mixing_strategy="mix", max_participants=10)

voice = VoiceChannel(
    "voice",
    backend=fastrtc_backend,
    stt=stt,
    bridge=bridge_config,
)
kit.register_channel(voice)

# Wire SIP inbound audio into the same pipeline so SIP participants
# get STT transcription and their audio flows through the bridge.
sip_backend.on_audio_received(voice._on_audio_received)
sip_backend.on_session_ready(voice._on_session_ready)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
async def on_session_started(event: Any, ctx: Any) -> None:
    name = _participant_name(event.session)
    count = voice._bridge.get_participant_count(ROOM_ID) if voice._bridge else 0
    logger.info("Joined: %s (total: %d)", name, count)


@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def on_transcription(event: Any, ctx: Any) -> HookResult:
    name = _participant_name(event.session)
    transcript.append((name, event.text))
    logger.info("[STT %s] %s", name, event.text)
    return HookResult.allow()


# ---------------------------------------------------------------------------
# SIP call handling
# ---------------------------------------------------------------------------


@sip_backend.on_call
async def handle_sip_call(session: Any) -> None:
    """Incoming SIP call — add to bridge room."""
    session.metadata["transport"] = "sip"
    name = _participant_name(session)
    logger.info("SIP call: %s (session=%s)", name, session.id)

    # backend=sip_backend so the bridge sends audio to this session
    # via the SIP backend, not the FastRTC backend.
    await kit.join(ROOM_ID, "voice", session=session, backend=sip_backend)

    count = voice._bridge.get_participant_count(ROOM_ID) if voice._bridge else 0
    logger.info("Participants: %d", count)


@sip_backend.on_call_disconnected
async def handle_sip_disconnect(session: Any) -> None:
    """SIP hangup — remove from bridge, maybe summarize."""
    name = _participant_name(session)
    await kit.leave(session)
    await _maybe_summarize(name)


# ---------------------------------------------------------------------------
# FastRTC session factory (WebSocket / WebRTC connections)
# ---------------------------------------------------------------------------

_web_session_counter = 0


async def fastrtc_session_factory(connection_id: str) -> Any:
    """Create a voice session when a browser connects via FastRTC."""
    global _web_session_counter  # noqa: PLW0603
    _web_session_counter += 1
    participant_id = f"web-user-{_web_session_counter}"

    session = await kit.join(ROOM_ID, "voice", participant_id=participant_id)

    name = _participant_name(session)
    count = voice._bridge.get_participant_count(ROOM_ID) if voice._bridge else 0
    logger.info("Web client joined: %s (total: %d)", name, count)
    return session


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


async def _maybe_summarize(who_left: str) -> None:
    """Generate AI summary when the last participant leaves."""
    count = voice._bridge.get_participant_count(ROOM_ID) if voice._bridge else 0
    logger.info("%s left (remaining: %d)", who_left, count)

    if count > 0 or len(transcript) == 0:
        return

    logger.info("All participants left. Generating summary...")
    transcript_text = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)
    logger.info("Full transcript:\n%s", transcript_text)

    ai = AIChannel(
        "ai-summarizer",
        provider=AnthropicAIProvider(
            AnthropicConfig(
                api_key=ANTHROPIC_API_KEY,
                model=CLAUDE_MODEL,
                max_tokens=1024,
            )
        ),
        system_prompt=(
            "You are a meeting assistant. Given a conversation "
            "transcript, produce a concise summary with:\n"
            "- Key topics discussed\n"
            "- Decisions made\n"
            "- Action items (if any)\n"
            "Keep it brief and actionable.\n\n"
            "IMPORTANT: Write the summary in the same language "
            "as the transcript."
        ),
    )
    kit.register_channel(ai)
    await kit.attach_channel(ROOM_ID, "ai-summarizer")

    await kit.process_inbound(
        InboundMessage(
            channel_id="voice",
            sender_id="system",
            content=TextContent(body="Summarize this meeting transcript:\n\n" + transcript_text),
            room_id=ROOM_ID,
        )
    )

    events = await kit.store.list_events(ROOM_ID)
    ai_events = [
        e
        for e in events
        if isinstance(e.content, TextContent) and e.source.channel_id == "ai-summarizer"
    ]
    if ai_events:
        summary = ai_events[-1].content.body
        logger.info(
            "\n========== MEETING SUMMARY ==========\n%s\n=====================================",
            summary,
        )
    else:
        logger.warning("No AI summary generated")

    transcript.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any):
    """Start SIP backend and RoomKit on startup."""
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")
    await sip_backend.start()

    logger.info("=== Multi-Transport Audio Bridge ===")
    logger.info("SIP:    %s:%d", SIP_LISTEN_ADDR, SIP_LISTEN_PORT)
    logger.info("HTTP:   http://0.0.0.0:%d", HTTP_PORT)
    logger.info("WebRTC: http://localhost:%d/voice/webrtc/offer", HTTP_PORT)
    logger.info("WS:     ws://localhost:%d/voice/websocket/offer", HTTP_PORT)
    logger.info("UI:     open examples/voice_agent_ui.html in a browser")
    logger.info(
        "STT: Deepgram nova-3 (lang=%s) | AI: Claude (%s)",
        STT_LANGUAGE,
        CLAUDE_MODEL,
    )
    logger.info("Connect from SIP phones and/or browsers. When all leave, a summary is generated.")
    yield
    await sip_backend.close()
    await fastrtc_backend.close()
    await kit.close()


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount FastRTC endpoints at /voice
    mount_fastrtc_voice(
        app,
        fastrtc_backend,
        path="/voice",
        session_factory=fastrtc_session_factory,
    )

    # Serve the web UI
    @app.get("/")
    async def serve_ui():
        ui_path = os.path.join(os.path.dirname(__file__), "voice_agent_ui.html")
        return FileResponse(ui_path, media_type="text/html")

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    missing = []
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(
            "\nDEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... "
            "uv run python examples/voice_multibackend_bridge.py"
        )
        sys.exit(1)

    import uvicorn

    config = uvicorn.Config(
        create_app(),
        host="0.0.0.0",  # nosec B104
        port=HTTP_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    serve_task = asyncio.create_task(server.serve())
    await stop.wait()
    server.should_exit = True
    await serve_task


if __name__ == "__main__":
    asyncio.run(main())

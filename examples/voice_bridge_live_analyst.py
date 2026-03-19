"""Live AI analyst on a bridged call — sentiment, topics, and alerts.

Bridges SIP and WebRTC participants while an AI agent silently monitors
the conversation in real time.  Every few turns, the AI produces a
structured analysis (sentiment, topics, action items) and pushes it to
a WebSocket dashboard channel.  The AI can also speak into the call
when it detects something critical (e.g. negative sentiment spike).

Architecture::

    SIP phone  ──► SIP backend  ──┐
    Browser    ──► FastRTC      ──┼──► AudioBridge (N-party mix)
                                  │        │
                                  │        ├──► STT (Deepgram)
                                  │        │       │
                                  │        │       └──► ON_TRANSCRIPTION hook
                                  │        │               │
                                  │        │          AI analyst (Claude)
                                  │        │           │           │
                                  │        │      analysis     critical?
                                  │        │        ↓              ↓
                                  │        │   WebSocket       voice.say()
                                  │        │   dashboard       "Please hold..."
                                  │        │
                                  └────────┘ (participants hear each other)

Web UI: open examples/voice_agent_ui.html in a browser, set:
  - Server URL: http://localhost:8000
  - WS endpoint: /voice

Requirements:
    pip install 'roomkit[sip,fastrtc]'

Run with:
    DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/voice_bridge_live_analyst.py

Environment variables:
    DEEPGRAM_API_KEY   -- Deepgram API key (required)
    ANTHROPIC_API_KEY  -- Anthropic API key (required for live analysis)
    STT_LANGUAGE       -- Language code for STT (default: multi)
    CLAUDE_MODEL       -- Claude model ID (default: claude-sonnet-4-20250514)
    ANALYSIS_INTERVAL  -- Analyze every N transcriptions (default: 5)
    SIP_LISTEN_ADDR    -- SIP listen IP (default: 0.0.0.0)
    SIP_LISTEN_PORT    -- SIP listen port (default: 5060)
    RTP_IP             -- RTP bind IP (default: 0.0.0.0)
    HTTP_PORT          -- HTTP port (default: 8000)
"""

from __future__ import annotations

import asyncio
import json
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
logger = logging.getLogger("voice_bridge_analyst")

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    VoiceChannel,
    WebSocketChannel,
)
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
ANALYSIS_INTERVAL = int(os.environ.get("ANALYSIS_INTERVAL", "5"))
SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8000"))

ROOM_ID = "bridge-room"

ANALYST_SYSTEM_PROMPT = """\
You are a real-time conversation analyst monitoring a live phone call.
You receive conversation excerpts and produce a brief JSON analysis.

Respond ONLY with valid JSON, no markdown, no explanation:
{
  "sentiment": "positive" | "neutral" | "negative",
  "sentiment_score": -1.0 to 1.0,
  "topics": ["topic1", "topic2"],
  "action_items": ["item1"],
  "alert": null or "string describing urgent issue",
  "summary": "one-sentence summary of what was discussed"
}

If someone expresses frustration, anger, or dissatisfaction, set
sentiment to "negative" and include an alert describing the issue.
"""

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

kit = RoomKit()
transcript: list[tuple[str, str]] = []
_analysis_pending = False


def _participant_name(session: Any) -> str:
    meta = session.metadata
    transport = meta.get("transport", "unknown")
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
# Voice channel (bridge + STT)
# ---------------------------------------------------------------------------

bridge_config = AudioBridgeConfig(mixing_strategy="mix", max_participants=10)

voice = VoiceChannel(
    "voice",
    backend=fastrtc_backend,
    stt=stt,
    bridge=bridge_config,
)
kit.register_channel(voice)

sip_backend.on_audio_received(voice._on_audio_received)
sip_backend.on_session_ready(voice._on_session_ready)

# ---------------------------------------------------------------------------
# WebSocket channel for dashboard
# ---------------------------------------------------------------------------

dashboard = WebSocketChannel("dashboard")
kit.register_channel(dashboard)


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------


async def _run_analysis() -> None:
    """Send recent transcript to the AI analyst and push results."""
    global _analysis_pending  # noqa: PLW0603
    if _analysis_pending or not ANTHROPIC_API_KEY:
        return

    _analysis_pending = True
    try:
        recent = transcript[-ANALYSIS_INTERVAL * 2 :]
        excerpt = "\n".join(f"{speaker}: {text}" for speaker, text in recent)

        # Call AI provider directly (not through room broadcast) to avoid
        # triggering the AI for every message in the room.
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=ANALYST_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this conversation excerpt:\n\n{excerpt}",
                }
            ],
        )
        raw = response.content[0].text
        logger.info("AI Analysis: %s", raw)

        # Parse and push to dashboard
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            analysis = {"summary": raw, "sentiment": "unknown"}

        await kit.process_inbound(
            InboundMessage(
                channel_id="dashboard",
                sender_id="analyst",
                content=TextContent(body=json.dumps({"type": "analysis", **analysis})),
                room_id=ROOM_ID,
            )
        )

        # If alert detected, announce to all participants
        alert = analysis.get("alert")
        if alert:
            logger.warning("ALERT: %s", alert)
            await kit.process_inbound(
                InboundMessage(
                    channel_id="dashboard",
                    sender_id="system",
                    content=TextContent(body=json.dumps({"type": "alert", "message": alert})),
                    room_id=ROOM_ID,
                )
            )

        sentiment_score = analysis.get("sentiment_score", 0)
        if isinstance(sentiment_score, (int, float)) and sentiment_score < -0.7:
            logger.warning("Negative sentiment (%.1f), notifying via voice", sentiment_score)
            bridge = voice._bridge
            if bridge:
                for session, _backend in bridge.get_bridged_sessions(ROOM_ID):
                    if session.metadata.get("transport") == "sip":
                        continue  # Don't interrupt the unhappy caller
                    await voice.say(
                        session,
                        "Attention: the system has detected elevated frustration. "
                        "Please consider de-escalation.",
                    )

    finally:
        _analysis_pending = False


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
async def on_session_started(event: Any, ctx: Any) -> None:
    name = _participant_name(event.session)
    count = voice._bridge.get_participant_count(ROOM_ID) if voice._bridge else 0
    logger.info("Joined: %s (total: %d)", name, count)

    await kit.process_inbound(
        InboundMessage(
            channel_id="dashboard",
            sender_id="system",
            content=TextContent(
                body=json.dumps({"type": "join", "participant": name, "count": count})
            ),
            room_id=ROOM_ID,
        )
    )


@kit.hook(HookTrigger.ON_TRANSCRIPTION)
async def on_transcription(event: Any, ctx: Any) -> HookResult:
    name = _participant_name(event.session)
    transcript.append((name, event.text))
    logger.info("[STT %s] %s", name, event.text)

    # Push live transcription to dashboard
    await kit.process_inbound(
        InboundMessage(
            channel_id="dashboard",
            sender_id="system",
            content=TextContent(
                body=json.dumps({"type": "transcription", "speaker": name, "text": event.text})
            ),
            room_id=ROOM_ID,
        )
    )

    # Trigger analysis every N transcriptions
    if len(transcript) % ANALYSIS_INTERVAL == 0:
        asyncio.create_task(_run_analysis(), name="analysis")

    return HookResult.allow()


# ---------------------------------------------------------------------------
# SIP call handling
# ---------------------------------------------------------------------------


@sip_backend.on_call
async def handle_sip_call(session: Any) -> None:
    session.metadata["transport"] = "sip"
    name = _participant_name(session)
    logger.info("SIP call: %s", name)

    await kit.join(ROOM_ID, "voice", session=session, backend=sip_backend)


@sip_backend.on_call_disconnected
async def handle_sip_disconnect(session: Any) -> None:
    name = _participant_name(session)
    await kit.leave(session)
    logger.info("%s left", name)


# ---------------------------------------------------------------------------
# FastRTC session factory
# ---------------------------------------------------------------------------

_web_session_counter = 0


async def fastrtc_session_factory(connection_id: str) -> Any:
    global _web_session_counter  # noqa: PLW0603
    _web_session_counter += 1
    participant_id = f"web-user-{_web_session_counter}"

    session = await kit.join(ROOM_ID, "voice", participant_id=participant_id)
    return session


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any):
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")
    await kit.attach_channel(ROOM_ID, "dashboard")
    await sip_backend.start()

    logger.info("=== Voice Bridge + Live AI Analyst ===")
    logger.info("SIP:       %s:%d", SIP_LISTEN_ADDR, SIP_LISTEN_PORT)
    logger.info("HTTP:      http://0.0.0.0:%d", HTTP_PORT)
    logger.info("Dashboard: ws://localhost:%d/ws/dashboard", HTTP_PORT)
    logger.info("Analysis every %d transcriptions", ANALYSIS_INTERVAL)
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — AI analysis disabled, transcription only")
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

    mount_fastrtc_voice(
        app,
        fastrtc_backend,
        path="/voice",
        session_factory=fastrtc_session_factory,
    )

    @app.get("/")
    async def serve_ui():
        ui_path = os.path.join(os.path.dirname(__file__), "voice_agent_ui.html")
        return FileResponse(ui_path, media_type="text/html")

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not DEEPGRAM_API_KEY:
        print("Missing: DEEPGRAM_API_KEY")
        print(
            "\nDEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... "
            "uv run python examples/voice_bridge_live_analyst.py"
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

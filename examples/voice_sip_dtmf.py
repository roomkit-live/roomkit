#!/usr/bin/env python3
"""Send DTMF tones during a SIP call via AI tool calling.

Demonstrates how an AI agent can send DTMF digits into an active SIP
call — useful for navigating IVR menus, entering PINs, or interacting
with phone systems programmatically.

The AI is given a ``send_dtmf`` tool via binding metadata.  When it
decides to press a key (e.g. "press 1 for English"), it calls the tool,
which delegates to ``VoiceChannel.send_dtmf()`` →
``SIPVoiceBackend.send_dtmf()`` → RFC 4733 RTP telephone-event.

Requirements:
    pip install roomkit[sip]

Run with:
    uv run python examples/voice_sip_dtmf.py

Environment variables (all optional):
    SIP_PROXY_HOST  — SIP proxy IP        (default: 127.0.0.1)
    SIP_PROXY_PORT  — SIP proxy port      (default: 5060)
    SIP_FROM_URI    — caller SIP URI      (default: sip:bot@example.com)
    SIP_TO_URI      — callee SIP URI      (default: sip:ivr@example.com)
    OPENAI_API_KEY  — OpenAI key for AI   (uses mock if unset)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("voice_sip_dtmf")

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.backends.sip import PT_PCMU, SIPVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.pipeline.dtmf import MockDTMFDetector
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIP_PROXY_HOST = os.environ.get("SIP_PROXY_HOST", "127.0.0.1")
SIP_PROXY_PORT = int(os.environ.get("SIP_PROXY_PORT", "5060"))
FROM_URI = os.environ.get("SIP_FROM_URI", "sip:bot@example.com")
TO_URI = os.environ.get("SIP_TO_URI", "sip:ivr@example.com")

SYSTEM_PROMPT = (
    "You are an AI agent navigating a phone IVR system. "
    "When you hear menu options (e.g. 'press 1 for sales'), "
    "use the send_dtmf tool to press the correct key. "
    "You can send digits 0-9, *, and #."
)

# Tool definition as a raw dict (passed via binding metadata)
DTMF_TOOL = {
    "name": "send_dtmf",
    "description": (
        "Send a DTMF tone (key press) into the active phone call. "
        "Use this to navigate IVR menus or enter numeric codes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "digit": {
                "type": "string",
                "description": "The DTMF digit to send: 0-9, *, or #",
                "enum": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "*",
                    "#",
                ],
            },
            "duration_ms": {
                "type": "integer",
                "description": "Tone duration in milliseconds (default 160)",
                "default": 160,
            },
        },
        "required": ["digit"],
    },
}


async def main() -> None:
    kit = RoomKit()
    room_id = "dtmf-demo"

    # --- SIP backend ----------------------------------------------------------
    backend = SIPVoiceBackend(
        local_sip_addr=("0.0.0.0", 5070),  # nosec B104
        local_rtp_ip="0.0.0.0",  # nosec B104
    )

    # --- Pipeline: VAD + DTMF (mock for demo) ---------------------------------
    vad = MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            None,
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"demo-audio",
                duration_ms=2000.0,
            ),
        ]
    )
    dtmf_detector = MockDTMFDetector()
    pipeline = AudioPipelineConfig(vad=vad, dtmf=dtmf_detector)

    # --- STT + TTS (mock for demo) --------------------------------------------
    stt = MockSTTProvider(
        transcripts=["Press 1 for sales, press 2 for support."],
    )
    tts = MockTTSProvider()

    # --- Voice channel --------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    # --- DTMF tool handler ----------------------------------------------------
    # Capture the active session so the tool handler can send DTMF.
    active_session: dict[str, VoiceSession] = {}

    async def handle_tool(name: str, arguments: dict) -> str:
        if name == "send_dtmf":
            digit = arguments["digit"]
            duration_ms = arguments.get("duration_ms", 160)
            session = active_session.get(room_id)
            if session is None:
                return json.dumps({"error": "No active voice session"})
            # send_dtmf is synchronous (RFC 4733 packets are queued)
            voice.send_dtmf(session, digit, duration_ms)
            return json.dumps(
                {
                    "status": "sent",
                    "digit": digit,
                    "duration_ms": duration_ms,
                }
            )
        return json.dumps({"error": f"Unknown tool: {name}"})

    # --- AI channel -----------------------------------------------------------
    ai_provider = MockAIProvider(
        responses=["I'll press 1 for sales now."],
    )
    ai = AIChannel(
        "ai",
        provider=ai_provider,
        system_prompt=SYSTEM_PROMPT,
        tool_handler=handle_tool,
    )
    kit.register_channel(ai)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=room_id)
    await kit.attach_channel(room_id, "voice")
    await kit.attach_channel(
        room_id,
        "ai",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [DTMF_TOOL]},
    )

    # --- Hooks ----------------------------------------------------------------
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("Transcription: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_DTMF, execution=HookExecution.ASYNC)
    async def on_dtmf(event, ctx):
        logger.info(
            "DTMF received: digit=%s duration=%sms",
            event.digit,
            event.duration_ms,
        )

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("AI says: %s", text)
        return HookResult.allow()

    # --- Dial -----------------------------------------------------------------
    await backend.start()

    logger.info(
        "Dialing %s from %s via %s:%d ...",
        TO_URI,
        FROM_URI,
        SIP_PROXY_HOST,
        SIP_PROXY_PORT,
    )
    try:
        session = await backend.dial(
            to_uri=TO_URI,
            from_uri=FROM_URI,
            proxy_addr=(SIP_PROXY_HOST, SIP_PROXY_PORT),
            codec=PT_PCMU,
            timeout=30.0,
        )
    except (TimeoutError, RuntimeError) as exc:
        logger.error("Call failed: %s", exc)
        await backend.close()
        return

    # Bind session to pipeline, then expose to the tool handler
    binding = ChannelBinding(
        room_id=room_id,
        channel_id="voice",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, room_id, binding)
    active_session[room_id] = session

    logger.info(
        "Call active — session=%s. Waiting for IVR prompts...",
        session.id,
    )

    # --- Keep running ---------------------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    await backend.close()
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())

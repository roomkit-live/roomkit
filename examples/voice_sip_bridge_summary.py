#!/usr/bin/env python3
"""Bridge SIP calls with live transcription and AI meeting summary.

Accepts incoming SIP calls and bridges them in a conference room.
Audio flows directly between participants while Deepgram transcribes
each speaker in real time.  When the last participant hangs up, the
accumulated transcript is sent to Claude which generates a meeting
summary in the same language the participants spoke.

Language detection is automatic — Deepgram Nova-3 with ``language=multi``
identifies the spoken language, and Claude is instructed to write the
summary in that same language.

Architecture::

    Caller A ──mic──► Pipeline ──► STT (Deepgram) ──► transcript log
                          │                                  │
                          └──bridge──► Caller B               │
                                                              │
                     (last caller hangs up) ──────► Claude (summary)

Requirements:
    pip install roomkit[sip]

Run with:
    DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... \
        uv run python examples/voice_sip_bridge_summary.py

Environment variables:
    DEEPGRAM_API_KEY   — Deepgram API key (required)
    ANTHROPIC_API_KEY  — Anthropic API key (required)
    STT_LANGUAGE       — Language code for STT (default: multi = auto-detect)
    CLAUDE_MODEL       — Claude model ID (default: claude-sonnet-4-20250514)
    SIP_LISTEN_ADDR    — SIP listen IP   (default: 0.0.0.0)
    SIP_LISTEN_PORT    — SIP listen port (default: 5060)
    RTP_IP             — RTP bind IP     (default: 0.0.0.0)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

logger = setup_logging("voice_sip_bridge_summary")

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
from roomkit.voice.backends.sip import SIPVoiceBackend
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "multi")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
SIP_LISTEN_ADDR = os.environ.get("SIP_LISTEN_ADDR", "0.0.0.0")
SIP_LISTEN_PORT = int(os.environ.get("SIP_LISTEN_PORT", "5060"))
RTP_IP = os.environ.get("RTP_IP", "0.0.0.0")

ROOM_ID = "bridge-room"


def _caller_name(session) -> str:
    """Extract a human-readable name from a SIP session."""
    meta = session.metadata
    return (
        meta.get("caller_display_name")
        or meta.get("caller_user")
        or session.participant_id
        or session.id
    )


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY")

    kit = RoomKit()
    console_cleanup = setup_console(kit)

    # --- Transcript accumulator -----------------------------------------------
    transcript: list[tuple[str, str]] = []

    # --- SIP backend ----------------------------------------------------------
    backend = SIPVoiceBackend(
        local_sip_addr=(SIP_LISTEN_ADDR, SIP_LISTEN_PORT),  # nosec B104
        local_rtp_ip=RTP_IP,  # nosec B104
    )

    # --- STT: Deepgram streaming ----------------------------------------------
    # language="multi" enables automatic language detection — Deepgram
    # transcribes in whatever language the caller speaks.
    stt = DeepgramSTTProvider(
        config=DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model="nova-3",
            language=STT_LANGUAGE,
            punctuate=True,
            smart_format=True,
            endpointing=300,
        )
    )

    # --- Voice channel with bridge + STT (no TTS — humans only) ---------------
    voice = VoiceChannel(
        "voice",
        backend=backend,
        stt=stt,
        bridge=True,
    )
    kit.register_channel(voice)

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "voice")

    # --- Hooks: live transcript capture ---------------------------------------
    @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
    async def on_session_started(event, ctx):
        name = _caller_name(event.session)
        logger.info("Session started: %s", name)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        name = _caller_name(event.session)
        transcript.append((name, event.text))
        logger.info("[TRANSCRIPT %s] %s", name, event.text)
        return HookResult.allow()

    # --- Handle incoming SIP calls --------------------------------------------
    @backend.on_call
    async def handle_call(session):
        name = _caller_name(session)
        logger.info("Incoming call: %s (session=%s)", name, session.id)
        await kit.join(ROOM_ID, "voice", session=session)
        count = voice._bridge.get_participant_count(ROOM_ID)
        logger.info(
            "Participants in room: %d — %s",
            count,
            "conference active" if count >= 2 else "waiting for more callers",
        )

    # --- Handle hangups: summarize when last caller leaves --------------------
    @backend.on_call_disconnected
    async def handle_disconnect(session):
        name = _caller_name(session)
        await kit.leave(session)
        count = voice._bridge.get_participant_count(ROOM_ID)
        logger.info("%s hung up (remaining: %d)", name, count)

        if count > 0 or len(transcript) == 0:
            return

        # Last participant left — generate AI summary
        logger.info("All participants left. Generating summary...")

        transcript_text = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)
        logger.info("Full transcript:\n%s", transcript_text)

        # Attach the AI channel and request a summary
        ai = AIChannel(
            "ai-summarizer",
            provider=AnthropicAIProvider(
                AnthropicConfig(
                    api_key=env["ANTHROPIC_API_KEY"],
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
                "as the transcript. If the conversation is in French, "
                "summarize in French. If in Spanish, summarize in "
                "Spanish. Always match the language of the speakers."
            ),
        )
        kit.register_channel(ai)
        await kit.attach_channel(
            ROOM_ID,
            "ai-summarizer",
        )

        await kit.process_inbound(
            InboundMessage(
                channel_id="voice",
                sender_id="system",
                content=TextContent(
                    body=("Summarize this meeting transcript:\n\n" + transcript_text)
                ),
                room_id=ROOM_ID,
            )
        )

        # Retrieve the AI summary from the room events
        events = await kit.store.list_events(ROOM_ID)
        ai_events = [
            e
            for e in events
            if isinstance(e.content, TextContent) and e.source.channel_id == "ai-summarizer"
        ]
        if ai_events:
            summary = ai_events[-1].content.body
            logger.info(
                "\n========== MEETING SUMMARY ==========\n%s\n"
                "=====================================",
                summary,
            )
        else:
            logger.warning("No AI summary generated")

        # Clear transcript for next conference
        transcript.clear()

    # --- Start ----------------------------------------------------------------
    await backend.start()
    logger.info(
        "Listening for SIP calls on %s:%d",
        SIP_LISTEN_ADDR,
        SIP_LISTEN_PORT,
    )
    logger.info(
        "STT: Deepgram nova-3 (language=%s) | AI: Claude (%s)",
        STT_LANGUAGE,
        CLAUDE_MODEL,
    )
    logger.info(
        "Call in from multiple phones to start a conference. "
        "When all hang up, a meeting summary is generated."
    )

    async def cleanup():
        if console_cleanup:
            await console_cleanup()
        await backend.close()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

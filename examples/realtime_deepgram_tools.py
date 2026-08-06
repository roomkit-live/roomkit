"""RoomKit — Deepgram Voice Agent with tool calling.

Ask the agent about the weather and it calls your Python function mid-conversation.

Tool calling on Deepgram needs no dashboard setup, unlike ElevenLabs: the schemas
travel in the ``Settings`` message as ``agent.think.functions``, Deepgram answers
with a ``FunctionCallRequest``, and the value your ``tool_handler`` returns goes
back as a ``FunctionCallResponse``. Hooks, gates and result truncation apply
exactly as on the other providers.

A function declared with an ``endpoint`` is the other half of the story: Deepgram
calls that HTTP endpoint itself and never tells RoomKit. That is the right shape
for a lookup the agent should make without a round-trip through your process —
see the commented block below.

Requirements:
    pip install roomkit[realtime-deepgram,local-audio] aec-audio-processing

Run with:
    DEEPGRAM_API_KEY=... uv run python examples/realtime_deepgram_tools.py

Environment variables:
    DEEPGRAM_API_KEY      (required) Deepgram API key
    DEEPGRAM_VOICE        Aura voice (default: aura-2-thalia-en)
    DEEPGRAM_LANGUAGE     Transcription language, e.g. fr-CA
    DEEPGRAM_THINK_MODEL  LLM model (default: gpt-4o-mini)
    AEC                   webrtc (default) | speex | 0 to disable
    AEC_DELAY_MS          Optional measured speaker-to-mic delay (default: auto)
    AUDIO_PREBUFFER_MS    Speaker jitter buffer (default: 240)
    MUTE_MIC              0 to keep the mic open during playback, 1 to force muting
                          (default: muted only when AEC is unavailable)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    build_aec,
    build_pipeline,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.deepgram import DeepgramAgentConfig, DeepgramAgentProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_deepgram_tools")

TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment for a caller on a given day",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "day": {"type": "string", "description": "ISO date, e.g. 2026-08-12"},
            },
            "required": ["name", "day"],
        },
    },
    # Server-side variant — Deepgram calls this itself and RoomKit never sees it:
    # {
    #     "name": "lookup_account",
    #     "description": "Look up a customer account",
    #     "parameters": {...},
    #     "endpoint": {
    #         "url": "https://api.example.com/accounts",
    #         "method": "POST",
    #         "headers": {"authorization": "Bearer ..."},
    #     },
    # },
]


async def handle_tool(name: str, arguments: dict) -> str:
    """Execute a tool call from the agent and return its result as JSON."""
    logger.info("Tool call: %s(%s)", name, arguments)

    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        return json.dumps({"city": city, "temperature_c": 22, "condition": "sunny"})

    if name == "book_appointment":
        return json.dumps(
            {
                "confirmed": True,
                "name": arguments.get("name"),
                "day": arguments.get("day"),
                "time": "14:30",
            }
        )

    # The agent hears this back and can recover rather than stalling its turn.
    return json.dumps({"error": f"Unknown tool: {name}"})


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    config = DeepgramAgentConfig(
        api_key=env["DEEPGRAM_API_KEY"],
        listen_language=os.environ.get("DEEPGRAM_LANGUAGE"),
        think_model=os.environ.get("DEEPGRAM_THINK_MODEL", "gpt-4o-mini"),
        speak_model=os.environ.get("DEEPGRAM_VOICE", "aura-2-thalia-en"),
        greeting="Hi! Ask me about the weather, or book an appointment.",
    )
    provider = DeepgramAgentProvider(config)

    # --- Audio (see realtime_voice_local_deepgram.py for the full rationale) ---
    sample_rate = 24000
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    pipeline = build_pipeline(aec=aec)
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        rt_prebuffer_ms=max(0, int(os.environ.get("AUDIO_PREBUFFER_MS", "240"))),
        aec=aec,
    )

    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a friendly voice assistant. Use your tools when asked about "
            "the weather or to book an appointment. Keep answers short.",
        ),
        voice=config.speak_model,
        tools=TOOLS,
        tool_handler=handle_tool,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(channel)

    await kit.create_room(room_id="tool-demo")
    await kit.attach_channel("tool-demo", "voice")

    session = await channel.start_session("tool-demo", "local-user", connection=None)

    logger.info("Deepgram Voice Agent session started with %d tools", len(TOOLS))
    logger.info('Try: "What is the weather in Montreal?" — Ctrl+C to stop.\n')

    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

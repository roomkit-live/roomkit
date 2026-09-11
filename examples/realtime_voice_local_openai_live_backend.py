"""RoomKit — OpenAI GPT-Live with your own reasoning backend (Claude).

The live model holds the spoken conversation. Anything about the user's
flights goes to a ``ReasoningBackend`` running Claude with its own tools, in
this process, through ``IntegratorReasoning`` (RFC §12.4.1). GPT-Live sends no
task text: the channel hands the backend the transcript recorded since the
previous delegation, and the backend works out the request from it.

Rebooking a cancelled flight takes the backend three tool calls, each needing
the last one's answer. What the backend says between calls is relayed as
spoken progress (``SPOKEN_PROGRESS=1``, the default) or kept as silent context
(``SPOKEN_PROGRESS=0``); its final answer is always spoken, in the live model's
own words. Every tool call passes the channel's pre-execution gate, so a
``BEFORE_TOOL_USE`` hook denies a backend call the same way it denies any other.

Say something like: "My flight UA482 this morning — can you check it, and get
me on something else if it's not running?"

Requirements:
    pip install roomkit[realtime-openai,anthropic,local-audio] aec-audio-processing

Run with:
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \\
        uv run python examples/realtime_voice_local_openai_live_backend.py

Environment variables:
    OPENAI_API_KEY      (required) OpenAI API key for GPT-Live
    ANTHROPIC_API_KEY   (required) Anthropic API key for the backend
    ANTHROPIC_MODEL     Backend model (default: claude-sonnet-5)
    OPENAI_LIVE_VOICE   Voice (default: cedar). Others: quartz, ripple, vesper,
                        willow, stone, gleam, meridian, delta, cinder, beacon,
                        bossa, tempo (fixed for the session)
    SYSTEM_PROMPT       Custom instructions for the live model (write them in the
                        user's language to pin the spoken language)
    SPOKEN_PROGRESS     1 (default) to voice the backend's intermediate steps
    AEC                 webrtc (default) | speex | 0 to disable
    DENOISE             webrtc (default) | rnnoise | sherpa | 0 to disable
    MUTE_MIC            1 to mute the mic during playback (default: only when
                        AEC is unavailable)

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
    build_denoiser,
    build_pipeline,
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import (
    AIProviderReasoningBackend,
    HookExecution,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomKit,
)
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.providers.openai.live import IntegratorReasoning, OpenAILiveProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_openai_live_backend")

FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Speak warmly and
naturally, at an unhurried pace, one or two sentences at a time, and let the
user finish before responding. Be clear and direct, not overly cheerful.
Speak the language the user speaks, and keep to it.

Answer simple conversational questions directly. Delegate anything about the
user's flights or bookings — checking one, changing one, finding another. The
backend reads the conversation, so hand off as soon as you know the request is
for it. While it works, keep the conversation going; relay what it sends you
as it arrives, and ignore results the conversation has already moved past.

Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Each message you receive
is the recent voice conversation between the user and the assistant, as a
transcript. Work out what is being asked and do it. The transcript may contain
transcription errors; use the most likely intent. Answer in the language the
user speaks.

Rebooking runs in steps: check the flight, find what else flies that route,
then book a seat on the earliest one that has them. Each step needs the one
before it, and the user has asked you to see it through, so carry on to the
booking rather than coming back with a menu. Before each tool call, say in one
short sentence what you are doing. Reply with the verified result in concise,
conversational plain text — no Markdown, no raw JSON — and never claim an
action completed without a tool result confirming it."""

TOOLS = [
    {
        "name": "check_flight_status",
        "description": "Check whether a flight is running, and what route it flies",
        "parameters": {
            "type": "object",
            "properties": {"flight_number": {"type": "string", "description": "e.g. UA482"}},
            "required": ["flight_number"],
        },
    },
    {
        "name": "find_alternative_flights",
        "description": "Find later flights on a route today",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Airport code, e.g. SFO"},
                "destination": {"type": "string", "description": "Airport code, e.g. JFK"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "rebook_flight",
        "description": "Move the booking onto another flight",
        "parameters": {
            "type": "object",
            "properties": {"flight_number": {"type": "string", "description": "e.g. UA716"}},
            "required": ["flight_number"],
        },
    },
]


async def handle_tool(name: str, arguments: dict) -> str:
    """The three steps of a rebooking, each slow enough that the user notices."""
    logger.info("Backend tool call: %s(%s)", name, arguments)
    await asyncio.sleep(2)
    if name == "check_flight_status":
        return json.dumps(
            {
                "flight": arguments.get("flight_number"),
                "status": "cancelled",
                "reason": "crew shortage",
                "origin": "SFO",
                "destination": "JFK",
            }
        )
    if name == "find_alternative_flights":
        return json.dumps(
            {
                "flights": [
                    {"flight": "UA716", "departs": "16:15", "seats": 2},
                    {"flight": "AA229", "departs": "19:05", "seats": 11},
                ]
            }
        )
    if name == "rebook_flight":
        return json.dumps(
            {
                "flight": arguments.get("flight_number"),
                "status": "confirmed",
                "seat": "14C",
                "confirmation": "X7K2QP",
            }
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


async def main() -> None:
    env = require_env("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- The backend: Claude with the channel's tools, in a tool loop ---
    backend = AIProviderReasoningBackend(
        AnthropicAIProvider(
            AnthropicConfig(
                api_key=env["ANTHROPIC_API_KEY"],
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            )
        ),
        system_prompt=BACKEND_INSTRUCTIONS,
        spoken_progress=env_bool("SPOKEN_PROGRESS", default=True),
    )

    # --- GPT-Live provider, integrator-side delegation ---
    provider = OpenAILiveProvider(api_key=env["OPENAI_API_KEY"], delegation=IntegratorReasoning())

    # --- Audio: AEC and noise suppression, no local VAD ---
    sample_rate = 24000
    block_ms = 20
    aec = build_aec(sample_rate, block_ms, default="webrtc")
    denoiser = build_denoiser(sample_rate, default="webrtc")
    pipeline = build_pipeline(aec=aec, denoiser=denoiser)
    mute_mic = env_bool("MUTE_MIC", default=aec is None)
    if mute_mic:
        logger.warning("Microphone muted during playback: the model cannot be talked over")

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )

    # --- Realtime voice channel: the tools are the backend's, the gate is the channel's ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get("SYSTEM_PROMPT", FRONTEND_INSTRUCTIONS),
        voice=os.environ.get("OPENAI_LIVE_VOICE", "cedar"),
        tools=TOOLS,
        tool_handler=handle_tool,
        reasoning_backend=backend,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(channel)

    @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, execution=HookExecution.ASYNC)
    async def on_delegation(event, ctx) -> None:  # noqa: ANN001
        logger.info("Delegated to the backend: %s", event.delegation_id)

    # --- Room ---
    await kit.create_room(room_id="rebooking-demo")
    await kit.attach_channel("rebooking-demo", "voice")

    session = await channel.start_session("rebooking-demo", "local-user", connection=None)
    await channel.inject_text(session, "Greet the user and ask how you can help.")

    logger.info("GPT-Live session started (Claude backend)")
    logger.info('Try: "My flight UA482 this morning — can you check it and rebook me?"\n')

    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

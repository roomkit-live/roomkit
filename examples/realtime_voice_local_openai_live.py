"""RoomKit — Full-duplex speech-to-speech with OpenAI GPT-Live (hosted reasoning).

GPT-Live listens and speaks at the same time: it handles being interrupted by
itself, can backchannel while you talk, and hands anything that needs tools or
careful reasoning to a backend model while the conversation keeps going. Here
OpenAI hosts that backend (``HostedReasoning``): the channel's tools become the
backend's tools, and its function calls run in this process through
``tool_handler`` exactly as with any other provider.

Because the model is full-duplex, RoomKit runs no barge-in path on this
session: no playback flush, no interrupt, no gating of the model's audio while
you speak (RFC §12.4.1). The microphone stays open during playback, so keep an
AEC (or headphones) so the model does not hear itself.

Requirements:
    pip install roomkit[realtime-openai,local-audio] aec-audio-processing

Run with:
    OPENAI_API_KEY=... uv run python examples/realtime_voice_local_openai_live.py

Environment variables:
    OPENAI_API_KEY      (required) OpenAI API key
    OPENAI_LIVE_MODEL   Live model (default: the provider's own default)
    OPENAI_LIVE_VOICE   Voice (default: marin). Others: quartz, ripple, vesper,
                        willow, stone, gleam, meridian, delta, cinder, beacon,
                        bossa, tempo (fixed for the session)
    BACKEND_MODEL       Hosted backend model (default: gpt-5.6-terra)
    SYSTEM_PROMPT       Custom instructions for the live model
    AEC                 webrtc (default) | speex | 0 to disable
    DENOISE             webrtc (default) | rnnoise | sherpa | 0 to disable
    AEC_DELAY_MS        Optional measured speaker-to-mic delay (default: auto)
    MUTE_MIC            1 to mute the mic during playback (default: only when
                        AEC is unavailable — muting defeats full-duplex)

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

from roomkit import HookExecution, HookTrigger, RealtimeVoiceChannel, RoomKit
from roomkit.providers.openai.live import HostedReasoning, OpenAILiveProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_openai_live")

# The live model only needs to know how to converse and when to delegate.
# Task knowledge, tools and business rules belong to the backend prompt.
FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Speak warmly and
naturally, at an unhurried pace, one or two sentences at a time, and let the
user finish before responding. Be clear and direct, not overly cheerful.
Speak the language the user speaks, and keep to it.

Answer simple conversational questions directly. Delegate when the user asks
for current information, such as the weather or a restaurant recommendation.
When delegating, include the user's goal and the exact details they gave, so
the request is self-contained. While the delegated work runs, keep the
conversation going and relay the result once it arrives; ignore results the
conversation has already moved past.

Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = """You are helping an assistant during a live voice conversation.
The request may contain transcription errors; use the most likely intent. Use
the available tools to answer questions about the weather and restaurants.
Answer in the language the user speaks. Return the verified result in concise, conversational plain text — no Markdown,
no raw JSON — and never claim an action completed without a tool result
confirming it."""

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
        "name": "get_restaurant_recommendation",
        "description": "Recommend a restaurant in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
]


async def handle_tool(name: str, arguments: dict) -> str:
    """Execute a backend function call and return its result as JSON."""
    logger.info("Tool call: %s(%s)", name, arguments)
    if name == "get_weather":
        return json.dumps(
            {"city": arguments.get("city"), "temperature_c": 22, "condition": "sunny"}
        )
    if name == "get_restaurant_recommendation":
        return json.dumps({"name": "The Golden Dragon", "city": arguments.get("city")})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def main() -> None:
    env = require_env("OPENAI_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- GPT-Live provider, OpenAI-hosted backend ---
    model = os.environ.get("OPENAI_LIVE_MODEL")
    provider = OpenAILiveProvider(
        api_key=env["OPENAI_API_KEY"],
        **({"model": model} if model else {}),
        delegation=HostedReasoning(
            model=os.environ.get("BACKEND_MODEL", "gpt-5.6-terra"),
            instructions=BACKEND_INSTRUCTIONS,
            reasoning_effort="low",
        ),
    )

    # --- Audio: AEC and noise suppression, no local VAD ---
    # The model takes turns itself; a pipeline VAD would only observe.
    sample_rate = 24000
    block_ms = 20
    aec = build_aec(sample_rate, block_ms, default="webrtc")
    denoiser = build_denoiser(sample_rate, default="webrtc")
    pipeline = build_pipeline(aec=aec, denoiser=denoiser)

    # Full-duplex needs the microphone open while the model speaks. Without
    # AEC the speaker would feed the model its own voice, so mute instead.
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

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get("SYSTEM_PROMPT", FRONTEND_INSTRUCTIONS),
        voice=os.environ.get("OPENAI_LIVE_VOICE", "marin"),
        tools=TOOLS,
        tool_handler=handle_tool,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(channel)

    @kit.hook(HookTrigger.ON_REALTIME_DELEGATION, execution=HookExecution.ASYNC)
    async def on_delegation(event, ctx) -> None:  # noqa: ANN001
        logger.info("Delegated to the %s backend: %s", event.target, event.delegation_id)

    # --- Room ---
    await kit.create_room(room_id="live-demo")
    await kit.attach_channel("live-demo", "voice")

    # --- Start session (connection=None for local transport) ---
    session = await channel.start_session("live-demo", "local-user", connection=None)

    # Spoken context is how GPT-Live is asked to open the conversation; the
    # model paraphrases it rather than reading it out.
    await channel.inject_text(session, "Greet the user and ask how you can help.")

    logger.info("GPT-Live session started (hosted backend)")
    logger.info('Try: "What is the weather in Montreal?" — talk over it, it listens.\n')

    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

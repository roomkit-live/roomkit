"""RoomKit — ElevenLabs Conversational AI with client-side tool calling.

Demonstrates how to handle tool calls from an ElevenLabs agent.
Tools are configured on the ElevenLabs dashboard as "client tools";
the agent invokes them and this script executes the handler.

Setup:
    1. Create an agent on the ElevenLabs dashboard
    2. Add a client tool named "get_weather" with a "city" string parameter
    3. Set the agent_id below

Requirements:
    pip install roomkit[realtime-elevenlabs,local-audio]

Run with:
    ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... \
        uv run python examples/realtime_elevenlabs_tools.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import run_until_stopped, setup_console, setup_logging
from shared.env import require_env

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_elevenlabs_tools")


async def handle_tool(name: str, arguments: dict) -> str:
    """Handle tool calls from the ElevenLabs agent."""
    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        logger.info("Tool call: get_weather(city=%s)", city)
        return json.dumps({"temperature": 22, "condition": "sunny", "city": city})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def main() -> None:
    env = require_env("ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    config = ElevenLabsRealtimeConfig(
        api_key=env["ELEVENLABS_API_KEY"],
        agent_id=env["ELEVENLABS_AGENT_ID"],
    )
    provider = ElevenLabsRealtimeProvider(config)

    sample_rate = 16000
    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=20,
    )

    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get("SYSTEM_PROMPT"),
        voice=os.environ.get("ELEVENLABS_VOICE_ID"),
        tool_handler=handle_tool,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
    )
    kit.register_channel(channel)

    await kit.create_room(room_id="tool-demo")
    await kit.attach_channel("tool-demo", "voice")

    session = await channel.start_session("tool-demo", "local-user", connection=None)

    logger.info("ElevenLabs session started with tool calling")
    logger.info("Ask about the weather to trigger the tool! Press Ctrl+C to stop.\n")

    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

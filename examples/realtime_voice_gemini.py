"""RoomKit — Realtime voice with Gemini Live API.

A minimal speech-to-speech example using Google Gemini Live.
Audio flows between the user's browser and Gemini; transcriptions
are emitted as RoomEvents so other channels see the conversation.

Requirements:
    pip install roomkit[realtime-gemini] fastapi uvicorn websockets

Run with:
    GOOGLE_API_KEY=... uv run python examples/realtime_voice_gemini.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, run_until_stopped, setup_console, setup_logging

from roomkit import (
    RealtimeVoiceChannel,
    RoomKit,
    WebSocketChannel,
)
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport

logger = setup_logging("realtime_voice_gemini")


async def main() -> None:
    env = require_env("GOOGLE_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- Gemini Live provider ---
    provider = GeminiLiveProvider(
        api_key=env["GOOGLE_API_KEY"],
        model="gemini-3.1-flash-live-preview",
    )
    transport = WebSocketRealtimeTransport()

    # --- Realtime voice channel ---
    realtime = RealtimeVoiceChannel(
        "realtime-voice",
        provider=provider,
        transport=transport,
        system_prompt="You are a friendly voice assistant. Be concise.",
        voice="Aoede",  # Gemini voice preset
    )
    kit.register_channel(realtime)

    # --- Optional: WebSocket channel for supervisor text ---
    ws_supervisor = WebSocketChannel("ws-supervisor")
    kit.register_channel(ws_supervisor)

    # --- Room setup ---
    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "realtime-voice")
    await kit.attach_channel("demo", "ws-supervisor")

    print("Room 'demo' created with realtime-voice and ws-supervisor channels.")
    print("Connect a WebSocket to start a voice session:")
    print("  session = await realtime.start_session('demo', 'user-1', ws)")
    print()
    print("Transcriptions will appear as RoomEvents in the room.")
    print("Messages sent via ws-supervisor will be injected into the voice AI.")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

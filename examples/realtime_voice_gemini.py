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
import os

from roomkit import (
    RealtimeVoiceChannel,
    RoomKit,
    WebSocketChannel,
)
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY to run this example.")
        return

    kit = RoomKit()

    # --- Gemini Live provider ---
    provider = GeminiLiveProvider(
        api_key=api_key,
        model="gemini-2.5-flash-native-audio-preview-12-2025",
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

    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

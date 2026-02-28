"""RoomKit -- Realtime voice with FastRTC (WebRTC) transport.

A speech-to-speech example using Google Gemini Live over WebRTC.
Audio flows between the user's browser and Gemini via FastRTC's
passthrough handler (no VAD -- Gemini handles speech detection).
Transcriptions are emitted as RoomEvents so other channels see
the conversation.

Demonstrates pluggable auth on the transport via ``auth=``.

Requirements:
    pip install roomkit[realtime-gemini,fastrtc] fastapi uvicorn

Run with:
    GOOGLE_API_KEY=... uv run uvicorn examples.realtime_voice_fastrtc:app
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.realtime.fastrtc_transport import (
    FastRTCRealtimeTransport,
    mount_fastrtc_realtime,
)

logging.basicConfig(level=logging.INFO)

kit = RoomKit()

# --- Gemini Live provider ---
provider = GeminiLiveProvider(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model="gemini-2.5-flash-native-audio-preview-12-2025",
)

# --- FastRTC transport (WebRTC, passthrough mode) ---
transport = FastRTCRealtimeTransport(
    input_sample_rate=16000,
    output_sample_rate=24000,
)

# --- Realtime voice channel ---
channel = RealtimeVoiceChannel(
    "realtime-voice",
    provider=provider,
    transport=transport,
    system_prompt="You are a friendly voice assistant. Be concise.",
    voice="Aoede",
)
kit.register_channel(channel)


# --- Auto-create session on WebRTC connect ---
async def on_client_connected(webrtc_id: str) -> None:
    """Called when a browser connects via WebRTC."""
    room = await kit.create_room()
    await kit.attach_channel(room.id, "realtime-voice")
    session = await channel.start_session(room.id, "user-1", connection=webrtc_id)
    logging.info(
        "Session started: session=%s, room=%s, webrtc_id=%s",
        session.id,
        room.id,
        webrtc_id,
    )


transport.on_client_connected(on_client_connected)


# --- Optional: pluggable auth on the transport ---
async def authenticate_connection(ctx: object) -> dict[str, object] | None:
    """Example auth callback for WebRTC connections.

    In production, inspect headers/tokens from the connection context.
    Return a metadata dict on success, or None to reject.
    """
    # Accept all connections in this example
    return {"authenticated": True}


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mount WebRTC endpoints with auth
    mount_fastrtc_realtime(app, transport, path="/rtc-realtime", auth=authenticate_connection)
    yield
    await kit.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return {
        "message": "RoomKit Realtime Voice (FastRTC/WebRTC)",
        "webrtc_endpoint": "/rtc-realtime",
    }

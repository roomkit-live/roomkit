"""Anam AI Avatar — Realtime Audio+Video Channel

Run (ephemeral persona — full control over components):
    export ANAM_API_KEY="your-api-key"
    export ANAM_AVATAR_ID="your-avatar-id"
    export ANAM_VOICE_ID="your-voice-id"
    export ANAM_LLM_ID="your-llm-id"
    python examples/realtime_av_anam.py

Run (pre-defined persona from Anam Lab):
    export ANAM_API_KEY="your-api-key"
    export ANAM_PERSONA_ID="your-persona-id"
    python examples/realtime_av_anam.py

Requires:
    pip install roomkit[anam,websocket]

This example creates a RealtimeAudioVideoChannel backed by Anam AI.
It accepts a WebSocket connection, streams audio to Anam, and receives
synchronized avatar audio+video back.

Configure components at https://lab.anam.ai (avatars, voices, LLMs).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, run_until_stopped, setup_console, setup_logging

from roomkit import RealtimeAudioVideoChannel, RoomKit
from roomkit.providers.anam import AnamConfig, AnamRealtimeProvider
from roomkit.voice.realtime.mock import MockRealtimeTransport

logger = setup_logging("realtime_av_anam")


async def main() -> None:
    env = require_env("ANAM_API_KEY")

    persona_id = os.environ.get("ANAM_PERSONA_ID")
    avatar_id = os.environ.get("ANAM_AVATAR_ID")
    voice_id = os.environ.get("ANAM_VOICE_ID")
    llm_id = os.environ.get("ANAM_LLM_ID")

    if not persona_id and not (avatar_id and voice_id and llm_id):
        logger.error(
            "Set either ANAM_PERSONA_ID (pre-defined persona) or "
            "ANAM_AVATAR_ID + ANAM_VOICE_ID + ANAM_LLM_ID (ephemeral persona)"
        )
        return

    # Configure Anam provider
    config = AnamConfig(
        api_key=env["ANAM_API_KEY"],
        persona_id=persona_id,
        avatar_id=avatar_id,
        voice_id=voice_id,
        llm_id=llm_id,
        system_prompt=("You are a helpful AI avatar. Keep responses conversational and concise."),
    )
    provider = AnamRealtimeProvider(config)

    # For local testing, use a mock transport.
    # In production, replace with a WebSocket or WebRTC transport.
    transport = MockRealtimeTransport()

    # Create the realtime audio+video channel
    channel = RealtimeAudioVideoChannel(
        "avatar-anam",
        provider=provider,
        transport=transport,
    )

    # Log video frames
    frame_count = 0

    def on_video(session: object, frame: object) -> None:
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 1:
            logger.info("Video frame #%d: %s", frame_count, type(frame).__name__)

    channel.add_video_media_tap(on_video)  # type: ignore[arg-type]

    # Set up RoomKit
    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    kit.register_channel(channel)

    room = await kit.create_room("avatar-room")
    await kit.attach_channel(room.id, channel.channel_id)

    logger.info("Room %s ready. Starting avatar session...", room.id)

    # Start a session (in production, the connection would be a real WebSocket)
    session = await channel.start_session(
        room.id,
        participant_id="user-1",
        connection="mock-ws",
    )
    logger.info("Session %s active. Avatar is listening.", session.id)

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)
        logger.info("Done. Received %d video frames.", frame_count)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

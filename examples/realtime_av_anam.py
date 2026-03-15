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
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from roomkit import (
    AnamConfig,
    AnamRealtimeProvider,
    RealtimeAudioVideoChannel,
    RoomKit,
)
from roomkit.voice.realtime.mock import MockRealtimeTransport

logger = logging.getLogger(__name__)


async def main() -> None:
    api_key = os.environ.get("ANAM_API_KEY", "")
    if not api_key:
        logger.error("Set ANAM_API_KEY environment variable")
        return

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
        api_key=api_key,
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

    # Keep running until interrupted
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await channel.end_session(session)
        await kit.close()
        logger.info("Done. Received %d video frames.", frame_count)


if __name__ == "__main__":
    asyncio.run(main())

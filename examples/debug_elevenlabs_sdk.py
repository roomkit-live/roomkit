"""Test AsyncConversation with real mic/speakers — no RoomKit.

Uses the SDK's own AsyncDefaultAudioInterface (pyaudio) so audio
is identical to the working sync version.

Run with:
    ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... \
        python examples/debug_elevenlabs_sdk.py
"""

import asyncio
import logging
import os

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import AsyncConversation
from elevenlabs.conversational_ai.default_audio_interface import AsyncDefaultAudioInterface

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
logging.getLogger("websockets").setLevel(logging.DEBUG)

api_key = os.environ.get("ELEVENLABS_API_KEY", "")
agent_id = os.environ.get("ELEVENLABS_AGENT_ID", "")

if not api_key or not agent_id:
    print("Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID")
    raise SystemExit(1)

try:
    from elevenlabs.version import __version__

    print(f"SDK version: {__version__}")
except Exception:
    print("SDK version: unknown")


async def on_agent_response(text):
    print(f"Agent: {text}")


async def on_user_transcript(text):
    print(f"User: {text}")


async def on_latency(ms):
    print(f"Latency: {ms}ms")


async def main():
    client = ElevenLabs(api_key=api_key)

    conversation = AsyncConversation(
        client=client,
        agent_id=agent_id,
        requires_auth=False,
        audio_interface=AsyncDefaultAudioInterface(),
        callback_agent_response=on_agent_response,
        callback_user_transcript=on_user_transcript,
        callback_latency_measurement=on_latency,
    )

    print("\nStarting AsyncConversation with real mic/speakers...")
    print("Speak into your microphone! Press Ctrl+C to stop.\n")
    await conversation.start_session()

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    await conversation.end_session()
    conv_id = await conversation.wait_for_session_end()
    print(f"\nConversation ended. ID: {conv_id}")


asyncio.run(main())

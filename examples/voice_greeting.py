"""RoomKit -- Voice greeting patterns.

Demonstrates how to greet callers when the voice session audio path is
ready, using the ON_VOICE_SESSION_READY hook and Agent auto_greet.

Three patterns are shown:

1. **Agent auto_greet** — set ``greeting`` on the Agent and it speaks
   automatically when the session is ready.  No extra code needed.

2. **Explicit hook** — register ON_VOICE_SESSION_READY and call
   send_greeting() yourself.  Maximum control.

3. **Manual say()** — use the hook to call voice._send_tts() directly
   for pre-rendered audio or non-agent greetings.

All greetings go directly through TTS — no LLM round-trip, so the
caller hears exactly the text you set with near-zero latency.

This example uses mock providers so it runs without external dependencies.

Run with:
    uv run python examples/voice_greeting.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RoomKit,
    VoiceChannel,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice import AudioPipelineConfig
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.events import VoiceSessionReadyEvent
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_greeting")


async def pattern_agent_auto_greet() -> None:
    """Pattern 1: Agent auto_greet (recommended).

    Just set ``greeting`` on the Agent — it speaks automatically when
    the voice session is ready.  No hooks or extra code needed.
    """
    logger.info("--- Pattern 1: Agent auto_greet ---")

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["hello"])
    tts = MockTTSProvider()

    kit = RoomKit(stt=stt, tts=tts, voice=backend)

    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
    )
    agent = Agent(
        "agent",
        provider=MockAIProvider(responses=["Hi! How can I help you today?"]),
        greeting="Welcome to Acme Support!",
        # auto_greet=True is the default
    )
    kit.register_channel(voice)
    kit.register_channel(agent)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "agent")

    # Greeting fires automatically when the audio path is ready
    session = await kit.connect_voice(room.id, "caller-1", "voice")
    await asyncio.sleep(0.2)

    logger.info("Session %s greeted via Agent auto_greet", session.id)
    await kit.close()


async def pattern_explicit_hook() -> None:
    """Pattern 2: Explicit ON_VOICE_SESSION_READY hook."""
    logger.info("--- Pattern 2: Explicit hook ---")

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["hello"])
    tts = MockTTSProvider()

    kit = RoomKit(stt=stt, tts=tts, voice=backend)

    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
    )
    agent = Agent(
        "agent",
        provider=MockAIProvider(responses=["Hi! How can I help you today?"]),
        greeting="Welcome to Acme Support!",
        auto_greet=False,  # Disable auto — we'll call send_greeting ourselves
    )
    kit.register_channel(voice)
    kit.register_channel(agent)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "agent")

    @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
    async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
        logger.info("Session ready: %s — sending greeting", event.session.id)
        await kit.send_greeting(room.id)

    session = await kit.connect_voice(room.id, "caller-1", "voice")
    await asyncio.sleep(0.2)

    logger.info("Session %s greeted via explicit hook", session.id)
    await kit.close()


async def pattern_manual_tts() -> None:
    """Pattern 3: Manual TTS via the session-ready hook."""
    logger.info("--- Pattern 3: Manual TTS ---")

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["hello"])
    tts = MockTTSProvider()

    kit = RoomKit(stt=stt, tts=tts, voice=backend)

    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
    )
    kit.register_channel(voice)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")

    @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
    async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
        logger.info("Session ready — speaking directly via TTS")
        await voice._send_tts(event.session, "Please hold while we connect you.")

    session = await kit.connect_voice(room.id, "caller-1", "voice")
    await asyncio.sleep(0.2)

    logger.info("Session %s greeted via manual TTS", session.id)
    await kit.close()


async def main() -> None:
    await pattern_agent_auto_greet()
    await pattern_explicit_hook()
    await pattern_manual_tts()
    logger.info("All greeting patterns demonstrated successfully.")


if __name__ == "__main__":
    asyncio.run(main())

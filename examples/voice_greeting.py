"""RoomKit -- Voice greeting patterns.

Demonstrates how to greet callers when the voice session audio path is
ready, using the ON_VOICE_SESSION_READY hook and Agent auto_greet.

Four patterns are shown:

1. **Agent auto_greet** — set ``greeting`` on the Agent and it speaks
   automatically when the session is ready.  No extra code needed.

2. **Explicit hook** — register ON_VOICE_SESSION_READY and call
   send_greeting() yourself.  Maximum control.

3. **Manual say()** — use the hook to call voice.say() directly for
   pre-rendered audio or non-agent greetings.

4. **LLM-generated greeting** — inject a synthetic user message via
   process_inbound() to trigger an LLM round-trip.  The AI generates
   a contextual greeting instead of a static string.

Patterns 1–3 go directly through TTS — no LLM round-trip, so the caller
hears exactly the text you set with near-zero latency.  Pattern 4 adds
an LLM round-trip for dynamic, context-aware greetings.

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
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
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


async def pattern_manual_say() -> None:
    """Pattern 3: Manual say() via the session-ready hook."""
    logger.info("--- Pattern 3: Manual say() ---")

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
        logger.info("Session ready — speaking directly via say()")
        await voice.say(event.session, "Please hold while we connect you.")

    session = await kit.connect_voice(room.id, "caller-1", "voice")
    await asyncio.sleep(0.2)

    logger.info("Session %s greeted via manual say()", session.id)
    await kit.close()


async def pattern_llm_greeting() -> None:
    """Pattern 4: LLM-generated greeting via process_inbound().

    Inject a synthetic user message when the session is ready so the AI
    generates a dynamic, context-aware greeting instead of a static string.
    """
    logger.info("--- Pattern 4: LLM-generated greeting ---")

    backend = MockVoiceBackend()
    stt = MockSTTProvider(transcripts=["hello"])
    tts = MockTTSProvider()

    kit = RoomKit(stt=stt, tts=tts, voice=backend)

    voice = VoiceChannel(
        "voice", stt=stt, tts=tts, backend=backend, pipeline=AudioPipelineConfig()
    )
    agent = Agent(
        "agent",
        provider=MockAIProvider(
            responses=["Welcome! I'm your virtual assistant. How can I help?"]
        ),
        auto_greet=False,  # No static greeting — let the LLM generate one
    )
    kit.register_channel(voice)
    kit.register_channel(agent)

    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.attach_channel(room.id, "agent")

    @kit.hook(HookTrigger.ON_VOICE_SESSION_READY, HookExecution.ASYNC)
    async def on_ready(event: VoiceSessionReadyEvent, context: object) -> None:
        logger.info("Session ready — injecting synthetic message for LLM greeting")
        inbound = InboundMessage(
            channel_id="voice",
            sender_id=event.session.participant_id,
            content=TextContent(body="[session started]"),
            metadata={"voice_session_id": event.session.id, "source": "greeting"},
        )
        await kit.process_inbound(inbound, room_id=room.id)

    session = await kit.connect_voice(room.id, "caller-1", "voice")
    await asyncio.sleep(0.2)

    logger.info("Session %s greeted via LLM-generated response", session.id)
    await kit.close()


async def main() -> None:
    await pattern_agent_auto_greet()
    await pattern_explicit_hook()
    await pattern_manual_say()
    await pattern_llm_greeting()
    logger.info("All greeting patterns demonstrated successfully.")


if __name__ == "__main__":
    asyncio.run(main())

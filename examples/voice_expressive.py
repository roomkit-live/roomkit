"""RoomKit -- ElevenLabs expressive voice assistant.

Demonstrates ElevenLabs v3 Conversational TTS with expressive mode.
The AI responds with emotion tags ([laughs], [whispers], [sighs], etc.)
that ElevenLabs renders with matching tone, timing, and delivery.

Audio flows:

  Mic -> Deepgram STT -> Claude -> ElevenLabs v3 TTS -> Speaker

The system prompt instructs Claude to use expressive tags naturally.
Tags last ~4-5 words before reverting to normal delivery.

Supported tags:
  [laughs]    — laughter
  [whispers]  — soft/quiet speech
  [sighs]     — audible sigh
  [slow]      — slower pacing
  [excited]   — energetic delivery

Requirements:
    pip install roomkit[local-audio,anthropic]

Run with:
    ANTHROPIC_API_KEY=... \\
    DEEPGRAM_API_KEY=... \\
    ELEVENLABS_API_KEY=... \\
    uv run python examples/voice_expressive.py

Environment variables:
    ANTHROPIC_API_KEY    (required) Anthropic API key
    DEEPGRAM_API_KEY     (required) Deepgram API key
    ELEVENLABS_API_KEY   (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID  Voice ID (default: Rachel)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import AIChannel, ChannelCategory, HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = logging.getLogger("roomkit")

SYSTEM_PROMPT = """\
You are a warm, expressive conversational assistant. Use the following
tags naturally to convey emotion — but don't overuse them:

  [laughs]    when something is genuinely funny
  [whispers]  for dramatic effect or secrets
  [sighs]     for sympathy, tiredness, or resignation
  [slow]      for emphasis or gravitas
  [excited]   for enthusiasm and energy

Example: "[excited] Oh, that's such a great idea! [laughs] I love it."

Keep responses short and conversational. Use tags only when they fit
the emotional context — not every sentence needs one.
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    kit = RoomKit()

    # --- AI (Claude) ----------------------------------------------------------
    ai = AIChannel(
        "ai",
        provider=AnthropicAIProvider(
            AnthropicConfig(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                model="claude-sonnet-4-20250514",
            )
        ),
        system_prompt=SYSTEM_PROMPT,
    )
    kit.register_channel(ai)

    # --- TTS (ElevenLabs expressive) ------------------------------------------
    tts = ElevenLabsTTSProvider(
        ElevenLabsConfig(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            output_format="pcm_24000",
            expressive=True,  # <-- enables v3 Conversational + expressive tags
        )
    )

    # --- STT (Deepgram) -------------------------------------------------------
    from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider

    stt = DeepgramSTTProvider(
        DeepgramConfig(api_key=os.environ["DEEPGRAM_API_KEY"], model="nova-3")
    )

    # --- Voice channel --------------------------------------------------------
    # No local VAD needed — Deepgram handles endpointing natively.
    from roomkit.voice import get_local_audio_backend

    LocalAudioBackend = get_local_audio_backend()
    backend = LocalAudioBackend(input_sample_rate=16000, output_sample_rate=24000)

    voice = VoiceChannel(
        "voice",
        tts=tts,
        stt=stt,
        backend=backend,
        # NOTE: do NOT use tts_filter=StripBrackets() with expressive mode
        # — it would strip the [laughs], [whispers], etc. tags.
    )
    kit.register_channel(voice)

    # --- Logging hook ---------------------------------------------------------
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def log_events(event, ctx):
        body = getattr(event.content, "body", "") or ""
        logger.info("[%s] %s", event.source.channel_id, body[:80])

    # --- Room -----------------------------------------------------------------
    await kit.create_room(room_id="expressive-demo")
    await kit.attach_channel("expressive-demo", "ai", category=ChannelCategory.INTELLIGENCE)
    await kit.attach_channel("expressive-demo", "voice")

    logger.info("Expressive voice assistant ready — speak into your microphone!")
    logger.info("ElevenLabs model: %s", tts._config.model_id)

    # Wait for Ctrl+C
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    logger.info("Stopping...")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

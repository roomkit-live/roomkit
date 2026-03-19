"""RoomKit -- ElevenLabs expressive voice assistant.

Demonstrates ElevenLabs v3 Conversational TTS with expressive mode.
The AI responds with emotion tags ([laughs], [whispers], [sighs], etc.)
that ElevenLabs renders with matching tone, timing, and delivery.

Audio flows:

  Mic -> [Denoiser] -> VAD -> STT -> Claude -> ElevenLabs v3 TTS -> Speaker

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
    ELEVENLABS_API_KEY=... \\
    VAD_MODEL=path/to/ten-vad.onnx \\
    uv run python examples/voice_expressive.py

Environment variables:
    ANTHROPIC_API_KEY    (required) Anthropic API key
    ELEVENLABS_API_KEY   (required) ElevenLabs API key
    ELEVENLABS_VOICE_ID  Voice ID (default: Rachel)
    VAD_MODEL            Path to sherpa-onnx VAD .onnx model
    VAD_MODEL_TYPE       Model type: ten | silero (default: ten)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from roomkit import AIChannel, HookExecution, HookTrigger, Room, RoomKit, VoiceChannel
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

    # --- STT (Deepgram or sherpa-onnx) ----------------------------------------
    from roomkit.voice.stt import get_sherpa_stt_provider

    SherpaSTTProvider = get_sherpa_stt_provider()
    stt = SherpaSTTProvider()

    # --- VAD ------------------------------------------------------------------
    from roomkit.voice.pipeline.vad import SherpaVADConfig, SherpaVADProvider

    vad_config = SherpaVADConfig(
        model_path=os.environ["VAD_MODEL"],
        model_type=os.environ.get("VAD_MODEL_TYPE", "ten"),
        threshold=float(os.environ.get("VAD_THRESHOLD", "0.35")),
    )
    vad = SherpaVADProvider(vad_config)

    # --- Voice channel --------------------------------------------------------
    from roomkit.voice import get_local_audio_backend
    from roomkit.voice.pipeline import AudioPipelineConfig

    LocalAudioBackend = get_local_audio_backend()
    backend = LocalAudioBackend(sample_rate=24000)

    voice = VoiceChannel(
        "voice",
        tts=tts,
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
        # NOTE: do NOT use tts_filter=StripBrackets() with expressive mode
        # — it would strip the [laughs], [whispers], etc. tags.
    )
    kit.register_channel(voice)

    # --- Logging hook ---------------------------------------------------------
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def log_events(event, ctx):
        logger.info(
            "[%s] %s: %s", event.channel_id, event.sender_id, event.text[:80] if event.text else ""
        )

    # --- Room -----------------------------------------------------------------
    room = Room(room_id="expressive-demo")
    room = await kit.store.create_room(room)
    await kit.join(room.room_id, "voice")
    await kit.join(room.room_id, "ai")

    logger.info("Expressive voice assistant ready — speak into your microphone!")
    logger.info("ElevenLabs model: %s", tts._config.model_id)

    # Wait for Ctrl+C
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    await kit.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

"""RoomKit — Speech-to-speech with Deepgram Voice Agent using local mic/speakers.

Talk to a Deepgram agent using your system microphone — the agent's audio plays
through your speakers.  Deepgram handles turn detection and barge-in server-side.
WebRTC AEC strips speaker echo so the mic stays open during playback.

Unlike single-model speech-to-speech APIs, a Deepgram agent is assembled from
three stages you pick independently: ``listen`` (Nova/Flux transcription),
``think`` (the LLM) and ``speak`` (an Aura voice).  Each is a separate env var
below, and each can be swapped mid-conversation.

Requirements:
    pip install roomkit websockets sounddevice numpy aec-audio-processing

    `aec-audio-processing` is not optional on open speakers: without it the mic is
    muted while the agent talks, so you cannot interrupt it. With it, barge-in works.

Run with:
    DEEPGRAM_API_KEY=... uv run python examples/realtime_voice_local_deepgram.py

In French — the three stages are set independently, and the prompt matters as much
as the voice: an Aura French voice reading an English answer is still English.

    DEEPGRAM_API_KEY=... DEEPGRAM_LANGUAGE=fr-CA DEEPGRAM_VOICE=aura-2-agathe-fr \
    DEEPGRAM_GREETING="Bonjour ! Comment puis-je vous aider ?" \
    SYSTEM_PROMPT="Tu es un assistant vocal. Réponds toujours en français, brièvement." \
    uv run python examples/realtime_voice_local_deepgram.py

Environment variables:
    DEEPGRAM_API_KEY      (required) Deepgram API key
    DEEPGRAM_VOICE        Aura voice (default: aura-2-thalia-en)
    DEEPGRAM_LANGUAGE     Transcription language, e.g. fr-CA (default: the model's own)
    DEEPGRAM_LISTEN_MODEL Speech-to-text model (default: nova-3)
    DEEPGRAM_THINK_MODEL  LLM model (default: gpt-4o-mini)
    DEEPGRAM_GREETING     Line the agent speaks first (default: a short hello)
    SYSTEM_PROMPT         Custom system prompt
    AEC                   webrtc (default) | speex | 0 to disable
    MUTE_MIC              0 to keep the mic open during playback, 1 to force muting
                          (default: muted only when AEC is unavailable)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    build_aec,
    build_pipeline,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.deepgram import DeepgramAgentConfig, DeepgramAgentProvider
from roomkit.voice.backends.local import LocalAudioBackend

logger = setup_logging("realtime_voice_local_deepgram")


async def main() -> None:
    env = require_env("DEEPGRAM_API_KEY")

    kit = RoomKit()

    # --- Console dashboard (set CONSOLE=1 to enable) ---
    console_cleanup = setup_console(kit)

    # --- Deepgram Voice Agent provider (speech-to-speech) ---
    # The three stages are configured independently — that is the point of this
    # provider. Swap the LLM without touching the voice, or vice versa.
    config = DeepgramAgentConfig(
        api_key=env["DEEPGRAM_API_KEY"],
        listen_model=os.environ.get("DEEPGRAM_LISTEN_MODEL", "nova-3"),
        listen_language=os.environ.get("DEEPGRAM_LANGUAGE"),
        think_model=os.environ.get("DEEPGRAM_THINK_MODEL", "gpt-4o-mini"),
        speak_model=os.environ.get("DEEPGRAM_VOICE", "aura-2-thalia-en"),
        greeting=os.environ.get("DEEPGRAM_GREETING", "Hi! What can I do for you?"),
    )
    provider = DeepgramAgentProvider(config)

    # Browse the catalog offline — no API key or network needed:
    #   for voice in DeepgramAgentProvider.available_voices():
    #       print(voice.id, voice.language, voice.description)

    # --- Audio pipeline (AEC + noise suppression) ---
    sample_rate = 24000
    block_ms = 20

    aec = build_aec(sample_rate, block_ms, default="webrtc")
    pipeline = build_pipeline(aec=aec)

    # When AEC is active it removes speaker echo from the mic signal, so the mic
    # can stay open during playback. Without AEC the mic is muted while the agent
    # speaks: Deepgram's turn detection cannot tell the agent's own voice from the
    # caller's, so an open mic transcribes the agent back as user speech, fires
    # UserStartedSpeaking, and the barge-in loop never stops.
    # MUTE_MIC=0|1 overrides the auto behaviour (same as the OpenAI example).
    mute_env = os.environ.get("MUTE_MIC")
    mute_mic = mute_env != "0" if mute_env is not None else aec is None

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=block_ms,
        mute_mic_during_playback=mute_mic,
        aec=aec,
    )

    # --- Realtime voice channel ---
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=transport,
        system_prompt=os.environ.get(
            "SYSTEM_PROMPT",
            "You are a friendly voice assistant. Be concise and helpful.",
        ),
        voice=config.speak_model,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(channel)

    # --- Room ---
    await kit.create_room(room_id="local-demo")
    await kit.attach_channel("local-demo", "voice")

    # --- Start session ---
    session = await channel.start_session(
        "local-demo",
        "local-user",
        connection=None,
    )

    logger.info(
        "Deepgram Voice Agent session started (listen=%s, think=%s, speak=%s)",
        config.listen_model,
        config.think_model,
        config.speak_model,
    )
    logger.info("Speak into your microphone! Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C ---
    async def cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

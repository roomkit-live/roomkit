"""RoomKit -- a voice agent on Microsoft's MAI models: talk to it through your microphone.

    MAI-Transcribe-2-Streaming  hears you  (AzureMAISTTProvider)
    Claude Haiku 5.5            answers    (AIChannel)
    MAI-Voice-2.1-Flash         speaks     (AzureSpeechTTSProvider)

MAI-Transcribe detects no turns: a pipeline VAD finds where you stop, and
the provider commits the utterance then; the final transcript comes about
0.13 s later. So this example always runs a VAD (``VAD=0`` is refused).

One Foundry resource in a region serving both models (``swedencentral``)
answers both providers with the same key. MAI-Transcribe-2-Streaming must be
deployed on it (Foundry portal: Build, Models, Deploy a base model).

Requirements:
    pip install roomkit[azure-speech,anthropic,local-audio] aec-audio-processing

Environment variables:
    ANTHROPIC_API_KEY     (required) Anthropic API key
    AZURE_SPEECH_KEY      (required) the Foundry resource's key
    AZURE_SPEECH_REGION   (required) the resource's region, e.g. swedencentral
    AZURE_MAI_ENDPOINT    (required) https://<resource>.services.ai.azure.com
    AZURE_MAI_DEPLOYMENT  The transcription deployment
                          (default: MAI-Transcribe-2-Streaming)

    --- Voice (optional) ---
    VOICE_LANGUAGE        Language to transcribe, e.g. fr-CA (default: detected
                          utterance by utterance)
    VOICE                 Voice name (default: en-US-Harper:MAI-Voice-2.1-Flash;
                          fr-FR-Soleil:MAI-Voice-2.1-Flash speaks French)
    STYLE                 A style the voice supports, e.g. customer_call_center

    --- Audio (optional) ---
    VAD                   energy | silero | ten (default: energy)
    AEC                   webrtc | speex | 0 (default: webrtc)

Run with:
    ANTHROPIC_API_KEY=... AZURE_SPEECH_KEY=... AZURE_SPEECH_REGION=swedencentral \\
    AZURE_MAI_ENDPOINT=https://<resource>.services.ai.azure.com \\
        uv run --extra azure-speech --extra anthropic --extra local-audio \\
        python examples/voice_azure_mai_agent.py

Use headphones, or keep AEC on: without echo cancellation the agent hears
itself. Press Ctrl+C to stop.
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
    build_vad,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
    voice_language,
)

from roomkit import ChannelCategory, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.stt.azure_mai import (
    DEFAULT_DEPLOYMENT,
    AzureMAISTTConfig,
    AzureMAISTTProvider,
)
from roomkit.voice.tts.azure_speech import (
    DEFAULT_VOICE,
    AzureSpeechTTSConfig,
    AzureSpeechTTSProvider,
)

logger = setup_logging("voice_azure_mai_agent")

SAMPLE_RATE = 16000
"""Microphone rate: one of the two MAI-Transcribe takes as is."""

OUTPUT_RATE = 24000


async def main() -> None:
    env = require_env(
        "ANTHROPIC_API_KEY", "AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_MAI_ENDPOINT"
    )

    # --- Audio: local mic + speakers, AEC so the agent does not hear itself ----
    aec = build_aec(SAMPLE_RATE)
    backend = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=OUTPUT_RATE,
        channels=1,
        block_duration_ms=20,
        aec=aec,
        mute_mic_during_playback=aec is None,
    )
    vad = build_vad(SAMPLE_RATE)
    if vad is None:
        sys.exit("MAI-Transcribe detects no turns: run with a VAD (VAD=energy|silero|ten).")
    # No aec= on the pipeline: the backend feeds the echo reference itself.
    pipeline = build_pipeline(vad=vad)

    # --- MAI-Transcribe: one stream per utterance, committed when it ends ------
    language = voice_language(None)
    stt = AzureMAISTTProvider(
        AzureMAISTTConfig(
            endpoint=env["AZURE_MAI_ENDPOINT"],
            api_key=env["AZURE_SPEECH_KEY"],
            deployment=os.environ.get("AZURE_MAI_DEPLOYMENT", DEFAULT_DEPLOYMENT),
            language=language,
        )
    )

    # --- MAI-Voice: rendered through Azure Speech, streamed as PCM -------------
    tts = AzureSpeechTTSProvider(
        AzureSpeechTTSConfig(
            api_key=env["AZURE_SPEECH_KEY"],
            region=env["AZURE_SPEECH_REGION"],
            voice=os.environ.get("VOICE", DEFAULT_VOICE),
            style=os.environ.get("STYLE") or None,
            sample_rate=OUTPUT_RATE,
        )
    )
    logger.info("Hearing in %s, speaking with %s", language or "any language", tts.default_voice)

    # --- Channels ---------------------------------------------------------------
    voice = VoiceChannel("voice", stt=stt, tts=tts, backend=backend, pipeline=pipeline)
    ai = AIChannel(
        "ai",
        provider=AnthropicAIProvider(
            AnthropicConfig(
                api_key=env["ANTHROPIC_API_KEY"],
                model="claude-haiku-5-5",
                max_tokens=300,
            )
        ),
        system_prompt=(
            "You are a friendly voice assistant. Answer in the language the user "
            "spoke, in one or two short sentences, without markdown or lists."
        ),
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)
    kit.register_channel(voice)
    kit.register_channel(ai)
    await kit.create_room(room_id="mai-demo")
    await kit.attach_channel("mai-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        logger.info("You said: %s", event.text)
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("mai-demo", "voice")

    logger.info("")
    logger.info("Speak: MAI-Transcribe hears you, Claude Haiku 5.5 answers, MAI-Voice speaks.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())

"""RoomKit -- Deepgram STT that detects the language, then locks on it.

Deepgram Nova-3 transcribes code-switched speech with ``language=multi``,
but a stream pinned to the speaker's language (``fr-CA``) is measurably
better. Deepgram fixes the language when the WebSocket opens, and RoomKit
opens one per utterance -- so the channel starts every session in ``multi``,
reads the language Deepgram reports, and pins the next utterance to it:

  utterance 1  Deepgram (multi)  -> "bonjour"  reported as fr  -> lock fr-CA
  utterance 2  Deepgram (fr-CA)  -> "je voudrais..."
  ...          Deepgram (fr-CA)  -> two utterances that stop fitting -> multi

``STTLanguageLock`` runs that loop. The ON_TRANSCRIPTION hook below shows
the signal it works from (``event.language``) and where it left the session
(``voice.get_stt_language``); a policy of your own would read the first and
call ``voice.set_stt_language`` from the same hook.

Requirements:
    pip install roomkit[local-audio,anthropic,deepgram,elevenlabs] aec-audio-processing

Environment variables:
    ANTHROPIC_API_KEY   (required) Anthropic API key
    DEEPGRAM_API_KEY    (required) Deepgram API key
    ELEVENLABS_API_KEY  (required) ElevenLabs API key (multilingual voice)
    ELEVENLABS_VOICE_ID Voice ID (default: Rachel)

    --- Language (optional) ---
    DETECT_LANGUAGE     Language every session starts in (default: multi)
    PREFER              Reported -> locked codes, "fr=fr-CA,en=en-US"
                        (default: fr=fr-CA)
    LOCK_AFTER          Agreeing utterances before locking (default: 1)
    RELEASE_AFTER       Misses before going back to detecting (default: 2)

    --- Audio (optional) ---
    VAD                 energy | silero | ten | 0 (default: energy). With 0,
                        Deepgram endpoints server-side: one stream per turn.
    AEC                 webrtc | speex | 0 (default: webrtc)

Run with:
    ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... ELEVENLABS_API_KEY=... \\
        uv run python examples/voice_deepgram_language_lock.py

Speak French, then English: the log shows the language Deepgram reported,
the language the next stream opens with, and when the lock lets go.

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
    build_vad,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    RoomKit,
    STTLanguageLock,
    VoiceChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.deepgram import DeepgramConfig, DeepgramSTTProvider
from roomkit.voice.tts.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider

logger = setup_logging("voice_deepgram_language_lock")


def _parse_prefer(spec: str) -> dict[str, str]:
    """``"fr=fr-CA,en=en-US"`` -> ``{"fr": "fr-CA", "en": "en-US"}``."""
    prefer: dict[str, str] = {}
    for pair in spec.split(","):
        reported, sep, locked = pair.partition("=")
        if sep and reported.strip() and locked.strip():
            prefer[reported.strip()] = locked.strip()
    return prefer


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")

    sample_rate = 16000
    output_rate = 24000  # ElevenLabs native rate

    # --- Audio: local mic + speakers, AEC so the bot does not hear itself -----
    aec = build_aec(sample_rate)
    backend = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=output_rate,
        channels=1,
        block_duration_ms=20,
        aec=aec,
        mute_mic_during_playback=aec is None,
    )
    vad = build_vad(sample_rate)
    pipeline = build_pipeline(aec=aec, vad=vad) or AudioPipelineConfig()
    logger.info(
        "STT mode: %s",
        "VAD, one stream per utterance" if vad else "continuous, one stream per turn",
    )

    # --- Deepgram: Nova-3 in ``multi``; the lock pins sessions from there -----
    detect_language = os.environ.get("DETECT_LANGUAGE", "multi")
    stt = DeepgramSTTProvider(
        DeepgramConfig(
            api_key=env["DEEPGRAM_API_KEY"],
            model="nova-3",
            language=detect_language,
            smart_format=True,
            # Deepgram recommends 100 ms endpointing for code-switching
            endpointing=100 if detect_language == "multi" else 300,
        )
    )

    # --- The policy: detect, lock, release -------------------------------------
    lock = STTLanguageLock(
        detect_language=detect_language,
        prefer=_parse_prefer(os.environ.get("PREFER", "fr=fr-CA")),
        lock_after=int(os.environ.get("LOCK_AFTER", "1")),
        release_after=int(os.environ.get("RELEASE_AFTER", "2")),
    )
    logger.info(
        "Language lock: start in %s, prefer %s, lock after %d, release after %d",
        lock.detect_language,
        lock.prefer,
        lock.lock_after,
        lock.release_after,
    )

    # --- TTS: a multilingual voice follows the answer's language --------------
    tts = ElevenLabsTTSProvider(
        ElevenLabsConfig(
            api_key=env["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model_id="eleven_multilingual_v2",
            output_format=f"pcm_{output_rate}",
        )
    )

    # --- Channels ---------------------------------------------------------------
    voice = VoiceChannel(
        "voice",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=pipeline,
        stt_language_lock=lock,
    )
    ai = AIChannel(
        "ai",
        provider=AnthropicAIProvider(
            AnthropicConfig(
                api_key=env["ANTHROPIC_API_KEY"],
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
            )
        ),
        system_prompt=(
            "You are a friendly voice assistant. Answer in the language the user "
            "spoke, in one or two short sentences."
        ),
    )

    kit = RoomKit()
    console_cleanup = setup_console(kit)
    kit.register_channel(voice)
    kit.register_channel(ai)
    await kit.create_room(room_id="voice-demo")
    await kit.attach_channel("voice-demo", "ai", category=ChannelCategory.INTELLIGENCE)

    # --- Hooks: the signal and the outcome -------------------------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event, ctx):
        # ``event.language`` is what Deepgram reported: a code while the
        # stream detects, nothing once it is pinned. ``get_stt_language`` is
        # where the lock left the session for its next utterance. A policy
        # of your own would read the first and call
        # ``voice.set_stt_language(event.session, ...)`` right here.
        logger.info(
            "You said: %s  (reported=%s, next stream=%s)",
            event.text,
            event.language or "-",
            voice.get_stt_language(event.session),
        )
        return HookResult.allow()

    @kit.hook(HookTrigger.BEFORE_TTS)
    async def before_tts(text, ctx):
        logger.info("Assistant: %s", text)
        return HookResult.allow()

    # --- Attach voice channel (auto-starts session) ---------------------------
    await kit.attach_channel("voice-demo", "voice")

    logger.info("")
    logger.info("Speak French, then English — watch the language the next stream opens with.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("")

    await run_until_stopped(kit, cleanup=console_cleanup)


if __name__ == "__main__":
    asyncio.run(main())

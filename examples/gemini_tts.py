"""Google Gemini text-to-speech example.

Synthesizes speech three ways and writes each result to a playable WAV file:

1. ``synthesize()`` — one request, whole clip.
2. ``synthesize_stream()`` — audio deltas measured as they arrive, so you can
   see the real time-to-first-audio.
3. ``style_prompt`` — a natural-language delivery direction. Gemini TTS is a
   generative model, so the instruction is intended to steer the voice rather
   than be spoken (the 3.1 preview model can still occasionally read it).

Gemini TTS trades latency for expressiveness: expect seconds, not
milliseconds, before the first audio byte. That makes it a good fit for
prompts, announcements and generated audio messages, and a poor one for live
turn-taking — use ElevenLabs/Gradium, or Gemini Live's speech-to-speech path,
for conversation.

Requires:
    pip install roomkit[gemini]

Environment variables:
    GEMINI_API_KEY — Google Gemini API key

Run with:
    uv run python examples/gemini_tts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import base64
import tempfile
import time

from shared import require_env

from roomkit.voice.tts.audio_utils import wrap_wav
from roomkit.voice.tts.gemini import (
    GEMINI_TTS_MODELS,
    GeminiTTSConfig,
    GeminiTTSProvider,
)

OUT_DIR = Path(tempfile.mkdtemp(prefix="roomkit_gemini_tts_"))
TEXT = "Bonjour, vous êtes bien chez RoomKit. Comment puis-je vous aider aujourd'hui?"


def _write_wav(name: str, wav_bytes: bytes) -> Path:
    path = OUT_DIR / name
    path.write_bytes(wav_bytes)
    return path


async def main() -> None:
    env = require_env("GEMINI_API_KEY")

    print(f"Available TTS models : {', '.join(GEMINI_TTS_MODELS)}")
    voices = GeminiTTSProvider.available_voices()
    print(
        f"Prebuilt voices      : {len(voices)} (e.g. " + ", ".join(v.id for v in voices[:5]) + ")"
    )

    provider = GeminiTTSProvider(
        GeminiTTSConfig(
            api_key=env["GEMINI_API_KEY"],
            voice="Kore",
            language="fr-CA",
        )
    )

    # ── One-shot synthesis ───────────────────────────────────────────
    print("\n=== synthesize() ===")
    t0 = time.monotonic()
    result = await provider.synthesize(TEXT)
    print(f"  latency    : {(time.monotonic() - t0):.1f}s")
    print(f"  mime_type  : {result.mime_type}")
    print(f"  duration   : {result.duration_seconds:.2f}s")
    wav = base64.b64decode(result.url.split(",", 1)[1])
    print(f"  wrote      : {_write_wav('oneshot.wav', wav)}")

    # ── Streaming synthesis ──────────────────────────────────────────
    print("\n=== synthesize_stream() ===")
    t0 = time.monotonic()
    ttfa: float | None = None
    frames: list[bytes] = []
    sample_rate = 24000
    async for chunk in provider.synthesize_stream(TEXT, voice="Sulafat"):
        if chunk.is_final:
            continue
        if ttfa is None:
            ttfa = time.monotonic() - t0
        sample_rate = chunk.sample_rate
        frames.append(chunk.data)
    pcm = b"".join(frames)
    print(f"  first audio: {ttfa:.1f}s" if ttfa else "  no audio received")
    print(f"  complete   : {(time.monotonic() - t0):.1f}s")
    print(
        f"  chunks     : {len(frames)}  "
        f"audio: {len(pcm) / 2 / sample_rate:.2f}s @ {sample_rate} Hz"
    )
    print(f"  wrote      : {_write_wav('streamed.wav', wrap_wav(pcm, sample_rate))}")

    # ── Style direction ─────────────────────────────────────────────
    print("\n=== style_prompt ===")
    styled = GeminiTTSProvider(
        GeminiTTSConfig(
            api_key=env["GEMINI_API_KEY"],
            voice="Puck",
            style_prompt="Lis ce texte d'une voix très joyeuse et enthousiaste",
        )
    )
    result = await styled.synthesize(TEXT)
    wav = base64.b64decode(result.url.split(",", 1)[1])
    print(f"  duration   : {result.duration_seconds:.2f}s")
    print(f"  wrote      : {_write_wav('styled.wav', wav)}")
    print("               (the instruction is intended to steer delivery, not be spoken)")
    await styled.close()

    await provider.close()
    print("\nDone — play the files in", OUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())

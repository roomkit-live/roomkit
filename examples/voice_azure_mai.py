"""Microsoft MAI voices both ways: MAI-Voice speaks a sentence, MAI-Transcribe hears it.

``AzureSpeechTTSProvider`` renders a sentence with a MAI-Voice-2.1-Flash voice,
16 kHz PCM streamed as it renders. That audio is then played to
``AzureMAISTTProvider`` in real time, 20 ms at a time, the way a VoiceChannel
behind a VAD streams one utterance: partials print while it "speaks", and the
final comes once the utterance ends and the provider commits it.

The run logs the two numbers a voice agent lives on: the TTS's first audio
after the request, and the STT's final after the end of speech.

One Foundry resource in a region serving both models (``swedencentral``)
answers both providers with the same key. MAI-Transcribe-2-Streaming must be
deployed on it (Foundry portal: Build, Models, Deploy a base model).

Requires:
    pip install roomkit[azure-speech]

Environment variables:
    AZURE_SPEECH_KEY      (required) the resource's key
    AZURE_SPEECH_REGION   (required) the resource's region, e.g. swedencentral
    AZURE_MAI_ENDPOINT    (required) https://<resource>.services.ai.azure.com
    AZURE_MAI_DEPLOYMENT  The transcription deployment (default:
                          MAI-Transcribe-2-Streaming)
    VOICE                 Voice name (default: en-US-Harper:MAI-Voice-2.1-Flash)
    TEXT                  What to say (default: an English sentence)

Run with:
    uv run --extra azure-speech python examples/voice_azure_mai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os
import time
from collections.abc import AsyncIterator

from shared import require_env, setup_logging

from roomkit.voice.base import AudioChunk
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

logger = setup_logging("voice_azure_mai")

SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE * 2 // 50  # 20 ms of mono 16-bit PCM

TEXT = "Hello! This sentence was spoken by MAI Voice and heard back by MAI Transcribe."


async def speak(tts: AzureSpeechTTSProvider, text: str) -> bytes:
    """Render *text*, logging when its first audio arrived."""
    started = time.perf_counter()
    first_audio: float | None = None
    pcm = bytearray()
    async for chunk in tts.synthesize_stream(text):
        if chunk.data and first_audio is None:
            first_audio = time.perf_counter() - started
        pcm += chunk.data
    logger.info(
        "TTS: %.1f s of audio, first audio after %.0f ms, whole after %.0f ms",
        len(pcm) / 2 / SAMPLE_RATE,
        (first_audio or 0) * 1000,
        (time.perf_counter() - started) * 1000,
    )
    return bytes(pcm)


async def speaker(pcm: bytes, ended: list[float]) -> AsyncIterator[AudioChunk]:
    """Play *pcm* in real time, then record when the speech ended."""
    for start in range(0, len(pcm), FRAME_BYTES):
        yield AudioChunk(data=pcm[start : start + FRAME_BYTES], sample_rate=SAMPLE_RATE)
        await asyncio.sleep(0.02)
    ended.append(time.perf_counter())


async def hear(stt: AzureMAISTTProvider, pcm: bytes) -> str:
    """Stream *pcm* as one utterance and return the final transcript."""
    ended: list[float] = []
    final = ""
    async for result in stt.transcribe_stream(speaker(pcm, ended)):
        if not result.is_final:
            logger.info("STT partial: %s", result.text)
            continue
        final = result.text
        logger.info(
            "STT final %.0f ms after the end of speech: %s",
            (time.perf_counter() - ended[0]) * 1000,
            final,
        )
    return final


async def main() -> None:
    env = require_env("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_MAI_ENDPOINT")
    tts = AzureSpeechTTSProvider(
        AzureSpeechTTSConfig(
            api_key=env["AZURE_SPEECH_KEY"],
            region=env["AZURE_SPEECH_REGION"],
            voice=os.environ.get("VOICE", DEFAULT_VOICE),
            sample_rate=SAMPLE_RATE,
        )
    )
    stt = AzureMAISTTProvider(
        AzureMAISTTConfig(
            endpoint=env["AZURE_MAI_ENDPOINT"],
            api_key=env["AZURE_SPEECH_KEY"],
            deployment=os.environ.get("AZURE_MAI_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        )
    )
    text = os.environ.get("TEXT", TEXT)
    try:
        pcm = await speak(tts, text)
        heard = await hear(stt, pcm)
        logger.info("Said:  %s", text)
        logger.info("Heard: %s", heard)
    finally:
        await tts.close()
        await stt.close()


if __name__ == "__main__":
    asyncio.run(main())

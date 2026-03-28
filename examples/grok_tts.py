"""Grok (xAI) text-to-speech example.

Demonstrates using the xAI Grok TTS provider for voice synthesis with
both REST and WebSocket streaming modes.

Requires:
    pip install roomkit httpx websockets

Environment variables:
    XAI_API_KEY — xAI API key

Run with:
    uv run python examples/grok_tts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env

from roomkit.voice.tts.grok import GrokTTSConfig, GrokTTSProvider


async def main() -> None:
    env = require_env("XAI_API_KEY")
    api_key = env["XAI_API_KEY"]

    provider = GrokTTSProvider(
        GrokTTSConfig(
            api_key=api_key,
            voice_id="eve",
            language="en",
            codec="pcm",
            sample_rate=24000,
        )
    )

    # ── REST synthesis ───────────────────────────────────────────────
    print("=== REST synthesis ===")
    result = await provider.synthesize("Hello from Grok TTS!")
    print(f"  mime_type  : {result.mime_type}")
    print(f"  transcript : {result.transcript}")
    print(f"  duration   : {result.duration_seconds:.2f}s" if result.duration_seconds else "")
    print(f"  data URL   : {result.url[:60]}...")

    # ── HTTP streaming ───────────────────────────────────────────────
    print("\n=== HTTP stream ===")
    total_bytes = 0
    chunk_count = 0
    async for chunk in provider.synthesize_stream("Streaming audio from Grok."):
        if not chunk.is_final:
            total_bytes += len(chunk.data)
            chunk_count += 1
    print(f"  chunks: {chunk_count}, total bytes: {total_bytes}")

    # ── WebSocket streaming (text deltas → audio deltas) ─────────────
    print("\n=== WebSocket stream input ===")

    async def text_chunks():
        for word in ["Real-time ", "streaming ", "text to speech ", "with Grok."]:
            yield word

    ws_bytes = 0
    ws_chunks = 0
    async for chunk in provider.synthesize_stream_input(text_chunks()):
        if not chunk.is_final:
            ws_bytes += len(chunk.data)
            ws_chunks += 1
    print(f"  chunks: {ws_chunks}, total bytes: {ws_bytes}")

    await provider.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

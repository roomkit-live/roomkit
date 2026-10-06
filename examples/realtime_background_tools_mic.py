"""RoomKit - hear a voice agent keep talking while a tool runs.

    GEMINI_API_KEY=... uv run --extra realtime-gemini --extra local-audio \
        python examples/realtime_background_tools_mic.py

Ask out loud: "how many blue widgets do we have in stock?"

The lookup takes six seconds on purpose. Through Gemini 3.1 that was six
seconds of silence on the line, which on a phone call reads as a dropped
connection. From 3.8 the tool runs in the background and the model keeps
talking: you hear it say it is checking, and carry on, while the terminal shows
the call still outstanding.

The terminal marks every assistant turn that happens while the tool is
running, so the thing you are listening for is also written down:

    [tool running] > Let me check the stock levels for you.
    [tool running] > Still looking, it is a large warehouse.
                   > We have 42 units in Lyon.

``examples/realtime_background_tools.py`` proves the same behaviour without a
microphone, for CI and for a machine with no audio device.

Requires:
    pip install roomkit[realtime-gemini,local-audio]

Environment variables:
    GEMINI_API_KEY         (required) Gemini API key
    GEMINI_MODEL           default: gemini-3.8-live-extended-thinking
    GEMINI_THINKING_LEVEL  low | medium | high (default: low). Required by the
                           extended-thinking model, which will not choose one.
    GEMINI_VOICE           default: Aoede
    TOOL_SECONDS           how long the fake lookup takes (default: 6)
    AEC                    webrtc (default, needs roomkit[webrtc-aec]) | speex | 0;
                           without AEC the mic is muted while the model speaks
    AEC_DELAY_MS           measured speaker-to-mic delay for WebRTC AEC (default: auto)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os
import time
from typing import Any

from shared import build_aec, build_pipeline, require_env, run_until_stopped, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.base import VoiceSession

logger = setup_logging("roomkit.examples.realtime_background_tools_mic")

SAMPLE_RATE = 24000
BLOCK_MS = 20
TOOL_SECONDS = float(os.environ.get("TOOL_SECONDS", "6"))

INVENTORY = {"blue widget": 42, "red widget": 7}

SYSTEM_PROMPT = (
    "You are a warehouse assistant on a phone call. When you need the stock "
    "lookup tool, say out loud that you are checking, then keep the caller "
    "company while it runs: never leave more than a couple of seconds of "
    "silence. Give the number as soon as you have it."
)


def inventory_tool() -> dict[str, Any]:
    return {
        "name": "check_inventory",
        "description": (
            "Look up warehouse stock for one item. Slow: takes several seconds. "
            "Tell the caller you are checking, then keep talking while it runs."
        ),
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    }


async def main() -> None:
    env = require_env("GEMINI_API_KEY")

    provider = GeminiLiveProvider(
        api_key=env["GEMINI_API_KEY"],
        model=os.environ.get("GEMINI_MODEL", "gemini-3.8-live-extended-thinking"),
    )

    # AEC only: the model's own server VAD handles barge-in on cleaned audio,
    # and without echo cancellation it would hear itself through the speakers.
    aec = build_aec(SAMPLE_RATE, BLOCK_MS, default="webrtc", enable_ns=False)
    pipeline = build_pipeline(aec=aec) if aec else None

    transport = LocalAudioBackend(
        input_sample_rate=SAMPLE_RATE,
        output_sample_rate=SAMPLE_RATE,
        block_duration_ms=BLOCK_MS,
        mute_mic_during_playback=aec is None,
    )

    tool_running = False
    started_at = time.monotonic()

    def stamp() -> str:
        return f"{time.monotonic() - started_at:6.1f}s"

    async def on_transcription(_session: VoiceSession, text: str, role: str, final: bool) -> None:
        if role != "assistant" or not final or not text.strip():
            return
        marker = "[tool running]" if tool_running else "              "
        print(f"{stamp()} {marker} > {text.strip()}", flush=True)

    # Registered before the session opens: a callback added afterwards is
    # never seen by the transport that already started.
    provider.on_transcription(on_transcription)

    async def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal tool_running
        item = str(args.get("item", "")).lower()
        print(f"{stamp()} [tool  START] {name}({item!r}) - {TOOL_SECONDS:.0f}s", flush=True)
        tool_running = True
        try:
            await asyncio.sleep(TOOL_SECONDS)
        finally:
            tool_running = False
        print(f"{stamp()} [tool    END] {name}", flush=True)
        return {"item": item, "in_stock": INVENTORY.get(item, 0), "warehouse": "Lyon"}

    channel = RealtimeVoiceChannel(
        "warehouse",
        provider=provider,
        transport=transport,
        tools=[inventory_tool()],
        tool_handler=handler,
        system_prompt=SYSTEM_PROMPT,
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        input_sample_rate=SAMPLE_RATE,
        pipeline=pipeline,
    )
    kit = RoomKit()
    kit.register_channel(channel)

    await kit.create_room(room_id="warehouse-demo")
    await kit.attach_channel("warehouse-demo", "warehouse")

    session = await channel.start_session(
        "warehouse-demo",
        "caller",
        connection=None,
        # The extended-thinking model refuses a session with no level.
        metadata={
            "provider_config": {
                "thinking_level": os.environ.get("GEMINI_THINKING_LEVEL", "low"),
            }
        },
    )

    logger.info('Ask out loud: "how many blue widgets do we have in stock?"')
    logger.info("Watch for assistant turns marked [tool running]. Ctrl+C to stop.\n")

    async def cleanup() -> None:
        await channel.end_session(session)

    await run_until_stopped(kit, cleanup=cleanup)


if __name__ == "__main__":
    asyncio.run(main())

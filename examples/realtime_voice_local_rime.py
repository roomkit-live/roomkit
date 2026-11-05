"""Talk to Rime Prism using a local microphone and headphones (English).

Install: pip install 'roomkit[rime,local-audio]'
Run: RIME_API_KEY=... uv run python examples/realtime_voice_local_rime.py

Optional: RIME_VOICE=astra|aurora|brando|wawona, SYSTEM_PROMPT='...'.
Use headphones: the microphone stays open during playback for barge-in.
Prism owns turn detection; input is 16 kHz and output is 24 kHz mono PCM16.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, run_until_stopped, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.realtime import RimeRealtimeConfig, RimeRealtimeProvider

logger = setup_logging("realtime_voice_local_rime")


async def main() -> None:
    key = require_env("RIME_API_KEY")["RIME_API_KEY"]
    kit = RoomKit()
    channel = RealtimeVoiceChannel(
        "voice",
        provider=RimeRealtimeProvider(
            RimeRealtimeConfig(
                api_key=key,
                voice_id=os.environ.get("RIME_VOICE", "astra"),
            )
        ),
        transport=LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=24000,
            block_duration_ms=20,
            mute_mic_during_playback=False,
        ),
        input_sample_rate=16000,
        output_sample_rate=24000,
        system_prompt=os.environ.get("SYSTEM_PROMPT", "Keep your spoken replies brief."),
    )
    kit.register_channel(channel)
    try:
        await kit.create_room(room_id="rime-demo")
        await kit.attach_channel("rime-demo", "voice")
        await channel.start_session("rime-demo", "local-user", connection=None)
        logger.info("Prism is ready. Speak into your microphone; Ctrl+C stops the session.")
        await run_until_stopped(kit)
    finally:
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

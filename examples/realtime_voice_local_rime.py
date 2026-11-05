"""Talk to Rime Prism using a local microphone and speakers (English).

Install: pip install 'roomkit[rime,local-audio,webrtc-aec,console]'
Run: RIME_API_KEY=... uv run python examples/realtime_voice_local_rime.py

Optional: RIME_VOICE=astra|aurora|brando|wawona, SYSTEM_PROMPT='...'.
Set CONSOLE=1 to enable the voice dashboard (disabled by default).
WebRTC AEC removes speaker echo so the microphone stays open for barge-in.
AEC=webrtc (default), speex or 0; AEC_DELAY_MS sets a measured echo delay.
Without AEC, playback mutes the microphone. MUTE_MIC=0|1 overrides this;
with headphones, AEC=0 MUTE_MIC=0 keeps capture open without cancellation.
Prism owns turn detection; input is 16 kHz and output is 24 kHz mono PCM16.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    build_aec,
    env_bool,
    require_env,
    run_until_stopped,
    setup_console,
    setup_logging,
)

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.backends import LocalAudioBackend
from roomkit.voice.realtime import RimeRealtimeConfig, RimeRealtimeProvider

logger = setup_logging("realtime_voice_local_rime")


def _create_channel(api_key: str) -> RealtimeVoiceChannel:
    # The backend converts the 24 kHz playback reference to the 16 kHz
    # capture rate, and applies AEC on the audio thread.
    aec = build_aec(16000, 20, default="webrtc")
    return RealtimeVoiceChannel(
        "voice",
        provider=RimeRealtimeProvider(
            RimeRealtimeConfig(
                api_key=api_key,
                voice_id=os.environ.get("RIME_VOICE", "astra"),
            )
        ),
        transport=LocalAudioBackend(
            input_sample_rate=16000,
            output_sample_rate=24000,
            block_duration_ms=20,
            aec=aec,
            mute_mic_during_playback=env_bool("MUTE_MIC", default=aec is None),
        ),
        input_sample_rate=16000,
        output_sample_rate=24000,
        system_prompt=os.environ.get("SYSTEM_PROMPT", "Keep your spoken replies brief."),
    )


async def main() -> None:
    key = require_env("RIME_API_KEY")["RIME_API_KEY"]
    async with AsyncExitStack() as cleanup:
        kit = RoomKit()
        cleanup.push_async_callback(kit.close)
        channel = _create_channel(key)
        kit.register_channel(channel)
        if console_cleanup := setup_console(kit):
            cleanup.push_async_callback(console_cleanup)
        await kit.create_room(room_id="rime-demo")
        await kit.attach_channel("rime-demo", "voice")
        await channel.start_session("rime-demo", "local-user", connection=None)
        logger.info("Prism is ready. Speak into your microphone; Ctrl+C stops the session.")
        await run_until_stopped(kit, cleanup=cleanup.aclose)


if __name__ == "__main__":
    asyncio.run(main())

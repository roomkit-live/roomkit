"""RoomKit — Voice AI in a Buzz huddle (speech-to-speech with Gemini Live).

``BuzzHuddleWatcher`` does the heavy lifting: it watches the parent channel
for huddle announcements (kind 48100) through the framework's Buzz source
(relay reconnection included), dials each announced huddle, bridges it to
the voice channel, and rejoins if the relay drops the call mid-huddle. The
transport ends each call by itself when the last human leaves.

The agent's identity must be a member of the watched channel (see buzzkit's
README for closed-relay onboarding); membership in each ephemeral huddle is
granted automatically by the relay via the parent channel.

Requirements:
    pip install roomkit[buzz,realtime-gemini]

Run with:
    GOOGLE_API_KEY=... BUZZ_RELAY_URL=wss://... BUZZ_NSEC=nsec1... \
        BUZZ_CHANNEL_ID=<uuid> uv run python examples/buzz_voice_agent.py

Environment variables:
    GOOGLE_API_KEY     (required) Google API key
    BUZZ_RELAY_URL     (required) Buzz relay WebSocket URL
    BUZZ_NSEC          (required) agent secret key (hex or nsec…)
    BUZZ_CHANNEL_ID    (required) parent channel UUID to watch for huddles
    BUZZ_HUDDLE_ID     join this huddle immediately instead of watching
    BUZZ_AUTH_TAG      NIP-OA owner attestation tag JSON (optional)
    GEMINI_MODEL       model name (default: gemini-3.1-flash-live-preview)
    GEMINI_VOICE       voice preset (default: Aoede)
    SYSTEM_PROMPT      custom system prompt

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import os

from shared import require_env, run_until_stopped, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.buzz import BuzzConfig
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend, BuzzHuddleWatcher

logger = setup_logging("buzz_voice_agent")

DEFAULT_PROMPT = (
    "You are a friendly voice assistant participating in a team huddle. "
    "Be concise and conversational."
)


async def main() -> None:
    env = require_env("GOOGLE_API_KEY", "BUZZ_RELAY_URL", "BUZZ_NSEC", "BUZZ_CHANNEL_ID")

    kit = RoomKit()
    voice = RealtimeVoiceChannel(
        "buzz-voice",
        provider=GeminiLiveProvider(
            api_key=env["GOOGLE_API_KEY"],
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
        ),
        transport=BuzzHuddleBackend(),
        system_prompt=os.environ.get("SYSTEM_PROMPT", DEFAULT_PROMPT),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        # No transport_sample_rate: the backend resamples 48 kHz huddle
        # audio to/from Gemini's 16 kHz input / 24 kHz output itself (soxr).
    )
    kit.register_channel(voice)
    await kit.create_room(room_id="buzz-huddle")
    await kit.attach_channel("buzz-huddle", "buzz-voice")

    watcher = BuzzHuddleWatcher(
        kit,
        voice_channel=voice,
        config=BuzzConfig(
            relay_url=env["BUZZ_RELAY_URL"],
            private_key=env["BUZZ_NSEC"],
            auth_tag=os.environ.get("BUZZ_AUTH_TAG"),
        ),
        parent_channel_id=env["BUZZ_CHANNEL_ID"],
        room_id="buzz-huddle",
    )

    explicit = os.environ.get("BUZZ_HUDDLE_ID")
    if explicit:  # direct-join mode: bridge one known huddle, then exit
        await watcher.bridge(explicit)
        return

    await watcher.start()
    await run_until_stopped(kit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

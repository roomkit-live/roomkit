"""RoomKit — Voice AI in a Buzz huddle (speech-to-speech with Gemini Live).

The agent watches a Buzz channel for huddle announcements (kind 48100).
When someone starts a huddle, it joins the huddle's audio room and bridges
it to Gemini Live: everything said in the huddle streams to Gemini, and
Gemini's voice plays back to every huddle participant. When the huddle
ends, the agent goes back to watching.

The agent's identity must be a member of the watched channel (see buzzkit's
README for closed-relay onboarding); membership in each ephemeral huddle is
granted automatically by the relay via ``parent_channel_id``.

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
    BUZZ_TAP_DIR       record the outbound audio path for offline analysis:
                       provider_out_24k.raw (Gemini output, pre-resample),
                       pacer_in_48k.raw (post-resample, pacer input),
                       wire_out_48k.raw (pacer output = exact Opus encoder
                       input), wire_send.csv (per-send timing), events.csv
                       (interrupts / end-of-response). Analyze with
                       lib-buzz-python/examples/analyze_audio_tap.py.
    GEMINI_MODEL       model name (default: gemini-3.1-flash-live-preview)
    GEMINI_VOICE       voice preset (default: Aoede)
    SYSTEM_PROMPT      custom system prompt

Press Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
from collections.abc import AsyncIterator
from typing import Any

from buzzkit import BuzzClient, HuddleClient, HuddleError

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend
from roomkit.voice.base import AudioChunk, VoiceSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("buzz_voice_agent")


class TappedBuzzHuddleBackend(BuzzHuddleBackend):
    """BuzzHuddleBackend that tees the outbound audio path to disk.

    Debug-only (single session): records the audio at each hop so a choppy
    result can be blamed on a specific link — provider output, resampler,
    pacer, or the wire beyond us. See BUZZ_TAP_DIR in the module docstring.
    """

    def __init__(self, tap_dir: pathlib.Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        tap_dir.mkdir(parents=True, exist_ok=True)
        self._tap_provider = (tap_dir / "provider_out_24k.raw").open("wb")
        self._tap_pacer_in = (tap_dir / "pacer_in_48k.raw").open("wb")
        self._tap_wire = (tap_dir / "wire_out_48k.raw").open("wb")
        self._tap_send_csv = (tap_dir / "wire_send.csv").open("w")
        self._tap_send_csv.write("t_mono_ns,bytes,all_zero\n")
        self._tap_events = (tap_dir / "events.csv").open("w")
        self._tap_events.write("t_mono_ns,event\n")
        logger.info("Audio tap enabled -> %s", tap_dir)

    def _tap_event(self, name: str) -> None:
        self._tap_events.write(f"{time.monotonic_ns()},{name}\n")
        self._tap_events.flush()

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        await super().accept(session, connection)
        inner = self._outbound_resample[session.id]

        def tee_resample(data: bytes) -> bytes:
            out = inner(data)
            self._tap_pacer_in.write(out)
            return out

        self._outbound_resample[session.id] = tee_resample

    def _make_wire_sender(self, client: Any) -> Any:
        inner = super()._make_wire_sender(client)

        async def send(pcm48k: bytes) -> None:
            zero = 1 if pcm48k.strip(b"\x00") == b"" else 0
            self._tap_send_csv.write(f"{time.monotonic_ns()},{len(pcm48k)},{zero}\n")
            self._tap_wire.write(pcm48k)
            await inner(pcm48k)

        return send

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        if isinstance(audio, bytes):
            self._tap_provider.write(audio)
            await super().send_audio(session, audio)
            return

        async def tee(it: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
            async for chunk in it:
                self._tap_provider.write(chunk.data)
                yield chunk

        await super().send_audio(session, tee(audio))

    def interrupt(self, session: VoiceSession) -> None:
        self._tap_event("interrupt")
        super().interrupt(session)

    def end_of_response(self, session: VoiceSession) -> None:
        self._tap_event("end_of_response")
        super().end_of_response(session)

    async def close(self) -> None:
        await super().close()
        for f in (
            self._tap_provider,
            self._tap_pacer_in,
            self._tap_wire,
            self._tap_send_csv,
            self._tap_events,
        ):
            f.close()

KIND_HUDDLE_STARTED = 48100

DEFAULT_PROMPT = (
    "You are a friendly voice assistant participating in a team huddle. "
    "Be concise and conversational."
)


async def run_huddle_session(
    channel: RealtimeVoiceChannel,
    ended: asyncio.Event,
    relay_url: str,
    nsec: str,
    huddle_id: str,
    parent_id: str,
    auth_tag: str | None,
) -> None:
    """Join one huddle and bridge it to Gemini until it ends."""
    # paced=False: the BuzzHuddleBackend drives outbound timing with its own
    # prebuffer + jitter-headroom pacer (prevents choppy playback under the
    # agent's GIL load); the client just relays frames to the wire.
    huddle = HuddleClient(
        relay_url,
        nsec,
        huddle_id,
        parent_channel_id=parent_id,
        auth_tag=auth_tag,
        paced=False,
    )
    try:
        await huddle.connect()
    except (HuddleError, OSError) as exc:
        logger.warning("Could not join huddle %s: %s", huddle_id, exc)
        return

    logger.info("Joined huddle %s (%d peer(s))", huddle_id, max(len(huddle.peers) - 1, 0))
    ended.clear()
    session = await channel.start_session("buzz-huddle", "buzz-agent", connection=huddle)

    async def stats_reporter() -> None:
        last_sent = 0.0
        while True:
            await asyncio.sleep(5)
            stats = dict(huddle.sender_stats)
            if stats["sent"] != last_sent:
                last_sent = stats["sent"]
                logger.info("sender stats: %s (queued=%d)", stats, huddle.queued_frames)

    reporter = asyncio.create_task(stats_reporter())
    try:
        await ended.wait()  # the huddle ended or the relay dropped us
    finally:
        reporter.cancel()
        await channel.end_session(session)
        logger.info("Left huddle %s — final sender stats: %s", huddle_id, huddle.sender_stats)


async def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY")
    relay_url = os.environ.get("BUZZ_RELAY_URL")
    nsec = os.environ.get("BUZZ_NSEC")
    parent_id = os.environ.get("BUZZ_CHANNEL_ID")
    if not (api_key and relay_url and nsec and parent_id):
        print("Set GOOGLE_API_KEY, BUZZ_RELAY_URL, BUZZ_NSEC and BUZZ_CHANNEL_ID.")
        return
    auth_tag = os.environ.get("BUZZ_AUTH_TAG")

    kit = RoomKit()
    provider = GeminiLiveProvider(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
    )
    tap_dir = os.environ.get("BUZZ_TAP_DIR")
    if tap_dir:
        backend: BuzzHuddleBackend = TappedBuzzHuddleBackend(pathlib.Path(tap_dir))
        # The pacer logs prebuffer sizes and underruns at DEBUG — that timing
        # narrative is exactly what a tap session is trying to correlate.
        logging.getLogger("roomkit.voice.realtime.pacer").setLevel(logging.DEBUG)
    else:
        backend = BuzzHuddleBackend()
    ended = asyncio.Event()
    backend.on_client_disconnected(lambda _session: ended.set())
    channel = RealtimeVoiceChannel(
        "buzz-voice",
        provider=provider,
        transport=backend,
        system_prompt=os.environ.get("SYSTEM_PROMPT", DEFAULT_PROMPT),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        # No transport_sample_rate: the backend resamples 48 kHz huddle
        # audio to/from Gemini's 16 kHz input / 24 kHz output itself (soxr).
    )
    kit.register_channel(channel)
    await kit.create_room(room_id="buzz-huddle")
    await kit.attach_channel("buzz-huddle", "buzz-voice")

    # Direct-join mode: bridge one known huddle, then exit.
    explicit = os.environ.get("BUZZ_HUDDLE_ID")
    if explicit:
        await run_huddle_session(channel, ended, relay_url, nsec, explicit, parent_id, auth_tag)
        return

    # Watch mode: join every huddle announced on the parent channel from now
    # on (`since` skips historical announcements replayed before EOSE).
    logger.info("Watching channel %s for huddles (kind 48100)…", parent_id)
    async with BuzzClient(relay_url, nsec, auth_tag=auth_tag) as bz:
        live_filter = {
            "kinds": [KIND_HUDDLE_STARTED],
            "#h": [parent_id],
            "since": int(time.time()),
        }
        async for event in bz.subscribe([live_filter]):
            try:
                huddle_id = json.loads(event.get("content", "{}"))["ephemeral_channel_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            await run_huddle_session(
                channel, ended, relay_url, nsec, huddle_id, parent_id, auth_tag
            )
            logger.info("Watching channel %s for the next huddle…", parent_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

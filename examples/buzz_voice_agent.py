"""RoomKit — Voice AI in a Buzz huddle (speech-to-speech with Gemini Live).

Everything flows through the framework:

* A ``BuzzRelaySource`` (+ ``BuzzChannel``) subscribes to the parent channel's
  huddle announcements (kind 48100) and emits them as room events —
  ``attach_source(auto_restart=True)`` reconnects to the relay by itself.
* An ``AFTER_BROADCAST`` hook reacts to each announcement by dialing the
  huddle: connect a ``buzzkit.HuddleClient`` and hand it to the
  ``RealtimeVoiceChannel`` (Gemini Live over a ``BuzzHuddleBackend``).
* The transport owns the end of the call: the backend ends the session when
  the last human leaves (``end_when_alone``) or when the relay drops the
  socket, and the channel cleans up. The example only reads WHY it ended
  (``session.metadata["buzz_end_reason"]``) to decide between rejoining the
  same huddle (connection loss) and waiting for the next one (call over).

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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import json
import logging
import os
import pathlib
import time
from collections.abc import AsyncIterator
from typing import Any

from buzzkit import KIND_HUDDLE_STARTED, HuddleClient, HuddleError
from shared import require_env, run_until_stopped, setup_logging

from roomkit import (
    BuzzChannel,
    HookExecution,
    HookResult,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.models.delivery import InboundMessage
from roomkit.providers.buzz import BuzzConfig, BuzzProvider
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.sources.buzz import BuzzMessageParser, BuzzRelaySource
from roomkit.voice.backends.buzz_huddle import BuzzHuddleBackend
from roomkit.voice.base import AudioChunk, VoiceSession

logger = setup_logging("buzz_voice_agent")

DEFAULT_PROMPT = (
    "You are a friendly voice assistant participating in a team huddle. "
    "Be concise and conversational."
)


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


def huddle_announcements(channel_id: str, *, started_after: int) -> BuzzMessageParser:
    """Parser for kind 48100: one InboundMessage per fresh huddle announcement.

    ``started_after`` drops announcements replayed from relay history (the
    subscription replays recent events before EOSE), so a restart doesn't
    try to join long-dead huddles.
    """

    def parser(event: dict[str, Any], own_pubkey: str | None) -> InboundMessage | None:
        if event.get("kind") != KIND_HUDDLE_STARTED:
            return None
        if int(event.get("created_at") or 0) < started_after:
            return None
        try:
            huddle_id = json.loads(event.get("content") or "{}")["ephemeral_channel_id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        event_id = str(event.get("id", ""))
        return InboundMessage(
            channel_id=channel_id,
            sender_id=str(event.get("pubkey", "")),
            content=TextContent(body=f"huddle started: {huddle_id}"),
            external_id=event_id,
            idempotency_key=event_id,
            metadata={"ephemeral_channel_id": str(huddle_id)},
        )

    return parser


async def main() -> None:
    env = require_env("GOOGLE_API_KEY", "BUZZ_RELAY_URL", "BUZZ_NSEC", "BUZZ_CHANNEL_ID")
    relay_url, nsec = env["BUZZ_RELAY_URL"], env["BUZZ_NSEC"]
    parent_id = env["BUZZ_CHANNEL_ID"]
    auth_tag = os.environ.get("BUZZ_AUTH_TAG")

    kit = RoomKit()

    # --- Voice channel: Gemini Live over huddle audio -------------------------
    tap_dir = os.environ.get("BUZZ_TAP_DIR")
    if tap_dir:
        backend: BuzzHuddleBackend = TappedBuzzHuddleBackend(pathlib.Path(tap_dir))
        # The pacer logs prebuffer sizes and underruns at DEBUG — that timing
        # narrative is exactly what a tap session is trying to correlate.
        logging.getLogger("roomkit.voice.realtime.pacer").setLevel(logging.DEBUG)
    else:
        backend = BuzzHuddleBackend()
    voice = RealtimeVoiceChannel(
        "buzz-voice",
        provider=GeminiLiveProvider(
            api_key=env["GOOGLE_API_KEY"],
            model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview"),
        ),
        transport=backend,
        system_prompt=os.environ.get("SYSTEM_PROMPT", DEFAULT_PROMPT),
        voice=os.environ.get("GEMINI_VOICE", "Aoede"),
        # No transport_sample_rate: the backend resamples 48 kHz huddle
        # audio to/from Gemini's 16 kHz input / 24 kHz output itself (soxr).
    )
    kit.register_channel(voice)

    # --- Announcement channel: huddle starts arrive as room events ------------
    config = BuzzConfig(relay_url=relay_url, private_key=nsec, auth_tag=auth_tag)
    source = BuzzRelaySource(
        config,
        "buzz-events",
        relay_channel_id=parent_id,
        kinds=[KIND_HUDDLE_STARTED],
        parser=huddle_announcements("buzz-events", started_after=int(time.time())),
    )
    kit.register_channel(BuzzChannel("buzz-events", provider=BuzzProvider(source)))

    await kit.create_room(room_id="buzz-huddle")
    await kit.attach_channel("buzz-huddle", "buzz-voice")
    await kit.attach_channel("buzz-huddle", "buzz-events")

    # --- Session lifecycle -----------------------------------------------------
    # The transport ends the session (last human left, or socket dropped) and
    # the channel cleans up on its own. We only track WHY it ended, to choose
    # between rejoining the same huddle and waiting for the next one.
    session_over = asyncio.Event()
    end_reason: dict[str, str | None] = {"value": None}

    def on_session_over(session: VoiceSession) -> None:
        end_reason["value"] = session.metadata.get("buzz_end_reason")
        session_over.set()

    backend.on_client_disconnected(on_session_over)

    busy = asyncio.Lock()

    async def bridge_huddle(huddle_id: str) -> None:
        """Dial the huddle and keep the bridge up until the call is over."""
        if busy.locked():
            logger.info("Already bridging a huddle — ignoring %s", huddle_id)
            return
        try:
            async with busy:
                retry = 0
                while True:
                    # paced=False: the backend's OutboundAudioPacer owns the
                    # outbound timing; the client just relays frames.
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
                    except HuddleError as exc:
                        logger.info("Huddle %s is over (%s)", huddle_id, exc)
                        return
                    except OSError as exc:
                        retry += 1
                        if retry > 3:
                            logger.warning(
                                "Could not join huddle %s: %s — giving up", huddle_id, exc
                            )
                            return
                        logger.info("Join failed (%s) — retry %d/3", exc, retry)
                        await asyncio.sleep(2 * retry)
                        continue
                    retry = 0
                    logger.info(
                        "Joined huddle %s (%d peer(s))", huddle_id, max(len(huddle.peers) - 1, 0)
                    )
                    session_over.clear()
                    await voice.start_session("buzz-huddle", "buzz-agent", connection=huddle)
                    await session_over.wait()
                    if end_reason["value"] == "alone":
                        logger.info("Huddle %s over — everyone left", huddle_id)
                        return
                    logger.info("Connection to huddle %s lost — rejoining…", huddle_id)
        except Exception:
            logger.exception("Huddle bridge failed for %s", huddle_id)

    background: set[asyncio.Task] = set()

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="join_huddle")
    async def join_huddle(event: RoomEvent, ctx: RoomContext) -> HookResult:
        huddle_id = event.metadata.get("ephemeral_channel_id")
        if huddle_id:
            task = asyncio.create_task(bridge_huddle(str(huddle_id)))
            background.add(task)
            task.add_done_callback(background.discard)
        return HookResult.allow()

    # Direct-join mode: bridge one known huddle, then exit.
    explicit = os.environ.get("BUZZ_HUDDLE_ID")
    if explicit:
        await bridge_huddle(explicit)
        return

    # Watch mode: the framework owns the announcement subscription, including
    # reconnecting to the relay (auto_restart) whenever it drops.
    await kit.attach_source("buzz-events", source, auto_restart=True)
    logger.info("Watching channel %s for huddles (kind %d)…", parent_id, KIND_HUDDLE_STARTED)
    await run_until_stopped(kit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

"""The AI answers the meeting — the real STT -> LLM -> TTS loop, end to end.

`conference_quickstart.py` shows the meeting being *heard*; this example closes
the loop: a human speaks, the transcription reaches an ``AIChannel`` as an
ordinary RoomEvent, the model answers, and the conference speaks the answer on
the bot's track. Nothing here wires those pieces together explicitly — that is
the point, and it is RFC §12.10.1 principle 2 at work: the conference is the
room, transcriptions are RoomEvents, and cross-channel broadcast applies with
no conference-specific exceptions. The one rule doing quiet work is the
channel's own: a conference speaks only events from an AI-typed source (pass
``speak_text_events=True`` to widen that), so an ``AIChannel``'s answer is
voiced with zero configuration.

A loop that answers a room it also speaks into needs two protections, and both
are on display:

1. **The media loop** — the bot's own TTS audio must not come back through a
   lane, be transcribed, and be answered again. Bot self-exclusion closes it:
   the channel never subscribes to a track its own session published
   (RFC §12.10.4). Watch the AI call count at the end: one response per human
   utterance, none for the bot's own voice.
2. **The event loop** — the AI must not answer its own answer. ``AIChannel``
   skips events it produced, and the framework bounds any AI-to-AI chain at
   ``max_chain_depth`` (default 5, RFC §8.3), so even two AIs in one room
   converse finitely.

``BEFORE_TTS`` is the orchestration point in between: it runs synchronously
just before synthesis, sees the final text, and may rewrite or block it. Here
it holds the bot silent while a "handoff" is pending — the answer is generated,
broadcast, stored — and simply never spoken.

Run with:
    uv run python examples/conference_ai_meeting.py
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Any

from roomkit import (
    ConferenceTranscription,
    HookResult,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "standup"

# Same audio arithmetic as conference_quickstart.py: the default pipeline is a
# 16 kHz mono contract with an energy VAD, and one utterance is speech loud
# enough, long enough — then enough silence for the VAD to call it finished.
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE * 20 // 1000  # 20 ms
SPEECH_FRAMES = 15  # 300 ms — past the energy VAD's 200 ms minimum
SILENCE_FRAMES = 30  # 600 ms — past its 500 ms end-of-speech threshold


def speech_frame() -> AudioFrame:
    """A frame loud enough for the energy VAD to call it speech."""
    samples = [8000, -8000] * (SAMPLES_PER_FRAME // 2)
    return AudioFrame(
        data=struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples), sample_rate=SAMPLE_RATE
    )


def silence_frame() -> AudioFrame:
    """Zeros — no VAD calls this speech."""
    return AudioFrame(data=b"\x00\x00" * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def speak(backend: MockConferenceBackend, track: Any) -> None:
    """Push one utterance's worth of frames through the backend."""
    for _ in range(SPEECH_FRAMES):
        await backend.simulate_audio(track, speech_frame())
    for _ in range(SILENCE_FRAMES):
        await backend.simulate_audio(track, silence_frame())


async def until(condition: Any, *, timeout: float = 5.0) -> None:
    """Wait for a condition the loop's background work will make true.

    The AI's answer travels on its own tasks — generation, re-entry,
    broadcast, synthesis — so the demo waits on the observable result rather
    than sleeping toward it.
    """
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the loop did not produce the expected result in time")
        await asyncio.sleep(0.01)


async def main() -> None:
    backend = MockConferenceBackend()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        stt=MockSTTProvider(
            transcripts=[
                "What's on the agenda today?",
                "Can you summarize where the release stands?",
                "Thanks everyone, let's wrap up.",
            ]
        ),
        tts=MockTTSProvider(),
    )
    # A real deployment swaps MockAIProvider for AnthropicAIProvider (or any
    # other) and changes nothing else — see examples/conference_livekit.py for
    # that wiring against a real SFU.
    provider = MockAIProvider(
        responses=[
            "Two items today: the release review, and the conference demo.",
            "The release is green: the suite passed and the changelog is ready.",
            "Enjoy the rest of your day!",
        ]
    )
    ai = AIChannel("ai", provider=provider, system_prompt="You are the meeting assistant.")

    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(ai)

    heard: list[ConferenceTranscription] = []
    spoken: list[str] = []
    held_back: list[str] = []
    handoff_pending = False

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(payload: ConferenceTranscription, ctx: Any) -> HookResult:
        heard.append(payload)
        print(f"      [stt] {payload.participant_id}: {payload.text}")
        return HookResult.allow()

    # BEFORE_TTS runs synchronously with the final text, just before synthesis:
    # the one moment orchestration can rewrite the bot's words or hold them
    # back entirely. Blocking here does not undo the answer — it is already a
    # RoomEvent in the store — it only keeps it out of the meeting's audio.
    @kit.hook(HookTrigger.BEFORE_TTS)
    async def orchestrate(text: str, ctx: Any) -> HookResult:
        if handoff_pending:
            held_back.append(text)
            print("      [orchestration] handoff pending -> the bot holds this answer back")
            return HookResult(action="block", reason="handoff pending")
        return HookResult.allow()

    # AFTER_TTS fires with what was actually said, once it has been published
    # on the bot's track — the pair BEFORE/AFTER stays matched, so an answer
    # blocked above never reaches here.
    @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
    async def voiced(text: str, ctx: Any) -> None:
        spoken.append(text)
        print(f'      [tts] the bot says: "{text}"')

    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "ai")

    await kit.ensure_participant(ROOM, "conf", "alice", display_name="Alice")
    await channel.mint_access(ROOM, "alice")
    await backend.simulate_participant_joined(ROOM, "alice", display_name="Alice")
    await backend.simulate_participant_joined(ROOM, "bob", display_name="Bob")
    alice_mic = await backend.simulate_track_published(ROOM, "alice")
    bob_mic = await backend.simulate_track_published(ROOM, "bob")

    print("alice asks, and the AI answers out loud:")
    await speak(backend, alice_mic)
    await until(lambda: len(spoken) == 1)

    print("\nbob asks during a handoff — the answer exists, the room stays quiet:")
    handoff_pending = True
    await speak(backend, bob_mic)
    await until(lambda: len(held_back) == 1)
    handoff_pending = False

    print("\nalice closes, and the bot speaks again:")
    await speak(backend, alice_mic)
    await until(lambda: len(spoken) == 2)

    # The whole exchange is ordinary room history: the humans attributed by
    # their tracks, the AI by its channel. The held-back answer is in here too
    # — blocked from the *audio*, not from the record.
    print("\nwhat the room recorded:")
    for event in await kit.store.list_events(ROOM):
        body = getattr(event.content, "body", None)
        who = event.source.participant_id or event.source.channel_id
        if body and who != "system":
            print(f"      [{who}] {body}")

    # The two loop protections, measured rather than asserted in prose: three
    # human utterances produced exactly three AI generations — the bot's own
    # voice (published below) triggered none, and no answer answered itself.
    bot = backend.bots[0]
    print(
        f"\nloops, closed: {len(heard)} human utterances -> {len(provider.calls)} AI "
        f"generations -> {len(backend.utterances_for(bot))} bot utterances published "
        f"({len(held_back)} held back by orchestration)"
    )

    info = channel.info()["rooms"][ROOM]
    print(
        f"disclosure: bot_present={info['bot_present']}"
        f" stt_active={info['stt_active']}"
        f" active_lanes={info['active_lanes']}"
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

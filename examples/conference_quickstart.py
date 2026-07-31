"""A conference end-to-end, with nothing real attached.

RoomKit never sits in the media path between humans — an external SFU routes
their audio, and RoomKit joins the meeting as one more participant (a bot) to
provide the intelligence: transcription attributed to whoever spoke, AI voice,
recording (RFC §12.10). This example runs that whole arrangement against
``MockConferenceBackend``: no SFU, no credentials, deterministic output.

Three things to watch for in the output:

1. **The bot joins lazily, because this channel has a need.** Minting an
   access credential is the framework's advance notice that a human is about
   to connect, and it is what brings the bot in — no one has to speak first.
   It brings the bot in *here* because this channel transcribes; a channel
   with no stt, tts or recording never joins at all — RoomKit stays the
   room's admission gate and roster, with no participant of its own in the
   meeting (pure transport, RFC §12.10.4).
2. **One utterance becomes one event, not one per frame.** Audio arrives as
   20 ms frames; the VAD finds the utterance boundary and the whole utterance
   goes to the STT as one block. The frame counts printed next to each
   transcription are the point.
3. **Silence produces nothing.** Frames of zeros — the obvious thing to reach
   for when simulating audio — are not speech, and the correct result is no
   event at all. If you write your own demo against the mock backend and see
   nothing, check this first.

Run with:
    uv run python examples/conference_quickstart.py
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from roomkit import (
    ConferenceTranscription,
    HookResult,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import HookTrigger
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.mock import MockSTTProvider

ROOM = "standup"

# The channel's default pipeline expects 16 kHz mono (the STT contract) and
# segments with an energy-threshold VAD. One utterance is speech loud enough,
# long enough — then enough silence for the VAD to call it finished.
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
    """Zeros. A VAD will never call this speech — see point 3 above."""
    return AudioFrame(data=b"\x00\x00" * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def deliver_utterance(backend: MockConferenceBackend, track: Any, *, speech: int) -> int:
    """Push one utterance's worth of frames through the backend; count them."""
    for _ in range(speech):
        await backend.simulate_audio(track, speech_frame())
    for _ in range(SILENCE_FRAMES if speech else 40):
        await backend.simulate_audio(track, silence_frame())
    return speech + (SILENCE_FRAMES if speech else 40)


async def main() -> None:
    backend = MockConferenceBackend()
    stt = MockSTTProvider(transcripts=["It's Alice.", "And this is Bob."])
    # No pipeline= argument: the channel builds a working default (16 kHz mono
    # contract + energy VAD). Pass an AudioPipelineConfig to choose your own
    # VAD — but a config without one is refused when an STT is configured.
    channel = ConferenceChannel("conf", backend=backend, stt=stt)

    kit = RoomKit()
    kit.register_channel(channel)

    heard: list[ConferenceTranscription] = []

    # ON_TRANSCRIPTION is synchronous: it sees the text BEFORE it reaches the
    # room, and may rewrite it (HookResult.modify with an updated payload) or
    # suppress it (HookResult.block) — this is where redaction lives. It fails
    # closed: a hook that raises blocks the text too. Here it only observes.
    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(payload: ConferenceTranscription, ctx: Any) -> HookResult:
        heard.append(payload)
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_SPEECH_START)
    async def speech_start(event: Any, ctx: Any) -> None:
        print(f"      [vad] {event.content.data['participant_id']} starts speaking")

    @kit.hook(HookTrigger.ON_SPEECH_END)
    async def speech_end(event: Any, ctx: Any) -> None:
        print(f"      [vad] {event.content.data['participant_id']} stops")

    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    # Mint access for a room participant (RFC §12.10.2): the credential is what
    # a real client would use to connect to the SFU directly — RoomKit is the
    # gate for issuing tokens, not for the connection itself. The mint also
    # starts the lazy bot join — it is the framework's own advance notice that
    # a human is about to arrive, and this channel has an stt to feed. Without
    # stt, tts or recording the mint would admit alice and bring no bot in.
    await kit.ensure_participant(ROOM, "conf", "alice", display_name="Alice")
    access = await channel.mint_access(ROOM, "alice")
    print(f"alice's credential: {access.url} token={access.token}")

    # The SFU now reports the humans connecting and publishing microphones.
    # (With a real backend these calls are what the SFU's events drive.)
    await backend.simulate_participant_joined(ROOM, "alice", display_name="Alice")
    await backend.simulate_participant_joined(ROOM, "bob", display_name="Bob")
    alice_mic = await backend.simulate_track_published(ROOM, "alice")
    bob_mic = await backend.simulate_track_published(ROOM, "bob")

    # Each subscribed audio track gets a lane of its own — queue, task, VAD
    # state — so one slow recognizer never delays another participant's audio.
    print("\nalice speaks:")
    frames = await deliver_utterance(backend, alice_mic, speech=SPEECH_FRAMES)
    await channel.active_lanes[alice_mic.id].drain()
    print(f"      {frames} frames delivered -> {len(heard)} transcription(s)")

    print("\nbob speaks:")
    frames = await deliver_utterance(backend, bob_mic, speech=SPEECH_FRAMES)
    await channel.active_lanes[bob_mic.id].drain()
    print(f"      {frames} frames delivered -> {len(heard) - 1} transcription(s)")

    print("\nbob's line stays open in silence:")
    frames = await deliver_utterance(backend, bob_mic, speech=0)
    await channel.active_lanes[bob_mic.id].drain()
    print(f"      {frames} frames delivered -> 0 transcriptions (silence is not speech)")

    # Transcriptions are ordinary RoomEvents, attributed to the participant
    # whose track carried the voice — track identity is what attributes speech,
    # which is why the conference pipeline needs no diarization stage.
    print("\nwhat the room recorded:")
    for event in await kit.store.list_events(ROOM):
        body = getattr(event.content, "body", None)
        if body and event.source.participant_id:
            print(f"      [{event.source.participant_id}] {body}")

    # info() answers RFC §17.7's disclosure questions: is a bot in the meeting,
    # is collection active, is anything being recorded — right now.
    info = channel.info()
    room_info = info["rooms"][ROOM]
    print(
        f"\ndisclosure: bot_present={room_info['bot_present']}"
        f" stt_active={room_info['stt_active']}"
        f" recording_active={room_info['recording_active']}"
        f" active_lanes={room_info['active_lanes']}"
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

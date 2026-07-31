"""A notetaker that enters when the host asks — and leaves when dismissed.

The differentiating primitive of RoomKit's conference layer (RFC §12.10.4):
a room is a transport, and the intelligence plugs into it *while the meeting
runs*. This meeting starts purely human — RoomKit mints the credentials and
keeps the roster, but no bot joins, nothing is transcribed, nothing recorded.
When the host asks for the notetaker, ``plug_stt()`` + ``plug_recording()``
turn the running channel into one: plugging a need is a first need, so the
occupied conference is joined at once and the tracks already published are
subscribed retroactively — the meeting is transcribed from the plug forward.
When the host dismisses it, unplugging the last need takes the bot out:
``conference_ended`` is announced, the recordings are finalized, and the
channel is pure transport again — same channel, same room, no rebuild.

Why this shape matters: mute, active-speaker and quality events exist only on
an SFU *connection*, so the bot is the one full-fidelity observer a meeting
can have — and §17.7's consent obligations want it present exactly when asked
for, not from construction. The host's request is the explicit gesture; the
plug is its implementation; ``info()`` answers the disclosure question with
the configuration in force, not the constructor's.

Run with:
    uv run python examples/conference_notetaker_on_demand.py
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

from roomkit import (
    ConferenceRecordingConfig,
    ConferenceTranscription,
    HookResult,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import HookTrigger
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.stt.mock import MockSTTProvider

ROOM = "board-meeting"

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE * 20 // 1000  # 20 ms
SPEECH_FRAMES = 15  # 300 ms — past the energy VAD's minimum
SILENCE_FRAMES = 30  # 600 ms — past its end-of-speech threshold


def speech_frame() -> AudioFrame:
    samples = [8000, -8000] * (SAMPLES_PER_FRAME // 2)
    return AudioFrame(
        data=struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples), sample_rate=SAMPLE_RATE
    )


def silence_frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x00" * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def say_something(backend: MockConferenceBackend, track: Any) -> None:
    """One utterance: speech, then enough silence for the VAD to close it."""
    for _ in range(SPEECH_FRAMES):
        await backend.simulate_audio(track, speech_frame())
    for _ in range(SILENCE_FRAMES):
        await backend.simulate_audio(track, silence_frame())


def disclosure(channel: ConferenceChannel) -> str:
    """The §17.7 answer, as an integrator's consent UI would read it."""
    room = channel.info()["rooms"][ROOM]
    return (
        f"bot_present={room['bot_present']}"
        f" stt_active={room['stt_active']}"
        f" recording_active={room['recording_active']}"
    )


async def main() -> None:
    backend = MockConferenceBackend()
    # No stt, no tts, no recording: pure transport. RoomKit is the admission
    # gate and the roster, and nothing more — no bot ever joins this way.
    channel = ConferenceChannel("conf", backend=backend)

    kit = RoomKit()
    kit.register_channel(channel)

    heard: list[str] = []

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(payload: ConferenceTranscription, ctx: Any) -> HookResult:
        heard.append(f"[{payload.participant_id}] {payload.text}")
        return HookResult.allow()

    @kit.on("conference_started")
    async def started(event: Any) -> None:
        print("      -> conference_started: the notetaker is in the meeting")

    @kit.on("conference_ended")
    async def ended(event: Any) -> None:
        print("      -> conference_ended: the notetaker has left")

    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    # The meeting starts without the framework in it: credentials go out,
    # humans connect and publish, the roster records them — and no bot joins,
    # because nothing plugged into the channel could use the connection.
    await kit.ensure_participant(ROOM, "conf", "alice", display_name="Alice")
    await channel.mint_access(ROOM, "alice")
    await backend.simulate_participant_joined(ROOM, "alice", display_name="Alice")
    await backend.simulate_participant_joined(ROOM, "bob", display_name="Bob")
    alice_mic = await backend.simulate_track_published(ROOM, "alice")
    print("the meeting runs, purely human:")
    print(f"      {disclosure(channel)}")

    # ------------------------------------------------------------------
    # The host asks for the notetaker. This is the explicit gesture §17.7
    # wants a consent decision tied to — and the plug is its implementation.
    # ------------------------------------------------------------------
    print("\nthe host asks for the notetaker:")
    recorder = MockMediaRecorder()
    await channel.plug_stt(MockSTTProvider(transcripts=["Approve the budget."]))
    await channel.plug_recording(ConferenceRecordingConfig(), recorder=recorder)
    # Plugging a need is a first need: the occupied conference was joined and
    # alice's already-published microphone subscribed — no re-publish needed.
    print(f"      {disclosure(channel)}")

    print("\nalice speaks, and the meeting is transcribed from the plug forward:")
    await say_something(backend, alice_mic)
    await channel.active_lanes[alice_mic.id].drain()
    for line in heard:
        print(f"      {line}")

    # ------------------------------------------------------------------
    # The host dismisses the notetaker. Unplugging the last need takes the
    # bot out — a session with nothing to consume and nothing to say is the
    # silent observer §17.7 exists to surface, so it does not linger.
    # ------------------------------------------------------------------
    print("\nthe host dismisses the notetaker:")
    await channel.unplug_stt()
    await channel.unplug_recording()
    print(f"      {disclosure(channel)}")
    print(
        f"      recordings finalized: {len(recorder.results)}, recorder closed: {recorder.closed}"
    )

    # The meeting carries on, purely human again — same channel, same room.
    await kit.ensure_participant(ROOM, "conf", "carol", display_name="Carol")
    await channel.mint_access(ROOM, "carol")
    print("\ncarol is admitted afterwards; still no bot:")
    print(f"      {disclosure(channel)}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

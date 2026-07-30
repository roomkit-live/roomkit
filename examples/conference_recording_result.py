"""Learning where a conference recording was written.

Framework-mode conference recording opens one recording per subscribed track,
attributed to the participant publishing it (RFC §12.10.8). This example shows
how an integrator finds out where those files went, which is the whole point of
recording something: ON_RECORDING_STARTED when a track's recording opens, and
ON_RECORDING_STOPPED when it closes, carrying the location, the duration and the
attribution.

One report per track, not one per conference — the tracks of a meeting do not
end together, and a participant who leaves halfway through has a finished file
while the meeting runs on. An integrator wanting the meeting's full list
accumulates it by room, as this example does.

Runs against MockConferenceBackend and MockMediaRecorder, so it needs no SFU and
no codec: swap in a real backend and PyAVMediaRecorder and the same handlers
report real paths on disk.

Run with:
    uv run python examples/conference_recording_result.py
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from roomkit import (
    ConferenceRecordingConfig,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
    HookExecution,
    HookTrigger,
    MockConferenceBackend,
    RoomContext,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.audio_frame import AudioFrame

ROOM = "board-meeting"


def speech() -> AudioFrame:
    """20 ms of 16 kHz PCM. Content does not matter here — arrival does."""
    return AudioFrame(data=b"\x01\x00" * 320, sample_rate=16000)


async def main() -> None:
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(
        ConferenceChannel(
            "conf",
            backend=backend,
            recorder=MockMediaRecorder(),
            # PyAVMediaRecorder(storage="./recordings") writes real files here.
            recording=ConferenceRecordingConfig(storage="./recordings", format="wav"),
        )
    )
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    # What a compliance archive would keep: one entry per file, by meeting.
    archive: dict[str, list[ConferenceRecordingStopped]] = defaultdict(list)

    @kit.hook(HookTrigger.ON_RECORDING_STARTED, execution=HookExecution.ASYNC)
    async def on_started(event: ConferenceRecordingStarted, ctx: RoomContext) -> None:
        print(f"  ▶ recording {event.id} opened for {event.participant_id} ({event.kind})")

    @kit.hook(HookTrigger.ON_RECORDING_STOPPED, execution=HookExecution.ASYNC)
    async def on_stopped(event: ConferenceRecordingStopped, ctx: RoomContext) -> None:
        archive[event.room_id].append(event)
        print(
            f"  ■ recording {event.id} closed for {event.participant_id} → "
            f"{event.url} ({event.duration_seconds:.1f}s, {event.size_bytes} bytes)"
        )

    print("Alice and Bob join and speak:")
    await backend.simulate_participant_joined(ROOM, "p-alice")
    await backend.simulate_participant_joined(ROOM, "p-bob")
    alice = await backend.simulate_track_published(ROOM, "p-alice")
    bob = await backend.simulate_track_published(ROOM, "p-bob")
    for _ in range(5):
        await backend.simulate_audio(alice, speech())
        await backend.simulate_audio(bob, speech())

    print("\nA participant who publishes but never speaks leaves no file:")
    await backend.simulate_participant_joined(ROOM, "p-carol")
    carol = await backend.simulate_track_published(ROOM, "p-carol")
    await backend.simulate_track_unpublished(carol.id)
    print("  (nothing reported for p-carol — the recording opens on the first frame)")

    print("\nAlice leaves early — her recording closes while the meeting runs on:")
    await backend.simulate_track_unpublished(alice.id)

    print("\nThe meeting ends:")
    await kit.detach_channel(ROOM, "conf")

    print(f"\nArchived for {ROOM}:")
    for entry in archive[ROOM]:
        print(f"  {entry.participant_id:>10}  {entry.url}")


if __name__ == "__main__":
    asyncio.run(main())

"""Testing a conference against a backend that misbehaves.

A conference that only ever meets a working SFU is a conference whose failure
paths are untested. MockConferenceBackend can therefore be made to refuse calls,
to take time answering them, to publish tracks in formats that disagree with
each other, and to say which chunks of the bot's voice belonged to which
utterance.

Four levers, one per thing a real conference does that a happy-path mock does
not:

1. ``fail(method, error, times=)``  — the SFU refuses, or times out
2. ``delay(operation, seconds)``    — the SFU, or a subscriber, is slow
3. ``MockTrackFormat``              — participants negotiate their own formats
4. ``backend.utterances``           — what the bot published, per utterance

The fifth scenario needs no lever at all: the storage is what is slow there, and
a recorder that blocks is a recorder written the only way the interface allows.

Run with:
    uv run python examples/conference_fault_injection.py
"""

from __future__ import annotations

import asyncio
import threading
import time

from roomkit import (
    ConferenceGrants,
    ConferenceRecordingConfig,
    MockConferenceBackend,
    MockTrackFormat,
    RoomKit,
)
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.conference.models import ConferenceTrack
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.recorder.base import MediaRecordingHandle, RecordingTrack
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "board-meeting"

SPEECH_FRAMES = 15
"""300 ms — past the energy VAD's minimum for an utterance."""

SILENCE_FRAMES = 30
"""600 ms — past its end-of-speech threshold, so the utterance closes."""


class AISource(Channel):
    """Stands in for an AIChannel, so the bot has something to say."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.AI

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class SlowRecorder(MockMediaRecorder):
    """A recorder whose writes take time, the only way a recorder can.

    ``MediaRecorder`` is synchronous throughout, so "slow storage" means a call
    that blocks — which is why the channel makes it somewhere other than the
    thread its event loop runs on.
    """

    def __init__(self, *, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self.threads: set[int] = set()

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        self.threads.add(threading.get_ident())
        time.sleep(self.seconds)
        super().on_data(handle, track, data, timestamp_ms)


async def say(backend: MockConferenceBackend, track: ConferenceTrack) -> None:
    """One utterance's worth of frames, in the track's own format."""
    for _ in range(SPEECH_FRAMES):
        await backend.simulate_audio(track, backend.frame_for(track))
    for _ in range(SILENCE_FRAMES):
        await backend.simulate_audio(track, backend.frame_for(track, amplitude=0.0))


async def failures() -> None:
    """1. The SFU refuses."""
    print("1. A backend that refuses\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    channel = kit.get_channel("conf")
    await kit.ensure_participant(ROOM, "conf", "p-alice", display_name="Alice")

    # Minting admission is the one call whose failure reaches the integrator:
    # a credential nobody received is better than one handed out blind.
    backend.fail("mint_access", TimeoutError("SFU unreachable"), times=1)
    try:
        await channel.mint_access(ROOM, "p-alice", grants=ConferenceGrants())
    except TimeoutError as exc:
        print(f"   mint_access → {type(exc).__name__}: {exc}")

    # times=1 retired the fault, so the retry finds a working SFU.
    access = await channel.mint_access(ROOM, "p-alice", grants=ConferenceGrants())
    print(f"   retry      → {access.token}")

    # The attempt is recorded even though it failed: the request did go out.
    attempts = [call for call in backend.calls if call.method == "mint_access"]
    print(f"   backend saw {len(attempts)} mint attempts, not {len(attempts) - 1}\n")

    await kit.detach_channel(ROOM, "conf")


async def latency() -> None:
    """2. Something is slow, and lane isolation is measured rather than assumed."""
    print("2. A slow track must not slow the others (RFC §12.10.4)\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend, stt=MockSTTProvider()))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    await backend.simulate_participant_joined(ROOM, "p-alice")
    await backend.simulate_participant_joined(ROOM, "p-bob")
    alice = await backend.simulate_track_published(ROOM, "p-alice")
    bob = await backend.simulate_track_published(ROOM, "p-bob")

    # A subscriber that does its work inside the delivery callback — which is
    # what a lane must not do — makes its own latency everyone's latency.
    async def slow_bystander(track, frame) -> None:
        if track.id == alice.id:
            await asyncio.sleep(0.01)

    backend.on_track_audio(slow_bystander)

    await say(backend, alice)
    await say(backend, bob)

    for participant, track in (("p-alice", alice), ("p-bob", bob)):
        held = [d.elapsed for d in backend.deliveries if d.track_id == track.id]
        print(f"   {participant}: slowest frame delivery {max(held) * 1000:.1f} ms")
    print("   (the channel's own lane is on the fast side — it accepts and returns)\n")

    await kit.detach_channel(ROOM, "conf")


async def formats() -> None:
    """3. Participants negotiate their own formats, and nothing makes them agree."""
    print("3. Three publishers, three formats\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(
        ConferenceChannel("conf", backend=backend, stt=MockSTTProvider(transcripts=["bonjour"]))
    )
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    published = {
        "p-dial-in": MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1),
        "p-laptop": MockTrackFormat(sample_rate=16_000, channels=1, sample_width=2),
        "p-studio": MockTrackFormat(sample_rate=48_000, channels=2, sample_width=4),
    }

    for participant, audio_format in published.items():
        await backend.simulate_participant_joined(ROOM, participant)
        track = await backend.simulate_track_published(
            ROOM, participant, audio_format=audio_format
        )
        print(f"   {participant:>10}: {audio_format.describe()}")
        await say(backend, track)

    await asyncio.sleep(0.2)  # let the lanes drain
    spoke = {
        event.source.participant_id
        for event in await kit.store.list_events(ROOM)
        if getattr(event.content, "body", None) == "bonjour"
    }
    print(f"\n   transcribed: {', '.join(sorted(spoke)) or 'nobody'}")
    print("   (format normalisation runs first in the lane, so the stages see one format)\n")

    await kit.detach_channel(ROOM, "conf")


async def utterances() -> None:
    """4. What the bot published, grouped by utterance."""
    print("4. Two answers on the bot's track\n")
    backend = MockConferenceBackend()
    kit = RoomKit()
    kit.register_channel(ConferenceChannel("conf", backend=backend, tts=MockTTSProvider()))
    kit.register_channel(AISource("ai"))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "ai")

    await kit.send_event(ROOM, "ai", TextContent(body="bonjour tout le monde"))
    await kit.send_event(ROOM, "ai", TextContent(body="je vous écoute"))

    for index, utterance in enumerate(backend.utterances, start=1):
        state = "complete" if utterance.complete else "unfinished"
        print(f"   utterance {index}: {len(utterance.chunks)} chunks, {state}")
    print(
        "   (one record per utterance: two answers published concurrently would "
        "share one,\n    which is how a test sees them run together)\n"
    )

    await kit.detach_channel(ROOM, "conf")


async def slow_disk() -> None:
    """5. The storage is slow, and the conference does not wait for it."""
    print("5. A recorder that blocks blocks nothing but itself (RFC §12.10.8)\n")
    backend = MockConferenceBackend()
    recorder = SlowRecorder(seconds=0.05)
    kit = RoomKit()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        recorder=recorder,
        recording=ConferenceRecordingConfig(),
        # A deliberately small backlog, so a dozen frames are enough to show
        # what overload does. The default is 100 — about two seconds of audio.
        max_queued_frames=4,
    )
    kit.register_channel(channel)
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    await backend.simulate_participant_joined(ROOM, "p-alice")
    alice = await backend.simulate_track_published(ROOM, "p-alice")

    loop = asyncio.get_running_loop()
    started = loop.time()
    for _ in range(20):
        await backend.simulate_audio(alice, backend.frame_for(alice))
    delivered_in = (loop.time() - started) * 1000

    dropped = channel.info()["rooms"][ROOM]["recording_dropped_frames"]
    print(f"   20 frames delivered in {delivered_in:.1f} ms")
    print(f"   (each one takes the recorder {recorder.seconds * 1000:.0f} ms to write)")
    print(f"   dropped so far, oldest first: {dropped}")

    # Detaching finalizes the recording, and what is still queued is written
    # first: a container closed over frames in flight would end early and say
    # nothing about it.
    await kit.detach_channel(ROOM, "conf")
    print(f"   frames the recorder ended up with: {len(recorder.chunks)}")
    print(f"   the loop runs on thread {threading.get_ident()}")
    print(f"   the writes ran on {sorted(recorder.threads)} — never on the loop's\n")


async def main() -> None:
    await failures()
    await latency()
    await formats()
    await utterances()
    await slow_disk()


if __name__ == "__main__":
    asyncio.run(main())

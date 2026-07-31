"""Speech-to-speech in a conference — the realtime AI as a participant.

`conference_ai_meeting.py` closes the STT -> LLM -> TTS loop; this example
replaces all three with one realtime provider (RFC §12.10.12): every
participant's track is mixed N->1 and fed to a single speech-to-speech
session, and the provider's voice publishes on the bot track like any other
utterance. Three arbitrations are on display:

1. **The provider hears a mix, so attribution ends at its boundary.** Its own
   transcription of what it heard names nobody and is discarded; the
   *attributed* transcript comes from the per-track STT lanes running in
   parallel — watch the room history at the end: the humans are attributed by
   their tracks, the AI's words by its channel.
2. **The per-lane VAD stays the barge-in sensor.** The provider brings its own
   turn-taking, but *who* may interrupt the bot is the conference's policy
   (§12.10.5). When Bob talks over the answer, the latch stops the chunk
   stream, ``stop_playback`` drops what the SFU had queued, and the provider
   is told to cancel the rest of its response.
3. **One voice per bot.** ``tts=`` and ``realtime=`` are mutually exclusive —
   the provider *is* the voice, so inbound text events are injected into its
   context instead of being synthesized over it.

The mock provider generates nothing by itself, so this demo plays the
provider's half explicitly — response start, audio deltas, transcript,
response end — which is exactly the callback sequence a real one emits. A
real deployment swaps two lines and changes nothing else::

    from roomkit import LiveKitConferenceBackend, LiveKitConfig
    from roomkit.providers.gemini import GeminiLiveProvider

    backend = LiveKitConferenceBackend(LiveKitConfig(url=..., api_key=..., api_secret=...))
    realtime = ConferenceRealtimeConfig(
        provider=GeminiLiveProvider(api_key=...),
        system_prompt="You are the meeting assistant.",
    )

Run with:
    uv run python examples/conference_realtime_ai.py
"""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Any

from roomkit import (
    ConferenceRealtimeConfig,
    ConferenceTranscription,
    HookResult,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.stt.mock import MockSTTProvider

ROOM = "standup"

# Same audio arithmetic as the other conference examples: 16 kHz mono contract,
# energy VAD, one utterance = loud speech then enough silence to close it.
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE * 20 // 1000  # 20 ms
SPEECH_FRAMES = 15  # 300 ms — past the energy VAD's 200 ms minimum
SILENCE_FRAMES = 30  # 600 ms — past its 500 ms end-of-speech threshold


def speech_frame() -> AudioFrame:
    samples = [8000, -8000] * (SAMPLES_PER_FRAME // 2)
    return AudioFrame(
        data=struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples), sample_rate=SAMPLE_RATE
    )


def silence_frame() -> AudioFrame:
    return AudioFrame(data=b"\x00\x00" * SAMPLES_PER_FRAME, sample_rate=SAMPLE_RATE)


async def speak(backend: MockConferenceBackend, track: Any) -> None:
    for _ in range(SPEECH_FRAMES):
        await backend.simulate_audio(track, speech_frame())
    for _ in range(SILENCE_FRAMES):
        await backend.simulate_audio(track, silence_frame())


async def until(condition: Any, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the loop did not produce the expected result in time")
        await asyncio.sleep(0.01)


async def main() -> None:
    backend = MockConferenceBackend()
    provider = MockRealtimeProvider()
    channel = ConferenceChannel(
        "conf",
        backend=backend,
        realtime=ConferenceRealtimeConfig(
            provider=provider,
            system_prompt="You are the meeting assistant.",
            voice="verse",
        ),
        # The RFC's SHOULD, taken: per-track STT lanes beside the mix are what
        # keep the transcript attributed once the provider boundary erases it.
        stt=MockSTTProvider(
            transcripts=[
                "What's on the agenda today?",
                "Sorry to cut in — one quick question first!",
            ]
        ),
    )

    kit = RoomKit()
    kit.register_channel(channel)

    interruptions: list[str] = []

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(payload: ConferenceTranscription, ctx: Any) -> HookResult:
        print(f"      [stt] {payload.participant_id}: {payload.text}")
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_BARGE_IN, execution=HookExecution.ASYNC)
    async def on_barge_in(payload: Any, ctx: Any) -> None:
        interruptions.append(payload.participant_id)
        print(f"      [barge-in] {payload.participant_id} cut the bot off")

    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")

    await backend.simulate_participant_joined(ROOM, "alice", display_name="Alice")
    await backend.simulate_participant_joined(ROOM, "bob", display_name="Bob")
    alice_mic = await backend.simulate_track_published(ROOM, "alice")
    bob_mic = await backend.simulate_track_published(ROOM, "bob")

    print("alice speaks; the mix reaches the provider and the session comes up lazily:")
    await speak(backend, alice_mic)
    await until(lambda: provider.sent_audio)
    session = channel._realtime.session_for(ROOM)
    assert session is not None
    print(f"      [mix] {len(provider.sent_audio)} mixed windows sent to the provider")

    print("\nthe provider answers — its voice is a bot-track utterance like any other:")
    await provider.simulate_response_start(session)
    await provider.simulate_audio(session, b"<24kHz pcm of the answer>")
    await provider.simulate_transcription(
        session, "Two items today: the release review, and this demo.", role="assistant"
    )
    await provider.simulate_response_end(session)
    bot = backend.bots[0]
    await until(lambda: any(u.complete for u in backend.utterances_for(bot)))
    print("      [voice] utterance published and closed on is_final")

    print("\nbob talks over the next answer — policy, latch, stop, cancel:")
    await provider.simulate_response_start(session)
    await provider.simulate_audio(session, b"<a much longer answer.........>")
    await until(lambda: len(backend.published_audio) > 1)
    await speak(backend, bob_mic)
    await until(lambda: backend.playback_stops)
    await until(lambda: any(c.method == "interrupt" for c in provider.calls))
    print("      [voice] stop_playback dropped the queued audio; the provider was cancelled")

    # The room's record: humans attributed by their tracks (the STT lanes),
    # the AI's words by its channel — and the provider's *user-side*
    # transcription of the mix is nowhere, because it names nobody.
    print("\nwhat the room recorded:")
    for event in await kit.store.list_events(ROOM):
        body = getattr(event.content, "body", None)
        who = event.source.participant_id or event.source.channel_id
        if body and who != "system":
            print(f"      [{who}] {body}")

    info = channel.info()
    room_info = info["rooms"][ROOM]
    print(
        f"\ndisclosure: realtime_configured={info['realtime_configured']}"
        f" provider={info['realtime_provider']}"
        f" realtime_active={room_info['realtime_active']}"
        f" interruptions={interruptions}"
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

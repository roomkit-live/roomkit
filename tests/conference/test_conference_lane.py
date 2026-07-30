"""Per-track audio lanes (RFC §12.10.4, §12.10.5).

A lane turns a stream of frames into utterances. These tests hold the two
properties that make it worth having and that are easy to lose silently:

- One utterance produces one transcription event, not one per 20 ms frame.
  A lane without segmentation still passes any test that only checks an event
  arrived, so the frame-to-event ratio is asserted explicitly.
- No track's processing delays another's. The backend awaits its subscribers
  in sequence, so any work done in the frame callback becomes every
  participant's latency.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import (
    ConferenceInterruptionConfig,
    ConferenceInterruptionScope,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels._conference_lane import ConferenceBargeIn
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.event import AudioContent, EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.interruption import InterruptionStrategy
from roomkit.voice.pipeline.agc.mock import MockAGCProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig, AudioPipelineContract
from roomkit.voice.pipeline.denoiser.mock import MockDenoiserProvider
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider
from roomkit.voice.pipeline.vad.mock import MockVADProvider
from roomkit.voice.stt.base import STTProvider
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.base import TTSProvider
from tests.conference.lane_audio import SAMPLE_RATE, drain, say, speech_frame

ROOM = "room-1"


class _Source(Channel):
    """A channel that only exists to originate AI events."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.AI

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _SlowSTT(STTProvider):
    """A recognizer that takes its time, the way a network one does."""

    def __init__(self, delay: float, transcripts: list[str]) -> None:
        self._delay = delay
        self._transcripts = transcripts
        self._index = 0

    async def transcribe(self, audio: Any) -> TranscriptionResult:
        await asyncio.sleep(self._delay)
        text = self._transcripts[self._index % len(self._transcripts)]
        self._index += 1
        return TranscriptionResult(text=text)


class _GatedTTS(TTSProvider):
    """Publishes one chunk, then waits to be released.

    Lets a test interleave a participant speaking with the bot speaking,
    without leaning on timing.
    """

    def __init__(self, chunks: int = 8) -> None:
        self._chunks = chunks
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    @property
    def default_voice(self) -> str:
        return "gated"

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        raise NotImplementedError

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append(text)
        for i in range(self._chunks):
            yield AudioChunk(
                data=b"\x00\x00", sample_rate=SAMPLE_RATE, is_final=(i == self._chunks - 1)
            )
            if i == 0:
                self.started.set()
                await self.release.wait()


_OPEN: list[ConferenceChannel] = []


@pytest.fixture(autouse=True)
async def _close_channels() -> AsyncIterator[None]:
    """Lanes own tasks; a test that leaves one running leaks it into the next."""
    _OPEN.clear()
    yield
    for channel in _OPEN:
        await channel.close()
    _OPEN.clear()


async def _kit(**channel_kwargs: Any) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    channel_kwargs.setdefault("stt", MockSTTProvider(transcripts=["bonjour"]))
    channel = ConferenceChannel("conf", backend=backend, **channel_kwargs)
    _OPEN.append(channel)
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src"))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    return kit, channel, backend


def _spoken(events: list[RoomEvent], body: str) -> list[RoomEvent]:
    return [e for e in events if getattr(e.content, "body", None) == body]


class TestSegmentation:
    async def test_one_utterance_becomes_one_event_not_one_per_frame(self) -> None:
        """An event per utterance, not per frame.

        The ratio is what carries the assertion: a lane that emitted one event
        per frame would satisfy any check that merely looks for an event.
        """
        kit, channel, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        frames = await say(backend, track)
        await drain(channel, track.id)

        spoken = _spoken(await kit.store.list_events(ROOM), "bonjour")
        assert len(spoken) == 1
        assert frames >= 10 * len(spoken)
        assert spoken[0].source.participant_id == "p-alice"
        assert spoken[0].metadata["conference_track_id"] == track.id

    async def test_silence_alone_produces_nothing(self) -> None:
        kit, channel, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, track, speech=0, silence=40)
        await drain(channel, track.id)

        assert _spoken(await kit.store.list_events(ROOM), "bonjour") == []

    async def test_two_speakers_produce_distinct_attributed_utterances(self) -> None:
        kit, channel, backend = await _kit(
            stt=MockSTTProvider(transcripts=["c'est Alice", "c'est Bob"])
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        await say(backend, alice)
        await drain(channel, alice.id)
        await say(backend, bob)
        await drain(channel, bob.id)

        events = await kit.store.list_events(ROOM)
        assert [
            (e.source.participant_id, e.content.body)
            for e in events
            if getattr(e.content, "body", None) in ("c'est Alice", "c'est Bob")
        ] == [("p-alice", "c'est Alice"), ("p-bob", "c'est Bob")]


class TestStageOrder:
    async def test_stages_run_in_the_canonical_order(self) -> None:
        """RFC §12.3: resampler, then AGC, then denoiser, then VAD.

        Each stage records what the previous ones left on the frame, which
        pins the order causally rather than by observation sequence.
        """
        order: list[str] = []
        seen_keys: dict[str, set[str]] = {}

        class _AGC(MockAGCProvider):
            def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
                order.append("agc")
                seen_keys["agc"] = set(frame.metadata)
                return super().process(frame, stream)

        class _Denoiser(MockDenoiserProvider):
            def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
                order.append("denoiser")
                seen_keys["denoiser"] = set(frame.metadata)
                return super().process(frame, stream)

        class _VAD(EnergyVADProvider):
            def process(self, frame: AudioFrame, stream: str) -> Any:
                order.append("vad")
                seen_keys["vad"] = set(frame.metadata)
                return super().process(frame, stream)

        _, channel, backend = await _kit(
            pipeline=AudioPipelineConfig(
                agc=_AGC(), denoiser=_Denoiser(), vad=_VAD(), contract=AudioPipelineContract()
            )
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(track, speech_frame())
        await drain(channel, track.id)

        assert order == ["agc", "denoiser", "vad"]
        # The resampler ran first: it is what stamps the original format.
        assert "original_sample_rate" in seen_keys["agc"]
        assert "agc" in seen_keys["denoiser"]
        assert {"agc", "denoiser"} <= seen_keys["vad"]

    async def test_neither_aec_nor_diarization_is_required(self) -> None:
        """RFC §12.10.4: no server-side echo path, and track identity already
        attributes speech, so a lane must work without either.
        """
        config = AudioPipelineConfig(vad=EnergyVADProvider(), contract=AudioPipelineContract())
        assert config.aec is None
        assert config.diarization is None

        kit, channel, backend = await _kit(pipeline=config)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await drain(channel, track.id)

        assert len(_spoken(await kit.store.list_events(ROOM), "bonjour")) == 1


class TestLaneIsolation:
    async def test_a_slow_recognizer_on_one_track_does_not_delay_another(self) -> None:
        """The backend awaits its subscribers in sequence. Transcribing inside
        that callback made one provider's latency everyone's latency.
        """
        delay = 0.2
        kit, channel, backend = await _kit(stt=_SlowSTT(delay, ["alice", "bob"]))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-bob")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        bob = await backend.simulate_track_published(ROOM, "p-bob")

        # Alice completes an utterance; her lane is now inside the recognizer.
        await say(backend, alice)
        await asyncio.sleep(0)

        started = time.perf_counter()
        await say(backend, bob)
        delivery = time.perf_counter() - started

        assert delivery < delay / 2, "frame delivery waited on another track's recognizer"

        await drain(channel, alice.id, bob.id)
        bodies = {
            e.content.body for e in await kit.store.list_events(ROOM) if hasattr(e.content, "body")
        }
        assert {"alice", "bob"} <= bodies


class TestBackpressure:
    async def test_a_full_queue_drops_the_oldest_frames_and_keeps_going(self) -> None:
        """A lane that falls behind stays close to live rather than growing an
        unbounded delay. The drops are counted so they are not silent.
        """
        _, channel, backend = await _kit(max_queued_frames=2)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        lane = channel.active_lanes[track.id]

        # Nothing awaits between deliveries, so the lane task never gets a turn
        # and the queue fills.
        for _ in range(10):
            await backend.simulate_audio(track, speech_frame())

        assert lane.dropped_frames >= 8

        await lane.drain()
        await backend.simulate_audio(track, speech_frame())
        await lane.drain()
        assert lane.dropped_frames >= 8  # still working, no further pile-up


class TestInterruption:
    async def _speak(self, kit: RoomKit) -> asyncio.Task[Any]:
        return asyncio.create_task(kit.send_event(ROOM, "src", TextContent(body="une réponse")))

    async def test_any_scope_lets_any_speaker_cut_the_bot(self) -> None:
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.ANY),
        )
        seen: list[ConferenceBargeIn] = []

        @kit.hook(HookTrigger.ON_BARGE_IN)
        async def _barge(payload: Any, ctx: Any) -> None:
            seen.append(payload)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = await self._speak(kit)
        await tts.started.wait()

        await say(backend, track, silence=0)
        await drain(channel, track.id)
        tts.release.set()
        await speaking

        spoken = [chunk for chunk in backend.published_audio if chunk.data]
        assert len(spoken) == 1, "the bot kept speaking after the interruption"
        assert [(b.participant_id, b.track_id) for b in seen] == [("p-alice", track.id)]

    async def test_being_cut_off_still_ends_the_utterance_for_the_backend(self) -> None:
        """A barge-in used to leave the loop without publishing anything: no
        ``is_final``, no cancellation, so the SFU went on believing the bot was
        mid-sentence. An empty final chunk is the whole message, and it is
        already what the ABC's ``is_final`` is for (RFC §12.10.3).
        """
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.ANY),
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = await self._speak(kit)
        await tts.started.wait()

        await say(backend, track, silence=0)
        await drain(channel, track.id)
        tts.release.set()
        await speaking

        assert backend.published_audio[-1].is_final is True
        assert backend.published_audio[-1].data == b""
        assert backend.published_audio[-1].sample_rate == SAMPLE_RATE
        assert backend.utterances[-1].complete is True

    async def test_being_cut_off_silences_the_answers_queued_behind(self) -> None:
        """Speaking over a room stops what the bot is saying in it — including
        the answer waiting its turn. Letting the queue drain into the silence
        someone just asked for is not what taking the floor means.
        """
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.ANY),
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = await self._speak(kit)
        await tts.started.wait()
        queued = asyncio.create_task(channel._voice.speak(ROOM, "et ensuite"))
        while len(channel._voice._speaking[ROOM].playbacks) < 2:
            await asyncio.sleep(0)

        await say(backend, track, silence=0)
        await drain(channel, track.id)
        tts.release.set()
        await asyncio.gather(speaking, queued)

        assert tts.calls == ["une réponse"], "the queued answer spoke over the interruption"

    async def test_none_scope_lets_the_bot_finish(self) -> None:
        """Presentation and IVR style: nobody talks over the bot."""
        tts = _GatedTTS(chunks=8)
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.NONE),
        )
        seen: list[ConferenceBargeIn] = []

        @kit.hook(HookTrigger.ON_BARGE_IN)
        async def _barge(payload: Any, ctx: Any) -> None:
            seen.append(payload)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = await self._speak(kit)
        await tts.started.wait()

        await say(backend, track, silence=0)
        await drain(channel, track.id)
        tts.release.set()
        await speaking

        assert len(backend.published_audio) == 8
        assert seen == []

    async def test_allowlist_scope_admits_only_the_listed_speakers(self) -> None:
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(
                scope=ConferenceInterruptionScope.ALLOWLIST, allowlist=["p-moderator"]
            ),
        )
        seen: list[ConferenceBargeIn] = []

        @kit.hook(HookTrigger.ON_BARGE_IN)
        async def _barge(payload: Any, ctx: Any) -> None:
            seen.append(payload)

        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_participant_joined(ROOM, "p-moderator")
        alice = await backend.simulate_track_published(ROOM, "p-alice")
        moderator = await backend.simulate_track_published(ROOM, "p-moderator")

        speaking = await self._speak(kit)
        await tts.started.wait()

        await say(backend, alice, silence=0)
        await drain(channel, alice.id)
        assert seen == [], "an unlisted participant interrupted the bot"

        await say(backend, moderator, silence=0)
        await drain(channel, moderator.id)
        tts.release.set()
        await speaking

        assert [b.participant_id for b in seen] == ["p-moderator"]
        assert len([chunk for chunk in backend.published_audio if chunk.data]) == 1

    async def test_speech_while_the_bot_is_silent_interrupts_nothing(self) -> None:
        """An answer the room holds is not an answer the room is saying.

        Between two utterances, and for as long as BEFORE_TTS is deciding, a
        room owns a playback that has published nothing. There is no speech to
        talk over in that gap, so ordinary conversation there must not cut the
        answer off — nor report a barge-in against words nobody has heard.
        """
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.ANY),
        )
        deciding, decided = asyncio.Event(), asyncio.Event()
        seen: list[ConferenceBargeIn] = []

        @kit.hook(HookTrigger.ON_BARGE_IN)
        async def _barge(payload: Any, ctx: Any) -> None:
            seen.append(payload)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def _hold(text: object, ctx: object) -> HookResult:
            deciding.set()
            await decided.wait()
            return HookResult.allow()

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        speaking = asyncio.create_task(channel._voice.speak(ROOM, "une réponse"))
        await asyncio.wait_for(deciding.wait(), timeout=5.0)

        await say(backend, track, silence=0)
        await drain(channel, track.id)
        decided.set()
        tts.release.set()
        await speaking

        assert seen == [], "a barge-in was reported against an utterance nobody heard"
        assert len(backend.published_audio) == 8, "the answer was cut off before it spoke"

    async def test_the_bot_never_interrupts_itself(self) -> None:
        """Bot self-exclusion means the bot's own track gets no lane, so its
        own speech cannot reach the interruption path.
        """
        tts = _GatedTTS()
        kit, channel, backend = await _kit(
            tts=tts,
            interruption=ConferenceInterruptionConfig(scope=ConferenceInterruptionScope.ANY),
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")

        bot_track = await backend.simulate_bot_echo(backend.bots[0])

        assert bot_track.id not in channel.active_lanes

        speaking = await self._speak(kit)
        await tts.started.wait()
        tts.release.set()
        await speaking

        assert len(backend.published_audio) == 8


class TestConfiguration:
    async def test_a_channel_without_speech_recognition_opens_no_lane(self) -> None:
        """Nothing else consumes an audio track, so there is nothing to run."""
        _, channel, backend = await _kit(stt=None)
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await backend.simulate_track_published(ROOM, "p-alice")

        assert channel.active_lanes == {}

    async def test_a_pipeline_without_a_contract_still_normalises_the_format(self) -> None:
        """RFC §12.10.4: format normalisation runs before the other stages.

        Participants negotiate their own formats with the SFU, so a 48 kHz
        track and a 16 kHz one can share a conference while every stage
        downstream assumes one format. Unlike the VAD this has an obvious
        default, so a config without a contract gets one rather than an error.
        """
        agc = MockAGCProvider()
        _, channel, backend = await _kit(
            pipeline=AudioPipelineConfig(agc=agc, vad=EnergyVADProvider())
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await backend.simulate_audio(track, AudioFrame(data=b"\x00\x00" * 960, sample_rate=48_000))
        await drain(channel, track.id)

        assert [f.sample_rate for f in agc.frames] == [16_000]

    async def test_a_pipeline_without_a_vad_is_refused(self) -> None:
        with pytest.raises(ValueError, match="requires a VAD"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                stt=MockSTTProvider(),
                pipeline=AudioPipelineConfig(denoiser=MockDenoiserProvider()),
            )

    async def test_semantic_interruption_is_refused(self) -> None:
        """It needs the transcript, which only exists once the utterance has
        ended — too late to interrupt anything.
        """
        with pytest.raises(ValueError, match="SEMANTIC"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                interruption=ConferenceInterruptionConfig(strategy=InterruptionStrategy.SEMANTIC),
            )


class TestLaneLifecycle:
    async def test_unpublishing_a_track_releases_its_stage_state(self) -> None:
        """Stage state is keyed by stream and some of it is native memory: a
        lane that ends without releasing leaks per track for the room's life.
        """
        vad = MockVADProvider()
        _, channel, backend = await _kit(
            pipeline=AudioPipelineConfig(vad=vad, contract=AudioPipelineContract())
        )
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await backend.simulate_audio(track, speech_frame())
        await drain(channel, track.id)

        await backend.simulate_track_unpublished(track.id)

        assert track.id not in channel.active_lanes
        assert vad.reset_count >= 1

    async def test_detaching_the_channel_stops_the_lanes(self) -> None:
        kit, channel, backend = await _kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        lane = channel.active_lanes[track.id]

        await kit.detach_channel(ROOM, "conf")

        assert channel.active_lanes == {}
        assert not lane.running

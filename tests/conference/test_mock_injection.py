"""What MockConferenceBackend can be made to do badly (RFC §12.10.3, §12.10.4).

The mock's own suite proves it behaves. This one proves it can be made to
misbehave, which is what the conference channel's failure paths need in order
to be testable at all: an SFU that refuses the bot, a control call that hangs,
a dial-in publishing 8 kHz 8-bit mono, two answers talking over each other on
the bot's track.

Each lever is covered here so the cards that use it can spend their tests on the
defect rather than on the scaffolding.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    ConferenceCapability,
    ConferenceGrants,
    MockConferenceBackend,
    MockTrackFormat,
)
from roomkit.conference import INJECTABLE_EMISSIONS, INJECTABLE_METHODS
from roomkit.conference.base import ConferenceBackend
from roomkit.conference.models import BotSession, ConferenceTrack
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

ROOM = "room-1"

SLOW = 0.03
"""Long enough to measure, short enough not to slow the suite down. Every
assertion on it is a lower bound: a sleep never returns early, so a loaded
machine cannot make one flaky."""


async def _joined_bot(backend: MockConferenceBackend) -> BotSession:
    await backend.ensure_room(ROOM)
    return await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())


async def _subscribed_track(
    backend: MockConferenceBackend, bot: BotSession, **kwargs: Any
) -> ConferenceTrack:
    track = await backend.simulate_track_published(ROOM, "p-alice", **kwargs)
    await backend.subscribe_track(bot, track.id)
    return track


def _invocations(
    backend: MockConferenceBackend, bot: BotSession, track: ConferenceTrack
) -> dict[str, Callable[[], Awaitable[Any]]]:
    """One plausible call per injectable method, so every one can be checked."""
    return {
        "ensure_room": lambda: backend.ensure_room(ROOM),
        "close_room": lambda: backend.close_room(ROOM),
        "mint_access": lambda: backend.mint_access(ROOM, "p-alice", ConferenceGrants()),
        "list_participants": lambda: backend.list_participants(ROOM),
        "remove_participant": lambda: backend.remove_participant(ROOM, "p-alice"),
        "mute_track": lambda: backend.mute_track(ROOM, track.id),
        "unmute_track": lambda: backend.unmute_track(ROOM, track.id),
        "join_as_bot": lambda: backend.join_as_bot(ROOM, "roomkit", ConferenceGrants()),
        "leave": lambda: backend.leave(bot),
        "subscribe_track": lambda: backend.subscribe_track(bot, track.id),
        "unsubscribe_track": lambda: backend.unsubscribe_track(bot, track.id),
        "publish_audio": lambda: backend.publish_audio(bot, AudioChunk(data=b"\x00\x00")),
        "stop_playback": lambda: backend.stop_playback(bot),
        "publish_video": lambda: backend.publish_video(
            bot, VideoFrame(data=b"", codec="raw_rgb24")
        ),
        "close": lambda: backend.close(),
    }


async def _elapsed(coro: Awaitable[Any]) -> float:
    loop = asyncio.get_running_loop()
    started = loop.time()
    await coro
    return loop.time() - started


class TestTheLeversCoverTheInterface:
    """A name the registries missed is a call nothing can inject into, and
    nothing would say so: the lever would simply refuse the name the day
    someone reached for it. So the registries are checked against the interface
    rather than maintained by hand.
    """

    def test_every_backend_call_is_injectable(self) -> None:
        calls = {
            name
            for name in ConferenceBackend.__abstractmethods__
            if inspect.iscoroutinefunction(getattr(ConferenceBackend, name))
        }

        assert calls == INJECTABLE_METHODS

    def test_every_emission_is_injectable(self) -> None:
        emitters = {
            name.removeprefix("_emit_")
            for name in vars(ConferenceBackend)
            if name.startswith("_emit_")
        }

        assert emitters == INJECTABLE_EMISSIONS


class TestFailureInjection:
    @pytest.mark.parametrize("method", sorted(INJECTABLE_METHODS))
    async def test_every_method_can_be_made_to_fail(self, method: str) -> None:
        """No method is exempt. A gap here is a failure path the channel's own
        tests could not reach, which is how the defect got in.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        track = await backend.simulate_track_published(ROOM, "p-alice")
        calls = _invocations(backend, bot, track)
        assert set(calls) == INJECTABLE_METHODS

        backend.fail(method, RuntimeError("SFU refused"))

        with pytest.raises(RuntimeError, match="SFU refused"):
            await calls[method]()

    async def test_the_attempt_is_recorded_before_it_fails(self) -> None:
        """The request did go out. A test asserting the channel tried to join
        needs the trace of the attempt, not just the exception.
        """
        backend = MockConferenceBackend()
        backend.fail("join_as_bot")

        with pytest.raises(RuntimeError):
            await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())

        assert backend.calls[-1].method == "join_as_bot"
        assert backend.calls[-1].args["identity"] == "roomkit"
        assert backend.bots == []

    async def test_the_fault_lands_before_capability_gating(self) -> None:
        """A backend that cannot reach its SFU fails there, not on a rule the
        framework applies afterwards.
        """
        backend = MockConferenceBackend(capabilities=ConferenceCapability.REMOTE_UNMUTE)
        backend.fail("unmute_track", TimeoutError)

        with pytest.raises(TimeoutError):
            await backend.unmute_track(ROOM, "tr-1")

    async def test_times_retires_the_fault(self) -> None:
        """One failure then success is the shape a retry — or a second teardown
        after a first that died — has to be tested against.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        backend.fail("leave", RuntimeError("transient"), times=1)

        with pytest.raises(RuntimeError, match="transient"):
            await backend.leave(bot)
        await backend.leave(bot)

        assert backend.bots == []

    async def test_an_error_may_be_an_instance_a_class_or_a_factory(self) -> None:
        backend = MockConferenceBackend()

        backend.fail("close_room", ValueError("instance"))
        with pytest.raises(ValueError, match="instance"):
            await backend.close_room(ROOM)

        backend.fail("close_room", TimeoutError)
        with pytest.raises(TimeoutError):
            await backend.close_room(ROOM)

        backend.fail("close_room", lambda: KeyError("built"))
        with pytest.raises(KeyError):
            await backend.close_room(ROOM)

    async def test_the_default_error_names_the_operation(self) -> None:
        backend = MockConferenceBackend()
        backend.fail("ensure_room")

        with pytest.raises(RuntimeError, match="ensure_room"):
            await backend.ensure_room(ROOM)

    async def test_an_unknown_method_is_refused(self) -> None:
        """A lever that silently does nothing is worse than no lever: the test
        passes believing it injected a failure, and the defect stays hidden.
        """
        backend = MockConferenceBackend()

        with pytest.raises(ValueError, match="join_bot"):
            backend.fail("join_bot")

    async def test_an_emission_cannot_be_made_to_fail(self) -> None:
        """A backend logs what its subscribers raise and carries on, so a
        failure injected into the fanout would never be observable.
        """
        backend = MockConferenceBackend()

        with pytest.raises(ValueError, match="emission"):
            backend.fail("track_audio")

    async def test_clearing_restores_the_happy_path(self) -> None:
        backend = MockConferenceBackend()
        backend.fail("ensure_room")
        backend.faults.clear("ensure_room")

        await backend.ensure_room(ROOM)

        assert ROOM in backend.rooms


class TestLatencyInjection:
    async def test_a_slow_method_takes_the_time(self) -> None:
        backend = MockConferenceBackend()
        backend.delay("list_participants", SLOW)

        assert await _elapsed(backend.list_participants(ROOM)) >= SLOW

    async def test_a_slow_call_still_fails_when_both_are_injected(self) -> None:
        """Delay first, exception second: that is the shape of a timeout, and
        an instant failure would leave no window for a test to act in.
        """
        backend = MockConferenceBackend()
        backend.delay("join_as_bot", SLOW)
        backend.fail("join_as_bot", TimeoutError)

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(TimeoutError):
            await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())

        assert loop.time() - started >= SLOW

    async def test_a_slow_emission_delays_delivery(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        track = await _subscribed_track(backend, bot)
        backend.delay("track_audio", SLOW)

        assert await _elapsed(backend.simulate_audio(track, backend.frame_for(track))) >= SLOW
        assert backend.deliveries[-1].elapsed >= SLOW

    async def test_a_slow_subscriber_is_measured_on_the_frame_it_held(self) -> None:
        """RFC §12.10.4 makes lane isolation checkable from outside — "delaying
        recognition on one track and measuring frame delivery on another" — and
        this is the measurement it needs. Subscribers are awaited in sequence,
        so a lane doing its work inline shows up as delivery time.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        slow = await _subscribed_track(backend, bot)
        quick = await backend.simulate_track_published(ROOM, "p-bob")
        await backend.subscribe_track(bot, quick.id)

        async def _hold(track, frame) -> None:
            if track.id == slow.id:
                await asyncio.sleep(SLOW)

        backend.on_track_audio(_hold)

        await backend.simulate_audio(slow, backend.frame_for(slow))
        await backend.simulate_audio(quick, backend.frame_for(quick))

        by_track = {delivery.track_id: delivery for delivery in backend.deliveries}
        assert by_track[slow.id].elapsed >= SLOW
        assert by_track[quick.id].elapsed < SLOW

    async def test_a_delivery_names_its_track_and_kind(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        track = await _subscribed_track(backend, bot)

        await backend.simulate_audio(track, backend.frame_for(track))

        delivery = backend.deliveries[-1]
        assert delivery.track_id == track.id
        assert delivery.kind is track.kind
        assert delivery.started_at > 0

    async def test_a_dropped_frame_is_not_a_delivery(self) -> None:
        """Nothing was delivered, so nothing took any time to deliver."""
        backend = MockConferenceBackend()
        await _joined_bot(backend)
        track = await backend.simulate_track_published(ROOM, "p-alice")

        assert await backend.simulate_audio(track, backend.frame_for(track)) is False
        assert backend.deliveries == []
        assert backend.dropped_frames == [track.id]

    async def test_delays_are_refused_where_they_would_mean_nothing(self) -> None:
        backend = MockConferenceBackend()

        with pytest.raises(ValueError, match="unknown operation"):
            backend.delay("track_sound", SLOW)
        with pytest.raises(ValueError, match="negative"):
            backend.delay("close", -1.0)

    async def test_every_emission_can_be_slowed(self) -> None:
        backend = MockConferenceBackend()

        for emission in INJECTABLE_EMISSIONS:
            backend.delay(emission, 0.0)


class TestHeterogeneousFormats:
    async def test_a_track_carries_the_format_its_publisher_negotiated(self) -> None:
        """Participants negotiate separately with the SFU and nothing obliges
        them to agree, so one conference can carry three formats at once.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        dial_in = await _subscribed_track(
            backend,
            bot,
            audio_format=MockTrackFormat(sample_rate=8_000, channels=1, sample_width=1),
        )
        studio = await backend.simulate_track_published(
            ROOM,
            "p-bob",
            audio_format=MockTrackFormat(sample_rate=48_000, channels=2, sample_width=4),
        )
        await backend.subscribe_track(bot, studio.id)
        seen: list[AudioFrame] = []
        backend.on_track_audio(lambda track, frame: seen.append(frame))

        await backend.simulate_audio(dial_in, backend.frame_for(dial_in))
        await backend.simulate_audio(studio, backend.frame_for(studio))

        assert (seen[0].sample_rate, seen[0].channels, seen[0].sample_width) == (8_000, 1, 1)
        assert (seen[1].sample_rate, seen[1].channels, seen[1].sample_width) == (48_000, 2, 4)

    async def test_a_synthesized_frame_is_a_valid_frame(self) -> None:
        """AudioFrame checks its own alignment, so a stereo 32-bit frame that
        constructs at all is one the pipeline can be handed.
        """
        backend = MockConferenceBackend()
        track = await backend.simulate_track_published(
            ROOM,
            "p-alice",
            audio_format=MockTrackFormat(sample_rate=24_000, channels=2, sample_width=4),
        )

        frame = backend.frame_for(track, ms=20)

        assert len(frame.data) == 24_000 * 20 // 1000 * 2 * 4

    async def test_amplitude_separates_speech_from_silence(self) -> None:
        backend = MockConferenceBackend()
        track = await backend.simulate_track_published(ROOM, "p-alice")

        assert set(backend.frame_for(track, amplitude=0.0).data) == {0}
        assert set(backend.frame_for(track).data) != {0}

    async def test_a_frame_in_another_format_cannot_arrive_on_the_track(self) -> None:
        """An SFU forwards what the publisher sent. A mock that let a 16 kHz
        frame land on an 8 kHz track would hide exactly the mismatch these
        formats exist to expose.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        track = await _subscribed_track(
            backend, bot, audio_format=MockTrackFormat(sample_rate=8_000)
        )

        with pytest.raises(ValueError, match="8000 Hz"):
            await backend.simulate_audio(track, AudioFrame(data=b"\x00\x00"))

    async def test_a_track_that_declares_nothing_stays_permissive(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)
        track = await _subscribed_track(backend, bot)

        assert await backend.simulate_audio(track, AudioFrame(data=b"\x00\x00")) is True

    async def test_24_bit_is_refused_and_says_why(self) -> None:
        """AudioFrame accepts 1, 2 or 4 bytes per sample and the resamplers map
        only int8/int16/int32, so a 24-bit publisher has no representation to
        be simulated with.
        """
        with pytest.raises(ValueError, match="24-bit"):
            MockTrackFormat(sample_width=3)

    async def test_unpublishing_forgets_the_format(self) -> None:
        backend = MockConferenceBackend()
        track = await backend.simulate_track_published(
            ROOM, "p-alice", audio_format=MockTrackFormat(sample_rate=8_000)
        )

        await backend.simulate_track_unpublished(track.id)

        assert track.id not in backend.track_formats


class TestUtteranceBoundaries:
    async def test_two_answers_in_turn_are_two_utterances(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        await backend.publish_audio(bot, AudioChunk(data=b"AA"))
        await backend.publish_audio(bot, AudioChunk(data=b"AA", is_final=True))
        await backend.publish_audio(bot, AudioChunk(data=b"BB"))
        await backend.publish_audio(bot, AudioChunk(data=b"BB", is_final=True))

        assert [utterance.data for utterance in backend.utterances] == [b"AAAA", b"BBBB"]
        assert all(utterance.complete for utterance in backend.utterances)

    async def test_two_answers_at_once_land_in_one_record(self) -> None:
        """This is the point of recording arrival order per utterance: a flat
        list of chunks cannot show that two responses ran together, and a
        single record whose contents alternate can.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        await backend.publish_audio(bot, AudioChunk(data=b"AA"))
        await backend.publish_audio(bot, AudioChunk(data=b"BB"))
        await backend.publish_audio(bot, AudioChunk(data=b"AA", is_final=True))

        assert len(backend.utterances) == 1
        assert backend.utterances[0].data == b"AABBAA"

    async def test_two_bots_publishing_at_once_are_two_utterances(self) -> None:
        """A bot is a track — one per conference room — so a channel serving two
        rooms publishes on two. Chunks alternating between them are two rooms
        talking at once, which is ordinary; a record that could not tell them
        apart would report it as one track carrying two answers, which is the
        defect these records exist to catch.
        """
        backend = MockConferenceBackend()
        first = await backend.join_as_bot("room-1", "roomkit", ConferenceGrants())
        second = await backend.join_as_bot("room-2", "roomkit", ConferenceGrants())

        await backend.publish_audio(first, AudioChunk(data=b"AA"))
        await backend.publish_audio(second, AudioChunk(data=b"BB"))
        await backend.publish_audio(first, AudioChunk(data=b"AA", is_final=True))
        await backend.publish_audio(second, AudioChunk(data=b"BB", is_final=True))

        assert [u.data for u in backend.utterances_for(first)] == [b"AAAA"]
        assert [u.data for u in backend.utterances_for(second)] == [b"BBBB"]
        assert all(u.complete for u in backend.utterances)

    async def test_an_utterance_nobody_finished_is_marked_incomplete(self) -> None:
        """A detach mid-utterance leaves one open, and a test asserting the
        bot was cut off needs to see that it never closed.
        """
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        await backend.publish_audio(bot, AudioChunk(data=b"AA"))

        assert backend.utterances[-1].complete is False

    async def test_the_flat_record_is_still_there(self) -> None:
        backend = MockConferenceBackend()
        bot = await _joined_bot(backend)

        await backend.publish_audio(bot, AudioChunk(data=b"AA", is_final=True))

        assert len(backend.published_audio) == 1
        assert backend.published_audio[0].is_final is True

"""LiveKit backend against a real SFU — the things a mock cannot prove.

Skipped unless a server is named. See the module docstring of
``roomkit.conference.livekit`` for the ``livekit.yaml`` this needs and why::

    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        -e LIVEKIT_CONFIG="$(cat livekit.yaml)" \\
        livekit/livekit-server --dev --bind 0.0.0.0

    ROOMKIT_LIVEKIT_URL=ws://127.0.0.1:7880 \\
    ROOMKIT_LIVEKIT_API_KEY=devkey ROOMKIT_LIVEKIT_API_SECRET=secret \\
        uv run pytest tests/conference/test_livekit_live.py -v

Every test here exists because the mock backend cannot stage it. The mock hands
the lane frames the test itself built, at a rate it chose, through a
subscription that took effect instantly, against grants nothing enforced. Here a
second real connection stands in for the human — publishing 48 kHz **stereo**,
which is what a browser actually sends and what RoomKit's own resampling path
had never seen — and the SFU decides what reaches whom.

What is still not proven here is that a human hears the bot: test 9 shows an
``AudioChunk`` reaching another participant's decoder, which is as far as an
automated test can carry it. A person in a room with a microphone is the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from array import array
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.conference.models import (
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.core.exceptions import ConferenceCapabilityError
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

rtc = pytest.importorskip("livekit.rtc")

URL = os.getenv("ROOMKIT_LIVEKIT_URL")
API_KEY = os.getenv("ROOMKIT_LIVEKIT_API_KEY")
API_SECRET = os.getenv("ROOMKIT_LIVEKIT_API_SECRET")

pytestmark = pytest.mark.skipif(
    not (URL and API_KEY and API_SECRET),
    reason=(
        "needs a LiveKit server: set ROOMKIT_LIVEKIT_URL, ROOMKIT_LIVEKIT_API_KEY "
        "and ROOMKIT_LIVEKIT_API_SECRET"
    ),
)

PUBLISH_RATE = 48_000
PUBLISH_CHANNELS = 2
"""Stereo on purpose: it is what a browser sends, and it is the shape RoomKit's
lane had only ever been handed by a mock that made it up."""

FRAME_MS = 10
TIMEOUT_S = 10.0
QUIET_S = 1.5
"""Long enough that an unsubscribed track would have delivered something."""


async def wait_for(predicate: Callable[[], bool], *, timeout: float = TIMEOUT_S) -> None:
    """Wait for a condition the SFU will bring about, or fail saying it did not."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"condition still false after {timeout}s")


def tone_frame(step: int) -> Any:
    """One 10 ms stereo frame of a 440 Hz tone, loud enough for a VAD to notice."""
    samples = array("h")
    per_channel = PUBLISH_RATE * FRAME_MS // 1000
    for index in range(per_channel):
        position = step * per_channel + index
        value = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * position / PUBLISH_RATE))
        samples.extend([value] * PUBLISH_CHANNELS)
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=PUBLISH_RATE,
        num_channels=PUBLISH_CHANNELS,
        samples_per_channel=per_channel,
    )


class Participant:
    """A second real connection, standing in for a person in the room.

    Its own ``rtc.Room``, its own token minted through the backend under test, so
    what it publishes travels the same path a browser's audio would.
    """

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.room: Any = rtc.Room()
        self.received: list[Any] = []
        self._tone: asyncio.Task[None] | None = None
        self._sink: asyncio.Task[None] | None = None
        self._source: Any | None = None

    async def join(self, backend: LiveKitConferenceBackend, room_id: str) -> None:
        access = await backend.mint_access(room_id, self.identity, ConferenceGrants())
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(access.url, access.token, rtc.RoomOptions(auto_subscribe=True))

    async def publish_tone(self) -> str:
        """Start publishing, and report the track sid the SFU gave it."""
        self._source = rtc.AudioSource(PUBLISH_RATE, PUBLISH_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track(f"{self.identity}-mic", self._source)
        publication = await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self._tone = asyncio.create_task(self._speak())
        return publication.sid

    async def _speak(self) -> None:
        step = 0
        while self._source is not None:
            await self._source.capture_frame(tone_frame(step))
            step += 1

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        self._sink = asyncio.create_task(self._listen(track))

    async def _listen(self, track: Any) -> None:
        stream = rtc.AudioStream.from_track(track=track, sample_rate=48_000, num_channels=1)
        try:
            async for event in stream:
                self.received.append(event.frame)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def leave(self) -> None:
        for task in (self._tone, self._sink):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tone = self._sink = None
        source, self._source = self._source, None
        if source is not None:
            with contextlib.suppress(Exception):
                await source.aclose()
        with contextlib.suppress(Exception):
            await self.room.disconnect()


class Observed:
    """What the backend told the framework, in the order it said it."""

    def __init__(self, backend: LiveKitConferenceBackend) -> None:
        self.joined: list[ConferenceParticipant] = []
        self.left: list[ConferenceParticipant] = []
        self.published: list[ConferenceTrack] = []
        self.unpublished: list[ConferenceTrack] = []
        self.audio: list[tuple[ConferenceTrack, AudioFrame]] = []
        self.speakers: list[str] = []
        self.order: list[str] = []
        backend.on_participant_joined(self._joined)
        backend.on_participant_left(self._left)
        backend.on_track_published(self._published)
        backend.on_track_unpublished(self._unpublished)
        backend.on_track_audio(self._audio)
        backend.on_active_speaker_changed(self._speaker)

    def _joined(self, room_id: str, participant: ConferenceParticipant) -> None:
        self.joined.append(participant)
        self.order.append(f"joined:{participant.participant_id}")

    def _left(self, room_id: str, participant: ConferenceParticipant) -> None:
        self.left.append(participant)

    def _published(self, room_id: str, track: ConferenceTrack) -> None:
        self.published.append(track)
        self.order.append(f"published:{track.participant_id}")

    def _unpublished(self, room_id: str, track: ConferenceTrack) -> None:
        self.unpublished.append(track)

    def _audio(self, track: ConferenceTrack, frame: AudioFrame) -> None:
        self.audio.append((track, frame))

    def _speaker(self, room_id: str, participant_id: str) -> None:
        self.speakers.append(participant_id)

    def audio_for(self, track_id: str) -> list[AudioFrame]:
        return [frame for track, frame in self.audio if track.id == track_id]


def make_backend(**overrides: Any) -> LiveKitConferenceBackend:
    settings: dict[str, Any] = {
        "url": URL,
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "audio_channels": PUBLISH_CHANNELS,
    }
    settings.update(overrides)
    return LiveKitConferenceBackend(LiveKitConfig(**settings))


@pytest.fixture
async def backend() -> AsyncIterator[LiveKitConferenceBackend]:
    instance = make_backend()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
async def room_id(backend: LiveKitConferenceBackend) -> AsyncIterator[str]:
    identifier = f"rmk-live-{uuid4().hex[:10]}"
    await backend.ensure_room(identifier)
    try:
        yield identifier
    finally:
        with contextlib.suppress(Exception):
            await backend.close_room(identifier)


@pytest.fixture
def observed(backend: LiveKitConferenceBackend) -> Observed:
    return Observed(backend)


@pytest.fixture
async def alice() -> AsyncIterator[Participant]:
    person = Participant("p-alice")
    try:
        yield person
    finally:
        await person.leave()


class TestControlPlane:
    async def test_a_created_room_is_empty_and_can_be_closed(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        assert await backend.list_participants(room_id) == []

        await backend.close_room(room_id)

        assert await backend.list_participants(room_id) == []

    async def test_creating_a_room_twice_is_idempotent(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        await backend.ensure_room(room_id)
        await backend.ensure_room(room_id, {"tenant": "acme"})

        assert await backend.list_participants(room_id) == []


class TestBotSession:
    async def test_the_bot_joins_and_the_server_sees_it(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        listed = await backend.list_participants(room_id)

        assert [p.participant_id for p in listed] == ["roomkit"]
        assert bot.identity == "roomkit"

    async def test_the_session_reports_when_the_sfu_says_it_joined(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """RFC section 12.10.2 asks a backend holding a better figure than
        construction time to use it, and the value must be aware. Checked against
        the wall clock because the unit LiveKit reports it in is exactly the sort
        of thing that only shows up against a real server — a millisecond field
        read as seconds lands in 1970, and the error would surface in a teardown
        as a missing announcement.
        """
        before = datetime.now(UTC)

        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        assert bot.joined_at.tzinfo is not None
        assert abs((bot.joined_at - before).total_seconds()) < 60

    async def test_leaving_takes_the_bot_out_of_the_room(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """Teardown observable from outside: the bot is gone from the list a
        human's client reads, not merely marked gone in our own state.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await wait_for(lambda: True)

        await backend.leave(bot)

        await wait_for(lambda: True, timeout=1.0)
        remaining = await backend.list_participants(room_id)
        assert "roomkit" not in [p.participant_id for p in remaining]

    async def test_a_join_the_server_refuses_leaves_nothing_running(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """A failed join is never registered, so the channel gets no handle to
        close it with — whatever it started has to be torn down on the way out
        or it runs for the life of the process.
        """
        before = len(asyncio.all_tasks())
        broken = make_backend(url="ws://127.0.0.1:1")

        with pytest.raises(Exception, match=r".*"):
            await broken.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        await asyncio.sleep(0.3)
        assert len(asyncio.all_tasks()) <= before
        await broken.close()

    async def test_an_observer_bot_is_hidden_from_the_room(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """``hidden`` is a grant the SFU enforces, and the point of asking it to."""
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.observer())
        await alice.join(backend, room_id)

        await wait_for(lambda: len(alice.room.remote_participants) >= 0, timeout=2.0)
        await asyncio.sleep(QUIET_S)

        assert "roomkit" not in alice.room.remote_participants

    async def test_a_revealed_bot_appears_to_connected_clients(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """The in-place reveal RFC §12.10.4 leans on: removing ``hidden``
        through ``update_bot_grants()`` makes the SFU announce the session to
        the clients already connected — no re-join needed. (The reverse is
        not true, which is why concealment replaces the session instead.)
        """
        appeared: list[str] = []
        alice.room.on(
            "participant_connected",
            lambda participant: appeared.append(participant.identity),
        )
        await alice.join(backend, room_id)
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.observer())
        await asyncio.sleep(QUIET_S)
        assert "roomkit" not in alice.room.remote_participants
        assert "roomkit" not in appeared

        await backend.update_bot_grants(bot, ConferenceGrants.for_bot(listens=True))

        await wait_for(lambda: "roomkit" in alice.room.remote_participants)
        assert "roomkit" in appeared


class TestPresenceAndTracks:
    async def test_a_participant_and_its_track_are_announced_in_order(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """A track arriving before its publisher would hand the roster a lane to
        open for someone it has never heard of. The serialized bridge is what
        keeps the order, and only a real SFU sends the two close enough together
        to test it.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await alice.publish_tone()

        await wait_for(lambda: bool(observed.published))

        assert observed.order.index("joined:p-alice") < observed.order.index("published:p-alice")
        assert observed.published[0].kind is TrackKind.AUDIO
        assert observed.published[0].participant_id == "p-alice"

    async def test_a_bot_that_arrives_late_still_sees_who_is_there(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """LiveKit announces arrivals, and everyone already in the room is not an
        arrival — so without the catch-up a bot joining a meeting in progress
        subscribes to nothing at all.
        """
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()

        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())

        await wait_for(lambda: bool(observed.published))
        assert [p.participant_id for p in observed.joined] == ["p-alice"]
        assert [t.id for t in observed.published] == [track_id]

    async def test_a_participant_leaving_is_announced(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await wait_for(lambda: bool(observed.joined))

        await alice.leave()

        await wait_for(lambda: bool(observed.left))
        assert observed.left[0].participant_id == "p-alice"

    async def test_a_participant_present_at_the_join_is_announced_once(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The catch-up and the arrival event overlap, and only a real server
        puts a participant in both at once. Announced twice, the roster would
        resolve one person's identity twice and open their lane against a second
        announcement of the same track.
        """
        await alice.join(backend, room_id)
        await alice.publish_tone()

        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await wait_for(lambda: bool(observed.published))
        await asyncio.sleep(QUIET_S)

        assert [p.participant_id for p in observed.joined] == ["p-alice"]
        assert len(observed.published) == 1

    async def test_a_departure_takes_its_tracks_off_the_books(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """A stale entry would answer for a track that no longer exists, and
        send a moderation call after somebody who has left.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await alice.leave()
        await wait_for(lambda: bool(observed.left))

        with pytest.raises(ValueError, match="nobody to moderate"):
            await backend.mute_track(room_id, track_id)


class TestSubscription:
    async def test_no_frames_arrive_before_the_framework_asks(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The framework's subscription set is the authoritative one, and the bot
        joins with auto-subscription off. Against a mock this is bookkeeping;
        here it is the SFU declining to forward.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await asyncio.sleep(QUIET_S)

        assert observed.audio == []

    async def test_frames_arrive_once_it_does(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.subscribe_track(bot, track_id)

        await wait_for(lambda: len(observed.audio_for(track_id)) > 5)

    async def test_frames_declare_the_format_they_arrive_in(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The reason this card exists. 48 kHz stereo, declared on every frame
        and not normalised by the transport — the lane's resampler finally meets
        audio it did not manufacture.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)

        await wait_for(lambda: len(observed.audio_for(track_id)) > 10)

        frames = observed.audio_for(track_id)
        assert {f.sample_rate for f in frames} == {PUBLISH_RATE}
        assert {f.channels for f in frames} == {PUBLISH_CHANNELS}
        assert {f.sample_width for f in frames} == {2}
        assert all(f.data for f in frames)
        assert all(len(f.data) % (f.sample_width * f.channels) == 0 for f in frames), (
            "a frame that is not whole samples would shift every one after it"
        )

    async def test_frame_timestamps_advance_with_the_audio(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)
        await wait_for(lambda: len(observed.audio_for(track_id)) > 10)

        stamps = [f.timestamp_ms for f in observed.audio_for(track_id)]

        assert stamps[0] == 0
        assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))

    async def test_unsubscribing_stops_the_frames(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))
        await backend.subscribe_track(bot, track_id)
        await wait_for(lambda: len(observed.audio_for(track_id)) > 5)

        await backend.unsubscribe_track(bot, track_id)
        await asyncio.sleep(0.3)
        settled = len(observed.audio_for(track_id))
        await asyncio.sleep(QUIET_S)

        assert len(observed.audio_for(track_id)) == settled

    async def test_subscribing_a_track_whose_publisher_left_does_not_raise(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """The race the card names: ``subscribe_track`` is genuinely asynchronous,
        and the publisher can already be gone. Ordinary in a conference, so it
        must not be an error the channel has to handle.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await alice.leave()
        await wait_for(lambda: bool(observed.unpublished) or bool(observed.left))
        await backend.subscribe_track(bot, track_id)

        await asyncio.sleep(QUIET_S)
        assert observed.audio_for(track_id) == []

    async def test_a_bot_denied_subscription_receives_nothing(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """Grant semantics enforced by the SFU rather than by us: a speak-only
        bot asks for no subscription right, and asking anyway gets nothing.
        """
        observed = Observed(backend)
        bot = await backend.join_as_bot(
            room_id, "roomkit", ConferenceGrants.for_bot(speaks=True, listens=False)
        )
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.subscribe_track(bot, track_id)
        await asyncio.sleep(QUIET_S)

        assert observed.audio_for(track_id) == []


class TestPublishing:
    async def test_the_bots_voice_reaches_another_participants_decoder(
        self, backend: LiveKitConferenceBackend, room_id: str, alice: Participant
    ) -> None:
        """As close to "audible" as an automated test reaches: PCM handed to
        ``publish_audio`` comes out of a *different* connection's Opus decoder.
        A person with a microphone is what is left, and it is not this test's.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        await alice.join(backend, room_id)
        await wait_for(lambda: "roomkit" in alice.room.remote_participants)

        for step in range(60):
            await backend.publish_audio(
                bot,
                AudioChunk(
                    data=_pcm_mono(step),
                    sample_rate=PUBLISH_RATE,
                    channels=1,
                    is_final=step == 59,
                ),
            )

        await wait_for(lambda: len(alice.received) > 5)
        assert sum(len(bytes(frame.data)) for frame in alice.received) > 0

    async def test_a_chunk_in_another_format_is_refused_mid_stream(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        """An ``rtc.AudioSource`` is fixed once published; republishing the track
        to follow a format change would drop the bot's voice out of the
        conference for as long as renegotiation takes.
        """
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        await backend.publish_audio(
            bot, AudioChunk(data=_pcm_mono(0), sample_rate=PUBLISH_RATE, channels=1)
        )

        with pytest.raises(ValueError, match="format is fixed"):
            await backend.publish_audio(
                bot, AudioChunk(data=_pcm_mono(1), sample_rate=16_000, channels=1)
            )

    async def test_publishing_after_leaving_is_refused(
        self, backend: LiveKitConferenceBackend, room_id: str
    ) -> None:
        bot = await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot(speaks=True))
        session = backend._sessions[bot.id]
        await backend.leave(bot)

        with pytest.raises(RuntimeError, match="has left"):
            await session.publish(AudioChunk(data=_pcm_mono(0), sample_rate=PUBLISH_RATE))


class TestModeration:
    async def test_muting_a_participants_track_takes_effect(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        await backend.mute_track(room_id, track_id)

        async def muted() -> bool:
            for participant in await backend.list_participants(room_id):
                for track in participant.tracks:
                    if track.id == track_id:
                        return track.muted
            return False

        deadline = asyncio.get_running_loop().time() + TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            if await muted():
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("the track was never reported muted")

    async def test_unmuting_is_refused_unless_the_server_allows_it(
        self,
        backend: LiveKitConferenceBackend,
        room_id: str,
        observed: Observed,
        alice: Participant,
    ) -> None:
        """LiveKit needs ``room.enable_remote_unmute`` server-side, which is the
        asymmetry ``REMOTE_UNMUTE`` exists to surface.
        """
        await backend.join_as_bot(room_id, "roomkit", ConferenceGrants.for_bot())
        await alice.join(backend, room_id)
        track_id = await alice.publish_tone()
        await wait_for(lambda: bool(observed.published))

        with pytest.raises(ConferenceCapabilityError):
            await backend.unmute_track(room_id, track_id)


def _pcm_mono(step: int) -> bytes:
    """10 ms of a 440 Hz mono tone, as the framework's TTS would hand it over."""
    samples = array("h")
    per_frame = PUBLISH_RATE * FRAME_MS // 1000
    for index in range(per_frame):
        samples.append(
            int(
                0.3
                * 32767
                * math.sin(2 * math.pi * 440 * (step * per_frame + index) / PUBLISH_RATE)
            )
        )
    return samples.tobytes()

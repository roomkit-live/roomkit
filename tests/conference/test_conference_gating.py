"""Binding gating and TTS hooks on the conference channel.

A binding the integrator has closed must stop the channel *collecting*, not
merely stop it writing: frames that keep arriving are still decoded, still sent
to speech recognition, and still cost what they cost. And the bot must not speak
without the hooks that hold it silent — during a pending handoff, or where the
text needs redacting — getting their say first.
"""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    ConferenceCapability,
    ConferenceGrants,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.enums import Access, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.hook import HookResult
from roomkit.models.session_event import SessionStartedEvent
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import AudioChunk
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conference.lane_audio import drain, say
from tests.conference.test_conference_outbound import _Source

ROOM = "room-1"


async def _kit(**kwargs: object) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, **kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src", ChannelType.AI))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    return kit, channel, backend


class TestCollectionGate:
    async def test_closing_the_binding_unsubscribes_the_tracks(self) -> None:
        kit, _, backend = await _kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        assert track.id in backend.subscriptions

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        assert track.id not in backend.subscriptions

    async def test_closed_binding_stops_transcription_immediately(self) -> None:
        """The unsubscribe is asynchronous, so frames already in flight would
        still arrive. The gate closes synchronously to leave no window.
        """
        kit, channel, backend = await _kit(stt=MockSTTProvider(transcripts=["bonjour"]))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await kit.set_access(ROOM, "conf", Access.NONE)
        await say(backend, track)
        await drain(channel, track.id)

        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "bonjour"] == []

    async def test_an_open_binding_does_transcribe(self) -> None:
        """The control the gating tests need: without it, a lane that never
        produced anything would satisfy every assertion above.
        """
        kit, channel, backend = await _kit(stt=MockSTTProvider(transcripts=["bonjour"]))
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, track)
        await drain(channel, track.id)

        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "bonjour"] != []

    async def test_muting_the_binding_stops_collection(self) -> None:
        kit, _, backend = await _kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await kit.mute(ROOM, "conf")
        await asyncio.sleep(0)

        assert track.id not in backend.subscriptions

    async def test_reopening_the_binding_resubscribes(self) -> None:
        kit, _, backend = await _kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        await kit.set_access(ROOM, "conf", Access.READ_WRITE)
        await asyncio.sleep(0)

        assert track.id in backend.subscriptions

    async def test_a_track_published_while_closed_is_not_subscribed(self) -> None:
        kit, _, backend = await _kit(stt=MockSTTProvider())
        await backend.simulate_participant_joined(ROOM, "p-alice")
        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)

        track = await backend.simulate_track_published(ROOM, "p-bob")

        assert track.id not in backend.subscriptions


class TestTTSHooks:
    async def test_before_tts_can_silence_the_bot(self) -> None:
        """Where orchestration holds the bot quiet during a pending handoff."""
        tts = MockTTSProvider()
        kit, _, backend = await _kit(tts=tts)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def _block(text: object, ctx: object) -> HookResult:
            return HookResult(action="block", reason="handoff pending")

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert tts.calls == []
        assert backend.published_audio == []

    async def test_before_tts_allows_by_default(self) -> None:
        tts = MockTTSProvider()
        kit, _, _ = await _kit(tts=tts)

        @kit.hook(HookTrigger.BEFORE_TTS)
        async def _observe(text: object, ctx: object) -> HookResult:
            return HookResult.allow()

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert len(tts.calls) == 1
        assert tts.calls[0]["text"] == "bonjour"

    async def test_after_tts_sees_the_final_text(self) -> None:
        kit, _, _ = await _kit(tts=MockTTSProvider())
        spoken: list[str] = []

        @kit.hook(HookTrigger.AFTER_TTS, execution=HookExecution.ASYNC)
        async def _record(text: object, ctx: object) -> None:
            spoken.append(str(text))

        await kit.send_event(ROOM, "src", TextContent(body="bonjour"))

        assert spoken == ["bonjour"]


class TestMockRejectsWhatTheInterfaceForbids:
    async def test_publish_audio_refuses_a_non_pcm_chunk(self) -> None:
        """A mock more permissive than the SFU it stands in for lets a channel
        pass its tests and fail against the real thing.
        """
        backend = MockConferenceBackend()
        bot = await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())

        with pytest.raises(ValueError, match="PCM"):
            await backend.publish_audio(bot, AudioChunk(data=b"\x00\x00", format="opus"))

    async def test_publish_video_refuses_an_encoded_frame(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.VIDEO_PUBLISH)
        bot = await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())

        with pytest.raises(ValueError, match="raw frame"):
            await backend.publish_video(bot, VideoFrame(data=b"\x00", codec="h264"))

    async def test_raw_frames_and_pcm_are_accepted(self) -> None:
        backend = MockConferenceBackend(capabilities=ConferenceCapability.VIDEO_PUBLISH)
        bot = await backend.join_as_bot(ROOM, "roomkit", ConferenceGrants())

        await backend.publish_audio(bot, AudioChunk(data=b"\x00\x00"))
        await backend.publish_video(bot, VideoFrame(data=b"\x00", codec="raw_rgb24"))

        assert len(backend.published_audio) == 1
        assert len(backend.published_video) == 1


class TestJoinRace:
    async def test_detaching_during_an_in_flight_join_leaves_no_bot(self) -> None:
        """The join holds the lock across the await, so a detach arriving
        mid-connection is serialised behind it — and the bot that comes back is
        released instead of being registered to a room nobody is attached to.
        """

        class _SlowBackend(MockConferenceBackend):
            async def join_as_bot(self, room_id, identity, grants):  # type: ignore[no-untyped-def]
                await asyncio.sleep(0.02)
                return await super().join_as_bot(room_id, identity, grants)

        backend = _SlowBackend()
        channel = ConferenceChannel("conf", backend=backend)
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")

        hooks: list[str] = []
        events: list[str] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: object, ctx: object) -> None:
            hooks.append("session_started")

        for name in ("conference_started", "conference_ended", "conference_participant_joined"):

            @kit.on(name)
            async def _record(event: object, name: str = name) -> None:
                events.append(name)

        joining = asyncio.create_task(backend.simulate_participant_joined(ROOM, "p-alice"))
        await asyncio.sleep(0.005)
        await kit.detach_channel(ROOM, "conf")
        await joining

        # No bot survives...
        assert all(room.bot is None for room in channel._rooms.values())
        assert backend.bots == []
        # ...and no lifecycle is announced for a session that never legitimately
        # existed: a started after an ended reads as a live conference to
        # anything watching, and the participant would come back with it.
        assert hooks == []
        assert events == []
        assert await kit.store.list_participants(ROOM) == []

    async def test_a_resubscription_in_flight_does_not_survive_detach(self) -> None:
        """Reopening a binding lists participants to resubscribe. If a detach
        lands while that listing is in flight, the work must be abandoned rather
        than resurrecting a subscription and a lane on a room the channel left.
        """

        class _SlowList(MockConferenceBackend):
            async def list_participants(self, room_id):  # type: ignore[no-untyped-def]
                await asyncio.sleep(0.02)
                return await super().list_participants(room_id)

        backend = _SlowList()
        channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)
        await kit.set_access(ROOM, "conf", Access.READ_WRITE)
        await asyncio.sleep(0.005)
        await kit.detach_channel(ROOM, "conf")
        await asyncio.sleep(0.05)

        assert track.id not in backend.subscriptions
        assert track.id not in channel._lanes

    async def test_a_subscription_in_flight_on_publish_does_not_survive_detach(self) -> None:
        """The other way into a subscription is a track being published.

        The re-subscription path re-reads the room after the backend returns,
        but a track published while the channel is live takes its own route,
        and a shielded subscribe there resumes after the detach that cancelled
        it. The lane it would open owns a task nobody will cancel and stage
        state nobody will release, so the subscription has to be undone.
        """

        class _ShieldedSubscribe(MockConferenceBackend):
            async def subscribe_track(self, bot, track_id):  # type: ignore[no-untyped-def]
                await asyncio.shield(asyncio.sleep(0.05))
                return await super().subscribe_track(bot, track_id)

        backend = _ShieldedSubscribe()
        channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")

        publishing = asyncio.create_task(backend.simulate_track_published(ROOM, "p-alice"))
        await asyncio.sleep(0.01)
        await kit.detach_channel(ROOM, "conf")
        await asyncio.sleep(0.1)
        track = await publishing

        assert track.id not in backend.subscriptions
        assert track.id not in channel._lanes


class TestSessionContract:
    async def test_session_started_carries_the_shared_event(self) -> None:
        """Auto-greeting reads event.session directly, so a synthetic room
        event shaped like a session would break it silently.
        """
        kit, _, backend = await _kit()
        received: list[object] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def _started(event: object, ctx: object) -> None:
            received.append(event)

        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert len(received) == 1
        event = received[0]
        assert isinstance(event, SessionStartedEvent)
        assert event.room_id == ROOM
        assert event.channel_id == "conf"
        assert event.channel_type is ChannelType.CONFERENCE
        assert event.session is backend.bots[0]


class TestFrameworkEventEnvelope:
    async def test_conference_events_carry_room_and_channel(self) -> None:
        kit, _, backend = await _kit()
        seen: list[tuple[str, str | None, str | None]] = []

        @kit.on("conference_started")
        async def _started(event: object) -> None:
            seen.append((event.type, event.room_id, event.channel_id))  # type: ignore[attr-defined]

        await backend.simulate_participant_joined(ROOM, "p-alice")

        assert seen == [("conference_started", ROOM, "conf")]


class TestTranscriptionFailsClosed:
    async def test_a_raising_hook_drops_the_text(self) -> None:
        """A redaction hook that fails must not let the unredacted transcript
        into the room — the usual log-and-continue would do exactly that.
        """
        kit, channel, backend = await _kit(stt=MockSTTProvider(transcripts=["numéro 4111"]))

        @kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def _boom(payload: object, ctx: object) -> object:
            raise RuntimeError("redaction backend down")

        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")
        await say(backend, track)
        await drain(channel, track.id)

        events = await kit.store.list_events(ROOM)
        assert [e for e in events if getattr(e.content, "body", None) == "numéro 4111"] == []


class TestSubscriptionSurvivesNothing:
    async def test_a_shielded_subscribe_is_undone_after_detach(self) -> None:
        """Cancellation is not a guarantee: an SDK may shield its network call,
        so the task resumes after the detach that cancelled it. The subscription
        it created must be undone rather than left streaming to a bot nobody
        reads.
        """

        class _Shielded(MockConferenceBackend):
            async def subscribe_track(self, bot, track_id):  # type: ignore[no-untyped-def]
                await asyncio.shield(asyncio.sleep(0.02))
                return await super().subscribe_track(bot, track_id)

        backend = _Shielded()
        channel = ConferenceChannel("conf", backend=backend, stt=MockSTTProvider())
        kit = RoomKit()
        kit.register_channel(channel)
        await kit.create_room(ROOM)
        await kit.attach_channel(ROOM, "conf")
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await kit.set_access(ROOM, "conf", Access.NONE)
        await asyncio.sleep(0)
        await kit.set_access(ROOM, "conf", Access.READ_WRITE)
        await asyncio.sleep(0.005)
        await kit.detach_channel(ROOM, "conf")
        await asyncio.sleep(0.08)

        assert track.id not in backend.subscriptions
        assert track.id not in channel._lanes

"""A door into a room's media reads the room with the caller's scope (RFC §17.2).

``kit.join()`` read the room unscoped, and a realtime channel's
``start_session()`` and a conference's ``mint_access()`` did not read it at
all: a host acting for one organization that knew a room id of another could
join a session to it, its audio entering that room and its track declared to
that room's recordings, or mint a credential into its conference (RMK-475).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from roomkit import RoomKit, VideoChannel, VoiceChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import (
    MockRealtimeAudioVideoProvider,
    MockRealtimeProvider,
    MockRealtimeTransport,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

Door = Callable[[str | None], Awaitable[object]]


class _AutoConnectBackend(MockVoiceBackend):
    """A backend whose sessions start when its channel is attached, once armed."""

    armed = False

    @property
    def auto_connect(self) -> bool:
        return self.armed


@dataclass
class Bench:
    """Tenant A's room ``r1``, recording, with one channel attached."""

    kit: RoomKit
    channel: Any
    recorder: MockMediaRecorder
    door: Door = field(init=False)
    started: list[FrameworkEvent] = field(default_factory=list)

    def sessions_started(self) -> set[str]:
        """The sessions announced in ``r1`` (a door may announce one twice:
        as voice and as video)."""
        return {e.data["session_id"] for e in self.started if e.room_id == "r1"}

    def sessions_bound(self) -> list[str]:
        """The sessions the channel holds for ``r1``, read from its own books."""
        if isinstance(self.channel, RealtimeVoiceChannel):
            return [s.id for s in self.channel.get_room_sessions("r1")]
        return [
            sid for sid, (room_id, _) in self.channel._session_bindings.items() if room_id == "r1"
        ]


async def _bench(channel: Any) -> Bench:
    kit = RoomKit()
    recorder = MockMediaRecorder()
    bench = Bench(kit, channel, recorder)

    @kit.on("voice_session_started")
    async def on_voice(event: FrameworkEvent) -> None:
        bench.started.append(event)

    @kit.on("video_session_started")
    async def on_video(event: FrameworkEvent) -> None:
        bench.started.append(event)

    kit.register_channel(channel)
    await kit.create_room(
        room_id="r1",
        organization_id="tenant-a",
        recorders=[RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())],
    )
    await kit.attach_channel("r1", channel.channel_id)
    return bench


def _voice_channel(backend: MockVoiceBackend | None = None) -> VoiceChannel:
    return VoiceChannel(
        "voice",
        stt=MockSTTProvider(),
        tts=MockTTSProvider(),
        backend=backend or MockVoiceBackend(),
    )


def _realtime(cls: type[RealtimeVoiceChannel], provider: Any) -> RealtimeVoiceChannel:
    return cls(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        recording=ChannelRecordingConfig(audio=True),
    )


async def _join_voice_pull() -> Bench:
    bench = await _bench(_voice_channel())
    bench.door = lambda org: bench.kit.join("r1", "voice", organization_id=org)
    return bench


async def _join_voice_push() -> Bench:
    """The SIP shape: the backend accepted the call, the host binds it."""
    bench = await _bench(_voice_channel())

    def door(org: str | None) -> Awaitable[object]:
        session = VoiceSession(
            id=f"sip-{org}",
            room_id="r1",
            participant_id="caller",
            channel_id="voice",
            state=VoiceSessionState.ACTIVE,
        )
        return bench.kit.join("r1", "voice", session=session, organization_id=org)

    bench.door = door
    return bench


async def _attach_auto_connect() -> Bench:
    """A backend that starts a session on attach: the attach is the door."""
    backend = _AutoConnectBackend()
    bench = await _bench(_voice_channel(backend))
    backend.armed = True
    bench.door = lambda org: bench.kit.attach_channel("r1", "voice", organization_id=org)
    return bench


async def _join_video() -> Bench:
    bench = await _bench(VideoChannel("video", backend=MockVideoBackend()))
    bench.door = lambda org: bench.kit.join("r1", "video", organization_id=org)
    return bench


async def _join_realtime() -> Bench:
    bench = await _bench(_realtime(RealtimeVoiceChannel, MockRealtimeProvider()))
    bench.door = lambda org: bench.kit.join("r1", "rt", connection="ws", organization_id=org)
    return bench


async def _inbound_realtime_session() -> Bench:
    """A stateful channel's session carried by an inbound message."""
    bench = await _bench(_realtime(RealtimeVoiceChannel, MockRealtimeProvider()))

    def door(org: str | None) -> Awaitable[object]:
        session = VoiceSession(
            id=f"in-{org}",
            room_id="r1",
            participant_id="u",
            channel_id="rt",
            state=VoiceSessionState.ACTIVE,
        )
        message = InboundMessage(
            channel_id="rt", sender_id="u", content=TextContent(body="hi"), session=session
        )
        return bench.kit.process_inbound(message, room_id="r1", organization_id=org)

    bench.door = door
    return bench


async def _start_realtime() -> Bench:
    channel = _realtime(RealtimeVoiceChannel, MockRealtimeProvider())
    bench = await _bench(channel)
    bench.door = lambda org: channel.start_session("r1", "user", "ws", organization_id=org)
    return bench


async def _start_realtime_av() -> Bench:
    channel = _realtime(RealtimeAudioVideoChannel, MockRealtimeAudioVideoProvider())
    bench = await _bench(channel)
    bench.door = lambda org: channel.start_session("r1", "user", "ws", organization_id=org)
    return bench


DOORS = pytest.mark.parametrize(
    "make",
    [
        _join_voice_pull,
        _join_voice_push,
        _attach_auto_connect,
        _join_video,
        _join_realtime,
        _inbound_realtime_session,
        _start_realtime,
        _start_realtime_av,
    ],
    ids=[
        "join-voice-pull",
        "join-voice-push",
        "attach-auto-connect",
        "join-video",
        "join-realtime",
        "inbound-realtime-session",
        "start-realtime",
        "start-realtime-av",
    ],
)


async def _until(predicate: Callable[[], bool]) -> None:
    """Wait for what a bind schedules (its session-started events) to land."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


@DOORS
async def test_another_organizations_room_is_not_found_and_nothing_joins_it(
    make: Callable[[], Awaitable[Bench]],
) -> None:
    bench = await make()

    with pytest.raises(RoomNotFoundError):
        await bench.door("tenant-b")

    # Refused before any bind: nothing was scheduled that could land later.
    assert bench.sessions_bound() == []
    assert bench.sessions_started() == set()
    assert bench.recorder.tracks == []
    await bench.kit.close()


@DOORS
@pytest.mark.parametrize("org", ["tenant-a", None], ids=["same-organization", "unscoped"])
async def test_the_rooms_organization_and_an_unscoped_call_join(
    make: Callable[[], Awaitable[Bench]], org: str | None
) -> None:
    bench = await make()

    await bench.door(org)
    await _until(lambda: len(bench.sessions_started()) == 1)

    assert len(bench.sessions_bound()) == 1
    assert len(bench.recorder.tracks) == 1
    await bench.kit.close()


async def test_a_scoped_start_on_a_channel_no_framework_registered_is_not_found() -> None:
    """With no framework there is no room to read, so the scope cannot hold."""
    channel = _realtime(RealtimeVoiceChannel, MockRealtimeProvider())

    with pytest.raises(RoomNotFoundError):
        await channel.start_session("r1", "user", "ws", organization_id="tenant-a")

    assert channel.get_room_sessions("r1") == []
    await channel.close()


async def _conference() -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend]:
    """Tenant A's room ``r1`` with a conference attached and ``p-x`` a member."""
    kit = RoomKit()
    backend = MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1", organization_id="tenant-a")
    await kit.attach_channel("r1", "conf", organization_id="tenant-a")
    await kit.add_member("r1", "conf", "p-x")
    return kit, channel, backend


def _mints(backend: MockConferenceBackend) -> list[Any]:
    return [call for call in backend.calls if call.method == "mint_access"]


async def test_a_conference_credential_is_not_minted_for_another_organizations_room() -> None:
    kit, channel, backend = await _conference()

    with pytest.raises(RoomNotFoundError):
        await channel.mint_access("r1", "p-x", organization_id="tenant-b")

    assert _mints(backend) == []
    await kit.close()


@pytest.mark.parametrize("org", ["tenant-a", None], ids=["same-organization", "unscoped"])
async def test_the_rooms_organization_and_an_unscoped_call_mint(org: str | None) -> None:
    kit, channel, backend = await _conference()

    await channel.mint_access("r1", "p-x", organization_id=org)

    assert len(_mints(backend)) == 1
    await kit.close()

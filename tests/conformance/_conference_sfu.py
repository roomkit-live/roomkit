"""The conference's scripted SFU and its bot (RFC §12.10.3, §12.10.4).

``conference.read`` returns what the scripted SFU received (each call, each
utterance published on a bot's track), the bot sessions it holds with their
grants, and the channel's disclosure surface.
``conference.ensure_bot``, ``plug``, ``unplug`` and ``set_bot_grants`` drive the
channel's own operations; ``conference.bot_lost`` and ``conference.echo`` make
the SFU report the bot's end, or the bot itself, through its callbacks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from roomkit import (
    BotSession,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceRealtimeConfig,
    ConferenceRecordingConfig,
    MockConferenceBackend,
    TrackKind,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conformance._calls import UnreadableError
from tests.conformance._conference_faults import SFUFaults
from tests.conformance._voice_synthesis import scripted_tts

if TYPE_CHECKING:
    from roomkit import ConferenceChannel, RoomKit
    from roomkit.conference import MockConferenceCall
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._harness import Harness


_GRANTS = (
    "publish_audio",
    "publish_video",
    "publish_screen_share",
    "publish_screen_share_audio",
    "subscribe",
    "moderate",
    "hidden",
)
_INFO = (
    "bot_identity",
    "bot_hidden",
    "bot_grant_update_in_place",
    "stt_configured",
    "realtime_configured",
    "recording_configured",
    "vision_configured",
    "e2ee",
)
# The calls ``conference.read`` lists, the names its ``calls`` may filter on.
_CALL_NAMES = frozenset(
    {
        "ensure_room",
        "close_room",
        "list_participants",
        "join_as_bot",
        "update_bot_grants",
        "leave",
        "stop_playback",
        "subscribe_track",
        "unsubscribe_track",
        "publish_audio",
        "publish_video",
        "mint_access",
        "remove_participant",
        "mute_track",
        "unmute_track",
        "close",
    }
)
_ROOM_INFO = (
    "bot_present",
    "bot_session_id",
    "leaving_session_ids",
    "bot_hidden",
    "collecting",
    "active_lanes",
    "stt_active",
    "realtime_active",
    "recording_active",
)


class ScriptedSFU(MockConferenceBackend):
    """The mock SFU, remembering which session each join returned.

    The mock records a call before it can fail, so a refused join has a
    record too, and no session: the map is keyed by the record itself.
    """

    def __init__(self, *, capabilities: ConferenceCapability = ConferenceCapability.NONE) -> None:
        super().__init__(capabilities=capabilities)
        self.joined: dict[int, str] = {}
        # The scenario's faults (``conference.fault``) and the ``joined_at``
        # of the sessions it returns (``conference.open``'s ``bot_joined_at``).
        self.held: SFUFaults | None = None
        self.joined_at: datetime | None = None

    async def _enter(self, method: str, **args: Any) -> None:
        # Received first, then held or failed as a fault says, then the mock's own.
        self._record(method, **args)
        if self.held is not None:
            await self.held.take(method, self.calls[-1])
        await self.faults.apply(method)

    async def join_as_bot(
        self, room_id: str, identity: str, grants: ConferenceGrants
    ) -> BotSession:
        # The mock records the call before its first suspension: the record
        # is the one appended at this position.
        position = len(self.calls)
        bot = await super().join_as_bot(room_id, identity, grants)
        self.joined[id(self.calls[position])] = bot.id
        if self.joined_at is not None:
            bot.joined_at = self.joined_at
        return bot


def _grants(grants: ConferenceGrants) -> dict[str, bool]:
    return {name: getattr(grants, name) for name in _GRANTS}


def _call(backend: ScriptedSFU, record: MockConferenceCall) -> dict[str, Any]:
    """One call the SFU received, as ``conference.read`` lists it."""
    read: dict[str, Any] = {"call": record.method}
    for name, value in record.args.items():
        if name == "metadata":
            continue
        if name == "bot":
            read["session"] = value
        elif isinstance(value, ConferenceGrants):
            read[name] = _grants(value)
        elif isinstance(value, Mapping):
            # A mint's attributes: an empty map reads as none passed.
            read[name] = dict(value) or None
        else:
            read[name] = value
    if record.method == "join_as_bot":
        read["session"] = backend.joined.get(id(record))
    return read


def _info(channel: ConferenceChannel) -> dict[str, Any]:
    info = channel.info
    rooms = {
        room_id: {name: room[name] for name in _ROOM_INFO}
        for room_id, room in info["rooms"].items()
    }
    return {**{name: info[name] for name in _INFO}, "rooms": rooms}


def _fixture(harness: Harness, spec: Mapping[str, Any]) -> ConferenceFixture:
    return harness.conferences[spec["channel_id"]]


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    fixture = _fixture(harness, spec)
    backend = fixture.backend
    assert isinstance(backend, ScriptedSFU)
    names = spec.get("calls")
    if names is not None and (unknown := set(names) - _CALL_NAMES):
        raise UnreadableError(f"conference.read names no call {sorted(unknown)}")
    calls = [
        _call(backend, record)
        for record in backend.calls
        if names is None or record.method in names
    ]
    utterances = [
        {"session": utterance.bot_id, "complete": utterance.complete}
        for utterance in backend.utterances
    ]
    # The sessions the SFU holds now, each with the grants it last gave it.
    bots = [
        {"session": bot.id, "room_id": bot.room_id, "grants": _grants(backend.bot_grants[bot.id])}
        for bot in backend.bots
    ]
    return {
        "calls": calls,
        "utterances": utterances,
        "bots": bots,
        "info": _info(fixture.channel),
    }


async def _ensure_bot(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    bot = await _fixture(harness, spec).channel.ensure_bot(spec["room_id"])
    return {"id": bot.id, "room_id": bot.room_id, "identity": bot.identity}


async def _plug(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    channel = _fixture(harness, spec).channel
    match spec["kind"]:
        case "stt":
            await channel.plug_stt(MockSTTProvider())
        case "tts":
            await channel.plug_tts(scripted_tts(spec.get("tts") or {}, harness.gates))
        case "recording":
            name = spec["recorder"]
            recorder = None if name is None else MockMediaRecorder()
            await channel.plug_recording(
                ConferenceRecordingConfig(),
                recorder=recorder,  # ty: ignore[invalid-argument-type]
            )
            if recorder is not None:
                harness.media_recorders[name] = recorder
        case "realtime":
            await channel.plug_realtime(ConferenceRealtimeConfig(provider=MockRealtimeProvider()))
        case kind:
            raise UnreadableError(f"no kind {kind!r} to plug")


async def _unplug(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    channel = _fixture(harness, spec).channel
    unplugs: dict[str, Callable[[], Awaitable[None]]] = {
        "stt": channel.unplug_stt,
        "tts": channel.unplug_tts,
        "recording": channel.unplug_recording,
        "realtime": channel.unplug_realtime,
    }
    if spec["kind"] not in unplugs:
        raise UnreadableError(f"no kind {spec['kind']!r} to unplug")
    await unplugs[spec["kind"]]()


async def _set_bot_grants(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    grants = spec["grants"]
    await _fixture(harness, spec).channel.set_bot_grants(
        None if grants is None else ConferenceGrants(**grants)
    )


def _live_bot(fixture: ConferenceFixture, room_id: str) -> BotSession:
    """The room's live bot session, as the backend holds it."""
    room = fixture.channel.info["rooms"].get(room_id) or {}
    session_id = room.get("bot_session_id")
    for bot in fixture.backend.bots:
        if bot.id == session_id:
            return bot
    raise UnreadableError(f"conference room {room_id!r} has no live bot session")


async def _bot_lost(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = _fixture(harness, spec)
    bot = _live_bot(fixture, spec["room_id"])
    await fixture.backend.simulate_bot_disconnected(bot, spec.get("reason", "connection lost"))


async def _echo(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The bot reported back as a participant and a publisher: the mock's
    ``simulate_bot_echo``, with the echoed track's id the scenario names."""
    fixture = _fixture(harness, spec)
    bot = _live_bot(fixture, spec["room_id"])
    await fixture.backend.simulate_participant_joined(bot.room_id, bot.identity)
    await fixture.backend.simulate_track_published(
        bot.room_id, bot.identity, TrackKind.AUDIO, track_id=spec["track_id"]
    )


def conference_work(fixtures: Mapping[str, ConferenceFixture]) -> list[asyncio.Task[Any]]:
    """The conference channels' background work an ``idle`` step waits for:
    each room's tasks (a join a trigger started, a re-join waiting out its
    backoff) and the detaches' teardowns."""
    tasks: list[asyncio.Task[Any]] = []
    for fixture in fixtures.values():
        channel = fixture.channel
        for room in list(channel._rooms.values()):
            tasks.extend(task for task in room.tasks if not task.done())
        tasks.extend(task for task in channel._teardowns if not task.done())
    return tasks


SFU_ARGUMENTS = {
    "conference.read": {"channel_id", "calls"},
    "conference.ensure_bot": {"channel_id", "room_id"},
    "conference.plug": {"channel_id", "kind", "tts", "recorder"},
    "conference.unplug": {"channel_id", "kind"},
    "conference.set_bot_grants": {"channel_id", "grants"},
    "conference.bot_lost": {"channel_id", "room_id", "reason"},
    "conference.echo": {"channel_id", "room_id", "track_id"},
}
SFU_CALLS = {
    "conference.read": _read,
    "conference.ensure_bot": _ensure_bot,
    "conference.plug": _plug,
    "conference.unplug": _unplug,
    "conference.set_bot_grants": _set_bot_grants,
    "conference.bot_lost": _bot_lost,
    "conference.echo": _echo,
}

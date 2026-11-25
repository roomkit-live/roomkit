"""A conference's SFU notifications and its model's tool-result wire."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit import ConferenceChannel, ConferenceRealtimeConfig, MockConferenceBackend, TrackKind
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conformance._conference_media import (
    ARRIVAL,
    MEDIA_ARGUMENTS,
    MEDIA_CALLS,
    MEDIA_OPEN,
    ConferenceMedia,
    arrival,
    media_backend,
    media_options,
    read_media,
)
from tests.conformance._conference_sfu import SFU_ARGUMENTS, SFU_CALLS

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


class _Model(MockRealtimeProvider):
    """Wake the step issuing a call when the model receives that call's result."""

    def __init__(self) -> None:
        super().__init__()
        self.replies: dict[str, asyncio.Future[str]] = {}

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        await super().submit_tool_result(session, call_id, result)
        if (reply := self.replies.get(call_id)) is not None and not reply.done():
            reply.set_result(result)

    async def call(self, session: VoiceSession, spec: Mapping[str, Any]) -> Any:
        call_id = spec["call_id"]
        if call_id in self.replies:
            raise ValueError(f"conference call {call_id} is already outstanding")
        reply = asyncio.get_running_loop().create_future()
        self.replies[call_id] = reply
        try:
            await self.simulate_tool_call(
                session, call_id, spec["name"], spec.get("arguments", {})
            )
            wire = await reply
        finally:
            del self.replies[call_id]
        try:
            return json.loads(wire)
        except json.JSONDecodeError:
            return wire


@dataclass
class ConferenceFixture:
    room: str
    channel: ConferenceChannel
    backend: MockConferenceBackend
    model: _Model | None
    media: ConferenceMedia | None = None


async def _open(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    backend = media_backend(spec)
    model = None
    realtime = None
    if "tools" in spec:
        model = _Model()
        tools = spec["tools"]

        async def answer(room_id: str, name: str, arguments: dict[str, Any]) -> str:
            return tools[name]

        realtime = ConferenceRealtimeConfig(
            provider=model,
            tools=[
                {"name": name, "description": name, "parameters": {"type": "object"}}
                for name in tools
            ],
            tool_handler=answer,
        )
    options, media = media_options(spec, harness, MockSTTProvider() if realtime is None else None)
    channel = ConferenceChannel(
        spec["channel_id"],
        backend=backend,
        realtime=realtime,
        speak_text_events=True,
        **options,
    )
    kit.register_channel(channel)
    await kit.attach_channel(spec["room_id"], channel.channel_id)
    harness.conferences[channel.channel_id] = ConferenceFixture(
        spec["room_id"], channel, backend, model, media
    )


CONFERENCE_ARGUMENTS = {
    "conference.open": {"channel_id", "room_id", "tools"} | MEDIA_OPEN,
    "conference.join": {"channel_id", "participant_id"} | ARRIVAL,
    "conference.leave": {"channel_id", "participant_id"},
    "conference.publish": {"channel_id", "participant_id", "track_id", "kind"},
    "conference.unpublish": {"channel_id", "track_id"},
    "conference.mute": {"channel_id", "track_id"},
    "conference.unmute": {"channel_id", "track_id"},
    "conference.active_speaker": {"channel_id", "participant_id"},
    "conference.connection_quality": {"channel_id", "participant_id", "quality"},
    "conference.tool_call": {"channel_id", "call_id", "name", "arguments"},
    **SFU_ARGUMENTS,
    **MEDIA_ARGUMENTS,
}


async def _notify(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    fixture = harness.conferences[spec["channel_id"]]
    backend, room = fixture.backend, fixture.room
    match spec["call"]:
        case "conference.join":
            await backend.simulate_participant_joined(
                room, spec["participant_id"], **arrival(spec)
            )
        case "conference.leave":
            await backend.simulate_participant_left(room, spec["participant_id"])
        case "conference.publish":
            await backend.simulate_track_published(
                room, spec["participant_id"], TrackKind(spec["kind"]), track_id=spec["track_id"]
            )
        case "conference.unpublish":
            await backend.simulate_track_unpublished(spec["track_id"])
        case "conference.mute":
            await backend.simulate_track_muted(spec["track_id"])
        case "conference.unmute":
            await backend.simulate_track_unmuted(spec["track_id"])
        case "conference.active_speaker":
            await backend.simulate_active_speaker(room, spec["participant_id"])
        case "conference.connection_quality":
            await backend.simulate_connection_quality(
                room, spec["participant_id"], spec["quality"]
            )
        case "conference.tool_call":
            sessions = fixture.channel.get_room_sessions(room)
            if fixture.model is None or not sessions:
                raise ValueError("the conference model must be connected first")
            return await fixture.model.call(sessions[0], spec)
        case _:
            raise ValueError(f"Unknown conference call {spec['call']}")
    return None


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    """``conference.read``: what the scripted SFU received and holds, and what
    the lanes' scripted stages and recognizer received."""
    sfu = await SFU_CALLS["conference.read"](kit, harness, spec)
    return sfu | await read_media(kit, harness, spec)


CONFERENCE_CALLS = {
    **dict.fromkeys(CONFERENCE_ARGUMENTS, _notify),
    "conference.open": _open,
    **SFU_CALLS,
    **MEDIA_CALLS,
    "conference.read": _read,
}

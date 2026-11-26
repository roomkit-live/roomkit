"""A conference's SFU notifications and its model's tool-result wire."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from roomkit import ConferenceChannel, MockConferenceBackend, TrackKind
from roomkit.voice.stt.mock import MockSTTProvider
from tests.conformance._conference_faults import (
    FAULT_ARGUMENTS,
    FAULT_CALLS,
    FAULT_OPEN,
    SFUFaults,
    arm_open_faults,
    bot_joined_at,
)
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
from tests.conformance._conference_model import (
    MODEL_OPEN,
    ConferenceModel,
    conference_model,
    model_options,
    pause,
    plug_model,
    read_model,
    realtime_settings,
    settle_mix,
    watch_mix,
)
from tests.conformance._conference_options import (
    OPTIONS_ARGUMENTS,
    OPTIONS_CALLS,
    OPTIONS_OPEN,
    ConferenceOptions,
    capabilities,
    channel_options,
    media_spec,
    read_options,
    realtime_config,
    seat_occupants,
)
from tests.conformance._conference_sfu import SFU_ARGUMENTS, SFU_CALLS

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


@dataclass
class ConferenceFixture:
    room: str
    channel: ConferenceChannel
    backend: MockConferenceBackend
    model: ConferenceModel | None
    media: ConferenceMedia | None = None
    options: ConferenceOptions | None = None


async def _open(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    backend = media_backend(spec, capabilities(spec))
    backend.held = SFUFaults(harness.gates)
    backend.joined_at = bot_joined_at(spec)
    arm_open_faults(backend.held, spec)
    await seat_occupants(backend, spec)
    model = conference_model(spec, harness.gates)
    options, media = media_options(
        media_spec(spec), harness, MockSTTProvider() if model is None else None
    )
    kept = channel_options(spec, harness, options, media)
    options.update(model_options(spec))
    realtime = None
    if model is not None:
        realtime = replace(realtime_config(spec, harness, model, kept), **realtime_settings(spec))
    channel = ConferenceChannel(spec["channel_id"], backend=backend, realtime=realtime, **options)
    watch_mix(channel)
    kit.register_channel(channel)
    await kit.attach_channel(spec["room_id"], channel.channel_id)
    harness.conferences[channel.channel_id] = ConferenceFixture(
        spec["room_id"], channel, backend, model, media, kept
    )


CONFERENCE_ARGUMENTS = {
    "conference.open": {"channel_id", "room_id", "tools"}
    | MEDIA_OPEN
    | OPTIONS_OPEN
    | FAULT_OPEN
    | MODEL_OPEN,
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
    **OPTIONS_ARGUMENTS,
    **FAULT_ARGUMENTS,
    "conference.read": SFU_ARGUMENTS["conference.read"] | {"stage_calls"},
    "conference.feed": MEDIA_ARGUMENTS["conference.feed"] | {"pause_ms"},
    "conference.settle": MEDIA_ARGUMENTS["conference.settle"] | {"mix"},
    "conference.plug": SFU_ARGUMENTS["conference.plug"] | {"realtime"},
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
    """``conference.read``: what the scripted SFU received and holds, what
    the lanes' scripted stages and recognizer received, and what the
    conference's own settings scripted."""
    sfu = await SFU_CALLS["conference.read"](kit, harness, spec)
    media = await read_media(kit, harness, spec)
    if (names := spec.get("stage_calls")) is not None:
        # Only the stage calls named: those a scenario compares, whatever else
        # the lanes' stages received.
        media["stages"] = [call for call in media["stages"] if call["call"] in names]
    fixture = harness.conferences[spec["channel_id"]]
    return sfu | media | read_options(fixture) | read_model(fixture)


async def _feed(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """``conference.feed``, its track silent for ``pause_ms`` first."""
    await pause(spec)
    await MEDIA_CALLS["conference.feed"](kit, harness, spec)


async def _settle(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """``conference.settle``, then, with ``mix``, the mix's sends to the model."""
    await MEDIA_CALLS["conference.settle"](kit, harness, spec)
    if spec.get("mix"):
        await settle_mix(harness.conferences[spec["channel_id"]])


async def _plug(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """``conference.plug``; a realtime model is the scripted one ``realtime`` describes."""
    if spec["kind"] == "realtime":
        await plug_model(harness, harness.conferences[spec["channel_id"]], spec)
    elif "realtime" in spec:
        raise ValueError("conference.plug's realtime scripts a realtime model")
    else:
        await SFU_CALLS["conference.plug"](kit, harness, spec)


CONFERENCE_CALLS = {
    **dict.fromkeys(CONFERENCE_ARGUMENTS, _notify),
    "conference.open": _open,
    **SFU_CALLS,
    **MEDIA_CALLS,
    **OPTIONS_CALLS,
    **FAULT_CALLS,
    "conference.read": _read,
    "conference.feed": _feed,
    "conference.settle": _settle,
    "conference.plug": _plug,
}

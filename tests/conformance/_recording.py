"""A room recorder and live feeds driven through the framework (§12.11)."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.recorder.base import RecordingTrack
from tests.conformance._conference_media import conference_recorder_calls, read_recordings
from tests.conformance._recording_room import (
    list_recordings,
    organization,
    recorder_calls,
    room_bindings,
)

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


async def _start(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await kit.start_room_recording(
        spec["room_id"], room_bindings(spec, harness), **organization(spec)
    )


async def _track(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    track = RecordingTrack(
        id=spec["track_id"],
        kind="audio",
        channel_id="capture",
        codec="pcm_s16le",
        sample_rate=48000,
    )
    feed = await kit.add_room_recording_track(spec["room_id"], track, **organization(spec))
    assert feed is not None
    harness.recording_feeds[spec["room_id"], track.id] = feed


async def _feed(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    feed = harness.recording_feeds[spec["room_id"], spec["track_id"]]
    feed.feed(base64.b64decode(spec["data"], validate=True))


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    if "channel_id" in spec:
        return read_recordings(harness, spec["channel_id"])
    return [
        base64.b64encode(chunk.data).decode("ascii")
        for chunk in harness.media_recorders[spec["recorder"]].chunks
    ]


async def _stop(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    results = await kit.stop_room_recording(spec["room_id"], **organization(spec))
    return [{"recording_id": r.id, "url": r.url} for r in results]


async def _calls(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    if "channel_id" in spec:
        return conference_recorder_calls(harness, spec)
    return recorder_calls(harness, spec)


RECORDING_CALLS = {
    "recording.start": _start,
    "recording.track": _track,
    "recording.feed": _feed,
    "recording.read": _read,
    "recording.stop": _stop,
    "recording.list": list_recordings,
    "recording.calls": _calls,
}
RECORDING_ARGUMENTS = {
    "recording.start": {"room_id", "recorder", "recorders"},
    "recording.track": {"room_id", "track_id"},
    "recording.feed": {"room_id", "track_id", "data"},
    "recording.read": {"recorder", "channel_id"},
    "recording.stop": {"room_id"},
    "recording.list": {"room_id"},
    "recording.calls": {"recorder", "channel_id", "calls"},
}

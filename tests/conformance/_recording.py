"""A room recorder and live feeds driven through the framework (§12.11)."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.recorder.base import (
    MediaRecordingConfig,
    MediaRecordingHandle,
    RecordingTrack,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


class _NamedRecorder(MockMediaRecorder):
    """The recorder chooses its handle id, a known one so its stop URL is observable."""

    def __init__(self, recording_id: str) -> None:
        super().__init__()
        self.recording_id = recording_id

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        handle = super().on_recording_start(config)
        handle.id = self.recording_id
        return handle


async def _start(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    name = spec["recorder"]
    recorder = _NamedRecorder(name)
    harness.media_recorders[name] = recorder
    await kit.start_room_recording(
        spec["room_id"], [RoomRecorderBinding(recorder=recorder, config=MediaRecordingConfig())]
    )


async def _track(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    track = RecordingTrack(
        id=spec["track_id"],
        kind="audio",
        channel_id="capture",
        codec="pcm_s16le",
        sample_rate=48000,
    )
    feed = await kit.add_room_recording_track(spec["room_id"], track)
    assert feed is not None
    harness.recording_feeds[spec["room_id"], track.id] = feed


async def _feed(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    feed = harness.recording_feeds[spec["room_id"], spec["track_id"]]
    feed.feed(base64.b64decode(spec["data"], validate=True))


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> list[str]:
    return [
        base64.b64encode(chunk.data).decode("ascii")
        for chunk in harness.media_recorders[spec["recorder"]].chunks
    ]


async def _stop(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    results = await kit.stop_room_recording(spec["room_id"])
    return [{"recording_id": r.id, "url": r.url} for r in results]


RECORDING_CALLS = {
    "recording.start": _start,
    "recording.track": _track,
    "recording.feed": _feed,
    "recording.read": _read,
    "recording.stop": _stop,
}
RECORDING_ARGUMENTS = {
    "recording.start": {"room_id", "recorder"},
    "recording.track": {"room_id", "track_id"},
    "recording.feed": {"room_id", "track_id", "data"},
    "recording.read": {"recorder"},
    "recording.stop": {"room_id"},
}

"""A room's recorders: bound at its creation or started together on an existing
room, one of them refusing its start, and the calls each received
(roomkit-specs ``conformance/README.md``, "Video sessions and room recording";
RFC §12.11).

``create_room`` and ``recording.start`` name their ``recorders``; a scripted
recorder notes every call of the MediaRecorder interface it receives, which
``recording.calls`` reads; ``recording.list`` lists the recordings a room runs.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.recorder.base import (
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from tests.conformance._realtime_io import ScriptedProviderError

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


class ScriptedRoomRecorder(MockMediaRecorder):
    """A room recorder known by its name, which it gives the recordings it
    opens, so their stop URL is observable; it notes each call it receives,
    and with *refuses* refuses its start as its own failure."""

    def __init__(self, name: str, refuses: str | None = None) -> None:
        super().__init__()
        self.recording_id = name
        self._refuses = refuses
        self.calls: list[dict[str, Any]] = []

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        if self._refuses is not None:
            self.calls.append({"call": "on_recording_start", "recording_id": None})
            raise ScriptedProviderError(self._refuses)
        handle = super().on_recording_start(config)
        handle.id = self.recording_id
        self.calls.append({"call": "on_recording_start", "recording_id": handle.id})
        return handle

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self.calls.append(track_call("on_track_added", handle, track))
        super().on_track_added(handle, track)

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self.calls.append(track_call("on_track_removed", handle, track))
        super().on_track_removed(handle, track)

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        self.calls.append(data_call(handle, track, data))
        super().on_data(handle, track, data, timestamp_ms)

    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        self.calls.append({"call": "on_recording_stop", "recording_id": handle.id})
        return super().on_recording_stop(handle)

    def close(self) -> None:
        self.calls.append({"call": "close"})
        super().close()


def recorded_track(track: RecordingTrack) -> dict[str, Any]:
    """A track a recorder was told, as ``recording.read`` and ``recording.calls`` read it."""
    return {
        "track_id": track.id,
        "kind": track.kind,
        "channel_id": track.channel_id,
        "participant_id": track.participant_id,
        "codec": track.codec,
        "sample_rate": track.sample_rate,
        "channels": track.channels,
    }


def track_call(call: str, handle: MediaRecordingHandle, track: RecordingTrack) -> dict[str, Any]:
    """An ``on_track_added`` or ``on_track_removed`` call as ``recording.calls`` reads it."""
    return {"call": call, "recording_id": handle.id, "track": recorded_track(track)}


def data_call(handle: MediaRecordingHandle, track: RecordingTrack, data: bytes) -> dict[str, Any]:
    """An ``on_data`` call as ``recording.calls`` reads it."""
    return {
        "call": "on_data",
        "recording_id": handle.id,
        "track_id": track.id,
        "data": base64.b64encode(data).decode("ascii"),
    }


def organization(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``organization_id`` a recording call acts for, when it names one."""
    return {"organization_id": spec["organization_id"]} if "organization_id" in spec else {}


def room_bindings(spec: Mapping[str, Any], harness: Harness) -> list[RoomRecorderBinding]:
    """The bindings of the recorders a call names, ``recorder`` or ``recorders``,
    each a scripted recorder the run then reads by its name."""
    named = spec["recorders"] if "recorders" in spec else [spec["recorder"]]
    bindings = []
    for recorder in named:
        if isinstance(recorder, str):
            scripted = ScriptedRoomRecorder(recorder)
        else:
            scripted = ScriptedRoomRecorder(recorder["name"], recorder.get("refuses"))
        harness.media_recorders[scripted.recording_id] = scripted
        bindings.append(RoomRecorderBinding(recorder=scripted, config=MediaRecordingConfig()))
    return bindings


async def list_recordings(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    handles = await kit.room_recordings(spec["room_id"], **organization(spec))
    return [{"recording_id": handle.id, "room_id": handle.room_id} for handle in handles]


def recorder_calls(harness: Harness, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``recording.calls`` of a room recorder: the calls it received, in order,
    only the named ones when ``calls`` names some."""
    recorder = harness.media_recorders[spec["recorder"]]
    if not isinstance(recorder, ScriptedRoomRecorder):
        raise ValueError(f"recorder {spec['recorder']} notes no calls")
    return only_named(recorder.calls, spec)


def only_named(calls: list[dict[str, Any]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The calls a ``recording.calls`` step reads: those it names, or all."""
    names = spec.get("calls")
    return [call for call in list(calls) if names is None or call["call"] in names]

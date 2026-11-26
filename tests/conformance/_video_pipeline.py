"""A video channel's scripted stages, bridge configuration, recorder and late
joiners (RFC §12.8.4, §12.8.5, §12.8.9, §12.8.10), and a realtime channel given
the room's vision (§12.8.7).

``video.open`` names the stages: ``transforms`` and ``filters`` in their order,
the vision results' ``vision_labels``, the channel's ``vision_interval_ms``, a
``recorder`` and ``bridge`` as a VideoBridgeConfig. The transport counts the
keyframe requests it receives, and keeps the session it made for each
participant, joined or refused.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from roomkit.video import (
    MockVideoBackend,
    MockVideoRecorder,
    VideoBridgeConfig,
    VideoFrame,
    VideoRecordingConfig,
    VideoRecordingHandle,
    VideoSession,
    setup_realtime_vision,
)
from roomkit.video.pipeline import (
    FilterContext,
    VideoFilterProvider,
    VideoPipelineConfig,
    VideoTransformProvider,
)
from tests.conformance._realtime_io import ScriptedProviderError

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _replaced(frame: VideoFrame, data: str | None) -> VideoFrame:
    """The frame a stage hands on: the one it was handed, or one of *data*'s bytes."""
    if data is None:
        return frame
    return replace(frame, data=base64.b64decode(data, validate=True))


class ScriptedVideoTransport(MockVideoBackend):
    """The scripted transport: it keeps the session it made for each participant,
    whether the kit's join of it succeeded or not, and counts the keyframe
    requests each session receives (§12.8.9)."""

    def __init__(self) -> None:
        super().__init__()
        self.made: dict[str, VideoSession] = {}
        self.keyframe_requests: dict[str, int] = {}

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VideoSession:
        session = await super().connect(room_id, participant_id, channel_id, metadata=metadata)
        self.made[participant_id] = session
        return session

    def request_keyframe(self, session: VideoSession) -> None:
        self.keyframe_requests[session.id] = self.keyframe_requests.get(session.id, 0) + 1


class _Transform(VideoTransformProvider):
    """A transform that records each frame it is handed, and hands on its own
    bytes in its place when it has some."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._name: str = spec["name"]
        self._data: str | None = spec.get("data")
        self.handed: list[VideoFrame] = []

    @property
    def name(self) -> str:
        return self._name

    def transform(self, frame: VideoFrame) -> VideoFrame:
        self.handed.append(frame)
        return _replaced(frame, self._data)


class _Filter(VideoFilterProvider):
    """A filter that records each frame it is handed with what its context held
    then, and hands on its own bytes in its place when it has some."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self._name: str = spec["name"]
        self._data: str | None = spec.get("data")
        self.handed: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def filter(self, frame: VideoFrame, context: FilterContext) -> VideoFrame:
        seen = context.last_vision_result
        self.handed.append(
            {
                "sequence": frame.sequence,
                "data": _b64(frame.data),
                "vision": None if seen is None else seen.description,
                "labels": sorted(context.labels_detected),
            }
        )
        return _replaced(frame, self._data)


class _Recorder(MockVideoRecorder):
    """A session recorder that refuses to start the recordings of the
    participants it is told to, with its own failure (§12.8.10)."""

    def __init__(self, refuses: list[str]) -> None:
        super().__init__()
        self.refuses = set(refuses)

    def start(self, session: VideoSession, config: VideoRecordingConfig) -> VideoRecordingHandle:
        if session.participant_id in self.refuses:
            raise ScriptedProviderError(f"recording of {session.participant_id} refused")
        return super().start(session, config)


@dataclass
class VideoStages:
    """The scripted stages one ``video.open`` built."""

    transforms: list[_Transform] = field(default_factory=list)
    filters: list[_Filter] = field(default_factory=list)
    recorder: _Recorder | None = None

    def config(self, first_filters: list[VideoFilterProvider]) -> VideoPipelineConfig | None:
        """The pipeline these stages make, after *first_filters*; None for none."""
        filters = [*first_filters, *self.filters]
        if not (self.transforms or filters or self.recorder):
            return None
        return VideoPipelineConfig(
            transforms=list(self.transforms),
            filters=filters,
            recorder=self.recorder,
            recording_config=VideoRecordingConfig() if self.recorder else None,
        )


def video_stages(spec: Mapping[str, Any]) -> VideoStages:
    """``video.open``'s scripted transforms, filters and recorder."""
    recorder = spec.get("recorder")
    return VideoStages(
        transforms=[_Transform(stage) for stage in spec.get("transforms", [])],
        filters=[_Filter(stage) for stage in spec.get("filters", [])],
        recorder=None if recorder is None else _Recorder(recorder.get("refuses", [])),
    )


def video_bridge(spec: Mapping[str, Any]) -> bool | VideoBridgeConfig:
    """``video.open``'s ``bridge``: a boolean, or a VideoBridgeConfig (§12.8.9)."""
    bridge = spec.get("bridge", False)
    if isinstance(bridge, Mapping):
        return VideoBridgeConfig(**bridge)
    return bool(bridge)


def read_stages(
    stages: VideoStages,
    transport: ScriptedVideoTransport,
    sessions: Mapping[str, VideoSession],
    analysed: list[VideoFrame],
) -> dict[str, Any]:
    """What ``video.read`` adds: the frames each scripted stage and the vision
    provider were handed, the keyframe requests, the frames recorded."""
    owners = {session.id: name for name, session in sessions.items()}
    recorded: dict[str, list[str]] = {}
    if stages.recorder is not None:
        for entry in stages.recorder.recordings.values():
            name = owners.get(entry.handle.session_id, entry.handle.session_id)
            recorded.setdefault(name, []).extend(_b64(frame.data) for frame in entry.frames)
    return {
        "analysed": [_b64(frame.data) for frame in analysed],
        "transforms": {
            stage.name: [_b64(frame.data) for frame in stage.handed] for stage in stages.transforms
        },
        "filters": {f.name: list(f.handed) for f in stages.filters},
        "keyframe_requests": {
            name: transport.keyframe_requests.get(session.id, 0)
            for name, session in sessions.items()
        },
        "recorded": recorded,
    }


async def _join(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The kit's ``join`` of a late participant; the session the transport made
    is the participant's even when the join is refused."""
    fixture = harness.video_fixtures[spec["channel_id"]]
    participant = spec["participant_id"]
    room_id = next(iter(fixture.sessions.values())).room_id
    try:
        await kit.join(room_id, fixture.channel.channel_id, participant_id=participant)
    finally:
        made = fixture.backend.made.get(participant)
        if made is not None:
            fixture.sessions[participant] = made
        await fixture.settle()


async def _realtime_vision(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    """The room's vision results given to the realtime channel's sessions (§12.8.7)."""
    prefix = {"context_prefix": spec["context_prefix"]} if "context_prefix" in spec else {}
    setup_realtime_vision(kit, spec["room_id"], spec["channel_id"], **prefix)


VIDEO_PIPELINE_CALLS = {
    "video.join": _join,
    "realtime.vision": _realtime_vision,
}
VIDEO_PIPELINE_ARGUMENTS = {
    "video.join": {"channel_id", "participant_id"},
    "realtime.vision": {"channel_id", "room_id", "context_prefix"},
}
VIDEO_STAGE_ARGUMENTS = {
    "vision_labels",
    "vision_interval_ms",
    "transforms",
    "filters",
    "recorder",
}

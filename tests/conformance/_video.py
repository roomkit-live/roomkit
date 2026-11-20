"""A video channel driven through its transport, filter and vision provider (§12.8)."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from roomkit.channels.video import VideoChannel
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.vision_event import VisionEvent
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.base import VideoSession
from roomkit.video.events import BridgeVideoEvent, VideoDetectionEvent
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import FaceDetection
from roomkit.video.vision.mock import MockVisionProvider
from tests.channel_tasks import settled

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


@dataclass
class VideoFixture:
    channel: VideoChannel
    backend: MockVideoBackend
    sessions: dict[str, VideoSession] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    async def vision_result(self, event: FrameworkEvent) -> None:
        if event.data["channel_id"] == self.channel.channel_id:
            self.results.append(dict(event.data))


async def _open(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    channel_id, room_id = spec["channel_id"], spec["room_id"]
    backend = MockVideoBackend()
    pipeline = None
    if "detections" in spec:
        by_frame: dict[int, list[VideoDetectionEvent]] = {}
        for detection in spec["detections"]:
            by_frame.setdefault(detection["frame_sequence"], []).append(
                VideoDetectionEvent(**detection)
            )
        pipeline = VideoPipelineConfig(filters=[MockFaceTouchFilter(by_frame)])
    channel = VideoChannel(
        channel_id,
        backend=backend,
        pipeline=pipeline,
        bridge=spec.get("bridge", False),
        vision=MockVisionProvider(spec["vision"]) if "vision" in spec else None,
        vision_interval_ms=0,
    )
    fixture = VideoFixture(channel, backend)
    harness.video_fixtures[channel_id] = fixture
    kit.on("video_vision_result")(fixture.vision_result)
    kit.register_channel(channel)
    await kit.attach_channel(room_id, channel_id)
    for name in spec["participants"]:
        session = await kit.join(room_id, channel_id, participant_id=name)
        assert isinstance(session, VideoSession)
        fixture.sessions[name] = session
    await settled(channel)


async def _frame(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.video_fixtures[spec["channel_id"]]
    frame = VideoFrame(
        data=base64.b64decode(spec["data"], validate=True),
        sequence=spec["sequence"],
        codec="raw_rgb24",
        width=1,
        height=1,
        keyframe=True,
    )
    await fixture.backend.simulate_video_received(fixture.sessions[spec["participant_id"]], frame)
    await settled(fixture.channel)


async def _disconnect(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.video_fixtures[spec["channel_id"]]
    await fixture.backend.simulate_client_disconnected(fixture.sessions[spec["participant_id"]])
    await settled(fixture.channel)


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    fixture = harness.video_fixtures[spec["channel_id"]]
    return {
        "forwarded": {
            name: [
                base64.b64encode(data).decode("ascii")
                for sid, data in fixture.backend.sent_video
                if sid == session.id
            ]
            for name, session in fixture.sessions.items()
        },
        "vision": fixture.results,
    }


def video_payload(payload: Any) -> dict[str, Any] | None:
    session = getattr(payload, "session", None)
    if not isinstance(session, VideoSession):
        return None
    data: dict[str, Any] = {"participant_id": session.participant_id}
    if isinstance(payload, BridgeVideoEvent):
        data["sequence"] = payload.frame.sequence
    if isinstance(payload, VideoDetectionEvent):
        data.update(
            kind=payload.kind,
            frame_sequence=payload.frame_sequence,
            confidence=payload.confidence,
            labels=list(payload.labels),
            metadata=dict(payload.metadata),
        )
    if isinstance(payload, VisionEvent):
        data.update(description=payload.description, labels=list(payload.labels))
    return data


def patched_video_event(
    payload: BridgeVideoEvent | VisionEvent, patch: Mapping[str, Any]
) -> BridgeVideoEvent | VisionEvent:
    if isinstance(payload, BridgeVideoEvent):
        frame = dict(patch["frame"])
        frame["data"] = base64.b64decode(frame["data"], validate=True)
        return replace(payload, frame=replace(payload.frame, **frame))
    changes = dict(patch)
    if "faces" in changes:
        changes["faces"] = [FaceDetection(**face) for face in changes["faces"]]
    return replace(payload, **changes)


VIDEO_CALLS = {
    "video.open": _open,
    "video.frame": _frame,
    "video.disconnect": _disconnect,
    "video.read": _read,
}
VIDEO_ARGUMENTS = {
    "video.open": {"channel_id", "room_id", "participants", "bridge", "vision", "detections"},
    "video.frame": {"channel_id", "participant_id", "sequence", "data"},
    "video.disconnect": {"channel_id", "participant_id"},
    "video.read": {"channel_id"},
}

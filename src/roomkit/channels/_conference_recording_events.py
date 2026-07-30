"""Telling an integrator where a conference recording went.

:class:`~roomkit.channels._conference_recording.ConferenceRecording` writes the
files and is synchronous throughout — it runs inside the backend's emission
loop, and every method of it is chosen to return rather than await. Announcing
is the opposite shape: it builds a room context, runs integrator hooks and
emits a framework event, all of which await. So the two live apart, and what
the recording returns is what this announces.

One announcement per track, at the two moments a track's recording has: opened,
and closed with a result. The conference is not the unit — a participant who
leaves halfway through has a finished recording while the meeting runs on, and
holding those results back for a single report at the end would deliver them
after the point an observer could act on them.

See RFC section 12.10.8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.channels._conference_recording import FinishedRecording, TrackRecording
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.channels.conference")


@dataclass
class ConferenceRecordingStarted:
    """A track's recording has opened.

    Carried to ON_RECORDING_STARTED. Names the track and the participant
    publishing it, which is what a conference has instead of the session the
    voice path's event carries (RFC 12.10.8).
    """

    room_id: str
    track_id: str
    participant_id: str
    id: str
    kind: str
    sample_rate: int | None
    channels: int | None = None
    """Audio channel count of the track, as the recording was opened on it."""

    codec: str = ""
    """Sample format of the track, e.g. ``pcm_s16le``.

    With :attr:`sample_rate` and :attr:`channels` it is the whole of what the
    recording was opened on, and this event is the only place an integrator
    learns it: two participants in one conference need not have negotiated the
    same one.
    """


@dataclass
class ConferenceRecordingStopped:
    """A track's recording has closed, and this is where it went.

    Carried to ON_RECORDING_STOPPED. ``url`` is what the recorder reported —
    a path, an object-store URL, whatever it writes to — and it is the whole
    point of the event: without it the files exist and nothing says where.
    """

    room_id: str
    track_id: str
    participant_id: str
    id: str
    url: str
    duration_seconds: float
    size_bytes: int
    format: str


class ConferenceRecordingEvents:
    """Announces what a conference channel's recordings did.

    Every announcement is wrapped, and the hooks are not what the wrapping is
    for — the hook engine logs and drops what integrator code raises. What is
    left is the room context, which reads the store: a channel closing
    announces the last of its recordings against a room an integrator may
    already have deleted, and a lookup that fails there must not take the
    closing down with it. The file is written by then, and the teardown that
    called this has a bot to remove from a conference afterwards.
    """

    def __init__(self, channel_id: str) -> None:
        self._channel_id = channel_id
        self._framework: RoomKit | None = None

    def set_framework(self, framework: RoomKit) -> None:
        """Wire the hooks this fires — ON_RECORDING_STARTED, ON_RECORDING_STOPPED."""
        self._framework = framework

    async def started(self, recording: TrackRecording) -> None:
        """Announce a recording that just opened on a track's first frame.

        Nothing is announced for a recording that has no handle. Opening one is
        the recorder's own work, on its own thread, and this is only reached
        once that has succeeded — a recording with no id is one there is nothing
        to say about yet, and the event exists to carry that id.
        """
        handle = recording.handle
        if handle is None:
            return
        event = ConferenceRecordingStarted(
            room_id=recording.room_id,
            track_id=recording.track.id,
            participant_id=recording.track.participant_id or "",
            id=handle.id,
            kind=recording.track.kind,
            sample_rate=recording.track.sample_rate,
            channels=recording.track.channels,
            codec=recording.track.codec,
        )
        await self._announce(
            recording.room_id,
            HookTrigger.ON_RECORDING_STARTED,
            event,
            "recording_started",
            {
                "id": event.id,
                "track_id": event.track_id,
                "participant_id": event.participant_id,
            },
        )

    async def stopped(self, finished: FinishedRecording) -> None:
        """Announce one closed recording, with where it was written."""
        event = ConferenceRecordingStopped(
            room_id=finished.room_id,
            track_id=finished.track.id,
            participant_id=finished.track.participant_id or "",
            id=finished.result.id,
            url=finished.result.url,
            duration_seconds=finished.result.duration_seconds,
            size_bytes=finished.result.size_bytes,
            format=finished.result.format,
        )
        await self._announce(
            finished.room_id,
            HookTrigger.ON_RECORDING_STOPPED,
            event,
            "recording_stopped",
            {
                "id": event.id,
                "track_id": event.track_id,
                "participant_id": event.participant_id,
                "url": event.url,
                "duration_seconds": event.duration_seconds,
                "size_bytes": event.size_bytes,
            },
        )

    async def stopped_all(self, finished: list[FinishedRecording]) -> None:
        """Announce several closed recordings, in the order they were finalized."""
        for recording in finished:
            await self.stopped(recording)

    async def _announce(
        self,
        room_id: str,
        trigger: HookTrigger,
        event: ConferenceRecordingStarted | ConferenceRecordingStopped,
        framework_event: str,
        data: dict[str, object],
    ) -> None:
        """Run the hooks, then emit the framework event, then give up quietly.

        The hooks come first because they are the documented interface and the
        framework event is the cross-channel echo of it; both are ASYNC, so
        neither can refuse a recording that has already been written.
        """
        if self._framework is None:
            return
        try:
            context = await self._framework._build_context(room_id)
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                trigger,
                event,
                context,
                skip_event_filter=True,
            )
            await self._framework._emit_framework_event(
                framework_event,
                room_id=room_id,
                channel_id=self._channel_id,
                data=data,
            )
        except Exception:
            logger.exception("Error announcing conference recording (%s)", trigger.value)

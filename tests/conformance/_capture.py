"""Scripted capture sources: the integrator's own sources, the marks a
scenario names in their backlog, and a realtime transport that subscribes
each session it accepts from the mark its metadata carries (roomkit-specs
``conformance/README.md``, "Scripted capture sources"; RFC §12.12).

A source is roomkit's public :class:`MockCaptureSource`, fed on the event
loop's thread, which stands in for the capture thread. The transport's
subscription callback only enqueues the frame, as a subscriber must (RFC
§12.12); the transport then hands each queued frame to the channel through
its ordinary inbound callback, in order.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.voice import AudioFrame, CaptureMark, CaptureSubscription, MockCaptureSource
from tests.conformance._realtime_io import _forwarded
from tests.conformance._realtime_reads import RecordingTransport

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.voice.base import VoiceSession
    from tests.conformance._harness import Harness

# The members of a setup's capture source, and of a channel's ``realtime``
# spec this module reads.
CAPTURE_SOURCE = {"id"}
CHANNEL_MEMBERS = {"capture_source"}
# The session metadata key a transport reads its mark from, as roomkit's
# local audio transport reads it.
CAPTURE_SINCE = "capture_since"


class CaptureTransport(RecordingTransport):
    """A realtime transport that captures from a shared source: it subscribes
    each session it accepts, from the mark the session's metadata carries,
    and unsubscribes it at its disconnect."""

    def __init__(self, source: MockCaptureSource) -> None:
        super().__init__()
        self._source = source
        self._subscriptions: dict[str, CaptureSubscription] = {}
        self._captured: list[tuple[VoiceSession, AudioFrame]] = []

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        await super().accept(session, connection)
        # A replay runs inside subscribe(): its frames are queued before it returns.
        self._subscriptions[session.id] = self._source.subscribe(
            partial(self._enqueue, session),
            since=session.metadata.get(CAPTURE_SINCE),
            name=f"session:{session.id}",
        )
        await self.deliver()

    async def disconnect(self, session: VoiceSession) -> None:
        subscription = self._subscriptions.pop(session.id, None)
        if subscription is not None:
            subscription.unsubscribe()
        await super().disconnect(session)

    def _enqueue(self, session: VoiceSession, frame: AudioFrame) -> None:
        self._captured.append((session, frame))

    async def deliver(self) -> None:
        """Hand each queued frame to the channel through the transport's
        ordinary inbound callback, in capture order."""
        while self._captured:
            session, frame = self._captured.pop(0)
            await self.simulate_client_audio(session, frame.data)


class CaptureSources:
    """The setup's capture sources, the marks named in them, and the realtime
    channels whose transports capture from each."""

    def __init__(self, specs: list[Mapping[str, Any]]) -> None:
        self._sources = {spec["id"]: MockCaptureSource() for spec in specs}
        self._marks: dict[str, CaptureMark] = {}
        self._channels: dict[str, list[tuple[CaptureTransport, RealtimeVoiceChannel]]] = {}
        for source in self._sources.values():
            source.start()

    def transport(self, source_id: str) -> CaptureTransport:
        """A transport capturing from the source *source_id*."""
        return CaptureTransport(self._source(source_id))

    def attach(self, source_id: str, channel: RealtimeVoiceChannel) -> None:
        """Count *channel*, whose transport captures from *source_id*, among
        those ``capture.feed`` waits for."""
        transport = channel.transport
        if not isinstance(transport, CaptureTransport):
            raise TypeError("a capturing channel holds a capture transport")
        self._channels.setdefault(source_id, []).append((transport, channel))

    def mark(self, name: str) -> CaptureMark:
        """The mark ``capture.mark`` named *name*."""
        if name not in self._marks:
            # A scenario's mistake, which maps to no error name.
            raise KeyError(f"no capture mark named {name}")
        return self._marks[name]

    def close(self) -> None:
        for source in self._sources.values():
            source.close()

    def _source(self, source_id: str) -> MockCaptureSource:
        if source_id not in self._sources:
            raise KeyError(f"no capture source {source_id} in the setup")
        return self._sources[source_id]

    async def call(self, spec: Mapping[str, Any]) -> None:
        source = self._source(spec["source_id"])
        if spec["call"] == "capture.mark":
            self._marks[spec["mark"]] = source.mark()
            return
        frames = [base64.b64decode(frame, validate=True) for frame in spec["frames"]]
        channels = self._channels.get(spec["source_id"], [])
        before = [set(channel._scheduled_tasks) for _, channel in channels]
        for frame in frames:
            source.feed(frame)
            for transport, _ in channels:
                await transport.deliver()
        # Returns once each channel forwarded or retained what it was handed.
        for (_, channel), started in zip(channels, before, strict=True):
            await _forwarded(channel, started)


CAPTURE_ARGUMENTS = {
    "capture.mark": {"source_id", "mark"},
    "capture.feed": {"source_id", "frames"},
}


async def capture_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    await harness.capture.call(spec)
    return None


CAPTURE_CALLS = dict.fromkeys(CAPTURE_ARGUMENTS, capture_call)

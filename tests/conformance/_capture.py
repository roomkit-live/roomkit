"""Scripted capture sources: the integrator's own sources, the marks a
scenario names in their backlog, a realtime transport that subscribes each
session it accepts from the mark its metadata carries, and scripted
subscribers (roomkit-specs ``conformance/README.md``, "Scripted capture
sources"; RFC §12.12).

A source is roomkit's public :class:`MockCaptureSource`, fed on the event
loop's thread, which stands in for the capture thread. The transport's
subscription callback only enqueues the frame, as a subscriber must (RFC
§12.12); the transport then hands each queued frame to the channel through
its ordinary inbound callback, in order. A scripted subscriber's callback
records the frame and returns, or raises once it recorded it.
"""

from __future__ import annotations

import base64
import threading
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

# The members of a setup's capture source, the source's constructor
# arguments beside its id, and of a channel's ``realtime`` spec this module
# reads.
SOURCE_OPTIONS = {
    "sample_rate",
    "channels",
    "block_duration_ms",
    "backlog_seconds",
    "max_backlog_bytes",
}
CAPTURE_SOURCE = {"id"} | SOURCE_OPTIONS
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


class ScriptedSubscriber:
    """A subscriber a scenario attaches to a source: its callback records
    each frame it is handed, then returns or, with ``raises``, raises. With
    frames to capture during its replay, the source captures them on another
    thread as the callback records its first replayed frame, and the callback
    waits for that capture before it returns."""

    def __init__(self, source_id: str, source: MockCaptureSource, spec: Mapping[str, Any]) -> None:
        self.source_id = source_id
        self.frames: list[AudioFrame] = []
        self.subscription: CaptureSubscription | None = None
        self._source = source
        self._raises = bool(spec.get("raises", False))
        self._during_replay = _frames(spec.get("captures_during_replay", []))
        self._replaying = False

    def subscribe(self, name: str, since: CaptureMark | None) -> None:
        # A replay runs inside subscribe(): what the callback receives before
        # it returns, from a mark, is the replay.
        self._replaying = since is not None
        try:
            self.subscription = self._source.subscribe(self._record, since=since, name=name)
        finally:
            self._replaying = False

    def unsubscribe(self) -> None:
        assert self.subscription is not None
        self.subscription.unsubscribe()

    def _record(self, frame: AudioFrame) -> None:
        self.frames.append(frame)
        if self._replaying and self._during_replay:
            frames, self._during_replay = self._during_replay, []
            capture = threading.Thread(target=self._capture, args=(frames,))
            capture.start()
            capture.join()
        if self._raises:
            raise RuntimeError("the scripted subscriber raises")

    def _capture(self, frames: list[bytes]) -> None:
        for frame in frames:
            self._source.feed(frame)

    def read(self) -> dict[str, Any]:
        subscription = self.subscription
        assert subscription is not None
        return {
            "frames": [
                {
                    "data": base64.b64encode(frame.data).decode(),
                    "sample_rate": frame.sample_rate,
                    "channels": frame.channels,
                    "sample_width": frame.sample_width,
                }
                for frame in self.frames
            ],
            "replayed_bytes": subscription.replayed_bytes,
            "truncated": subscription.truncated,
            "active": subscription.active,
        }


def _frames(encoded: list[str]) -> list[bytes]:
    return [base64.b64decode(frame, validate=True) for frame in encoded]


class CaptureSources:
    """The setup's capture sources, the marks named in them, the realtime
    channels whose transports capture from each, and the scripted
    subscribers."""

    def __init__(self, specs: list[Mapping[str, Any]]) -> None:
        self._sources = {
            spec["id"]: MockCaptureSource(
                **{name: value for name, value in spec.items() if name in SOURCE_OPTIONS}
            )
            for spec in specs
        }
        self._marks: dict[str, CaptureMark] = {}
        self._channels: dict[str, list[tuple[CaptureTransport, RealtimeVoiceChannel]]] = {}
        self._subscribers: dict[str, ScriptedSubscriber] = {}
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

    async def call(self, spec: Mapping[str, Any]) -> Any:
        name = spec["call"]
        if name == "capture.unsubscribe":
            self._subscriber(spec["subscriber"]).unsubscribe()
            return None
        source = self._source(spec["source_id"])
        if name == "capture.mark":
            self._marks[spec["mark"]] = source.mark()
        elif name == "capture.feed":
            await self._feed(spec["source_id"], source, _frames(spec["frames"]))
        elif name == "capture.subscribe":
            self._subscribe(spec, source)
        elif name == "capture.start":
            source.start()
        elif name == "capture.stop":
            source.stop()
        else:
            return self._read(spec["source_id"], source)
        return None

    async def _feed(self, source_id: str, source: MockCaptureSource, frames: list[bytes]) -> None:
        channels = self._channels.get(source_id, [])
        before = [set(channel._scheduled_tasks) for _, channel in channels]
        for frame in frames:
            source.feed(frame)
            for transport, _ in channels:
                await transport.deliver()
        # Returns once each channel forwarded or retained what it was handed.
        for (_, channel), started in zip(channels, before, strict=True):
            await _forwarded(channel, started)

    def _subscribe(self, spec: Mapping[str, Any], source: MockCaptureSource) -> None:
        name = spec["subscriber"]
        if name in self._subscribers:
            # A scenario's mistake, which maps to no error name.
            raise KeyError(f"a scripted subscriber named {name} already subscribed")
        since = self.mark(spec["since"]) if "since" in spec else None
        subscriber = ScriptedSubscriber(spec["source_id"], source, spec)
        self._subscribers[name] = subscriber
        subscriber.subscribe(name, since)

    def _subscriber(self, name: str) -> ScriptedSubscriber:
        if name not in self._subscribers:
            raise KeyError(f"no scripted subscriber named {name}")
        return self._subscribers[name]

    def _read(self, source_id: str, source: MockCaptureSource) -> dict[str, Any]:
        return {
            "started": source.started,
            "start_count": source.start_count,
            "subscribers": {
                name: subscriber.read()
                for name, subscriber in self._subscribers.items()
                if subscriber.source_id == source_id
            },
        }


CAPTURE_ARGUMENTS = {
    "capture.mark": {"source_id", "mark"},
    "capture.feed": {"source_id", "frames"},
    "capture.subscribe": {"source_id", "subscriber", "since", "raises", "captures_during_replay"},
    "capture.unsubscribe": {"subscriber"},
    "capture.start": {"source_id"},
    "capture.stop": {"source_id"},
    "capture.read": {"source_id"},
}


async def capture_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    return await harness.capture.call(spec)


CAPTURE_CALLS = dict.fromkeys(CAPTURE_ARGUMENTS, capture_call)

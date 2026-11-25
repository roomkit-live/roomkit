"""A conference's media: its arrivals' attributes, the audio its SFU delivers on
a track, the lanes that recognize it and the per-track recordings
(roomkit-specs ``conformance/README.md``, "Scripted conference media";
RFC §12.10.2, §12.10.4, §12.10.8).

``conference.open``'s media inputs build the lanes' scripted VAD, stages and
recognizer and a scripted recorder whose writes a gate may hold;
``conference.feed`` delivers frames on a published track, ``conference.settle``
waits for the lanes and recordings to finish what they were handed,
``conference.read`` reads what the lanes' stages and recognizer received, and
``recording.read`` with ``channel_id`` reads what the recorder wrote.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from roomkit import (
    ConferenceRecordingConfig,
    ConferenceRecordingStarted,
    ConferenceRecordingStopped,
    ConferenceTranscription,
)
from roomkit.models.event import RoomEvent, SystemContent
from roomkit.recorder import (
    MediaRecorder,
    MediaRecordingConfig,
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
)
from roomkit.voice.pipeline import AudioPipelineConfig, RecordingEncryption
from tests.conformance._conference_sfu import ScriptedSFU, conference_work
from tests.conformance._voice import _vad
from tests.conformance._voice_pipeline import Trace, TracedVAD, add_pipeline_stages
from tests.conformance._voice_stt import ScriptedSTT
from tests.conformance._voice_transport import feed_frame

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates

# The longest a recorder write waits at its gate: a gate no step opens must not
# keep the writer's thread, and with it the run's end, waiting for ever.
_WRITE_GATE_BOUND_S = 10.0

# The conference.open members this module reads.
MEDIA_OPEN = {"vad", "stt", "agc", "denoiser", "recording", "max_queued_frames", "address_keys"}
# The conference.join members a scripted SFU's arrival carries (§12.10.2).
ARRIVAL = {"display_name", "metadata", "client_metadata", "asserts_provenance"}


class _AddressedBackend(ScriptedSFU):
    """The scripted SFU, naming a caller's address under keys of its own."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = tuple(keys)

    @property
    def address_keys(self) -> tuple[str, ...]:
        return self._keys


class _NamedEncryption(RecordingEncryption):
    """A recording encryption known by its name alone; it leaves files as they are."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def encrypt_file(self, path: str) -> str:
        return path


@dataclass
class _Recording:
    """What the scripted recorder holds of one recording it opened."""

    config: MediaRecordingConfig
    tracks: list[RecordingTrack] = field(default_factory=list)
    chunks: list[bytes] = field(default_factory=list)
    stopped: bool = False


class ConferenceRecorder(MediaRecorder):
    """Records each recording it opens, in order, and holds a track's writes at
    that track's gate until a step opens it.

    The writes come on the channel's own thread, never the event loop's: a
    write there waits for the gate on the loop, bounded. A recording takes no
    frame once finalized, as a container closed over its streams takes none.
    """

    def __init__(self, gates: Gates, write_gates: Mapping[str, str]) -> None:
        self._gates = gates
        self._write_gates = dict(write_gates)
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self.recordings: dict[str, _Recording] = {}

    @property
    def name(self) -> str:
        return "ScriptedConferenceRecorder"

    def on_recording_start(self, config: MediaRecordingConfig) -> MediaRecordingHandle:
        handle = MediaRecordingHandle(
            id=f"rec{len(self.recordings) + 1}",
            room_id="",
            started_at=datetime.now(UTC),
        )
        self.recordings[handle.id] = _Recording(config)
        return handle

    def on_track_added(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        self.recordings[handle.id].tracks.append(track)

    def on_track_removed(self, handle: MediaRecordingHandle, track: RecordingTrack) -> None:
        return None

    def on_data(
        self,
        handle: MediaRecordingHandle,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float | None,
    ) -> None:
        if (gate := self._write_gates.get(track.id)) is not None:
            self._hold(gate)
        recording = self.recordings[handle.id]
        if not recording.stopped:
            recording.chunks.append(bytes(data))

    def _hold(self, gate: str) -> None:
        # On the loop's own thread the write cannot wait for a step to open
        # the gate: it says it reached it and goes on.
        if threading.get_ident() == self._loop_thread:
            self._loop.call_soon(self._gates.reach, gate)
            return
        waiting = asyncio.run_coroutine_threadsafe(self._gates.hold(gate), self._loop)
        try:
            waiting.result(timeout=_WRITE_GATE_BOUND_S)
        except TimeoutError:
            waiting.cancel()

    def on_recording_stop(self, handle: MediaRecordingHandle) -> MediaRecordingResult:
        recording = self.recordings[handle.id]
        recording.stopped = True
        return MediaRecordingResult(
            id=handle.id,
            url=f"mock://{handle.id}",
            tracks=list(recording.tracks),
            format=recording.config.format,
            size_bytes=sum(len(chunk) for chunk in recording.chunks),
        )

    def read(self) -> list[dict[str, Any]]:
        return [
            {
                "tracks": [_track(track) for track in recording.tracks],
                "config": _config(recording.config),
                "chunks": [base64.b64encode(chunk).decode("ascii") for chunk in recording.chunks],
                "stopped": recording.stopped,
            }
            for recording in list(self.recordings.values())
        ]


def _track(track: RecordingTrack) -> dict[str, Any]:
    return {
        "track_id": track.id,
        "kind": track.kind,
        "channel_id": track.channel_id,
        "participant_id": track.participant_id,
        "codec": track.codec,
        "sample_rate": track.sample_rate,
        "channels": track.channels,
    }


def _config(config: MediaRecordingConfig) -> dict[str, Any]:
    return {
        "format": config.format,
        "storage": config.storage,
        "metadata": dict(config.metadata),
        "encryption": config.encryption.name if config.encryption is not None else None,
        "storage_encrypted_at_rest": config.storage_encrypted_at_rest,
    }


@dataclass
class ConferenceMedia:
    """What the scenario scripted of a conference's media, kept to be read."""

    trace: Trace = field(default_factory=list)
    recognizer: ScriptedSTT | None = None
    recorder: ConferenceRecorder | None = None


def media_backend(spec: Mapping[str, Any]) -> ScriptedSFU:
    """The scripted SFU ``conference.open`` names: with its own address keys."""
    if "address_keys" in spec:
        return _AddressedBackend(list(spec["address_keys"]))
    return ScriptedSFU()


def media_options(
    spec: Mapping[str, Any], harness: Harness, default_stt: Any
) -> tuple[dict[str, Any], ConferenceMedia]:
    """The ConferenceChannel keywords of ``conference.open``'s media inputs."""
    media = ConferenceMedia()
    options: dict[str, Any] = {"stt": default_stt}
    if "stt" in spec:
        if spec["stt"] is not None:
            media.recognizer = ScriptedSTT(spec["stt"], harness.gates)
        options["stt"] = media.recognizer
    if {"vad", "agc", "denoiser"} & spec.keys():
        decisions = [_vad(d) if d is not None else None for d in spec.get("vad", [])]
        pipeline = AudioPipelineConfig(vad=TracedVAD(media.trace, decisions))
        add_pipeline_stages(
            pipeline, {k: spec[k] for k in ("agc", "denoiser") if k in spec}, media.trace
        )
        options["pipeline"] = pipeline
    if "recording" in spec:
        recording = spec["recording"]
        encryption = recording.get("encryption")
        options["recording"] = ConferenceRecordingConfig(
            storage=recording.get("storage", "local"),
            format=recording.get("format", "wav"),
            metadata=dict(recording.get("metadata", {})),
            encryption=_NamedEncryption(encryption) if encryption is not None else None,
            storage_encrypted_at_rest=recording.get("storage_encrypted_at_rest", False),
        )
        media.recorder = ConferenceRecorder(harness.gates, recording.get("write_gates", {}))
        options["recorder"] = media.recorder
    if "max_queued_frames" in spec:
        options["max_queued_frames"] = spec["max_queued_frames"]
    return options, media


def arrival(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The keywords of the scripted SFU's arrival ``conference.join`` names."""
    return {member: spec[member] for member in ARRIVAL if member in spec}


def _fixture(harness: Harness, spec: Mapping[str, Any]) -> ConferenceFixture:
    return harness.conferences[spec["channel_id"]]


async def _feed(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = _fixture(harness, spec)
    track = fixture.backend.tracks[spec["track_id"]]
    for _ in range(spec.get("repeat", 1)):
        for encoded in spec["frames"]:
            await fixture.backend.simulate_audio(track, feed_frame(spec, encoded))


async def _room_work(fixture: ConferenceFixture) -> None:
    """The channel's own work for its rooms: what a binding change or a
    track's end started (its unsubscriptions, its lanes and recordings closed)."""
    pending = conference_work({fixture.channel.channel_id: fixture})
    if pending:
        await asyncio.wait(pending, timeout=_WRITE_GATE_BOUND_S)


async def settle_tracks(fixture: ConferenceFixture, track_ids: list[str] | None) -> None:
    """Waits until the channel's own work for its rooms ended, and the named
    tracks' lanes and recordings (every one when None) took up what they were
    handed.

    It reads the channel's internals: the Python core offers no public wait
    for a lane's or a recording's work."""
    channel = fixture.channel
    await _room_work(fixture)
    for track_id, lane in list(channel.active_lanes.items()):
        if track_ids is None or track_id in track_ids:
            await lane.drain()
    recordings = channel._recorder
    if recordings is not None:
        for track_id, recording in list(recordings._open.items()):
            if track_ids is None or track_id in track_ids:
                await recording.writer.drain()
    await _room_work(fixture)


async def _settle(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await settle_tracks(_fixture(harness, spec), spec.get("track_ids"))
    await harness.idle()


async def read_media(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    """``conference.read``'s members of the lanes: what their scripted stages
    and recognizer received."""
    media = _fixture(harness, spec).media
    recognizer = media.recognizer if media is not None else None
    return {
        "stages": list(media.trace) if media is not None else [],
        "stt_streams": recognizer.read() if recognizer is not None else [],
    }


def read_recordings(harness: Harness, channel_id: str) -> dict[str, Any]:
    """``recording.read`` of a conference: what its scripted recorder wrote,
    and the loss the channel exposes for the conference's room."""
    fixture = harness.conferences[channel_id]
    recorder = fixture.media.recorder if fixture.media is not None else None
    if recorder is None:
        raise ValueError(f"conference {channel_id} was opened without a recording")
    rooms = fixture.channel.info["rooms"]
    dropped = (
        rooms[fixture.room]["recording_dropped_frames"]
        if fixture.room in rooms
        else fixture.channel._recorder.dropped_frames(fixture.room)  # ty: ignore[possibly-missing-attribute]
    )
    return {"recordings": recorder.read(), "dropped": dropped}


def conference_payload(payload: Any) -> dict[str, Any] | None:
    """What a conference's lane or recording hook names (README, "Scripted
    conference media"): its room, track and participant, and the
    transcription's text or the recording's facts."""
    if isinstance(payload, ConferenceTranscription):
        return {
            "room_id": payload.room_id,
            "track_id": payload.track_id,
            "participant_id": payload.participant_id,
            "text": payload.text,
        }
    if isinstance(payload, ConferenceRecordingStarted):
        return {
            "room_id": payload.room_id,
            "track_id": payload.track_id,
            "participant_id": payload.participant_id,
            "recording_id": payload.id,
            "kind": getattr(payload.kind, "value", payload.kind),
            "sample_rate": payload.sample_rate,
            "channels": payload.channels,
            "codec": payload.codec,
        }
    if isinstance(payload, ConferenceRecordingStopped):
        return {
            "room_id": payload.room_id,
            "track_id": payload.track_id,
            "participant_id": payload.participant_id,
            "recording_id": payload.id,
            "url": payload.url,
            "format": payload.format,
        }
    if isinstance(payload, RoomEvent) and isinstance(payload.content, SystemContent):
        data = payload.content.data or {}
        if payload.content.code in {"speech_start", "speech_end"} and "track_id" in data:
            return {
                "room_id": payload.room_id,
                "track_id": data["track_id"],
                "participant_id": data.get("participant_id"),
            }
    return None


MEDIA_CALLS = {"conference.feed": _feed, "conference.settle": _settle}
MEDIA_ARGUMENTS = {
    "conference.feed": {"channel_id", "track_id", "frames", "sample_rate", "channels", "repeat"},
    "conference.settle": {"channel_id", "track_ids"},
}

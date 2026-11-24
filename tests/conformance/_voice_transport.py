"""The scripted voice transport's traits and session steps for RFC §12.2 voice.

``voice.open`` names the capabilities the transport declares, the ids it gives
its sessions and where the AEC reference comes from; ``voice.join``,
``voice.leave`` and ``voice.dtmf_received`` drive its sessions.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.voice import AudioChunk, AudioFrame, VoiceCapability, VoiceSession, VoiceSessionState
from roomkit.voice.backends import MockVoiceBackend, MockVoiceCall
from roomkit.voice.pipeline import DTMFEvent

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness

_CAPABILITIES = {
    "interruption": VoiceCapability.INTERRUPTION,
    "barge_in": VoiceCapability.BARGE_IN,
    "native_aec": VoiceCapability.NATIVE_AEC,
    "native_agc": VoiceCapability.NATIVE_AGC,
    "dtmf_inband": VoiceCapability.DTMF_INBAND,
    "dtmf_signaling": VoiceCapability.DTMF_SIGNALING,
    "native_bridge": VoiceCapability.NATIVE_BRIDGE,
}
_DEFAULT_CAPABILITIES = ["interruption", "barge_in"]


class PipelineTransport(MockVoiceBackend):
    """The scripted transport's traits ``voice.open`` names: capabilities, the
    sessions' ids, where the AEC reference comes from, out-of-band DTMF."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        capabilities = VoiceCapability.NONE
        for name in spec.get("capabilities", _DEFAULT_CAPABILITIES):
            capabilities |= _CAPABILITIES[name]
        super().__init__(capabilities=capabilities)
        self._feeds_aec_reference = bool(spec.get("feeds_aec_reference", False))
        self._playback_callback = bool(spec.get("playback_callback", False))
        self._reference_continues = bool(spec.get("reference_continues", False))
        self.session_ids: dict[str, str] = dict(spec.get("session_ids", {}))
        self.chunks: list[tuple[str, AudioChunk]] = []
        self._played_callbacks: list[Callable[[VoiceSession, AudioFrame], None]] = []
        self._dtmf_callbacks: list[Callable[[VoiceSession, DTMFEvent], Any]] = []

    @property
    def feeds_aec_reference(self) -> bool:
        return self._feeds_aec_reference

    @property
    def supports_playback_callback(self) -> bool:
        return self._playback_callback

    def on_audio_played(
        self, callback: Callable[[VoiceSession, AudioFrame], None]
    ) -> Callable[[], None] | None:
        self._played_callbacks.append(callback)
        return lambda: self._played_callbacks.remove(callback)

    def on_dtmf_received(self, callback: Callable[[VoiceSession, DTMFEvent], Any]) -> None:
        self._dtmf_callbacks.append(callback)

    async def simulate_dtmf_received(self, session: VoiceSession, event: DTMFEvent) -> None:
        for callback in self._dtmf_callbacks:
            result = callback(session, event)
            if hasattr(result, "__await__"):
                await result

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """The mock's connect, its session named by ``session_ids`` when it names one."""
        wanted = self.session_ids.pop(participant_id, None)
        if wanted is None:
            return await super().connect(room_id, participant_id, channel_id, metadata=metadata)
        session = VoiceSession(
            id=wanted,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        self.calls.append(
            MockVoiceCall(
                method="connect",
                args={
                    "room_id": room_id,
                    "participant_id": participant_id,
                    "channel_id": channel_id,
                    "metadata": metadata,
                },
            )
        )
        for callback in self._session_ready_callbacks:
            result = callback(session)
            if hasattr(result, "__await__"):
                task = asyncio.get_running_loop().create_task(result)
                task.add_done_callback(log_task_exception)
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        """Hang up: the session the transport handed out reads ENDED."""
        session.state = VoiceSessionState.ENDED
        await super().disconnect(session)

    async def played(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Each chunk sent, logged, then reported played when the transport reports blocks."""
        if isinstance(audio, bytes):
            audio = _one_chunk(audio)
        previous: AudioChunk | None = None
        async for chunk in audio:
            self.chunks.append((session.id, chunk))
            if previous is not None:
                self._report_played(session, previous, ended=False)
            previous = chunk
            yield chunk
        if previous is not None:
            self._report_played(session, previous, ended=True)

    def _report_played(self, session: VoiceSession, chunk: AudioChunk, *, ended: bool) -> None:
        if not self._playback_callback:
            return
        metadata: dict[str, Any] = {}
        if ended:
            metadata["playback_ended"] = True
        if self._reference_continues:
            metadata["reference_continues"] = True
        frame = AudioFrame(
            data=chunk.data,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            metadata=metadata,
        )
        for callback in list(self._played_callbacks):
            callback(session, frame)

    def send_audio_sync(self, session: VoiceSession, chunk: AudioChunk) -> None:
        self.chunks.append((session.id, chunk))
        super().send_audio_sync(session, chunk)


async def _one_chunk(data: bytes) -> AsyncIterator[AudioChunk]:
    yield AudioChunk(data=data)


def chunk_record(chunk: AudioChunk) -> dict[str, Any]:
    return {
        "data": base64.b64encode(chunk.data).decode("ascii"),
        "sample_rate": chunk.sample_rate,
        "channels": chunk.channels,
    }


def feed_frame(spec: Mapping[str, Any], encoded: str) -> AudioFrame:
    """One inbound frame of ``voice.feed``, in the format the step names."""
    return AudioFrame(
        data=base64.b64decode(encoded, validate=True),
        sample_rate=spec.get("sample_rate", 16000),
        channels=spec.get("channels", 1),
        timestamp_ms=spec.get("timestamp_ms"),
    )


async def _join(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    participant = spec["participant_id"]
    if "session_id" in spec:
        fixture.backend.session_ids[participant] = spec["session_id"]
    session = await kit.join(
        fixture.room_id, fixture.channel.channel_id, participant_id=participant
    )
    assert isinstance(session, VoiceSession)
    fixture.sessions[participant] = session
    fixture.joined.setdefault(participant, []).append(session)


async def _leave(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    await kit.leave(fixture.sessions[spec["participant_id"]])


async def _dtmf_received(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    await fixture.backend.simulate_dtmf_received(
        fixture.sessions[spec["participant_id"]],
        DTMFEvent(digit=spec["digit"], duration_ms=spec["duration_ms"]),
    )


TRANSPORT_CALLS = {
    "voice.join": _join,
    "voice.leave": _leave,
    "voice.dtmf_received": _dtmf_received,
}
TRANSPORT_ARGUMENTS = {
    "voice.join": {"channel_id", "participant_id", "session_id"},
    "voice.leave": {"channel_id", "participant_id"},
    "voice.dtmf_received": {"channel_id", "participant_id", "digit", "duration_ms"},
}
TRANSPORT_OPEN_ARGUMENTS = {
    "capabilities",
    "feeds_aec_reference",
    "playback_callback",
    "reference_continues",
    "session_ids",
}
FEED_ARGUMENTS = {"sample_rate", "channels", "timestamp_ms"}

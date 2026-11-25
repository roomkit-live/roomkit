"""Scripted transport, recognition and pipeline inputs for RFC §9 voice hooks."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from roomkit import VoiceChannel
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel.base import BackchannelDecision
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.pipeline.dtmf.base import DTMFEvent
from roomkit.voice.pipeline.turn.base import TurnDecision
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.context import TTSContext
from roomkit.voice.tts.mock import MockTTSProvider
from tests.conformance._voice_bridge import (
    BRIDGE_ARGUMENTS,
    BRIDGE_CALLS,
    BRIDGE_OPEN_ARGUMENTS,
    bridge_config,
    read_transports,
)
from tests.conformance._voice_pipeline import (
    READ_ARGUMENTS,
    STAGE_ARGUMENTS,
    Trace,
    TracedDiarization,
    TracedDTMF,
    TracedTurnDetector,
    TracedVAD,
    add_pipeline_stages,
    read_stages,
)
from tests.conformance._voice_stt import (
    OPEN_ARGUMENTS,
    STT_ARGUMENTS,
    STT_CALLS,
    ScriptedSTT,
    recognizer,
    settle_voice,
    turn_detector,
)
from tests.conformance._voice_transport import (
    FEED_ARGUMENTS,
    PRIMARY,
    TRANSPORT_ARGUMENTS,
    TRANSPORT_CALLS,
    TRANSPORT_OPEN_ARGUMENTS,
    PipelineTransport,
    chunk_record,
    feed_frame,
    transport_of,
)

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


class _Transport(PipelineTransport):
    """An optional gate holds a real playback open while the caller talks."""

    def __init__(self, gates: Gates, gate: str | None, spec: Mapping[str, Any]) -> None:
        super().__init__(spec)
        self.gates = gates
        self.gate = gate

    async def send_audio(
        self, session: VoiceSession, audio: bytes | AsyncIterator[AudioChunk]
    ) -> None:
        await super().send_audio(session, self.played(session, audio))
        if self.gate:
            self.start_playing(session)
            try:
                await self.gates.hold(self.gate)
            finally:
                self.stop_playing(session)


class _StreamingSTT(MockSTTProvider):
    """A recognizer emitting the configured partials on audio, then its final at EOF."""

    def __init__(self, transcripts: list[str], partials: list[str]) -> None:
        super().__init__(transcripts, streaming=True)
        self.partials = partials

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        first = True
        async for chunk in audio_stream:
            self.calls.append(chunk)
            if first:
                first = False
                for text in self.partials:
                    yield TranscriptionResult(text=text, is_final=False)
        text, reported = self._next()
        yield TranscriptionResult(text=text, language=reported)


class _StreamingTTS(MockTTSProvider):
    """Synthesize each sentence actually supplied by the channel."""

    @property
    def supports_streaming_input(self) -> bool:
        return True

    async def synthesize_stream_input(
        self,
        text_stream: AsyncIterator[str],
        *,
        voice: str | None = None,
        context: TTSContext | None = None,
    ) -> AsyncIterator[AudioChunk]:
        async for text in text_stream:
            async for chunk in self.synthesize_stream(text, voice=voice, context=context):
                yield chunk


@dataclass
class VoiceFixture:
    channel: VoiceChannel
    backend: _Transport
    tts: MockTTSProvider | None
    sessions: dict[str, VoiceSession]
    room_id: str
    trace: Trace
    joined: dict[str, list[VoiceSession]]
    recognizer: ScriptedSTT | None = None
    # Every transport by name, the one built here as PRIMARY (voice.add_backend).
    transports: dict[str, PipelineTransport] = field(default_factory=dict)
    new_transport: Callable[[Mapping[str, Any]], PipelineTransport] | None = None


def _vad(spec: Mapping[str, Any]) -> VADEvent:
    event = VADEvent(
        type=VADEventType(spec["type"]), **{k: v for k, v in spec.items() if k != "type"}
    )
    if event.type in (VADEventType.SPEECH_START, VADEventType.SPEECH_END):
        # The provider's pre-roll / speech buffer, as returned by a real VAD.
        event.audio_bytes = b"\x01\x02" * 160
    return event


async def _open(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    trace: Trace = []
    # A scripted recognizer without a VAD runs the channel in continuous mode.
    vad = None
    if "vad" in spec or "stt" not in spec:
        vad = TracedVAD(trace, [_vad(s) if s is not None else None for s in spec.get("vad", [])])
    pipeline = AudioPipelineConfig(vad=vad)
    if "turn_incomplete_wait_ms" in spec:
        pipeline.turn_incomplete_wait_ms = spec["turn_incomplete_wait_ms"]
    if "speakers" in spec:
        pipeline.diarization = TracedDiarization(
            trace, [DiarizationResult(**s) for s in spec["speakers"]]
        )
    if "tones" in spec:
        pipeline.dtmf = TracedDTMF(trace, [DTMFEvent(**t) for t in spec["tones"]])
    if "turns" in spec:
        pipeline.turn_detector = TracedTurnDetector(
            trace, [TurnDecision(is_complete=v) for v in spec["turns"]]
        )
    add_pipeline_stages(pipeline, spec, trace)
    if (scripted_turns := turn_detector(spec, harness.gates)) is not None:
        pipeline.turn_detector = scripted_turns
    interruption = InterruptionConfig(
        strategy=InterruptionStrategy.IMMEDIATE, allow_during_first_ms=0
    )
    if "backchannels" in spec:
        interruption = InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC,
            allow_during_first_ms=0,
            backchannel_detector=MockBackchannelDetector(
                [BackchannelDecision(is_backchannel=v) for v in spec["backchannels"]]
            ),
        )
    backend = _Transport(harness.gates, spec.get("playback_gate"), spec)
    # ``tts: null`` builds the channel with no TTS provider (§12.7.3).
    tts = None if "tts" in spec else _StreamingTTS()
    scripted = recognizer(spec, harness.gates)
    stt: MockSTTProvider | ScriptedSTT | None = scripted
    if "transcripts" in spec:
        if "partials" in spec:
            stt = _StreamingSTT(spec["transcripts"], spec["partials"])
        else:
            stt = MockSTTProvider(spec["transcripts"])
    channel = VoiceChannel(
        spec["channel_id"],
        backend=backend,
        tts=tts,
        stt=stt,
        pipeline=pipeline,
        interruption=interruption,
        bridge=bridge_config(spec),
        batch_mode=spec.get("batch_mode", False),
        pipeline_speakers=spec.get("pipeline_speakers", False),
    )
    kit.register_channel(channel)
    await kit.attach_channel(spec["room_id"], channel.channel_id)
    fixture = VoiceFixture(channel, backend, tts, {}, spec["room_id"], trace, {}, scripted)
    fixture.transports[PRIMARY] = backend
    fixture.new_transport = lambda traits: _Transport(
        harness.gates, traits.get("playback_gate"), traits
    )
    harness.voice_fixtures[channel.channel_id] = fixture
    for participant in spec["participants"]:
        session = await kit.join(spec["room_id"], channel.channel_id, participant_id=participant)
        assert isinstance(session, VoiceSession)
        fixture.sessions[participant] = session
        fixture.joined[participant] = [session]


async def _feed(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    session = fixture.sessions[spec["participant_id"]]
    transport = transport_of(fixture, spec["participant_id"])
    for _ in range(spec.get("repeat", 1)):
        for encoded in spec["frames"]:
            await transport.simulate_audio_received(session, feed_frame(spec, encoded))


async def _settle(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await settle_voice(harness.voice_fixtures[spec["channel_id"]].channel)


async def _say(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    await fixture.channel.say(fixture.sessions[spec["participant_id"]], spec["text"])


async def _stream(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    room_id = next(iter(fixture.sessions.values())).room_id
    binding = await kit.store.get_binding(room_id, fixture.channel.channel_id)
    assert binding is not None
    room = await kit.get_room(room_id)
    assert room is not None
    response = RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id="assistant", channel_type=ChannelType.AI),
        content=TextContent(body=""),
    )

    async def chunks() -> AsyncIterator[str]:
        for chunk in spec["chunks"]:
            yield chunk

    await fixture.channel.deliver_stream(
        chunks(), response, binding, RoomContext(room=room, bindings=[binding])
    )


async def _barge_in(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    await transport_of(fixture, spec["participant_id"]).simulate_barge_in(
        fixture.sessions[spec["participant_id"]]
    )


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    fixture = harness.voice_fixtures[spec["channel_id"]]
    transports = list(fixture.transports.values())
    read: dict[str, Any] = {}
    if fixture.recognizer is not None:
        read["stt_streams"] = fixture.recognizer.read()
    return read | {
        "synthesized": [call["text"] for call in fixture.tts.calls] if fixture.tts else [],
        "audio": {
            name: [
                base64.b64encode(audio).decode("ascii")
                for transport in transports
                for sid, audio in transport.sent_audio
                if sid == session.id
            ]
            for name, session in fixture.sessions.items()
        },
        "chunks": {
            name: [
                chunk_record(chunk)
                for transport in transports
                for sid, chunk in transport.chunks
                if sid == session.id
            ]
            for name, session in fixture.sessions.items()
        },
        "transports": read_transports(fixture),
        "sessions": {
            name: [{"session_id": s.id, "state": s.state.value} for s in joined]
            for name, joined in fixture.joined.items()
        },
        "stages": read_stages(fixture.trace, spec),
    }


def voice_payload(payload: Any) -> dict[str, Any] | None:
    session = payload if isinstance(payload, VoiceSession) else getattr(payload, "session", None)
    if not isinstance(session, VoiceSession):
        return None
    data: dict[str, Any] = {"participant_id": session.participant_id}
    if is_dataclass(payload) and not isinstance(payload, VoiceSession):
        for field in fields(payload):
            if field.name not in {"session", "timestamp", "participant_id"}:
                value = getattr(payload, field.name)
                if isinstance(value, (str, float, int, bool)) or value is None:
                    data[field.name] = value
    return data


VOICE_CALLS = {
    "voice.open": _open,
    "voice.feed": _feed,
    "voice.say": _say,
    "voice.stream": _stream,
    "voice.barge_in": _barge_in,
    "voice.settle": _settle,
    "voice.read": _read,
    **TRANSPORT_CALLS,
    **STT_CALLS,
    **BRIDGE_CALLS,
}
VOICE_ARGUMENTS = {
    "voice.open": {
        "channel_id",
        "room_id",
        "participants",
        "vad",
        "transcripts",
        "partials",
        "speakers",
        "tones",
        "turns",
        "bridge",
        "backchannels",
        "playback_gate",
        *STAGE_ARGUMENTS,
        *TRANSPORT_OPEN_ARGUMENTS,
        *OPEN_ARGUMENTS,
        *BRIDGE_OPEN_ARGUMENTS,
    },
    "voice.feed": {"channel_id", "participant_id", "frames", "repeat", *FEED_ARGUMENTS},
    "voice.say": {"channel_id", "participant_id", "text"},
    "voice.stream": {"channel_id", "chunks"},
    "voice.barge_in": {"channel_id", "participant_id"},
    "voice.settle": {"channel_id"},
    "voice.read": {"channel_id", *READ_ARGUMENTS},
    **TRANSPORT_ARGUMENTS,
    **STT_ARGUMENTS,
    **BRIDGE_ARGUMENTS,
}

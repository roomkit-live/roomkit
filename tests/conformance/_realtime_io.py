"""A realtime session's provider traits, its audio both ways, its speech,
transcriptions and responses, a held or failed start, and an audio pipeline
on its channel (roomkit-specs ``conformance/README.md``, "Scripted realtime
sessions"; RFC §12.4, §12.4.1).

The scripted provider (:class:`IOProvider`) declares the traits the channel's
``realtime`` spec names, records each connect it is asked, holds or fails
the connect a ``realtime.start`` scripts, and answers ``inject_text`` and a
tool result's submission as the spec scripts them. ``realtime.read`` reads
what the provider and the transport received.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conformance._conference_model import MODEL_CALLS, model_call
from tests.conformance._realtime_reads import configuration
from tests.conformance._voice import _vad
from tests.conformance._voice_pipeline import (
    Trace,
    TracedDiarization,
    TracedVAD,
    add_pipeline_stages,
    read_stages,
)

if TYPE_CHECKING:
    from roomkit import ConferenceChannel, RoomKit
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates


class ScriptedProviderError(RuntimeError):
    """The scripted provider's own failure, which the caller is handed
    (README, Errors: ``provider_failed``)."""


# The members of a channel's ``realtime`` spec this module reads.
SESSION_MEMBERS = {
    "full_duplex",
    "supports_tools",
    "supports_mid_session_reconfigure",
    "inject_result",
    "submit_gate",
    "submit_error",
    "mute_on_tool_call",
    "pipeline",
}
INJECT_RESULT_MEMBERS = {"status", "reason", "retryable"}
PIPELINE_MEMBERS = {"vad", "speakers", "aec", "agc", "denoiser", "recorder"}
# What ``realtime.start`` and ``realtime.read`` take beside their own members.
START_MEMBERS = {"gate", "fail"}
READ_MEMBERS = {"stages", "calls"}


class IOProvider(MockRealtimeProvider):
    """The provider's traits, its connects and its scripted answers."""

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        super().__init__(full_duplex=spec.get("full_duplex", False))
        self._spec = spec
        self._gates = gates
        self.connects: list[dict[str, Any]] = []
        # The session a start is creating, from its connect on: the steps of
        # a held start address it (README, "Scripted realtime sessions").
        self.connecting: VoiceSession | None = None
        self.hold: Mapping[str, Any] = {}

    @property
    def supports_tools(self) -> bool:
        return self._spec.get("supports_tools", True)

    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return self._spec.get("supports_mid_session_reconfigure", True)

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        self.connecting = session
        self.connects.append(
            {"server_vad": kwargs.get("server_vad", True), **configuration(session, kwargs)}
        )
        if (gate := self.hold.get("gate")) is not None:
            await self._gates.hold(gate)
        if (message := self.hold.get("fail")) is not None:
            raise ScriptedProviderError(message)
        await super().connect(session, **kwargs)

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult | None:
        result = await super().inject_text(session, text, role=role, silent=silent)
        if "inject_result" not in self._spec:
            return result
        scripted = self._spec["inject_result"]
        return None if scripted is None else VoiceInjectionResult(**scripted)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        if (gate := self._spec.get("submit_gate")) is not None:
            await self._gates.hold(gate)
        if (message := self._spec.get("submit_error")) is not None:
            raise ScriptedProviderError(message)
        await super().submit_tool_result(session, call_id, result)


def channel_options(spec: Mapping[str, Any], trace: Trace) -> dict[str, Any]:
    """The channel's keywords the ``realtime`` spec names: the muted input
    and the pipeline, whose scripted stages log into *trace*."""
    options: dict[str, Any] = {}
    if "mute_on_tool_call" in spec:
        options["mute_on_tool_call"] = spec["mute_on_tool_call"]
    if "pipeline" in spec:
        options["pipeline"] = _pipeline(spec["pipeline"], trace)
    return options


def _pipeline(spec: Mapping[str, Any], trace: Trace) -> AudioPipelineConfig:
    """The pipeline the spec scripts, written as on ``voice.open``; no VAD
    without ``vad``."""
    vad = None
    if "vad" in spec:
        vad = TracedVAD(trace, [_vad(s) if s is not None else None for s in spec["vad"]])
    pipeline = AudioPipelineConfig(vad=vad)
    if "speakers" in spec:
        pipeline.diarization = TracedDiarization(
            trace, [DiarizationResult(**s) for s in spec["speakers"]]
        )
    add_pipeline_stages(pipeline, spec, trace)
    return pipeline


def read(
    provider: IOProvider, channel: RealtimeVoiceChannel, trace: Trace, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """What ``realtime.read`` adds: the audio the provider and the transport
    received, each mute of the input, the provider's connects, the stages'
    calls."""
    transport = _transport(channel)
    return {
        "provider_audio": [_b64(audio) for _, audio in provider.sent_audio],
        "transport_audio": [_b64(audio) for _, audio in transport.sent_audio],
        "input_muted": [
            call.args["muted"] for call in transport.calls if call.method == "set_input_muted"
        ],
        "connects": list(provider.connects),
        "stages": read_stages(trace, spec),
    }


def parked_send_workers(channel: RealtimeVoiceChannel) -> tuple[set[asyncio.Task[Any]], bool]:
    """The channel's resident audio send workers, and whether any of them
    still has work: a worker parked on its empty queue runs nothing, and
    lives until its session ends."""
    workers: set[asyncio.Task[Any]] = set()
    working = False
    for session_id, worker in list(channel._audio_send_workers.items()):
        workers.add(worker)
        queue = channel._audio_send_queues.get(session_id)
        parked = (
            queue is not None
            and queue.empty()
            and any(
                not getter.done()
                for getter in queue._getters  # ty: ignore[unresolved-attribute]
            )
        )
        working = working or not (worker.done() or parked)
    return workers, working


IO_ARGUMENTS = {
    "realtime.transcribe": {"channel_id", "text", "role", "is_final"},
    "realtime.speech": {"channel_id", "edge"},
    "realtime.audio": {"channel_id", "source", "frames", "room_id"},
    "realtime.response": {"channel_id", "edge"},
    "realtime.wait_idle": {"channel_id", "room_id", "timeout_ms"},
}


async def io_call(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    if spec["channel_id"] in harness.conferences:
        return await _conference_model_call(harness.conferences[spec["channel_id"]], spec)
    fixture = harness.realtime_sessions[spec["channel_id"]]
    provider, session = fixture.provider, fixture.session
    match spec["call"]:
        case "realtime.transcribe":
            await provider.simulate_transcription(
                session, spec["text"], spec["role"], spec.get("is_final", True)
            )
        case "realtime.speech":
            edge = (
                provider.simulate_speech_start
                if spec["edge"] == "start"
                else provider.simulate_speech_end
            )
            await edge(session)
        case "realtime.response":
            await _response(provider, session, spec["edge"])
        case "realtime.audio":
            # The session of the room the call names, the channel's first
            # when it names none (README, ``realtime.audio``).
            if "room_id" in spec:
                session = fixture.session_in(spec["room_id"])
            await _audio(fixture.channel, provider, session, spec)
        case "realtime.wait_idle":
            return await _went_idle(fixture.channel, spec)
        case _:
            raise ValueError(f"Unknown realtime call {spec['call']}")
    return None


async def _conference_model_call(fixture: ConferenceFixture, spec: Mapping[str, Any]) -> Any:
    """A ``realtime.*`` call naming a conference: its model's (README,
    "Scripted conference model")."""
    if spec["call"] not in MODEL_CALLS or fixture.model is None:
        raise ValueError(f"{spec['call']} needs a realtime voice channel or a conference model")
    if spec["call"] == "realtime.wait_idle":
        return await _went_idle(fixture.channel, spec)
    await model_call(fixture, spec)
    return None


async def _response(provider: MockRealtimeProvider, session: VoiceSession, edge: str) -> None:
    if edge == "start":
        await provider.simulate_response_start(session)
    else:
        await provider.simulate_response_end(session)


async def _went_idle(
    channel: RealtimeVoiceChannel | ConferenceChannel, spec: Mapping[str, Any]
) -> bool:
    """The channel's ``wait_idle``: whether the room's sessions went idle in time."""
    try:
        await channel.wait_idle(spec["room_id"], spec["timeout_ms"] / 1000)
    except TimeoutError:
        return False
    return True


async def _audio(
    channel: RealtimeVoiceChannel,
    provider: IOProvider,
    session: VoiceSession,
    spec: Mapping[str, Any],
) -> None:
    """The caller's frames the transport receives, returned once the channel
    forwarded or retained each one, or the provider's output."""
    frames = [base64.b64decode(frame, validate=True) for frame in spec["frames"]]
    before = set(channel._scheduled_tasks)
    for frame in frames:
        if spec["source"] == "transport":
            await _transport(channel).simulate_client_audio(session, frame)
        else:
            await provider.simulate_audio(session, frame)
    await _forwarded(channel, before)


async def _forwarded(channel: RealtimeVoiceChannel, before: set[asyncio.Task[Any]]) -> None:
    """Wait for the tasks that forward the caller's audio to the provider,
    started since *before*: a frame the channel retains starts none."""
    workers = set(channel._audio_send_workers.values())
    forwarding = [
        task
        for task in channel._scheduled_tasks - before
        if not task.done()
        and task not in workers
        and task.get_name().startswith(_FORWARDING_TASKS)
    ]
    if forwarding:
        await asyncio.wait(forwarding, timeout=_FORWARD_BOUND_S)


# The channel's tasks that forward the caller's audio to the provider: the
# direct path's and the pipeline's (the provider's own audio has its resident
# send worker, left to the idle wait).
_FORWARDING_TASKS = ("rt_client_audio:", "rt_send_audio:")
_FORWARD_BOUND_S = 5.0


def _transport(channel: RealtimeVoiceChannel) -> MockRealtimeTransport:
    transport = channel.transport
    if not isinstance(transport, MockRealtimeTransport):
        raise TypeError("a scripted realtime channel holds the mock transport")
    return transport


def _b64(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")

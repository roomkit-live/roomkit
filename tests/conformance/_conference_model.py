"""The conference's realtime model, scripted (roomkit-specs
``conformance/README.md``, "Scripted conference model"; RFC §12.10.12).

``conference.open``'s ``realtime`` scripts the model its ``tools`` build: the
provider's full-duplex trait, the input rate it declares, and what each
connect it is asked does. The provider's own callbacks are the ``realtime.*``
steps that name the conference (a response, its audio, a transcription, its
speech detection, a delegation); ``conference.read``'s ``model`` reads what
the provider received. ``conference.settle``'s ``mix`` waits for the mix to
send the model what its tracks hold, ``conference.feed``'s ``pause_ms`` leaves
a track silent first, ``conference.plug``'s ``realtime`` plugs a scripted
model, and ``conference.open``'s ``default_grants`` are the channel's.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from roomkit import ConferenceGrants, ConferenceRealtimeConfig
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.mock import MockRealtimeProvider

if TYPE_CHECKING:
    from roomkit import ConferenceChannel
    from roomkit.channels._conference_mixer import ConferenceMixer
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates

# The conference.open members this module reads.
MODEL_OPEN = frozenset({"realtime", "default_grants"})
_REALTIME_MEMBERS = frozenset({"full_duplex", "input_sample_rate", "connects"})
_CONNECT_MEMBERS = frozenset({"gate", "fail", "error_type"})
# The provider calls ``conference.read``'s ``model.calls`` lists.
_LISTED_CALLS = ("interrupt", "disconnect")
# The ``realtime.*`` calls that may name a conference opened with tools.
MODEL_CALLS = frozenset(
    {
        "realtime.response",
        "realtime.wait_idle",
        "realtime.transcribe",
        "realtime.speech",
        "realtime.audio",
        "realtime.delegate",
    }
)
# How long ``conference.settle``'s ``mix`` waits for the mix to send what it holds.
_MIX_BOUND_S = 5.0


def _error_named(name: str) -> type[Exception]:
    """An error type whose name is *name*: the type a failed connect's
    ON_ERROR reports by name (RFC §12.10.12)."""
    return type(name, (RuntimeError,), {})


class ConferenceModel(MockRealtimeProvider):
    """The conference's realtime model: its trait, its connects and what it
    received, and the step issuing a tool call woken by its result."""

    def __init__(self, spec: Mapping[str, Any], gates: Gates) -> None:
        if unknown := set(spec) - _REALTIME_MEMBERS:
            raise ValueError(f"conference realtime has no member {sorted(unknown)}")
        super().__init__(full_duplex=spec.get("full_duplex", False))
        self._gates = gates
        self._scripts: list[Mapping[str, Any]] = list(spec.get("connects", []))
        for script in self._scripts:
            if unknown := set(script) - _CONNECT_MEMBERS:
                raise ValueError(f"a connect script has no member {sorted(unknown)}")
            if ("fail" in script) != ("error_type" in script):
                raise ValueError("a connect's fail and error_type go together")
        self.connects: list[dict[str, Any]] = []
        self.injections: list[dict[str, Any]] = []
        # The session a connect is establishing, while the connect runs: the
        # steps of a held connect address it.
        self.connecting: VoiceSession | None = None
        # The last session connected: a late callback names it once its room
        # let it go.
        self.last: VoiceSession | None = None
        self.replies: dict[str, asyncio.Future[str]] = {}

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        script = (
            self._scripts[len(self.connects)] if len(self.connects) < len(self._scripts) else {}
        )
        self.connects.append(
            {
                "session": session.id,
                "room_id": session.room_id,
                "input_sample_rate": kwargs.get("input_sample_rate"),
                "tools": [tool["name"] for tool in kwargs.get("tools") or []],
                "failed": "fail" in script,
            }
        )
        self.connecting = session
        try:
            if (gate := script.get("gate")) is not None:
                await self._gates.hold(gate)
            if (message := script.get("fail")) is not None:
                raise _error_named(script["error_type"])(message)
            await super().connect(session, **kwargs)
            self.last = session
        finally:
            self.connecting = None

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult:
        self.injections.append(
            {"session": session.id, "text": text, "role": role, "silent": silent}
        )
        return await super().inject_text(session, text, role=role, silent=silent)

    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        await super().submit_tool_result(session, call_id, result)
        if (reply := self.replies.get(call_id)) is not None and not reply.done():
            reply.set_result(result)

    async def call(self, session: VoiceSession, spec: Mapping[str, Any]) -> Any:
        """``conference.tool_call``: issue the call, return what the model received."""
        call_id = spec["call_id"]
        if call_id in self.replies:
            raise ValueError(f"conference call {call_id} is already outstanding")
        reply = asyncio.get_running_loop().create_future()
        self.replies[call_id] = reply
        try:
            await self.simulate_tool_call(
                session, call_id, spec["name"], spec.get("arguments", {})
            )
            wire = await reply
        finally:
            del self.replies[call_id]
        try:
            return json.loads(wire)
        except json.JSONDecodeError:
            return wire

    def read(self) -> dict[str, Any]:
        """``conference.read``'s ``model``: what the provider received, in order."""
        return {
            "connects": [dict(connect) for connect in self.connects],
            "audio": [
                {"session": session_id, "data": _b64(data), "samples": len(data) // 2}
                for session_id, data in self.sent_audio
            ],
            "heard": {
                "data": _b64(b"".join(data for _, data in self.sent_audio)),
                "samples": sum(len(data) for _, data in self.sent_audio) // 2,
            },
            "injections": [dict(injection) for injection in self.injections],
            "results": [
                {"session": session_id, "call_id": call_id, "result": result}
                for session_id, call_id, result in self.tool_results
            ],
            "delegation_outputs": [
                {
                    "session": session_id,
                    "delegation_id": delegation_id,
                    "text": text,
                    "spoken": spoken,
                }
                for session_id, delegation_id, text, spoken in self.delegation_outputs
            ],
            "calls": [
                {"call": call.method, "session": call.args.get("session_id")}
                for call in self.calls
                if call.method in _LISTED_CALLS
            ],
        }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def conference_model(spec: Mapping[str, Any], gates: Gates) -> ConferenceModel | None:
    """The model ``conference.open`` builds: one with ``tools``, scripted by
    its ``realtime``, which needs them."""
    if "tools" not in spec:
        if "realtime" in spec:
            raise ValueError("conference.open's realtime scripts the model tools build")
        return None
    return ConferenceModel(spec.get("realtime", {}), gates)


def model_options(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ConferenceChannel keywords of this module's ``conference.open`` members."""
    if "default_grants" not in spec:
        return {}
    return {"default_grants": ConferenceGrants(**spec["default_grants"])}


def realtime_settings(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ConferenceRealtimeConfig members ``conference.open``'s ``realtime`` sets."""
    rate = spec.get("realtime", {}).get("input_sample_rate")
    return {} if rate is None else {"input_sample_rate": rate}


class _CountedSends:
    """The mix's sends in flight: wraps the mixer's sender to count them."""

    def __init__(self, mixer: ConferenceMixer) -> None:
        self.in_flight = 0
        send = mixer._send

        async def counted(room_id: str, data: bytes) -> None:
            self.in_flight += 1
            try:
                await send(room_id, data)
            finally:
                self.in_flight -= 1

        mixer._send = counted  # ty: ignore[invalid-assignment]


_SENDS: weakref.WeakKeyDictionary[ConferenceMixer, _CountedSends] = weakref.WeakKeyDictionary()


def watch_mix(channel: ConferenceChannel) -> None:
    """Count the sends of the channel's mix, which ``conference.settle``'s
    ``mix`` waits out. It reads the channel's internals: the Python core
    offers no public wait for its mix."""
    mixer = channel._realtime.mixer
    _SENDS[mixer] = _CountedSends(mixer)


def _mixing(mixer: ConferenceMixer) -> bool:
    """Whether the mix still holds a full window of a track, or is sending one."""
    sends = _SENDS.get(mixer)
    if sends is not None and sends.in_flight:
        return True
    for room in mixer._rooms.values():
        if room.sample_rate is None:
            continue
        window = mixer._window_bytes(room.sample_rate)
        if any(len(ring.buffer) >= window for ring in room.rings.values()):
            return True
    return False


async def settle_mix(fixture: ConferenceFixture) -> None:
    """Wait until the mix sent the model every full window its tracks held, or
    found no session to send it to (README, ``conference.settle``)."""
    mixer = fixture.channel._realtime.mixer
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _MIX_BOUND_S
    while _mixing(mixer):
        if loop.time() > deadline:
            raise TimeoutError(f"the mix did not send what it held within {_MIX_BOUND_S} s")
        await asyncio.sleep(0.005)


def _session(fixture: ConferenceFixture) -> VoiceSession:
    """The room's model session: the connected one, the one a held connect is
    establishing, or, once the room let it go, the last one connected, which a
    late callback names."""
    sessions = fixture.channel.get_room_sessions(fixture.room)
    if sessions:
        return sessions[0]
    model = fixture.model
    if isinstance(model, ConferenceModel):
        if model.connecting is not None:
            return model.connecting
        if model.last is not None:
            return model.last
    raise ValueError("the conference model must be connected, or connecting, first")


async def model_call(fixture: ConferenceFixture, spec: Mapping[str, Any]) -> None:
    """A ``realtime.*`` provider callback of the conference's model."""
    model = fixture.model
    if model is None:
        raise ValueError(f"{spec['call']} names a conference opened with no tools")
    session = _session(fixture)
    match spec["call"]:
        case "realtime.response":
            if spec["edge"] == "start":
                await model.simulate_response_start(session)
            else:
                await model.simulate_response_end(session)
        case "realtime.transcribe":
            await model.simulate_transcription(
                session, spec["text"], spec["role"], spec.get("is_final", True)
            )
        case "realtime.speech":
            edge = (
                model.simulate_speech_start
                if spec["edge"] == "start"
                else model.simulate_speech_end
            )
            await edge(session)
        case "realtime.audio":
            if spec["source"] != "provider":
                raise ValueError("a conference model's audio is the provider's")
            for frame in spec["frames"]:
                await model.simulate_audio(session, base64.b64decode(frame, validate=True))
        case "realtime.delegate":
            await model.simulate_delegation(session, spec["delegation_id"], spec["target"])
        case call:
            raise ValueError(f"{call} does not name a conference's model")


async def plug_model(
    harness: Harness, fixture: ConferenceFixture, spec: Mapping[str, Any]
) -> None:
    """``conference.plug`` of a realtime model, scripted by its ``realtime``,
    which ``conference.read`` reads."""
    script = spec.get("realtime", {})
    model = ConferenceModel(script, harness.gates)
    settings = realtime_settings({"realtime": script})
    await fixture.channel.plug_realtime(ConferenceRealtimeConfig(provider=model, **settings))
    fixture.model = model


async def pause(spec: Mapping[str, Any]) -> None:
    """``conference.feed``'s ``pause_ms``: the track silent that long first."""
    if (ms := spec.get("pause_ms")) is not None:
        await asyncio.sleep(ms / 1000)


def read_model(fixture: ConferenceFixture) -> dict[str, Any]:
    """``conference.read``'s ``model``, null for a conference with none."""
    model = fixture.model
    return {"model": model.read() if isinstance(model, ConferenceModel) else None}


def model_work(fixtures: Iterable[ConferenceFixture]) -> tuple[list[asyncio.Task[Any]], bool]:
    """The conference models' background work an ``idle`` step waits for, and
    whether a response is still being published.

    The model's room tasks and its reports are waited for, but a response's
    own task, which lives until the provider ends it: that one is busy only
    while it has a chunk to publish in its turn, or its publication, or its
    closing, is in flight."""
    tasks: list[asyncio.Task[Any]] = []
    busy = False
    for fixture in fixtures:
        if fixture.model is None:
            continue
        channel = fixture.channel
        realtime = channel._realtime
        for room in list(realtime._rooms.values()):
            for task in list(room.tasks):
                if task.done():
                    continue
                if _speaks(task):
                    busy = busy or _publishing(task, channel)
                else:
                    tasks.append(task)
        tasks.extend(task for task in realtime._reports if not task.done())
        busy = busy or bool(channel._voice._activity._rooms)
    return tasks, busy


def _speaks(task: asyncio.Task[Any]) -> bool:
    return getattr(task.get_coro(), "__qualname__", "") == "ConferenceRealtime._speak"


def _publishing(task: asyncio.Task[Any], channel: ConferenceChannel) -> bool:
    """Whether a response's task has work it does now: not yet begun, its turn
    free to take, or in its turn with a chunk or its end waiting."""
    frame = task.get_coro().cr_frame  # ty: ignore[possibly-missing-attribute]
    utterance = frame.f_locals.get("utterance") if frame is not None else None
    if utterance is None:
        return False
    playback = utterance.playback
    if playback is None:
        return True
    if playback.bot is None:
        room = channel._voice._speaking.get(playback.room_id)
        return room is None or not room.turn.locked()
    return not utterance.queue.empty()

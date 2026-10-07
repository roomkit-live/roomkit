"""A conference with a realtime model keeps the contracts a realtime voice
channel keeps with the same provider (RMK-523, RFC §12.10.12, §12.4, §7.5).

Three behaviours, each run on both hosts: a muted or output-muted binding
withholds the model's answer and its audio; a call the provider issues while
``connect()`` runs is served once the session is up, and reported cancelled
when the start fails; a provider error reaches ON_ERROR, and a session the
provider ended is let go.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    HookExecution,
    HookTrigger,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels import _conference_realtime
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.injection import VoiceInjectionResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import _Source, until

ROOM = "r"
HOSTS = pytest.mark.parametrize("host", ["voice", "conference"])

_TOOLS = [
    {
        "name": "lookup",
        "description": "look up",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


class _Provider(MockRealtimeProvider):
    """Records each injection's silence. From ``connect()`` (a receive loop
    the connect started can) it may issue a call, abandon it, end the
    session, wait on *hold*, then fail the start."""

    def __init__(
        self,
        *,
        call_on_connect: bool = False,
        cancel_on_connect: bool = False,
        end_on_connect: bool = False,
        hold: asyncio.Event | None = None,
        fail_connect: bool = False,
        delegate_on_connect: bool = False,
        full_duplex: bool = False,
    ) -> None:
        super().__init__(full_duplex=full_duplex)
        self.injections: list[tuple[str, bool]] = []
        self.connecting = asyncio.Event()
        self._call_on_connect = call_on_connect
        self._cancel_on_connect = cancel_on_connect
        self._end_on_connect = end_on_connect
        self._hold = hold
        self._fail_connect = fail_connect
        self._delegate_on_connect = delegate_on_connect

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        if self._call_on_connect:
            await self.simulate_tool_call(session, "c0", "lookup", {"q": "first"})
        if self._cancel_on_connect:
            await self.simulate_tool_call_cancellation(session, ["c0"])
        if self._delegate_on_connect:
            await self.simulate_delegation(session, "d0", "integrator")
        if self._end_on_connect:
            # From the receive loop the connect started, while the handshake
            # still waits on the server.
            await asyncio.ensure_future(self._end(session))
        self.connecting.set()
        if self._hold is not None:
            await self._hold.wait()
        if self._fail_connect:
            raise ConnectionError("handshake refused")

    async def _end(self, session: VoiceSession) -> None:
        session.state = VoiceSessionState.ENDED
        await self.simulate_error(session, "ws_1008", "policy violation")

    async def inject_text(  # type: ignore[override]
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> VoiceInjectionResult:
        self.injections.append((text, silent))
        return VoiceInjectionResult(status="sent")


class _Host:
    """One realtime host in a room, built the same way for both kinds."""

    def __init__(self, kind: str, provider: _Provider, handler_ran: list[Any]) -> None:
        self.kind = kind
        self.provider = provider
        self.kit = RoomKit()
        if kind == "voice":

            async def voice_handler(name: str, arguments: dict[str, Any]) -> str:
                handler_ran.append(arguments)
                return "served"

            self.channel: Any = RealtimeVoiceChannel(
                "host",
                provider=provider,
                transport=MockRealtimeTransport(),
                tools=_TOOLS,
                tool_handler=voice_handler,
            )
        else:

            async def conference_handler(
                room_id: str, name: str, arguments: dict[str, Any]
            ) -> str:
                handler_ran.append(arguments)
                return "served"

            config = ConferenceRealtimeConfig(
                provider=provider, tools=_TOOLS, tool_handler=conference_handler
            )
            self.channel = ConferenceChannel(
                "host", backend=MockConferenceBackend(), realtime=config
            )
        self.kit.register_channel(self.channel)
        self.kit.register_channel(_Source("src", ChannelType.AI))

    async def attach(self) -> None:
        await self.kit.create_room(room_id=ROOM)
        await self.kit.attach_channel(ROOM, "host")
        await self.kit.attach_channel(ROOM, "src")

    async def start(self) -> VoiceSession | None:
        """The host's session, ``None`` when the start failed."""
        if self.kind == "voice":
            try:
                return await self.channel.start_session(ROOM, "u1", "ws")
            except (ConnectionError, RuntimeError):
                return None
        return await self.channel._realtime.ensure_session(ROOM)

    def sessions(self) -> list[VoiceSession]:
        if self.kind == "voice":
            return list(self.channel.get_room_sessions(ROOM))
        session = self.channel._realtime.session_for(ROOM)
        return [] if session is None else [session]

    def serve_with(self, handler: Any) -> None:
        """Serve the room's calls with *handler* from now on."""
        if self.kind == "voice":
            self.channel._tool_handler = handler
        else:
            self.channel._realtime.config.tool_handler = handler

    def audio_heard(self) -> int:
        if self.kind == "voice":
            return sum(len(audio) for _, audio in self.channel.transport.sent_audio)
        return sum(len(chunk.data) for chunk in self.channel._backend.published_audio)


def _observe_calls(kit: RoomKit) -> list[ToolCallEvent]:
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def observe(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


def _observe_errors(kit: RoomKit) -> list[dict[str, Any]]:
    heard: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def on_error(event: Any, ctx: Any) -> None:
        heard.append({"channel": event.source.channel_id, **event.metadata})

    return heard


async def _settle() -> None:
    for _ in range(30):
        await asyncio.sleep(0)


# -- a muted binding withholds the answer ------------------------------------


@HOSTS
@pytest.mark.parametrize(
    ("mute", "silent"), [(None, False), ("mute", True), ("mute_output", True)]
)
async def test_a_broadcast_is_injected_silently_under_a_muted_binding(
    host: str, mute: str | None, silent: bool
) -> None:
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    await rt.start()
    if mute is not None:
        await getattr(rt.kit, mute)(ROOM, "host")

    await rt.kit.send_event(ROOM, "src", TextContent(body="hello"), chain_depth=1)
    await until(lambda: bool(provider.injections))
    await rt.kit.close()

    assert provider.injections == [("hello", silent)]


@HOSTS
async def test_no_provider_audio_reaches_the_people_under_output_mute(host: str) -> None:
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    await rt.kit.mute_output(ROOM, "host")

    await provider.simulate_response_start(session)
    for _ in range(3):
        await provider.simulate_audio(session, b"\x01\x00" * 480)
    await provider.simulate_response_end(session)
    await asyncio.sleep(0.1)
    heard = rt.audio_heard()
    await rt.kit.close()

    assert heard == 0


# -- a call issued while connect() runs ----------------------------------------


@HOSTS
async def test_a_call_issued_during_connect_is_served_once_the_session_is_up(
    host: str,
) -> None:
    provider = _Provider(call_on_connect=True)
    ran: list[Any] = []
    rt = _Host(host, provider, ran)
    await rt.attach()
    observed = _observe_calls(rt.kit)

    assert await rt.start() is not None
    await until(lambda: bool(observed) and bool(provider.tool_results))
    await rt.kit.close()

    assert ran == [{"q": "first"}]
    assert [(call_id, result) for _, call_id, result in provider.tool_results] == [
        ("c0", "served")
    ]
    assert [(e.tool_call_id, e.cancelled, e.result) for e in observed] == [("c0", False, "served")]


@HOSTS
async def test_a_call_issued_during_a_failed_connect_is_reported_cancelled(
    host: str,
) -> None:
    provider = _Provider(call_on_connect=True, fail_connect=True)
    rt = _Host(host, provider, [])
    await rt.attach()
    observed = _observe_calls(rt.kit)

    assert await rt.start() is None
    await until(lambda: bool(observed))
    await rt.kit.close()

    # Nothing answers the model, and observers hear the call once, cancelled.
    assert provider.tool_results == []
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c0", True)]


# -- a provider error ------------------------------------------------------------


@HOSTS
async def test_a_provider_error_reaches_on_error_and_the_session_stays(host: str) -> None:
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    heard = _observe_errors(rt.kit)

    await provider.simulate_error(session, "rate_limited", "slow down")
    await until(lambda: bool(heard))
    sessions = rt.sessions()
    await rt.kit.close()

    assert heard == [
        {
            "channel": "host",
            "error": "slow down",
            "error_type": "rate_limited",
            "error_category": "realtime_provider",
        }
    ]
    assert sessions == [session]


@HOSTS
async def test_a_session_the_provider_ended_is_let_go_and_its_call_reported(
    host: str,
) -> None:
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    observed = _observe_calls(rt.kit)
    heard = _observe_errors(rt.kit)
    gate = asyncio.Event()
    channel_handler_ran: list[Any] = []

    async def block(*args: Any) -> str:
        channel_handler_ran.append(args)
        await gate.wait()
        return "late"

    rt.serve_with(block)
    await provider.simulate_tool_call(session, "c1", "lookup", {"q": "x"})
    await until(lambda: bool(channel_handler_ran))

    session.state = VoiceSessionState.ENDED
    await provider.simulate_error(session, "ws_1008", "policy violation")
    await until(lambda: not rt.sessions() and bool(observed) and bool(heard))
    disconnected = [c.args.get("session_id") for c in provider.calls if c.method == "disconnect"]
    await rt.kit.close()

    assert heard[0]["error_type"] == "ws_1008"
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c1", True)]
    assert provider.tool_results == []
    assert session.id in disconnected


# -- what a start that does not go through leaves --------------------------------


async def _never(*args: Any) -> str:
    await asyncio.Event().wait()
    return "never"


@HOSTS
async def test_a_call_the_provider_abandons_while_it_starts_is_never_answered(host: str) -> None:
    provider = _Provider(call_on_connect=True, cancel_on_connect=True)
    rt = _Host(host, provider, [])
    rt.serve_with(_never)
    await rt.attach()
    observed = _observe_calls(rt.kit)

    assert await rt.start() is not None
    await until(lambda: bool(observed))
    await _settle()
    await rt.kit.close()

    assert provider.tool_results == []
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c0", True)]


@HOSTS
async def test_a_start_cancelled_reports_its_call_and_disconnects(host: str) -> None:
    provider = _Provider(call_on_connect=True, hold=asyncio.Event())
    rt = _Host(host, provider, [])
    rt.serve_with(_never)
    await rt.attach()
    observed = _observe_calls(rt.kit)

    start = asyncio.ensure_future(rt.start())
    await provider.connecting.wait()
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    await until(lambda: bool(observed))
    disconnects = [c for c in provider.calls if c.method == "disconnect"]
    await rt.kit.close()

    assert provider.tool_results == []
    assert [(e.tool_call_id, e.cancelled) for e in observed] == [("c0", True)]
    assert disconnects


@HOSTS
async def test_a_session_the_provider_ends_while_it_starts_is_not_held(host: str) -> None:
    provider = _Provider(end_on_connect=True)
    rt = _Host(host, provider, [])
    await rt.attach()
    heard = _observe_errors(rt.kit)

    assert await rt.start() is None
    await until(lambda: bool(heard))
    sessions = rt.sessions()
    await rt.kit.close()

    assert heard[0]["error_type"] == "ws_1008"
    assert sessions == []


@HOSTS
async def test_an_on_error_hook_that_ends_the_session_does_not_wait_for_itself(
    host: str,
) -> None:
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    ended = asyncio.Event()

    @rt.kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def tear_down(event: Any, ctx: Any) -> None:
        if host == "voice":
            await rt.channel.end_session(session)
        else:
            await rt.channel.unplug_realtime()
        ended.set()

    await provider.simulate_error(session, "rate_limited", "slow down")
    await asyncio.wait_for(ended.wait(), timeout=2.0)
    sessions = rt.sessions()
    await rt.kit.close()

    assert sessions == []


async def test_the_conference_reconnects_once_the_provider_ended_its_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_conference_realtime, "CONNECT_COOLDOWN_S", 0.0)
    provider = _Provider()
    rt = _Host("conference", provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None

    session.state = VoiceSessionState.ENDED
    await provider.simulate_error(session, "ws_1011", "server error")
    await until(lambda: not rt.sessions())
    again = await rt.start()
    await rt.kit.close()

    assert again is not None and again is not session


# -- an injection, as the hook hears it ---------------------------------------


@HOSTS
async def test_a_broadcast_is_announced_as_the_injection_it_is(host: str) -> None:
    """RMK-530: one event on every host, naming the session, the role and
    the event the broadcast came from."""
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    heard: list[Any] = []

    @rt.kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
    async def audit(event: Any, ctx: Any) -> None:
        heard.append(event)

    sent = await rt.kit.send_event(ROOM, "src", TextContent(body="hello"), chain_depth=1)
    await until(lambda: bool(heard))
    await rt.kit.close()

    [event] = heard
    assert event.source.channel_id == "host"
    assert event.metadata == {
        "injected_role": "system",
        "session_id": session.id,
        "injected_from": {"channel_id": "src", "event_id": sent.id},
    }


@HOSTS
@pytest.mark.parametrize("ending", ["let-go", "provider-ended"])
async def test_an_injection_into_an_ended_session_is_not_sent(host: str, ending: str) -> None:
    """Refused whether the host let the session go or the provider ended it
    and the host still holds it."""
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    heard: list[Any] = []

    @rt.kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
    async def audit(event: Any, ctx: Any) -> None:
        heard.append(event)

    if ending == "provider-ended":
        session.state = VoiceSessionState.ENDED
        assert rt.sessions() == [session]
    elif host == "voice":
        await rt.channel.end_session(session)
    else:
        session.state = VoiceSessionState.ENDED
        rt.channel._realtime.detach_room(ROOM)
    result = await rt.channel.inject_text(session, "late", role="system")
    await _settle()
    await rt.kit.close()

    assert (result.status, result.reason) == ("not_sent", "realtime_session_gone")
    assert provider.injections == []
    assert heard == []


# -- a delegation, and what idle waits for ---------------------------------------


@HOSTS
async def test_a_delegation_with_no_backend_is_answered_aloud_and_announced(host: str) -> None:
    """RMK-528: the model never waits for an answer that does not come."""
    provider = _Provider()
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    heard: list[tuple[str, str]] = []

    @rt.kit.hook(HookTrigger.ON_REALTIME_DELEGATION, execution=HookExecution.ASYNC)
    async def announced(event: Any, ctx: Any) -> None:
        heard.append((event.delegation_id, event.target))

    await provider.simulate_delegation(session, "d1", "integrator")
    await until(
        lambda: (
            bool(heard)
            and any(call.method == "submit_delegation_output" for call in provider.calls)
        )
    )
    outputs = [
        (call.args.get("delegation_id"), call.args.get("spoken"))
        for call in provider.calls
        if call.method == "submit_delegation_output"
    ]
    await rt.kit.close()

    assert heard == [("d1", "integrator")]
    assert outputs == [("d1", True)]


@HOSTS
async def test_a_host_is_not_idle_while_a_call_runs_nor_before_its_answer(host: str) -> None:
    """RMK-528: a hand-back waiting for idle does not land mid-call."""
    release = asyncio.Event()

    async def slow(*args: Any) -> str:
        await release.wait()
        return "done"

    provider = _Provider()
    rt = _Host(host, provider, [])
    rt.serve_with(slow)
    await rt.attach()
    session = await rt.start()
    assert session is not None

    async def idle() -> bool:
        try:
            await rt.channel.wait_idle(ROOM, timeout=0.1)
        except TimeoutError:
            return False
        return True

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await _settle()
    while_running = await idle()
    release.set()
    await until(lambda: bool(provider.tool_results))
    before_answer = await idle()
    await provider.simulate_response_start(session)
    await provider.simulate_response_end(session)
    await _settle()
    after_answer = await idle()
    await rt.kit.close()

    assert (while_running, before_answer, after_answer) == (False, False, True)


LOUD = b"\x00\x40" * 480  # PCM16 well above the activity floor


async def _is_idle(rt: _Host) -> bool:
    try:
        await rt.channel.wait_idle(ROOM, timeout=0.3)
    except TimeoutError:
        return False
    return True


async def _done(*args: Any) -> str:
    return "done"


@HOSTS
@pytest.mark.parametrize("answer", ["audio", "transcript"])
@pytest.mark.parametrize("sent", ["fallback", "result"])
async def test_an_answer_said_inside_the_open_turn_ends_the_wait(
    host: str, sent: str, answer: str
) -> None:
    """A full-duplex model may answer inside the response already open, with
    no new response start: its audio or its words start the answer (RFC
    §12.4.1)."""
    provider = _Provider(full_duplex=True)
    rt = _Host(host, provider, [])
    rt.serve_with(_done)
    await rt.attach()
    session = await rt.start()
    assert session is not None
    await provider.simulate_response_start(session)
    if sent == "fallback":
        await provider.simulate_delegation(session, "d1", "integrator")
        await until(lambda: bool(provider.delegation_outputs))
    else:
        await provider.simulate_tool_call(session, "c1", "lookup", {})
        await until(lambda: bool(provider.tool_results))
    await _settle()
    if answer == "audio":
        await provider.simulate_audio(session, LOUD)
    else:
        await provider.simulate_transcription(session, "Sorry, I cannot", "assistant", False)
    await provider.simulate_response_end(session)
    await _settle()
    idle = await _is_idle(rt)
    await rt.kit.close()

    assert idle


class _AnswersBeforeSubmitReturns(_Provider):
    async def submit_tool_result(self, session: VoiceSession, call_id: str, result: str) -> None:
        await super().submit_tool_result(session, call_id, result)
        await self.simulate_response_start(session)


@HOSTS
async def test_an_answer_that_starts_before_the_send_returns_ends_the_wait(host: str) -> None:
    provider = _AnswersBeforeSubmitReturns()
    rt = _Host(host, provider, [])
    rt.serve_with(_done)
    await rt.attach()
    session = await rt.start()
    assert session is not None
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: bool(provider.tool_results))
    await provider.simulate_response_end(session)
    await _settle()
    idle = await _is_idle(rt)
    await rt.kit.close()

    assert idle


class _FallbackRefused(_Provider):
    async def submit_delegation_output(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("socket closed")


@HOSTS
async def test_a_fallback_that_could_not_be_sent_is_not_waited_for(host: str) -> None:
    provider = _FallbackRefused(full_duplex=True)
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    await provider.simulate_delegation(session, "d1", "integrator")
    await _settle()
    idle = await _is_idle(rt)
    await rt.kit.close()

    assert idle


@HOSTS
async def test_a_delegation_issued_while_connecting_is_answered(host: str) -> None:
    provider = _Provider(delegate_on_connect=True, full_duplex=True)
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    await until(lambda: bool(provider.delegation_outputs))
    outputs = [(d_id, spoken) for _, d_id, _, spoken in provider.delegation_outputs]
    await rt.kit.close()

    assert outputs == [("d0", True)]


@HOSTS
async def test_a_delegation_from_a_session_the_provider_ended_is_not_announced(
    host: str,
) -> None:
    provider = _Provider(full_duplex=True)
    rt = _Host(host, provider, [])
    await rt.attach()
    session = await rt.start()
    assert session is not None
    heard: list[str] = []

    @rt.kit.hook(HookTrigger.ON_REALTIME_DELEGATION, execution=HookExecution.ASYNC)
    async def announced(event: Any, ctx: Any) -> None:
        heard.append(event.delegation_id)

    session.state = VoiceSessionState.ENDED
    await provider.simulate_delegation(session, "d1", "integrator")
    await _settle()
    await rt.kit.close()

    assert (heard, provider.delegation_outputs) == ([], [])

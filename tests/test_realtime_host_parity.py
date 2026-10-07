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
    """Records each injection's silence; may issue a call from ``connect()``
    (a receive loop the connect started can) and then fail the start."""

    def __init__(self, *, call_on_connect: bool = False, fail_connect: bool = False) -> None:
        super().__init__()
        self.injections: list[tuple[str, bool]] = []
        self._call_on_connect = call_on_connect
        self._fail_connect = fail_connect

    async def connect(self, session: VoiceSession, **kwargs: Any) -> None:
        await super().connect(session, **kwargs)
        if self._call_on_connect:
            await self.simulate_tool_call(session, "c0", "lookup", {"q": "first"})
        if self._fail_connect:
            raise ConnectionError("handshake refused")

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
            except ConnectionError:
                return None
        return await self.channel._realtime.ensure_session(ROOM)

    def sessions(self) -> list[VoiceSession]:
        if self.kind == "voice":
            return list(self.channel.get_room_sessions(ROOM))
        session = self.channel._realtime.session_for(ROOM)
        return [] if session is None else [session]

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

    if host == "voice":
        rt.channel._tool_handler = block
    else:
        rt.channel._realtime.config.tool_handler = block
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

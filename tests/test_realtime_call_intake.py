"""Every realtime provider hands every call to the channel, which decides it
(RMK-440, RFC §12.4).

A call without an id, or under an id still in flight, a call the output cap
cut, and a call to a name RoomKit did not declare all reach the channel: it
refuses and reports each, sending nothing where no result can be named.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.elevenlabs import sdk_patch
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_openai_live import TOOL, _provider, _response_event
from tests.test_openai_live import _connect as live_connect

LOOKUP = [{"name": "lookup", "parameters": {"type": "object"}}]


def _observe(kit: RoomKit) -> list[Any]:
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append(event)

    return seen


async def test_the_channel_refuses_a_call_without_an_id_and_sends_nothing() -> None:
    ran: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=LOOKUP,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen = _observe(kit)
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "", "lookup", {})
    await until(lambda: bool(seen))

    assert ran == []
    assert provider.tool_results == []
    [event] = seen
    assert (event.name, event.is_error, event.cancelled) == ("lookup", True, False)
    assert "without an id" in str(event.result)
    await kit.close()


async def test_a_conference_refuses_a_call_without_an_id() -> None:
    ran: list[str] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=LOOKUP, tool_handler=handler)
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    seen = _observe(kit)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "", "lookup", {})
    await until(lambda: bool(seen))

    assert ran == []
    assert provider.tool_results == []
    assert [(e.name, e.is_error) for e in seen] == [("lookup", True)]
    await kit.close()


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="p1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


async def _gpt_live_calls(
    items: list[dict[str, Any] | tuple[str, dict[str, Any]]],
) -> tuple[dict[str, str], list[Any]]:
    """GPT-Live's books after *items* (an item, or a delegation id and an
    item), and what the channel heard."""
    provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
    session = _session()
    heard: list[Any] = []
    provider.on_tool_call(lambda _s, call_id, name, arguments: heard.append((call_id, arguments)))
    ws, _ = await live_connect(provider, session, tools=[TOOL])
    for entry in items:
        delegation, item = entry if isinstance(entry, tuple) else ("d1", entry)
        ws.push(_response_event({"type": "response.created"}, delegation_id=delegation))
        event = {"type": "response.output_item.done", "item": item}
        ws.push(_response_event(event, delegation_id=delegation))
    await asyncio.sleep(0.1)
    # Read before the disconnect clears the book: each open call with the
    # delegation it answers.
    booked = provider._open_tool_calls.get(session.id, {})
    open_calls = {call_id: key for call_id, (_, key) in booked.items()}
    await provider.disconnect(session)
    return open_calls, heard


async def test_gpt_live_hands_on_a_call_the_output_cap_cut() -> None:
    cut = '{"city": "Par'
    open_calls, heard = await _gpt_live_calls(
        [
            {
                "type": "function_call",
                "status": "incomplete",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": cut,
            }
        ]
    )

    # The fragment as text: the channel refuses it as unreadable, and answers.
    assert heard == [("c1", cut)]
    assert list(open_calls) == ["c1"]


async def test_gpt_live_hands_on_a_cut_call_whose_arguments_read_whole() -> None:
    _, heard = await _gpt_live_calls(
        [
            {
                "type": "function_call",
                "status": "incomplete",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            }
        ]
    )

    # Whole arguments are whole, cut or not (RFC §6.4), as on OpenAI Realtime.
    assert heard == [("c1", {"city": "Paris"})]


async def test_gpt_live_hands_on_id_less_and_duplicate_calls_untracked() -> None:
    call = {"type": "function_call", "name": "get_weather", "arguments": "{}"}
    open_calls, heard = await _gpt_live_calls(
        [{**call, "call_id": "c1"}, {**call, "call_id": "c1"}, {**call, "call_id": ""}]
    )

    assert [call_id for call_id, _ in heard] == ["c1", "c1", ""]
    # Only the call the channel may answer holds the response open.
    assert list(open_calls) == ["c1"]


async def test_gpt_live_keeps_an_id_in_flight_on_its_first_delegation() -> None:
    call = {"type": "function_call", "name": "get_weather", "arguments": "{}", "call_id": "c1"}
    open_calls, heard = await _gpt_live_calls([("d1", call), ("d2", call)])

    assert [call_id for call_id, _ in heard] == ["c1", "c1"]
    assert open_calls == {"c1": "d1"}


async def test_openai_realtime_books_only_a_call_it_can_answer() -> None:
    provider = OpenAIRealtimeProvider(api_key="sk-test")
    session = _session()
    provider._connections[session.id] = AsyncMock()
    provider._sessions[session.id] = session
    heard: list[str] = []
    provider.on_tool_call(lambda _s, call_id, *_: heard.append(call_id))

    for call_id in ("c1", "c1", ""):
        item = {"type": "function_call", "call_id": call_id, "name": "get_weather"}
        await provider._on_output_item_done(session, {"item": {**item, "arguments": "{}"}})

    assert heard == ["c1", "c1", ""]
    assert set(provider._open_tool_calls.get(session.id, {})) == {"c1"}
    assert provider._pending_responses[session.id].call_ids == {"c1"}


class _SdkLikeClientTools:
    """The SDK's ``ClientTools`` registry and dispatch, as elevenlabs 2.69.0
    writes them: an unregistered name is the SDK's own error."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.tools: dict[str, tuple[Any, bool]] = {}

    def register(self, tool_name: str, handler: Any, is_async: bool = False) -> None:
        self.tools[tool_name] = (handler, is_async)

    async def handle(self, tool_name: str, parameters: dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        handler, _ = self.tools[tool_name]
        return await handler(parameters)


async def test_elevenlabs_routes_an_unregistered_name_to_the_channel() -> None:
    routed: list[tuple[str, dict[str, Any]]] = []

    async def route(name: str, parameters: dict[str, Any]) -> str:
        routed.append((name, parameters))
        return "refused by the channel"

    async def declared(parameters: dict[str, Any]) -> str:
        return "declared"

    tools = sdk_patch.client_tools(
        _SdkLikeClientTools, loop=asyncio.get_running_loop(), route=route
    )
    tools.register("lookup", declared, is_async=True)

    assert await tools.handle("lookup", {}) == "declared"
    assert await tools.handle("secret_op", {"tool_call_id": "t1"}) == "refused by the channel"
    assert routed == [("secret_op", {"tool_call_id": "t1"})]


async def test_an_unregistered_elevenlabs_call_reaches_on_tool_call() -> None:
    provider = ElevenLabsRealtimeProvider(ElevenLabsRealtimeConfig(api_key="k", agent_id="a"))
    session = _session()
    heard: list[tuple[str, str]] = []

    def on_call(_s: VoiceSession, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        heard.append((call_id, name))

    provider.on_tool_call(on_call)
    routed = asyncio.create_task(
        provider._route_unregistered(session, "secret_op", {"tool_call_id": "t1"})
    )
    for _ in range(100):
        if heard:
            break
        await asyncio.sleep(0)
    await provider.submit_tool_result(session, "t1", '{"error": "not declared"}')

    assert heard == [("t1", "secret_op")]
    assert await routed == '{"error": "not declared"}'

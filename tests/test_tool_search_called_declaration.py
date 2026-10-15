"""A called tool stays declared after another search, even after a schema error."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolRefusedError
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_tool_search_text import _binding, _catalogue, _run

_TABLE = {
    "name": "tables_query_rows",
    "description": "Query dataset rows using filters.",
    "parameters": {
        "type": "object",
        "properties": {"filters": {"type": "array", "items": {"type": "object"}}},
        "required": ["filters"],
    },
}
_UNUSED = {"name": "tables_insert_row", "description": "Insert dataset rows."}
_BOARDS = {"name": "boards_list", "description": "List kanban boards."}
_TOOLS = [_TABLE, _UNUSED, _BOARDS, *_catalogue(5)]


def _round(call_id: str, name: str, arguments: dict[str, Any]) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)],
    )


@pytest.mark.parametrize("discovered", [True, False], ids=["found", "recovered"])
@pytest.mark.parametrize("invalid", [True, False], ids=["rejected", "served"])
async def test_a_later_search_keeps_called_schemas(
    streaming: bool, discovered: bool, invalid: bool
) -> None:
    calls: list[dict[str, Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        calls.append(arguments)
        return '{"rows": []}'

    responses = [_round("find", "find_tools", {"query": "dataset rows"})] if discovered else []
    responses += [
        _round("first", _TABLE["name"], {"filters": "[]" if invalid else []}),
        _round("other", "find_tools", {"query": "kanban boards"}),
        _round("retry", _TABLE["name"], {"filters": []}),
        AIResponse(content="done", finish_reason="stop"),
    ]
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)
    channel = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=handler)
    await _run(channel, _binding(_TOOLS))

    first = int(discovered)
    for context in provider.calls[first + 1 :]:
        tools = {tool.name: tool for tool in context.tools}
        assert tools[_TABLE["name"]].parameters == _TABLE["parameters"]
    after_search = {tool.name for tool in provider.calls[first + 2].tools}
    assert _BOARDS["name"] in after_search
    assert _UNUSED["name"] not in after_search
    assert calls == [{"filters": []}] * (1 if invalid else 2)


async def _session(
    **kwargs: Any,
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=_TOOLS,
        tool_search=True,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await channel.start_session(room.id, "user", object())
    provider.reconfigure = AsyncMock()  # type: ignore[method-assign]
    return kit, channel, provider, session


async def _call(
    channel: RealtimeVoiceChannel, session: Any, call_id: str, name: str, arguments: dict[str, Any]
) -> None:
    call = channel._provider_call(session, call_id, name, arguments)
    assert channel._open_tool_call(call)
    try:
        await channel._execute_tool_call(call)
    finally:
        channel._close_tool_call(call)


@pytest.mark.parametrize("discovered", [True, False], ids=["found", "recovered"])
@pytest.mark.parametrize("invalid", [True, False], ids=["rejected", "served"])
async def test_realtime_retains_called_schemas_after_delivering_the_result(
    discovered: bool,
    invalid: bool,
) -> None:
    called: list[dict[str, Any]] = []

    async def handler(name: str, args: dict[str, Any]) -> str:
        called.append(args)
        return '{"rows": []}'

    kit, channel, provider, session = await _session(tool_handler=handler)
    try:
        if discovered:
            await _call(channel, session, "find", "find_tools", {"query": "dataset rows"})
        results_before = len(provider.tool_results)

        async def reconfigure(*args: Any, **kwargs: Any) -> None:
            assert len(provider.tool_results) > results_before

        provider.reconfigure.side_effect = reconfigure
        await _call(
            channel, session, "first", _TABLE["name"], {"filters": "[]" if invalid else []}
        )
        await _call(channel, session, "other", "find_tools", {"query": "kanban boards"})
        declarations = provider.reconfigure.call_args.kwargs["tools"]
        tools = {t["name"]: t for t in declarations}
        assert tools[_TABLE["name"]]["parameters"] == _TABLE["parameters"]
        assert _BOARDS["name"] in tools
        assert _UNUSED["name"] not in tools
        await _call(channel, session, "retry", _TABLE["name"], {"filters": []})
        assert called == [{"filters": []}] * (1 if invalid else 2)
    finally:
        await kit.close()


@pytest.mark.parametrize("gate", ["policy", "hook", "handler"])
async def test_realtime_access_refusals_do_not_retain_a_hidden_tool(gate: str) -> None:
    async def handler(name: str, args: dict[str, Any]) -> str:
        if gate == "handler":
            raise ToolRefusedError("not allowed")
        raise AssertionError("the gate must block the call")

    options = {"tool_policy": ToolPolicy(deny=[_TABLE["name"]])} if gate == "policy" else {}
    kit, channel, provider, session = await _session(tool_handler=handler, **options)
    try:
        if gate == "hook":

            @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC)
            async def block(event: Any, ctx: Any) -> HookResult:
                return HookResult.block("not allowed")

        await _call(channel, session, "refused", _TABLE["name"], {"filters": []})
        if gate != "handler":
            assert "error" in json.loads(provider.tool_results[-1][2])
        provider.reconfigure.assert_not_called()
        support = channel._tool_search_support
        assert _TABLE["name"] not in {t["name"] for t in support.visible_tools(session.id, _TOOLS)}
    finally:
        await kit.close()


async def test_a_catalogue_replacement_clears_retained_tools() -> None:
    kit, channel, provider, session = await _session()
    support = channel._tool_search_support
    try:
        await _call(channel, session, "invalid", _TABLE["name"], {"filters": "[]"})
        assert _TABLE["name"] in {t["name"] for t in support.visible_tools(session.id, _TOOLS)}
        await channel.reconfigure_session(session, tools=_TOOLS)
        assert _TABLE["name"] not in {t["name"] for t in support.visible_tools(session.id, _TOOLS)}
    finally:
        await kit.close()
    assert session.id not in support._retained

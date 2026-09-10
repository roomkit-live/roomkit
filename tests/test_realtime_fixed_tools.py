"""Fixed declarations use the native execution gates and session lifecycle."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.hook import HookResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.testing import VoiceTrace
from tests.test_realtime_skills import _registry_with_skill


class FixedProvider(MockRealtimeProvider):
    @property
    def supports_mid_session_reconfigure(self) -> bool:
        return False


def tool(name: str, description: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "description": description or name,
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
            "additionalProperties": False,
        },
    }


@asynccontextmanager
async def channel_context(**kwargs: Any) -> AsyncIterator[tuple[Any, ...]]:
    provider = FixedProvider()
    provider.reconfigure = AsyncMock()
    handler = kwargs.pop("tool_handler", AsyncMock(return_value={"ok": True}))
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=kwargs.pop("tools", [tool("calendar"), tool("projects")]),
        tool_search=True,
        tool_handler=handler,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt")
    session = await channel.start_session(room.id, "participant", object())
    try:
        yield kit, channel, provider, session, handler
    finally:
        await kit.close()


async def call(channel: Any, provider: Any, session: Any, name: str, args: Any) -> Any:
    call_id = f"call-{len(provider.tool_results)}"
    await provider.simulate_tool_call(session, call_id, name, args)
    await asyncio.gather(*list(channel._scheduled_tasks))
    assert provider.tool_results[-1][:2] == (session.id, call_id)
    return json.loads(provider.tool_results[-1][2])


async def test_large_catalogue_uses_fixed_declarations_and_complete_schemas() -> None:
    calendar = tool("calendar", "Read calendar appointments. " + "schema detail " * 1800)
    tools = [calendar, tool("projects", "Read project information")]
    tools += [tool(f"filler_{i}") for i in range(110)]
    async with channel_context(tools=tools, tool_result_max_length=500) as ctx:
        kit, channel, provider, session, handler = ctx
        trace = VoiceTrace(kit, triggers=[HookTrigger.BEFORE_TOOL_USE, HookTrigger.ON_TOOL_CALL])
        connected = next(c.args for c in provider.calls if c.method == "connect")
        assert {t["name"] for t in connected["tools"]} == {"find_tools", "list_tools", "call_tool"}
        assert "filler_" not in json.dumps(connected)
        assert "arguments_json" in connected["system_prompt"]
        for name, query in [
            ("calendar", "calendar appointments"),
            ("projects", "project information"),
        ]:
            found = await call(channel, provider, session, "find_tools", {"query": query})
            assert found["matches"][0]["name"] == name
            assert "list_tools(name=" in found["_note"]
            schema = await call(channel, provider, session, "list_tools", {"name": name})
            assert schema["tool"] == next(t for t in tools if t["name"] == name)
            result = await call(
                channel,
                provider,
                session,
                "call_tool",
                {
                    "name": name,
                    "arguments_json": '{"action":"list"}',
                },
            )
            assert result == {"ok": True}
        assert [a.args[0] for a in handler.await_args_list] == ["calendar", "projects"]
        events = [e.payload for e in trace.entries(HookTrigger.ON_TOOL_CALL)]
        business = [e for e in events if e.name in {"calendar", "projects"}]
        assert [(e.name, e.tool_call_id, e.arguments) for e in business] == [
            ("calendar", "call-2", {"action": "list"}),
            ("projects", "call-5", {"action": "list"}),
        ]
        provider.reconfigure.assert_not_awaited()
        assert sum(c.method == "connect" for c in provider.calls) == 1
        trace.close()


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"name": "calendar"}, "missing required argument"),
        ({"name": "calendar", "arguments_json": {}}, "must be of type string"),
        ({"name": "calendar", "arguments_json": "{"}, "Invalid arguments_json"),
        ({"name": "calendar", "arguments_json": "[]"}, "expected a JSON object"),
        ({"name": "calendar", "arguments_json": '{"action":NaN}'}, "Invalid arguments_json"),
        ({"name": "calendar", "arguments_json": '{"action":3}'}, "Invalid arguments"),
        (
            {"name": "calendar", "arguments_json": '{"action":"list","extra":1}'},
            "unknown argument",
        ),
        ({"name": "excluded", "arguments_json": "{}"}, "unavailable in this session"),
        ({"name": "invented", "arguments_json": "{}"}, "unavailable in this session"),
        ({"name": "call_tool", "arguments_json": "{}"}, "unavailable in this session"),
        ({"name": "find_tools", "arguments_json": "{}"}, "unavailable in this session"),
    ],
)
async def test_transport_refusals_never_execute(args: Any, error: str) -> None:
    async with channel_context() as (_, channel, provider, session, handler):
        result = await call(channel, provider, session, "call_tool", args)
        assert error in result["error"]
        handler.assert_not_awaited()


@pytest.mark.parametrize("transported", [True, False])
@pytest.mark.parametrize("decision", ["deny", "rewrite", "invalid_rewrite"])
async def test_native_and_transported_calls_share_hook_validation(
    transported: bool, decision: str
) -> None:
    async with channel_context() as (kit, channel, provider, session, handler):
        seen = []

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC)
        async def gate(event: Any, context: Any) -> HookResult:
            seen.append((event.name, event.arguments, event.tool_call_id))
            if decision == "deny":
                return HookResult.block("Policy denies calendar access")
            return HookResult(
                action="allow",
                metadata={"arguments": {"action": "get" if decision == "rewrite" else 42}},
            )

        name, args = "calendar", {"action": "list"}
        if transported:
            name, args = "call_tool", {"name": name, "arguments_json": json.dumps(args)}
        result = await call(channel, provider, session, name, args)
        assert seen == [("calendar", {"action": "list"}, "call-0")]
        if decision == "rewrite":
            handler.assert_awaited_once_with("calendar", {"action": "get"})
            assert result == {"ok": True}
        else:
            handler.assert_not_awaited()
            assert (
                "Policy denies" if decision == "deny" else "Invalid rewritten arguments"
            ) in result["error"]


async def test_transported_calls_obey_skill_gating(tmp_path: Any) -> None:
    registry = _registry_with_skill(tmp_path, allowed_tools="calendar")
    async with channel_context(skills=registry, skill_delivery_mode="on_demand") as ctx:
        _, channel, provider, session, handler = ctx
        args = {"name": "calendar", "arguments_json": '{"action":"list"}'}
        refused = await call(channel, provider, session, "call_tool", args)
        assert "gated by a skill" in refused["error"]
        handler.assert_not_awaited()
        await call(channel, provider, session, "activate_skill", {"name": "test-skill"})
        assert await call(channel, provider, session, "call_tool", args) == {"ok": True}
        provider.reconfigure.assert_not_awaited()


async def test_empty_session_catalogue_cannot_recover_channel_defaults() -> None:
    async with channel_context() as (kit, channel, provider, session, handler):
        await kit.create_room(room_id="other-room")
        await kit.attach_channel("other-room", "rt")
        other = await channel.start_session(
            "other-room", "other", object(), metadata={"tools": []}
        )
        args = {"name": "calendar", "arguments_json": '{"action":"list"}'}
        assert "unavailable" in (await call(channel, provider, other, "call_tool", args))["error"]
        assert (
            "unavailable"
            in (await call(channel, provider, other, "list_tools", {"name": "calendar"}))["error"]
        )
        handler.assert_not_awaited()
        assert await call(channel, provider, session, "call_tool", args) == {"ok": True}


async def test_integration_failure_is_returned_without_a_success() -> None:
    handler = AsyncMock(side_effect=ConnectionError("integration disconnected"))
    async with channel_context(tool_handler=handler) as (_, channel, provider, session, _):
        result = await call(
            channel,
            provider,
            session,
            "call_tool",
            {
                "name": "calendar",
                "arguments_json": '{"action":"list"}',
            },
        )
        assert result["error"] == "Internal error handling tool call"
        assert result["tool"] == "calendar"
        assert "before retrying" in result["hint"]
        handler.assert_awaited_once()


async def test_call_tool_collision_is_rejected_in_session_overrides() -> None:
    async with channel_context() as (_, channel, provider, session, handler):
        with pytest.raises(ValueError, match="call_tool is reserved"):
            await channel.start_session(
                "unused", "other", object(), metadata={"tools": [tool("call_tool")]}
            )
        assert await call(
            channel,
            provider,
            session,
            "call_tool",
            {"name": "calendar", "arguments_json": '{"action":"list"}'},
        ) == {"ok": True}


@pytest.mark.parametrize("transported", [True, False])
async def test_hangup_cancels_only_its_calls_and_refuses_late_execution(transported: bool) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()
    executed = []

    async def handler(name: str, args: Any) -> dict[str, bool]:
        if args["action"] == "wait":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        executed.append(name)
        return {"ok": True}

    async with channel_context(tool_handler=handler) as (kit, channel, provider, session, _):
        await kit.create_room(room_id="other-room")
        await kit.attach_channel("other-room", "rt")
        other = await channel.start_session("other-room", "other", object())
        name, args = "calendar", {"action": "wait"}
        if transported:
            name, args = "call_tool", {"name": name, "arguments_json": json.dumps(args)}
        await provider.simulate_tool_call(session, "waiting", name, args)
        await asyncio.wait_for(started.wait(), 1)
        await channel.end_session(session)
        assert cancelled.is_set()
        await provider.simulate_tool_call(session, "late", name, args)
        await asyncio.gather(*list(channel._scheduled_tasks))
        assert not executed and not provider.tool_results
        assert await call(
            channel,
            provider,
            other,
            "call_tool",
            {
                "name": "projects",
                "arguments_json": '{"action":"list"}',
            },
        ) == {"ok": True}
        assert executed == ["projects"]

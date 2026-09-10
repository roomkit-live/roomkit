"""Instruction delivery is the prerequisite for opening session skill gates."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.hook import HookResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_realtime_fixed_tools import FixedProvider, call, tool
from tests.test_realtime_skills import _make_skill, _registry_with_skill


@asynccontextmanager
async def running(registry, *, provider=None, **kwargs):
    provider = provider or FixedProvider()
    provider.reconfigure = AsyncMock()
    provider.connect = AsyncMock(wraps=provider.connect)
    handler = AsyncMock(return_value={"items": ["appointment"]})
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        skills=registry,
        skill_delivery_mode="on_demand",
        tools=kwargs.pop("tools", [tool("calendar")]),
        tool_handler=handler,
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    session = await channel.start_session(room.id, "person", object())
    try:
        yield channel, provider, session, handler
    finally:
        await kit.close()


async def test_complete_body_references_and_prerequisite_schema_after_eighth_skill(tmp_path):
    body, reference = "Mandatory instruction.\n" * 1600, "Reference detail.\n" * 2500
    registry = _registry_with_skill(
        tmp_path,
        body=body,
        references=[("guide.md", reference)],
        allowed_tools="calendar",
    )
    for index in range(12):
        _make_skill(tmp_path, f"other-{index}", body=f"Other body {index}")
    registry.discover(tmp_path)
    registry.get_skill("test-skill").metadata.extra_metadata["requires"] = "['calendar']"
    calendar = tool("calendar", "Complete schema " * 1400)
    async with running(registry, tools=[calendar], tool_result_max_length=40) as ctx:
        channel, provider, session, handler = ctx
        connected = next(c.args for c in provider.calls if c.method == "connect")
        assert body.strip() not in connected["system_prompt"]
        assert all(f"other-{index}" in connected["system_prompt"] for index in range(12))
        assert provider.connect.call_args.kwargs["provider_config"]["preserve_context"] is True
        assert "call_tool" in {t["name"] for t in connected["tools"]}
        assert "calendar" not in {t["name"] for t in connected["tools"]}
        denied = await call(
            channel,
            provider,
            session,
            "call_tool",
            {
                "name": "calendar",
                "arguments_json": '{"action":"list"}',
            },
        )
        assert "error" in denied
        handler.assert_not_awaited()
        for _ in range(2):
            result = await call(
                channel, provider, session, "activate_skill", {"name": "test-skill"}
            )
            assert result["instructions"] == body.strip()
            assert result["required_tools"] == [calendar]
            assert result["references"] == ["guide.md"]
        assert result["already_active"] is True
        assert len(channel._skill_support._activated_bodies[session.id]) == 1
        result = await call(
            channel,
            provider,
            session,
            "read_skill_reference",
            {
                "skill_name": "test-skill",
                "filename": "guide.md",
            },
        )
        assert result["content"] == reference
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
        assert "error" not in result
        handler.assert_awaited_once_with("calendar", {"action": "list"})
        provider.reconfigure.assert_not_awaited()


@pytest.mark.parametrize("ending", ["send_error", "cancel", "hangup"])
async def test_unsuccessful_delivery_never_opens_gates(tmp_path, ending):
    registry = _registry_with_skill(tmp_path, allowed_tools="calendar")
    async with running(registry) as (channel, provider, session, handler):
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_send(*args):
            entered.set()
            await release.wait()
            if ending == "send_error":
                raise ConnectionError("delivery unavailable")

        provider.submit_tool_result = blocked_send
        task = asyncio.create_task(
            channel._handle_tool_call(
                session,
                "activation",
                "activate_skill",
                {"name": "test-skill"},
            )
        )
        await asyncio.wait_for(entered.wait(), 3)
        assert channel._skill_support.is_gated("calendar", session.id)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            if ending == "hangup":
                await channel.end_session(session)
            release.set()
            await asyncio.wait_for(task, 3)
        assert channel._skill_support.is_gated("calendar", session.id)
        assert not channel._skill_support._activated_skills.get(session.id)
        handler.assert_not_awaited()


async def test_native_update_failure_does_not_record_activation(tmp_path):
    registry = _registry_with_skill(tmp_path, allowed_tools="calendar")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx
        provider.reconfigure.side_effect = ConnectionError("update failed")
        await call(channel, provider, session, "activate_skill", {"name": "test-skill"})
        assert channel._skill_support.is_gated("calendar", session.id)
        assert not channel._skill_support._activated_skills[session.id]


async def test_simultaneous_native_activations_keep_both_bodies(tmp_path):
    registry = _registry_with_skill(tmp_path, body="First binding instruction.")
    _make_skill(tmp_path, "second", body="Second binding instruction.")
    registry.discover(tmp_path)
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx

        async def update(*args, **kwargs):
            await asyncio.sleep(0)

        provider.reconfigure.side_effect = update
        await asyncio.gather(
            *[
                channel._handle_tool_call(session, name, "activate_skill", {"name": name})
                for name in ["test-skill", "second"]
            ]
        )
        prompt = provider.reconfigure.call_args.kwargs["system_prompt"]
        assert "First binding instruction." in prompt
        assert "Second binding instruction." in prompt


@pytest.mark.parametrize("update", ["search", "handoff"])
@pytest.mark.parametrize("activation_first", [True, False])
async def test_activation_and_configuration_preserve_session_rules(
    tmp_path, update, activation_first
):
    body = "Mandatory rule that must survive every configuration update."
    registry = _registry_with_skill(tmp_path, body=body, allowed_tools="calendar")
    async with running(
        registry, provider=MockRealtimeProvider(), tool_search=True, system_prompt="Original role"
    ) as (channel, provider, session, _):
        entered, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        applied = []

        async def apply_config(*args, **kwargs):
            if not applied:
                entered.set()
                await release.wait()
            applied.append(kwargs)

        provider.reconfigure.side_effect = apply_config

        async def operation(activate, *, second=False):
            if second:
                second_started.set()
            if activate:
                await channel._handle_tool_call(
                    session, "activation", "activate_skill", {"name": "test-skill"}
                )
            elif update == "search":
                await channel._handle_tool_call(
                    session, "search", "find_tools", {"query": "calendar"}
                )
            else:
                await channel.reconfigure_session(session, system_prompt="New role")

        first = asyncio.create_task(operation(activation_first))
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.create_task(operation(not activation_first, second=True))
        await asyncio.wait_for(second_started.wait(), 3)
        try:
            # The second operation must wait for the first provider update and
            # its local commit, rather than preparing a stale prompt concurrently.
            assert provider.reconfigure.await_count == 1
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 3)
        final_prompt = applied[-1]["system_prompt"]
        assert final_prompt.count(body) == 1
        assert "test-skill" in final_prompt
        assert channel._tool_search_support.preamble in final_prompt
        assert ("New role" if update == "handoff" else "Original role") in final_prompt
        assert not channel._skill_support.is_gated("calendar", session.id)
        if update == "search":
            assert "calendar" in {tool["name"] for tool in applied[-1]["tools"]}
        await channel.end_session(session)
        assert session.id not in channel._session_config_locks


async def test_search_observer_can_reconfigure_without_deadlock(tmp_path):
    registry = _registry_with_skill(tmp_path, body="Persistent skill instructions.")
    async with running(registry, provider=MockRealtimeProvider(), tool_search=True) as ctx:
        channel, provider, session, _ = ctx
        await call(channel, provider, session, "activate_skill", {"name": "test-skill"})

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def observer(event, context):
            if event.name == "find_tools":
                await channel.reconfigure_session(session, system_prompt="Observer role")
            return HookResult.allow()

        await asyncio.wait_for(
            call(channel, provider, session, "find_tools", {"query": "calendar"}), 3
        )
        prompt = provider.reconfigure.call_args.kwargs["system_prompt"]
        assert "Observer role" in prompt
        assert "Persistent skill instructions." in prompt


@pytest.mark.parametrize("args", [{}, {"name": []}, {"name": 42}, {"name": "missing"}])
async def test_invalid_activation_never_claims_an_integration_outage(tmp_path, args):
    registry = _registry_with_skill(tmp_path, allowed_tools="calendar")
    async with running(registry) as (channel, provider, session, handler):
        result = await call(channel, provider, session, "activate_skill", args)
        assert "error" in result
        assert not result.get("ok")
        assert not channel._skill_support._activated_skills[session.id]
        handler.assert_not_awaited()


async def test_session_catalogue_limits_prerequisite_schemas(tmp_path):
    registry = _registry_with_skill(tmp_path)
    registry.get_skill("test-skill").metadata.extra_metadata["requires"] = "calendar, private_tool"
    async with running(registry) as (channel, provider, session, _):
        result = await call(channel, provider, session, "activate_skill", {"name": "test-skill"})
        assert "Required tools not available" in result["error"]
        assert "instructions" not in result
        assert not channel._skill_support._activated_skills[session.id]


def test_explicit_search_off_is_refused_for_fixed_skill_gates(tmp_path):
    registry = _registry_with_skill(tmp_path, allowed_tools="calendar")
    with pytest.raises(ValueError, match="tool_search=False"):
        RealtimeVoiceChannel(
            "voice",
            provider=FixedProvider(),
            transport=MockRealtimeTransport(),
            skills=registry,
            tools=[tool("calendar")],
            tool_search=False,
        )


def test_unsupported_fixed_provider_refuses_on_demand(tmp_path):
    class Unsupported(FixedProvider):
        @property
        def supports_context_preservation(self):
            return False

    with pytest.raises(ValueError, match="context preservation"):
        RealtimeVoiceChannel(
            "voice",
            provider=Unsupported(),
            transport=MockRealtimeTransport(),
            skills=_registry_with_skill(tmp_path),
            skill_delivery_mode="on_demand",
        )

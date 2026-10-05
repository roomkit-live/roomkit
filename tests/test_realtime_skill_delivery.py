"""Instruction delivery is the prerequisite for opening session skill gates."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels._realtime_tool_calls import RealtimeToolCall
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.hook import HookResult
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_realtime_fixed_tools import FixedProvider, call, tool
from tests.test_realtime_skills import _make_skill, _registry_with_skill


async def raw_call(channel, provider, session, name, args):
    """The text a call's result went out as, which a bound may have cut."""
    call_id = f"raw-{len(provider.tool_results)}"
    await provider.simulate_tool_call(session, call_id, name, args)
    await asyncio.gather(*list(channel._scheduled_tasks))
    assert provider.tool_results[-1][:2] == (session.id, call_id)
    return provider.tool_results[-1][2]


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


async def test_complete_body_and_prerequisite_schema_after_eighth_skill(tmp_path):
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
        # The refusal is bounded too (RFC §21.5): read as text, not as JSON;
        # a bound shorter than the truncation note cuts the text alone.
        denied = await raw_call(
            channel,
            provider,
            session,
            "call_tool",
            {
                "name": "calendar",
                "arguments_json": '{"action":"list"}',
            },
        )
        assert len(denied) == 40 and denied.startswith('{"error"')
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
        # A reference is data, bounded as any tool result is; the
        # instructions above are not (RFC §21.5, §24.4).
        read = await raw_call(
            channel,
            provider,
            session,
            "read_skill_reference",
            {"skill_name": "test-skill", "filename": "guide.md"},
        )
        assert len(read) == 40 and read.startswith('{"filename"')
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
            channel._execute_tool_call(
                RealtimeToolCall(session, "activation", "activate_skill", {"name": "test-skill"})
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
                channel._execute_tool_call(
                    RealtimeToolCall(session, name, "activate_skill", {"name": name})
                )
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
    # The search needs a match the session may call before the activation:
    # Tool Search never names the gated calendar (RFC §21.1).
    tools = [tool("calendar"), tool("agenda", "Read the calendar agenda")]
    async with running(
        registry,
        provider=MockRealtimeProvider(),
        tool_search=True,
        system_prompt="Original role",
        tools=tools,
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
                await channel._execute_tool_call(
                    RealtimeToolCall(
                        session, "activation", "activate_skill", {"name": "test-skill"}
                    )
                )
            elif update == "search":
                await channel._execute_tool_call(
                    RealtimeToolCall(session, "search", "find_tools", {"query": "calendar"})
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
            # The search's reveal survives the update that follows it: after the
            # activation it finds the calendar; before it, only the agenda.
            revealed = "calendar" if activation_first else "agenda"
            assert revealed in {tool["name"] for tool in applied[-1]["tools"]}
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


async def test_a_blocked_activation_is_refused_and_opens_no_gate(tmp_path):
    """ON_TOOL_CALL decides before the result goes out: a block reaches the
    model as the refusal, and the skill's gates stay closed (RMK-272)."""
    registry = _registry_with_skill(tmp_path, body="Payment rules.", allowed_tools="calendar")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def refuse(event, context):
            if event.name == "activate_skill":
                return HookResult.block(reason="skill not allowed for this user")
            return HookResult.allow()

        result = await call(channel, provider, session, "activate_skill", {"name": "test-skill"})

        assert result == {"error": "skill not allowed for this user"}
        assert channel._skill_support.is_gated("calendar", session.id)
        provider.reconfigure.assert_not_called()


async def test_a_hook_rewrites_a_skill_tool_result_before_it_is_sent(tmp_path):
    registry = _registry_with_skill(tmp_path, body="Rules.", references=[("guide.md", "SECRET")])
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def redact(event, context):
            if event.name == "read_skill_reference":
                return HookResult(action="allow", metadata={"result": '{"content": "[redacted]"}'})
            return HookResult.allow()

        result = await call(
            channel,
            provider,
            session,
            "read_skill_reference",
            {"skill_name": "test-skill", "filename": "guide.md"},
        )

        assert result == {"content": "[redacted]"}


async def test_a_hook_may_reconfigure_the_session_during_an_activation(tmp_path):
    """The hooks run outside the session's configuration lock: one that
    reconfigures the session from ON_TOOL_CALL does not wait on itself."""
    registry = _registry_with_skill(tmp_path, body="Rules.", allowed_tools="calendar")
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def observer(event, context):
            if event.name == "activate_skill":
                await channel.reconfigure_session(session, system_prompt="Observer role")
            return HookResult.allow()

        await asyncio.wait_for(
            call(channel, provider, session, "activate_skill", {"name": "test-skill"}), 3
        )

        assert not channel._skill_support.is_gated("calendar", session.id)


async def test_an_activation_rechecks_its_required_tools_before_delivery(tmp_path):
    """The hooks run outside the configuration lock: a handoff landing
    meanwhile may take a required tool away, and the activation must not then
    be delivered as if it were still there."""
    registry = _registry_with_skill(tmp_path, body="Rules.", allowed_tools="calendar")
    registry.get_skill("test-skill").metadata.extra_metadata["requires"] = "['calendar']"
    async with running(registry, provider=MockRealtimeProvider()) as ctx:
        channel, provider, session, _ = ctx

        @channel._framework.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def handoff(event, context):
            if event.name == "activate_skill":
                await channel.reconfigure_session(session, tools=[tool("agenda")])
            return HookResult.allow()

        result = await call(channel, provider, session, "activate_skill", {"name": "test-skill"})

        assert result == {"error": "Required tools not available: calendar"}
        assert channel._skill_support.is_gated("calendar", session.id)

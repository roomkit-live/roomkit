"""One strategy per room, installed and uninstalled while the room lives (RFC §19.7)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import Agent, Discussion, Loop, RoomKit, Supervisor, Swarm
from roomkit.core.hooks import HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.orchestration.base import Orchestration
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


def _agent(name: str, *answers: AIResponse | str) -> Agent:
    responses = [a if isinstance(a, AIResponse) else AIResponse(content=a) for a in answers]
    provider = MockAIProvider(ai_responses=responses or [AIResponse(content=f"{name} here")])
    return Agent(name, provider=provider, role=name)


def _handoff(target: str) -> AIResponse:
    arguments = {"target": target, "reason": "theirs", "summary": "over to you"}
    return AIResponse(
        content="",
        tool_calls=[
            AIToolCall(id=f"to-{target}", name="handoff_conversation", arguments=arguments)
        ],
    )


async def _live_room(kit: RoomKit, assistant: Agent) -> None:
    """A room already in use: a person and one assistant, two exchanges."""
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(assistant)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    await kit.attach_channel("r1", assistant.channel_id, category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "hello")
    await _say(kit, "checkout is failing")


async def _say(kit: RoomKit, body: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body))
    )


def _bodies(result: Any) -> list[str]:
    return [
        f"{e.source.channel_id}: {e.content.body}"
        for e in result.response_events
        if isinstance(e.content, TextContent)
    ]


async def _bound(kit: RoomKit) -> set[str]:
    return {b.channel_id for b in await kit.list_bindings("r1")}


async def test_a_swarm_installed_on_a_live_room_reaches_the_agent_it_hands_off_to() -> None:
    sales = _agent("sales", "hi", "noted", _handoff("billing"), "passing you to billing")
    billing = _agent("billing", "billing here")
    kit = RoomKit()
    await _live_room(kit, sales)

    await kit.install_strategy("r1", Swarm(agents=[sales, billing], entry="sales"))
    await _say(kit, "my invoice is wrong")
    result = await _say(kit, "it says 40 instead of 20")

    assert "billing" in await _bound(kit)
    assert _bodies(result) == ["billing: billing here"]
    await kit.close()


async def test_a_discussion_installed_on_a_live_room_reads_its_history() -> None:
    assistant = _agent("assistant", "hi", "noted")
    sre = _agent("sre", "the metrics spike at 14:05")
    kit = RoomKit()
    await _live_room(kit, assistant)

    await kit.install_strategy("r1", Discussion([assistant, sre]))
    await _say(kit, "@sre what do the metrics say?")
    async with asyncio.timeout(5):
        while not sre._provider.calls:  # type: ignore[attr-defined]
            await asyncio.sleep(0.02)

    read = " ".join(str(m.content) for m in sre._provider.calls[0].messages)  # type: ignore[attr-defined]
    assert "hello" in read and "checkout is failing" in read
    await kit.close()


async def test_a_room_moves_from_a_swarm_to_a_discussion_and_back() -> None:
    sales = _agent("sales", "hi", "noted", "sales again")
    billing = _agent("billing")
    kit = RoomKit()
    await _live_room(kit, sales)
    swarm = Swarm(agents=[sales, billing], entry="sales")
    await kit.install_strategy("r1", swarm)
    assert kit.hook_engine.has_router_hook("r1")

    assert await kit.uninstall_strategy("r1")

    # What the swarm added is gone: its router, its handoff tool, its state and
    # the agent it attached; the assistant bound before it stays.
    assert not kit.hook_engine.has_router_hook("r1")
    assert "handoff_conversation" not in sales._registry.room_tool_names("r1")
    assert get_conversation_state(await kit.get_room("r1")).active_agent_id is None
    assert await _bound(kit) == {"ops", "sales"}
    assert kit.room_strategy("r1") is None

    discussion = Discussion([sales])
    await kit.install_strategy("r1", discussion)
    assert kit.room_strategy("r1") is discussion and kit.speak_queue("r1") is not None
    await kit.uninstall_strategy("r1")
    assert kit.speak_queue("r1") is None
    await kit.install_strategy("r1", Swarm(agents=[sales, billing], entry="sales"))
    await kit.close()


async def test_a_room_holds_one_strategy_at_a_time() -> None:
    sales = _agent("sales")
    kit = RoomKit()
    await _live_room(kit, sales)
    await kit.install_strategy("r1", Discussion([sales]))

    with pytest.raises(ValueError, match="holds a strategy already"):
        await kit.install_strategy("r1", Swarm(agents=[sales], entry="sales"))
    with pytest.raises(ValueError, match="holds a strategy already"):
        await kit.install_strategy("r1", _Custom())
    await kit.close()


class _Custom(Orchestration):
    """A host's own strategy: a hook, a metadata key, nothing to undo itself."""

    def agents(self) -> list[Any]:
        return []

    async def install(self, kit: RoomKit, room_id: str) -> None:
        async def tag(event: RoomEvent, context: RoomContext) -> HookResult:
            return HookResult.allow()

        kit.hook_engine.add_room_hook(
            room_id,
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=tag,
                name="custom_tag",
            ),
        )
        await kit.store.patch_room_metadata(room_id, {"custom_state": {"on": True}})


async def test_a_hosts_own_strategy_is_taken_back_out_whole() -> None:
    kit = RoomKit()
    await _live_room(kit, _agent("assistant", "hi", "noted"))

    await kit.install_strategy("r1", _Custom())
    assert "custom_tag" in kit.hook_engine.room_hook_names("r1")
    await kit.uninstall_strategy("r1")

    assert "custom_tag" not in kit.hook_engine.room_hook_names("r1")
    assert (await kit.get_room("r1")).metadata.get("custom_state") is None
    await kit.close()


async def test_a_refused_install_takes_back_what_it_attached() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("phone", channel_type=ChannelType.VOICE))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "phone")

    with pytest.raises(ValueError, match="voice or realtime"):
        await kit.install_strategy("r1", Discussion([_agent("a"), _agent("b")]))

    assert await _bound(kit) == {"phone"}
    assert kit.room_strategy("r1") is None
    await kit.close()


async def test_create_room_installs_through_the_same_path() -> None:
    kit = RoomKit()
    swarm = Swarm(agents=[_agent("sales"), _agent("billing")], entry="sales")
    await kit.create_room(room_id="r1", orchestration=swarm)

    assert kit.room_strategy("r1") is swarm
    await kit.uninstall_strategy("r1")
    assert await _bound(kit) == set()
    await kit.close()


@pytest.mark.parametrize("kind", ["loop", "supervisor"])
async def test_a_strategy_that_runs_turns_gives_them_back(kind: str) -> None:
    writer = _agent("writer", "draft", "draft", "plain answer")
    kit = RoomKit()
    await _live_room(kit, writer)
    strategy: Orchestration = (
        Loop(agent=writer, reviewers=[_agent("editor")], max_iterations=1)
        if kind == "loop"
        else Supervisor(
            supervisor=writer,
            workers=[_agent("worker")],
            strategy="sequential",
            auto_delegate=True,
        )
    )
    await kit.install_strategy("r1", strategy)
    assert writer._registry.turn_runner("r1") is not None

    await kit.uninstall_strategy("r1")

    assert writer._registry.turn_runner("r1") is None
    result = await _say(kit, "just answer")
    assert [
        e.source.channel_id for e in result.response_events if e.type == EventType.MESSAGE
    ] == ["writer"]
    await kit.close()


async def test_an_uninstall_that_cannot_finish_keeps_the_room_claimed() -> None:
    sales = _agent("sales", "hi", "noted")
    kit = RoomKit()
    await _live_room(kit, sales)
    await kit.install_strategy("r1", Swarm(agents=[sales, _agent("billing")], entry="sales"))
    detach = kit.detach_channel
    failing = True

    async def flaky(room_id: str, channel_id: str, **kwargs: Any) -> bool:
        if failing:
            raise RuntimeError("store down")
        return await detach(room_id, channel_id, **kwargs)

    kit.detach_channel = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="still binds"):
        await kit.uninstall_strategy("r1")

    # Part of the swarm stays: the room is still its, never open to another.
    assert kit.room_strategy("r1") is not None
    with pytest.raises(ValueError, match="holds a strategy already"):
        await kit.install_strategy("r1", Discussion([sales]))
    failing = False
    assert await kit.uninstall_strategy("r1")
    assert kit.room_strategy("r1") is None and await _bound(kit) == {"ops", "sales"}
    await kit.close()


async def test_the_room_strategy_is_read_within_a_tenant() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1", organization_id="acme")
    swarm = Swarm(agents=[_agent("sales")], entry="sales")
    await kit.install_strategy("r1", swarm, organization_id="acme")

    assert kit.room_strategy("r1", organization_id="acme") is swarm
    assert kit.room_strategy("r1", organization_id="other") is None
    with pytest.raises(Exception, match="not found"):
        await kit.uninstall_strategy("r1", organization_id="other")
    await kit.close()

"""Integration tests for the full delegation flow via kit.delegate()."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    ChannelCategory,
    ChannelNotRegisteredError,
    RoomKit,
    RoomNotFoundError,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import HookExecution, HookTrigger, TaskStatus
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.tasks.models import DelegatedTaskResult

# -- Helpers ------------------------------------------------------------------


def _make_agent(agent_id: str, response: str) -> Agent:
    return Agent(
        agent_id,
        provider=MockAIProvider(responses=[response]),
        role="Test Agent",
        description=f"Agent {agent_id}",
    )


# -- Tests --------------------------------------------------------------------


class TestDelegateIntegration:
    async def test_full_delegate_flow(self):
        """Create kit → delegate → wait → verify child room, response, hooks."""
        kit = RoomKit()

        # Register agents
        voice_agent = AIChannel(
            "voice-assistant",
            provider=MockAIProvider(responses=["ok"]),
            system_prompt="You are a voice assistant.",
        )
        pr_reviewer = _make_agent("pr-reviewer", "PR looks good. Ready to merge.")

        kit.register_channel(voice_agent)
        kit.register_channel(pr_reviewer)

        # Create parent room
        await kit.create_room(room_id="call-room")
        await kit.attach_channel(
            "call-room", "voice-assistant", category=ChannelCategory.INTELLIGENCE
        )

        # Track hooks
        delegated_hooks: list[object] = []
        completed_hooks: list[object] = []

        @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
        async def on_delegated(event, ctx):
            delegated_hooks.append(event)

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def on_completed(event, ctx):
            completed_hooks.append(event)

        # Delegate
        task = await kit.delegate(
            room_id="call-room",
            agent_id="pr-reviewer",
            task="Review the latest PR on roomkit",
            context={"repo": "roomkit"},
            notify="voice-assistant",
        )

        assert task.parent_room_id == "call-room"
        assert task.agent_id == "pr-reviewer"
        assert task.child_room_id.startswith("call-room::task-")

        # Wait for result
        result = await task.wait(timeout=5.0)

        assert result.status == TaskStatus.COMPLETED
        assert "PR looks good" in (result.output or "")
        assert result.duration_ms > 0

        # Verify child room was created
        child_room = await kit.get_room(task.child_room_id)
        assert child_room.metadata["parent_room_id"] == "call-room"
        assert child_room.metadata["task_agent_id"] == "pr-reviewer"
        assert child_room.metadata["task_status"] == TaskStatus.COMPLETED

        # Verify hooks fired
        assert len(delegated_hooks) == 1
        assert len(completed_hooks) == 1

        # Verify notify binding was updated with result
        binding = await kit.store.get_binding("call-room", "voice-assistant")
        assert binding is not None
        prompt = binding.metadata.get("system_prompt", "")
        assert "BACKGROUND TASK COMPLETED" in prompt
        assert "PR looks good" in prompt

        await kit.close()

    async def test_shared_channels_appear_in_child_room(self):
        kit = RoomKit()

        agent = _make_agent("agent-a", "done")
        from roomkit.channels import EmailChannel

        email = EmailChannel("email-out", provider=MockEmailProvider(), from_address="a@b.com")

        kit.register_channel(agent)
        kit.register_channel(email)

        await kit.create_room(room_id="parent")
        await kit.attach_channel("parent", "agent-a", category=ChannelCategory.INTELLIGENCE)
        await kit.attach_channel("parent", "email-out", metadata={"from_": "a@b.com"})

        task = await kit.delegate(
            room_id="parent",
            agent_id="agent-a",
            task="do work",
            share_channels=["email-out"],
        )

        await task.wait(timeout=5.0)

        # Verify email channel was shared to child room
        child_bindings = await kit.store.list_bindings(task.child_room_id)
        child_channel_ids = {b.channel_id for b in child_bindings}
        assert "email-out" in child_channel_ids

        await kit.close()

    async def test_parallel_delegation(self):
        kit = RoomKit()

        agent_a = _make_agent("agent-a", "result A")
        agent_b = _make_agent("agent-b", "result B")
        main_ai = AIChannel(
            "main", provider=MockAIProvider(responses=["ok"]), system_prompt="main"
        )

        kit.register_channel(agent_a)
        kit.register_channel(agent_b)
        kit.register_channel(main_ai)

        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "main", category=ChannelCategory.INTELLIGENCE)

        task_a = await kit.delegate(
            room_id="room-1", agent_id="agent-a", task="task A", notify="main"
        )
        task_b = await kit.delegate(
            room_id="room-1", agent_id="agent-b", task="task B", notify="main"
        )

        result_a, result_b = await asyncio.gather(
            task_a.wait(timeout=5.0),
            task_b.wait(timeout=5.0),
        )

        assert result_a.status == TaskStatus.COMPLETED
        assert result_b.status == TaskStatus.COMPLETED
        assert result_a.output == "result A"
        assert result_b.output == "result B"

        # Both child rooms should exist
        await kit.get_room(task_a.child_room_id)
        await kit.get_room(task_b.child_room_id)

        await kit.close()

    async def test_invalid_room_raises(self):
        kit = RoomKit()
        agent = _make_agent("agent-a", "done")
        kit.register_channel(agent)

        with pytest.raises(RoomNotFoundError):
            await kit.delegate(
                room_id="nonexistent",
                agent_id="agent-a",
                task="do stuff",
            )

        await kit.close()

    async def test_invalid_agent_raises(self):
        kit = RoomKit()
        await kit.create_room(room_id="room-1")

        with pytest.raises(ChannelNotRegisteredError):
            await kit.delegate(
                room_id="room-1",
                agent_id="nonexistent-agent",
                task="do stuff",
            )

        await kit.close()

    async def test_on_complete_callback(self):
        kit = RoomKit()
        agent = _make_agent("agent-a", "done")
        kit.register_channel(agent)

        await kit.create_room(room_id="room-1")

        callback_results: list[DelegatedTaskResult] = []

        async def my_callback(result: DelegatedTaskResult) -> None:
            callback_results.append(result)

        task = await kit.delegate(
            room_id="room-1",
            agent_id="agent-a",
            task="do work",
            on_complete=my_callback,
        )

        await task.wait(timeout=5.0)
        assert len(callback_results) == 1
        assert callback_results[0].status == TaskStatus.COMPLETED

        await kit.close()

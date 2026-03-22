"""Tests for Supervisor wait_for_result=True (inline delegation).

Exercises the _run_worker_inline path end-to-end using MockAIProvider
so no real API calls are made.
"""

from __future__ import annotations

import json

from roomkit import Agent, RoomKit, Supervisor, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.state import get_conversation_state
from roomkit.providers.ai.mock import MockAIProvider

# -- Helpers ------------------------------------------------------------------


def _find_reply(events: list[RoomEvent], agent_id: str) -> RoomEvent | None:
    for event in events:
        if event.source.channel_id == agent_id:
            return event
    return None


# -- Tests --------------------------------------------------------------------


class TestWaitForResultBasic:
    """Supervisor with wait_for_result=True and a single worker."""

    async def test_inline_delegation_returns_worker_output(self) -> None:
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["Delegating now."]),
            role="Boss",
            memory=SlidingWindowMemory(max_events=50),
        )
        worker = Agent(
            "worker",
            provider=MockAIProvider(responses=["Worker result."]),
            role="Worker",
            description="Does work",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=True,
            ),
        )

        ws = WebSocketChannel("ws")
        inbox: list[RoomEvent] = []

        async def on_receive(_conn: str, event: RoomEvent) -> None:
            inbox.append(event)

        ws.register_connection("user", on_receive)
        kit.register_channel(ws)

        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        # Call the delegation tool directly
        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = await supervisor.tool_handler("delegate_to_worker", {"task": "Do the thing"})
        parsed = json.loads(result)

        assert parsed["status"] == "completed"
        assert parsed["worker"] == "worker"
        assert "Worker result." in parsed["result"]

        await kit.close()

    async def test_child_room_has_no_orchestration(self) -> None:
        """Child rooms must not inherit the parent's orchestration."""
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["ok"]),
            memory=SlidingWindowMemory(max_events=50),
        )
        worker = Agent(
            "worker",
            provider=MockAIProvider(responses=["done"]),
            description="Worker",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=True,
            ),
        )

        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = await supervisor.tool_handler("delegate_to_worker", {"task": "test"})
        parsed = json.loads(result)

        # If orchestration leaked to child room, the supervisor would
        # respond instead of the worker, causing infinite recursion.
        # A successful "completed" status proves isolation works.
        assert parsed["status"] == "completed"
        assert parsed["worker"] == "worker"

        await kit.close()


class TestWaitForResultHooks:
    """ON_TASK_DELEGATED and ON_TASK_COMPLETED hooks fire for inline delegation."""

    async def test_delegation_hooks_fire(self) -> None:
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["ok"]),
            memory=SlidingWindowMemory(max_events=50),
        )
        worker = Agent(
            "worker",
            provider=MockAIProvider(responses=["result"]),
            description="Worker",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=True,
            ),
        )

        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        hook_events: list[tuple[str, dict]] = []

        @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
        async def on_delegated(event: RoomEvent, _ctx: object) -> None:
            hook_events.append(("delegated", event.metadata))

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def on_completed(event: RoomEvent, _ctx: object) -> None:
            hook_events.append(("completed", event.metadata))

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        await supervisor.tool_handler("delegate_to_worker", {"task": "test"})

        assert len(hook_events) == 2

        assert hook_events[0][0] == "delegated"
        assert hook_events[0][1]["agent_id"] == "worker"

        assert hook_events[1][0] == "completed"
        assert hook_events[1][1]["agent_id"] == "worker"
        assert hook_events[1][1]["task_status"] == "completed"
        assert "duration_ms" in hook_events[1][1]

        await kit.close()


class TestWaitForResultMultipleWorkers:
    """Supervisor with two workers — both should be callable."""

    async def test_two_workers_sequential(self) -> None:
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["ok"]),
            memory=SlidingWindowMemory(max_events=50),
        )
        researcher = Agent(
            "researcher",
            provider=MockAIProvider(responses=["Research findings."]),
            description="Researches",
            memory=SlidingWindowMemory(max_events=50),
        )
        writer = Agent(
            "writer",
            provider=MockAIProvider(responses=["Article text."]),
            description="Writes",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[researcher, writer],
                wait_for_result=True,
            ),
        )

        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")

        # Delegate to researcher
        r1 = json.loads(
            await supervisor.tool_handler("delegate_to_researcher", {"task": "Research AI"})
        )
        assert r1["status"] == "completed"
        assert "Research findings." in r1["result"]

        # Delegate to writer
        r2 = json.loads(
            await supervisor.tool_handler("delegate_to_writer", {"task": "Write about AI"})
        )
        assert r2["status"] == "completed"
        assert "Article text." in r2["result"]

        await kit.close()


class TestWaitForResultFalse:
    """Default behavior (wait_for_result=False) still uses kit.delegate."""

    async def test_async_delegation_returns_task_id(self) -> None:
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["ok"]),
            memory=SlidingWindowMemory(max_events=50),
        )
        worker = Agent(
            "worker",
            provider=MockAIProvider(responses=["done"]),
            description="Worker",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                # wait_for_result defaults to False
            ),
        )

        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_to_worker", {"task": "Do it"}))

        # Async delegation returns immediately with task_id
        assert result["status"] == "delegated"
        assert "task_id" in result
        assert result["worker"] == "worker"

        await kit.close()


class TestInlineDelegationState:
    """Verify conversation state in parent room is unaffected."""

    async def test_parent_state_unchanged_after_delegation(self) -> None:
        supervisor = Agent(
            "boss",
            provider=MockAIProvider(responses=["ok"]),
            memory=SlidingWindowMemory(max_events=50),
        )
        worker = Agent(
            "worker",
            provider=MockAIProvider(responses=["done"]),
            description="Worker",
            memory=SlidingWindowMemory(max_events=50),
        )

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=True,
            ),
        )

        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        # State before delegation
        room = await kit.get_room("room")
        state_before = get_conversation_state(room)

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        await supervisor.tool_handler("delegate_to_worker", {"task": "test"})

        # State after delegation — should be unchanged
        room = await kit.get_room("room")
        state_after = get_conversation_state(room)

        assert state_after.active_agent_id == state_before.active_agent_id
        assert state_after.phase == state_before.phase

        await kit.close()

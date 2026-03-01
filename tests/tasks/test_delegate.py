"""Tests for delegation tool, handler, and setup_delegation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.orchestration.handoff import _room_id_var
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import (
    DELEGATE_TOOL,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
)
from roomkit.tasks.models import DelegatedTask

# -- Tool definition ----------------------------------------------------------


class TestDelegateTool:
    def test_tool_definition(self):
        assert DELEGATE_TOOL.name == "delegate_task"
        props = DELEGATE_TOOL.parameters["properties"]
        assert "agent" in props
        assert "task" in props
        assert "context" in props
        assert "share_channels" in props
        assert DELEGATE_TOOL.parameters["required"] == ["agent", "task"]

    def test_build_delegate_tool_empty_targets(self):
        tool = build_delegate_tool([])
        assert tool is DELEGATE_TOOL

    def test_build_delegate_tool_with_targets(self):
        tool = build_delegate_tool(
            [
                ("pr-reviewer", "Reviews PRs"),
                ("code-writer", None),
            ]
        )
        assert tool.name == "delegate_task"
        agent_prop = tool.parameters["properties"]["agent"]
        assert agent_prop["enum"] == ["pr-reviewer", "code-writer"]
        assert "pr-reviewer: Reviews PRs" in tool.description
        assert "code-writer" in tool.description


# -- DelegateHandler ----------------------------------------------------------


class TestDelegateHandler:
    async def test_handle_calls_kit_delegate(self):
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="pr-reviewer",
            task="review PR",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, notify="voice-agent")
        result = await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={
                "agent": "pr-reviewer",
                "task": "review PR #42",
                "context": {"repo": "roomkit"},
                "share_channels": ["email-out"],
            },
        )

        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"
        assert result["agent_id"] == "pr-reviewer"

        kit.delegate.assert_called_once_with(
            room_id="room-1",
            agent_id="pr-reviewer",
            task="review PR #42",
            context={"repo": "roomkit"},
            share_channels=["email-out"],
            notify="voice-agent",
            delivery_strategy=None,
        )

    async def test_handle_uses_default_share_channels(self):
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t2",
            child_room_id="child-2",
            parent_room_id="room-1",
            agent_id="agent-a",
            task="do stuff",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, default_share_channels=["email-out", "slack"])
        await handler.handle(
            room_id="room-1",
            calling_agent_id="agent-main",
            arguments={"agent": "agent-a", "task": "do stuff"},
        )

        _, kwargs = kit.delegate.call_args
        assert kwargs["share_channels"] == ["email-out", "slack"]


# -- setup_delegation ---------------------------------------------------------


class TestSetupDelegation:
    def test_injects_tool_and_wraps_handler(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_delegation(channel, handler)

        tool_names = [t.name for t in channel._extra_tools]
        assert "delegate_task" in tool_names
        assert channel._tool_handler is not None

    def test_double_setup_raises(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)

        setup_delegation(channel, handler)
        with pytest.raises(RuntimeError, match="already called"):
            setup_delegation(channel, handler)

    async def test_wrapped_handler_intercepts_delegate_task(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="pr-reviewer",
            task="review",
        )
        kit.delegate = AsyncMock(return_value=task_handle)

        handler = DelegateHandler(kit, notify="ai-main")
        setup_delegation(channel, handler)

        # Set room_id context
        token = _room_id_var.set("room-1")
        try:
            result_str = await channel._tool_handler(
                "delegate_task",
                {"agent": "pr-reviewer", "task": "review PR"},
            )
        finally:
            _room_id_var.reset(token)

        result = json.loads(result_str)
        assert result["status"] == "delegated"
        assert result["task_id"] == "t1"

    async def test_wrapped_handler_no_room_id_returns_error(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))
        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_delegation(channel, handler)

        token = _room_id_var.set(None)
        try:
            result_str = await channel._tool_handler(
                "delegate_task",
                {"agent": "a", "task": "b"},
            )
        finally:
            _room_id_var.reset(token)

        result = json.loads(result_str)
        assert "error" in result

    async def test_wrapped_handler_passes_through_other_tools(self):
        channel = AIChannel("ai-main", provider=MockAIProvider(responses=["hi"]))

        called = []

        async def original_handler(name: str, arguments: dict) -> str:
            called.append(name)
            return json.dumps({"ok": True})

        channel._tool_handler = original_handler

        kit = MagicMock()
        handler = DelegateHandler(kit)
        setup_delegation(channel, handler)

        result_str = await channel._tool_handler("some_other_tool", {})
        assert json.loads(result_str) == {"ok": True}
        assert called == ["some_other_tool"]

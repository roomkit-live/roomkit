"""Tests for the TaskPlanner (channels/_task_planner.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from roomkit.channels._task_planner import TaskPlanner
from roomkit.providers.ai.base import AITool

# ===========================================================================
# tool_definition
# ===========================================================================


class TestToolDefinition:
    def test_returns_aitool(self) -> None:
        tool = TaskPlanner.tool_definition()
        assert isinstance(tool, AITool)

    def test_tool_name(self) -> None:
        tool = TaskPlanner.tool_definition()
        assert tool.name == "plan_tasks"

    def test_tool_has_tasks_parameter(self) -> None:
        tool = TaskPlanner.tool_definition()
        assert "tasks" in tool.parameters["properties"]
        assert "tasks" in tool.parameters["required"]

    def test_tasks_is_array_of_objects(self) -> None:
        tool = TaskPlanner.tool_definition()
        tasks_schema = tool.parameters["properties"]["tasks"]
        assert tasks_schema["type"] == "array"
        items = tasks_schema["items"]
        assert items["type"] == "object"
        assert "title" in items["properties"]
        assert "status" in items["properties"]
        assert items["properties"]["status"]["enum"] == [
            "pending",
            "in_progress",
            "completed",
            "blocked",
        ]


# ===========================================================================
# handle_plan_tasks
# ===========================================================================


class TestHandlePlanTasks:
    async def test_stores_plan(self) -> None:
        planner = TaskPlanner()
        assert planner.current_plan is None

        tasks = [
            {"title": "Step 1", "status": "pending"},
            {"title": "Step 2", "status": "in_progress"},
        ]
        await planner.handle_plan_tasks({"tasks": tasks})

        assert planner.current_plan == tasks

    async def test_returns_json_with_counts(self) -> None:
        planner = TaskPlanner()
        tasks = [
            {"title": "A", "status": "pending"},
            {"title": "B", "status": "pending"},
            {"title": "C", "status": "completed"},
            {"title": "D", "status": "in_progress"},
            {"title": "E", "status": "blocked"},
        ]
        result = await planner.handle_plan_tasks({"tasks": tasks})
        data = json.loads(result)

        assert data["status"] == "ok"
        assert data["task_count"] == 5
        assert data["pending"] == 2
        assert data["completed"] == 1
        assert data["in_progress"] == 1
        assert data["blocked"] == 1

    async def test_empty_tasks(self) -> None:
        planner = TaskPlanner()
        result = await planner.handle_plan_tasks({"tasks": []})
        data = json.loads(result)

        assert data["task_count"] == 0
        assert data["pending"] == 0

    async def test_missing_tasks_key(self) -> None:
        planner = TaskPlanner()
        result = await planner.handle_plan_tasks({})
        data = json.loads(result)

        assert data["task_count"] == 0
        assert planner.current_plan == []

    async def test_publishes_ephemeral_event(self) -> None:
        planner = TaskPlanner()
        realtime = AsyncMock()
        tasks = [{"title": "Do thing", "status": "pending"}]

        await planner.handle_plan_tasks(
            {"tasks": tasks},
            realtime=realtime,
            room_id="room-1",
            channel_id="ai-1",
        )

        realtime.publish_to_room.assert_called_once()
        call_args = realtime.publish_to_room.call_args
        assert call_args[0][0] == "room-1"
        event = call_args[0][1]
        assert event.data["type"] == "plan_updated"
        assert event.data["tasks"] == tasks

    async def test_no_publish_without_realtime(self) -> None:
        planner = TaskPlanner()
        tasks = [{"title": "Do thing", "status": "pending"}]

        # Should not raise even without realtime
        result = await planner.handle_plan_tasks({"tasks": tasks})
        data = json.loads(result)
        assert data["status"] == "ok"

    async def test_no_publish_without_room_id(self) -> None:
        planner = TaskPlanner()
        realtime = AsyncMock()
        tasks = [{"title": "Do thing", "status": "pending"}]

        await planner.handle_plan_tasks(
            {"tasks": tasks},
            realtime=realtime,
            room_id=None,
        )

        realtime.publish_to_room.assert_not_called()

    async def test_publish_failure_is_suppressed(self) -> None:
        planner = TaskPlanner()
        realtime = AsyncMock()
        realtime.publish_to_room.side_effect = RuntimeError("publish failed")

        tasks = [{"title": "Do thing", "status": "pending"}]

        # Should not raise
        result = await planner.handle_plan_tasks(
            {"tasks": tasks},
            realtime=realtime,
            room_id="room-1",
        )
        data = json.loads(result)
        assert data["status"] == "ok"

    async def test_channel_id_in_ephemeral_event(self) -> None:
        planner = TaskPlanner()
        realtime = AsyncMock()

        await planner.handle_plan_tasks(
            {"tasks": [{"title": "X", "status": "pending"}]},
            realtime=realtime,
            room_id="room-1",
            channel_id="ch-42",
        )

        event = realtime.publish_to_room.call_args[0][1]
        assert event.channel_id == "ch-42"
        assert event.user_id == "ch-42"

    async def test_no_channel_id_uses_empty_string(self) -> None:
        planner = TaskPlanner()
        realtime = AsyncMock()

        await planner.handle_plan_tasks(
            {"tasks": [{"title": "X", "status": "pending"}]},
            realtime=realtime,
            room_id="room-1",
        )

        event = realtime.publish_to_room.call_args[0][1]
        assert event.user_id == ""


# ===========================================================================
# format_plan_prompt
# ===========================================================================


class TestFormatPlanPrompt:
    def test_formats_tasks(self) -> None:
        tasks = [
            {"title": "Analyze data", "status": "completed"},
            {"title": "Write report", "status": "in_progress"},
            {"title": "Review", "status": "pending"},
            {"title": "Blocked task", "status": "blocked"},
        ]
        result = TaskPlanner.format_plan_prompt(tasks)

        assert "## Current Task Plan" in result
        assert "- [x] Analyze data (completed)" in result
        assert "- [-] Write report (in_progress)" in result
        assert "- [ ] Review (pending)" in result
        assert "- [!] Blocked task (blocked)" in result

    def test_empty_tasks(self) -> None:
        result = TaskPlanner.format_plan_prompt([])
        assert "## Current Task Plan" in result
        # Only the header, no task lines
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_missing_status_defaults_to_pending(self) -> None:
        tasks = [{"title": "No status"}]
        result = TaskPlanner.format_plan_prompt(tasks)
        assert "- [ ] No status (pending)" in result

    def test_missing_title_defaults_to_untitled(self) -> None:
        tasks = [{"status": "completed"}]
        result = TaskPlanner.format_plan_prompt(tasks)
        assert "- [x] Untitled (completed)" in result

    def test_unknown_status_uses_default_icon(self) -> None:
        tasks = [{"title": "Weird", "status": "unknown_status"}]
        result = TaskPlanner.format_plan_prompt(tasks)
        assert "- [ ] Weird (unknown_status)" in result

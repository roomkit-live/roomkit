"""A delegated task is followed on the StatusBus, its hand-back names it, and
``task_status`` reads a room's tasks (RMK-537, RMK-538, RMK-539; RFC §23.3, §23.4)."""

from __future__ import annotations

import json
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.mixins.delegation import task_name
from roomkit.models.enums import ChannelCategory, EventType, TaskStatus
from roomkit.orchestration.status_bus import (
    RESULT_LIMIT,
    StatusEntry,
    StatusLevel,
    bounded_text,
)
from roomkit.orchestration.strategies.supervisor._inject_strategy import _follow_hint
from roomkit.providers.ai.base import AIContext, AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import TASK_STATUS_TOOL, TaskStatusTool
from roomkit.tasks.status import room_tasks
from roomkit.tools.compose import extract_tools
from roomkit.tools.context import tool_turn_context
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


class _FailingProvider(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        raise RuntimeError("secret provider detail")


async def _room(worker: Agent) -> RoomKit:
    """Room ``r``: a transport, a speaking agent, and *worker* registered."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("speaker", provider=MockAIProvider(responses=["Voilà."])))
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)
    return kit


async def _task_entries(kit: RoomKit) -> list[StatusEntry]:
    return [e for e in await kit.status_bus.recent(50) if e.action == "task"]


async def test_a_delegation_posts_pending_then_completed_with_its_task() -> None:
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=["8°C, averses."])))
    task = await kit.delegate("r", "worker", "La météo à Québec demain ?", notify="speaker")
    await task.wait()
    await until(lambda: task.result is not None, timeout=5)
    entries = await _task_entries(kit)
    await kit.close()

    pending, done = entries
    assert (pending.agent_id, pending.status, pending.detail) == (
        "worker",
        StatusLevel.PENDING,
        "La météo à Québec demain ?",
    )
    assert (done.status, done.detail) == (StatusLevel.COMPLETED, "8°C, averses.")
    for entry in entries:
        assert entry.metadata["room_id"] == "r"
        assert entry.metadata["task_id"] == task.id
        assert entry.metadata["child_room_id"] == task.child_room_id
    assert done.metadata["task_status"] == "completed"
    assert "duration_ms" in done.metadata


_FORECAST = (
    "À Montréal, il fait actuellement 6,5 °C avec un ciel dégagé et un vent de 12,7 km/h. "
    "Demain, mercredi 7 octobre, on attend de la pluie, avec 42 % de risque, une minimale "
    "de 4,9 °C et une maximale de 14,9 °C."
)


async def test_task_status_gives_the_whole_result_the_summary_cut() -> None:
    """RMK-556: the bus summary stopped at « une maximale de 1 », read back as 1 °C."""
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=[_FORECAST])))
    task = await kit.delegate("r", "worker", "La météo ?", notify="speaker")
    await task.wait()
    await until(lambda: task.result is not None, timeout=5)
    [_, done] = await _task_entries(kit)
    with tool_turn_context(room_id="r"):
        answer = json.loads(await TaskStatusTool(kit).handler(TASK_STATUS_TOOL, {}))
    await kit.close()

    assert len(done.detail) <= 200
    assert done.detail.endswith("…")
    assert done.metadata["result"] == _FORECAST
    assert answer["tasks"][0]["result"] == _FORECAST


def test_a_bounded_text_is_cut_at_a_word_and_marked() -> None:
    assert bounded_text("court", 200) == "court"
    cut = bounded_text(_FORECAST, 200)
    assert cut.endswith("…") and len(cut) <= 200
    assert _FORECAST.startswith(cut[:-1])
    assert cut[:-1].split()[-1] in _FORECAST.split()  # a whole word, not « 1 » of « 14,9 »
    assert bounded_text("x" * 300, 200) == "x" * 199 + "…"  # no word to cut at
    assert len(bounded_text(_FORECAST * 40, RESULT_LIMIT)) <= RESULT_LIMIT


async def test_a_failed_task_is_posted_failed_without_its_error_text() -> None:
    kit = await _room(Agent("worker", provider=_FailingProvider()))
    task = await kit.delegate("r", "worker", "Fais-le.", notify="speaker")
    result = await task.wait()
    entries = await _task_entries(kit)
    await kit.close()

    assert result.status == TaskStatus.FAILED
    done = entries[-1]
    assert (done.status, done.detail) == (StatusLevel.FAILED, "failed")
    assert "secret" not in json.dumps([e.model_dump(mode="json") for e in entries])


async def test_a_caller_that_follows_its_tasks_itself_gets_none_posted() -> None:
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=["ok"])))
    task = await kit.delegate("r", "worker", "Fais-le.", wait=True, post_status=False)
    entries = await _task_entries(kit)
    await kit.close()

    assert task.result is not None
    assert entries == []


async def test_the_hand_back_names_its_task() -> None:
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=["8°C."])))
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def capture(event: Any, ctx: Any) -> HookResult:
        seen.append(dict(event.metadata))
        return HookResult.allow()

    delivered: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def after(event: Any, ctx: Any) -> None:
        delivered.append(dict(event.metadata))

    task = await kit.delegate("r", "worker", "La météo ?", notify="speaker")
    await task.wait()
    await until(lambda: bool(seen) and bool(delivered), timeout=5)
    await kit.close()

    expected = {
        "task_id": task.id,
        "agent_id": "worker",
        "task_status": "completed",
        "task": "La météo ?",
    }
    assert expected.items() <= seen[0].items()
    assert expected.items() <= delivered[0].items()


async def test_the_hand_back_says_what_was_asked() -> None:
    """A result back after the conversation moved on is said for what was asked
    (RMK-550: Québec's weather was said as Montréal's)."""
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=["8°C, averses."])))
    texts: list[str] = []

    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def capture(event: Any, ctx: Any) -> HookResult:
        texts.append(event.content.body)
        return HookResult.allow()

    asked = "Météo de demain\nà Québec, " + "en détail " * 40
    task = await kit.delegate("r", "worker", asked, notify="speaker")
    await task.wait()
    await until(lambda: bool(texts), timeout=5)
    await kit.close()

    header = texts[0].splitlines()[0]
    assert header.startswith(
        "[Background task from worker completed. Task: \u201cMétéo de demain à Québec, en détail"
    )
    named = header.split("\u201c")[1].split("\u201d")[0]
    assert len(named) <= 200 and named.endswith("…")  # one line, bounded
    assert header.endswith("Share the outcome with the user.]")
    assert "8°C, averses." in texts[0]


def test_a_task_is_named_on_one_line_and_bounded() -> None:
    assert task_name("  La météo\n à  Québec ? ") == "La météo à Québec ?"
    assert len(task_name("x" * 500)) == 200


async def test_task_status_lists_the_room_tasks_only() -> None:
    kit = await _room(Agent("worker", provider=MockAIProvider(responses=["8°C."])))
    await kit.create_room(room_id="other")
    await kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    mine = await kit.delegate("r", "worker", "La météo ?", notify="speaker")
    theirs = await kit.delegate("other", "worker", "Leur secret ?", notify="speaker")
    await mine.wait()
    await theirs.wait()
    tool = TaskStatusTool(kit)

    with tool_turn_context(room_id="r"):
        answer = json.loads(await tool.handler(TASK_STATUS_TOOL, {}))
        one = json.loads(await tool.handler(TASK_STATUS_TOOL, {"task_id": mine.id}))
        unknown = json.loads(await tool.handler(TASK_STATUS_TOOL, {"task_id": theirs.id}))
    outside = json.loads(await tool.handler(TASK_STATUS_TOOL, {}))
    await kit.close()

    [line] = answer["tasks"]
    assert (line["task_id"], line["agent"], line["task"], line["status"], line["result"]) == (
        mine.id,
        "worker",
        "La météo ?",
        "completed",
        "8°C.",
    )
    assert one["tasks"] == answer["tasks"]
    assert unknown["tasks"] == []
    assert "error" in outside


def test_a_running_task_reads_running_and_an_orchestration_run_reads_as_one_task() -> None:
    def entry(agent: str, status: StatusLevel, detail: str, **meta: Any) -> StatusEntry:
        return StatusEntry(
            ts="2026-10-05T18:00:00+00:00",
            agent_id=agent,
            action="task",
            status=status,
            detail=detail,
            metadata={"room_id": "r", **meta},
        )

    tasks = room_tasks(
        [
            entry("worker", StatusLevel.PENDING, "La météo ?", task_id="t1"),
            entry("writer", StatusLevel.PENDING, "Rédige."),  # a worker run, no id yet
            entry("writer", StatusLevel.COMPLETED, "Le texte.", task_id="t2"),
        ]
    )
    assert [(t["task_id"], t["agent"], t["status"]) for t in tasks] == [
        ("t1", "worker", "running"),
        ("t2", "writer", "completed"),
    ]
    assert tasks[1]["task"] == "Rédige."


def test_the_supervisor_names_task_status_only_to_an_agent_that_has_it() -> None:
    with tool_turn_context(
        room_id="r", tools=[AITool(name=TASK_STATUS_TOOL, description="", parameters={})]
    ):
        assert TASK_STATUS_TOOL in _follow_hint("follow progress")
    with tool_turn_context(room_id="r", tools=[]):
        assert _follow_hint("follow progress") == ""
    assert _follow_hint("follow progress") == ""


class _EchoTool:
    """Another tool of the same channel, listed after task_status."""

    @property
    def definition(self) -> dict[str, Any]:
        return {"name": "echo", "description": "", "parameters": {"type": "object"}}

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        return f"echo {arguments}"


async def test_task_status_leaves_another_tools_call_to_that_tool() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r")
    _, handler = extract_tools([TaskStatusTool(kit), _EchoTool()])
    assert handler is not None

    with tool_turn_context(room_id="r"):
        echoed = await handler("echo", {"x": 1})
        listed = json.loads(str(await handler(TASK_STATUS_TOOL, {})))
    await kit.close()

    assert echoed == "echo {'x': 1}"
    assert listed == {"tasks": []}

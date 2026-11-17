"""A delegated worker that its round cap cuts has no answer (RMK-414).

RFC §23.3 step 6, RFC §6.4: the task fails, its error naming how the turn
ended, its output the worker's last narration and its metadata the
``loop_end_reason``; streamed or buffered, inline or in the background, with
a transport shared into the child room or not, whether the turn wrote text or
none. An ACP worker whose prompt stopped on any reason but ``end_turn`` is cut
too (RMK-418). A worker that owes a result and is cut without it fails,
never re-prompted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from roomkit import HookExecution, HookTrigger, RoomKit, TaskCutShortError
from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, TaskStatus
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.orchestration._worker_run import task_output
from roomkit.orchestration.strategies.loop import _execute_loop
from roomkit.orchestration.strategies.supervisor.supervised import _supervisor_dispatch
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult, task_work
from tests.test_channels.test_acp import _channel
from tests.test_framework import AILikeChannel, SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)
SILENT_LOOPING = LOOPING.model_copy(update={"content": ""})


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _kit(responses: list[AIResponse], *, streaming: bool) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(ai_responses=responses, streaming=streaming),
            tools=[LOOKUP],
            tool_handler=_found,
            tool_search=False,
            max_tool_rounds=1,
        )
    )
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    return kit


def _assert_cut(result: Any) -> None:
    assert result is not None
    assert result.status == TaskStatus.FAILED
    assert result.error == "The worker's turn ended max_rounds before its answer"
    assert result.output == "Still checking."
    assert result.metadata["loop_end_reason"] == "max_rounds"


@pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
async def test_an_inline_task_fails_with_its_narration(streaming: bool, shared: bool) -> None:
    kit = await _kit([LOOPING] * 3, streaming=streaming)

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=["email-out"] if shared else None,
        wait=True,
    )

    _assert_cut(task.result)
    await kit.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["generates", "streams"])
async def test_a_background_task_fails_with_its_narration(streaming: bool) -> None:
    kit = await _kit([LOOPING] * 3, streaming=streaming)

    task = await kit.delegate("parent", "worker", "Find it.")
    result = await task.wait(timeout=5)

    _assert_cut(result)
    await kit.close()


async def test_a_worker_that_finished_still_completes() -> None:
    kit = await _kit([AIResponse(content="Found it.")], streaming=True)

    task = await kit.delegate("parent", "worker", "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.COMPLETED, "Found it.")
    assert "loop_end_reason" not in task.result.metadata
    await kit.close()


async def test_a_worker_owing_a_result_fails_when_cut_without_one() -> None:
    kit = await _kit([LOOPING] * 6, streaming=True)

    task = await kit.delegate(
        "parent", "worker", "Find it.", wait=True, require_structured_result=True
    )

    _assert_cut(task.result)
    await kit.close()


async def test_a_loop_does_not_take_a_cut_producers_narration_for_its_work() -> None:
    """Every orchestration reader reads a failed task's work as none (RMK-414)."""
    kit = await _kit([LOOPING] * 3, streaming=True)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"] * 3))
    kit.register_channel(reviewer)
    producer = kit.channels["worker"]

    out = await _execute_loop(
        kit=kit,
        room_id="parent",
        producer=producer,
        reviewers=[reviewer],
        strategy=None,
        task_desc="Find it.",
        max_iterations=2,
    )

    assert out.approved is False
    assert out.output == ""
    await kit.close()


def test_a_supervisor_reads_a_cut_worker_as_failed() -> None:
    cut = DelegatedTaskResult(
        task_id="t",
        child_room_id="c",
        parent_room_id="p",
        agent_id="worker",
        status=TaskStatus.FAILED,
        output="Still checking.",
        error="The worker's turn ended max_rounds before its answer",
    )

    assert task_work(cut) == ""
    assert task_output(cut) == "The task failed."


@pytest.mark.parametrize("cut", [LOOPING, SILENT_LOOPING], ids=["narrated", "silent"])
async def test_a_supervisor_cut_framing_the_first_task_hands_on_the_goal(
    cut: AIResponse,
) -> None:
    kit = await _kit([cut] * 3, streaming=True)
    first = Agent("first", provider=MockAIProvider(responses=["done"]))
    kit.register_channel(first)

    framed = await _supervisor_dispatch(
        kit,
        kit.channels["worker"],
        "parent",
        goal="Find the flight.",
        workers=[first],
        share_channels=None,
        task_timeout=10.0,
    )

    assert framed == "Find the flight."
    await kit.close()


async def test_the_cut_is_logged_once_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    kit = await _kit([LOOPING] * 3, streaming=True)

    with caplog.at_level(logging.DEBUG, logger="roomkit.tasks"):
        await kit.delegate("parent", "worker", "Find it.", wait=True)

    [record] = [r for r in caplog.records if "failed" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    await kit.close()


async def test_on_task_completed_names_the_turns_end() -> None:
    kit = await _kit([LOOPING] * 3, streaming=True)
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def completed(event: Any, ctx: Any) -> None:
        seen.append(event.metadata)

    await kit.delegate("parent", "worker", "Find it.", wait=True)
    await asyncio.sleep(0.05)

    [metadata] = seen
    assert (metadata["task_status"], metadata["loop_end_reason"]) == ("failed", "max_rounds")
    await kit.close()


def test_the_error_names_the_turns_end() -> None:
    error = TaskCutShortError("budget_exceeded", "Halfway there.")

    assert (error.reason, error.narration) == ("budget_exceeded", "Halfway there.")
    assert "budget_exceeded" in str(error)


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "background"])
@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
async def test_a_task_cut_before_any_text_fails_naming_the_end(shared: bool, inline: bool) -> None:
    kit = await _kit([SILENT_LOOPING] * 3, streaming=True)
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def completed(event: Any, ctx: Any) -> None:
        seen.append(event.metadata)

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=["email-out"] if shared else None,
        wait=inline,
    )
    result = task.result if inline else await task.wait(timeout=5)
    await asyncio.sleep(0.05)

    assert result is not None
    assert (result.status, result.output) == (TaskStatus.FAILED, None)
    assert result.error == "The worker's turn ended max_rounds before its answer"
    assert result.metadata["loop_end_reason"] == "max_rounds"
    [metadata] = seen
    assert (metadata["error"], metadata["loop_end_reason"]) == (result.error, "max_rounds")
    await kit.close()


class _BufferedWorker(AILikeChannel):
    """A worker that answers buffered, its turn's end named on its record only."""

    def __init__(self, channel_id: str, narration: str | None) -> None:
        super().__init__(channel_id)
        self._narration = narration

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        rows = [] if self._narration is None else [_message(self.channel_id, self._narration)]
        return ChannelOutput(
            responded=True,
            response_events=rows,
            response_metadata=ResponseMetadata({"loop_end_reason": "timeout"}),
        )


def _message(channel_id: str, body: str) -> RoomEvent:
    return RoomEvent(
        room_id="child",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.AI),
        content=TextContent(body=body),
    )


@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
@pytest.mark.parametrize("narration", [None, "Halfway."], ids=["no-text", "narrated"])
async def test_a_buffered_worker_whose_record_names_a_cut_fails(
    narration: str | None, shared: bool
) -> None:
    kit = RoomKit()
    kit.register_channel(_BufferedWorker("worker", narration))
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=["email-out"] if shared else None,
        wait=True,
    )

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.FAILED, narration)
    assert task.result.error == "The worker's turn ended timeout before its answer"
    await kit.close()


def _acp_turn(stop_reason: str) -> Any:
    async def prompt(connection: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        await connection.client.session_update(
            session_id, acp.update_agent_message_text("Let me look into that.")
        )
        return PromptResponse(stop_reason=stop_reason)

    return prompt


def _interrupted_turn() -> Any:
    async def prompt(connection: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        await connection.client.session_update(
            session_id, acp.update_agent_message_text("Let me look into that.")
        )
        # The channel closing, or a cancel while its session is rebuilt.
        raise asyncio.CancelledError

    return prompt


async def _acp_worker_task(tmp_path: Any, turn: Any, *, shared: bool) -> Any:
    kit = RoomKit()
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    connection.prompt = lambda *a, **k: turn(connection, *a, **k)  # type: ignore[method-assign]
    kit.register_channel(channel)
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def completed(event: Any, ctx: Any) -> None:
        seen.append(event.metadata)

    task = await kit.delegate(
        "parent",
        channel.channel_id,
        "Find it.",
        share_channels=["email-out"] if shared else None,
    )
    result = await task.wait(timeout=5)
    await asyncio.sleep(0.05)
    await kit.close()
    [metadata] = seen
    assert metadata["loop_end_reason"] == result.metadata.get("loop_end_reason")
    return result


@pytest.mark.parametrize("shared", [False, True], ids=["trace", "shared-transport"])
@pytest.mark.parametrize(
    ("turn", "reason"),
    [
        *((_acp_turn(stop), stop) for stop in ("max_tokens", "max_turn_requests", "cancelled")),
        (_acp_turn("refusal"), "refusal"),
        (_interrupted_turn(), "interrupted"),
    ],
    ids=["max_tokens", "max_turn_requests", "cancelled", "refusal", "interrupted"],
)
async def test_an_acp_worker_that_stopped_short_fails_with_its_narration(
    tmp_path: Any, turn: Any, reason: str, shared: bool
) -> None:
    result = await _acp_worker_task(tmp_path, turn, shared=shared)

    assert (result.status, result.output) == (TaskStatus.FAILED, "Let me look into that.")
    assert result.error == f"The worker's turn ended {reason} before its answer"
    assert result.metadata["loop_end_reason"] == reason


async def test_an_acp_worker_that_ended_its_turn_completes(tmp_path: Any) -> None:
    kit = RoomKit()
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    turn = _acp_turn("end_turn")
    connection.prompt = lambda *a, **k: turn(connection, *a, **k)  # type: ignore[method-assign]
    kit.register_channel(channel)
    await kit.create_room(room_id="parent")

    task = await kit.delegate("parent", channel.channel_id, "Find it.", wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (
        TaskStatus.COMPLETED,
        "Let me look into that.",
    )
    await kit.close()


def _cutting_agent(
    channel_id: str, responses: list[AIResponse], *, streaming: bool = True
) -> Agent:
    return Agent(
        channel_id,
        provider=MockAIProvider(ai_responses=responses, streaming=streaming),
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
    )


_SUBMITTED = {"status": "completed", "summary": "Found it.", "data": {"x": 1}}

_SHARED = pytest.mark.parametrize(
    "shared", [["helper", "email-out"], ["helper"]], ids=["transport", "trace"]
)
_STREAMING = pytest.mark.parametrize("streaming", [True, False], ids=["streams", "generates"])


@_STREAMING
@_SHARED
@pytest.mark.parametrize("worker_cut", [True, False], ids=["worker-cut", "helper-cut"])
async def test_the_answer_and_its_end_are_read_off_one_responder(
    worker_cut: bool, shared: list[str], streaming: bool
) -> None:
    """The task's answer and its end are the worker's, never the helper's
    shared into the child room: a cut helper leaves the worker's answer, a
    cut worker fails the task whatever the helper said (RMK-834)."""
    done = [AIResponse(content="Found it.")] * 3
    kit = RoomKit()
    worker = [LOOPING] * 3 if worker_cut else done
    helper = done if worker_cut else [LOOPING] * 3
    kit.register_channel(_cutting_agent("worker", worker, streaming=streaming))
    kit.register_channel(_cutting_agent("helper", helper, streaming=streaming))
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    await kit.attach_channel("parent", "helper", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate("parent", "worker", "Find it.", share_channels=shared, wait=True)

    assert task.result is not None
    if worker_cut:
        _assert_cut(task.result)
    else:
        assert (task.result.status, task.result.output, task.result.error) == (
            TaskStatus.COMPLETED,
            "Found it.",
            None,
        )
    await kit.close()


@_SHARED
async def test_a_cut_helper_leaves_a_worker_that_owes_its_result(shared: list[str]) -> None:
    """The worker hands its result back on its second turn; the helper is cut
    on each: the worker is re-prompted and its result kept (RMK-834)."""
    submit = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="s", name="submit_result", arguments=_SUBMITTED)],
    )
    replies = [AIResponse(content="Working on it."), submit, AIResponse(content="Sent.")]
    kit = RoomKit()
    kit.register_channel(
        Agent("worker", provider=MockAIProvider(ai_responses=replies, streaming=True))
    )
    kit.register_channel(_cutting_agent("helper", [LOOPING] * 9))
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    await kit.attach_channel("parent", "helper", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate(
        "parent",
        "worker",
        "Find it.",
        share_channels=shared,
        wait=True,
        require_structured_result=True,
    )

    assert task.result is not None
    assert task.result.status == TaskStatus.COMPLETED
    assert json.loads(task.result.output)["summary"] == _SUBMITTED["summary"]
    await kit.close()


@_STREAMING
@_SHARED
async def test_a_delegation_keeps_its_agent_s_answer_when_another_returns_first(
    shared: list[str], streaming: bool
) -> None:
    """RMK-834: the helper shared into the child room returns before the
    worker: the task keeps the worker's answer, whatever the order its
    responders returned in (RFC §23.3 step 6)."""
    kit = RoomKit()
    helper = _Answers(
        "helper", provider=MockAIProvider(responses=["From the helper."], streaming=streaming)
    )
    worker = MockAIProvider(responses=["From the worker."], streaming=streaming)
    kit.register_channel(_After("worker", helper, provider=worker))
    kit.register_channel(helper)
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    await kit.attach_channel("parent", "helper", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate("parent", "worker", "Find it.", share_channels=shared, wait=True)

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.COMPLETED, "From the worker.")
    await kit.close()


class _Answers(Agent):
    """An agent that tells when its turn in a task's child room returned."""

    def __init__(self, channel_id: str, **kwargs: Any) -> None:
        super().__init__(channel_id, **kwargs)
        self.returned = asyncio.Event()

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        output = await super().on_event(event, binding, context)
        if event.room_id != "parent":
            self.returned.set()
        return output


class _After(Agent):
    """An agent whose turn returns once *first*'s has."""

    def __init__(self, channel_id: str, first: _Answers, **kwargs: Any) -> None:
        super().__init__(channel_id, **kwargs)
        self._first = first

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        await asyncio.wait_for(self._first.returned.wait(), 5)
        return await super().on_event(event, binding, context)


class _Down(MockAIProvider):
    """A helper whose model fails."""

    async def generate(self, context: Any) -> AIResponse:
        raise RuntimeError("the helper's model is down")


@_SHARED
@pytest.mark.parametrize("inline", [True, False], ids=["inline", "background"])
async def test_another_agent_s_failed_turn_leaves_the_task_its_agent_s(
    shared: list[str], inline: bool
) -> None:
    """RMK-879: a helper shared into the child room whose turn fails: the task
    is its delegated agent's and completes with its answer; the helper's
    failure stays its own turn in the child room, ON_ERROR firing there (RFC
    §23.3 step 6)."""
    kit = RoomKit()
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["From the worker."])))
    kit.register_channel(Agent("helper", provider=_Down()))
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    await kit.attach_channel("parent", "helper", category=ChannelCategory.INTELLIGENCE)
    errors: list[tuple[str, str]] = []

    @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
    async def failed(event: RoomEvent, ctx: Any) -> None:
        errors.append((event.room_id, event.source.channel_id))

    task = await kit.delegate("parent", "worker", "Find it.", share_channels=shared, wait=inline)
    result = task.result if inline else await task.wait(timeout=5)
    for _ in range(50):  # ON_ERROR runs as an async hook
        if errors:
            break
        await asyncio.sleep(0.02)
    await kit.close()

    assert result is not None
    assert (result.status, result.output, result.error) == (
        TaskStatus.COMPLETED,
        "From the worker.",
        None,
    )
    # With a transport, the helper also answers the worker's answer, and
    # fails again: each of its failed turns is reported, in the child room.
    assert errors
    assert set(errors) == {(task.child_room_id, "helper")}


class _AnswersBuffered(AILikeChannel):
    """A worker that answers buffered, its record naming no end."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id != "system":
            return ChannelOutput.empty()
        return ChannelOutput(
            responded=True, response_events=[_message(self.channel_id, "From the worker.")]
        )


async def test_a_streaming_helper_s_failure_is_never_a_buffered_worker_s() -> None:
    """RMK-879 review: with a transport shared, a helper's stream that fails
    is the helper's turn, never that of a worker that did not stream: the
    task completes with the worker's answer (RFC §23.3 step 6)."""
    kit = RoomKit()
    kit.register_channel(_AnswersBuffered("worker"))
    kit.register_channel(Agent("helper", provider=_Down(streaming=True)))
    kit.register_channel(SimpleChannel("email-out"))
    await kit.create_room(room_id="parent")
    await kit.attach_channel("parent", "email-out")
    await kit.attach_channel("parent", "helper", category=ChannelCategory.INTELLIGENCE)

    task = await kit.delegate(
        "parent", "worker", "Find it.", share_channels=["helper", "email-out"], wait=True
    )
    await kit.close()

    assert task.result is not None
    assert (task.result.status, task.result.output) == (TaskStatus.COMPLETED, "From the worker.")

"""A delegated task carries its worker's turn record under ``turns`` however
the turn ended, its last turn's when a result tool re-prompted it, ``cancelled``
when a cut from outside lands once it began, and a failed one keeps the error
its turn raised, its type unchanged and marked reported where its turn
reported it (RMK-529, RFC §23.3)."""

from __future__ import annotations

import asyncio
import copy
import pickle
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.core._failure_log import was_reported
from roomkit.core.exceptions import TaskCutShortError
from roomkit.models.enums import TaskStatus
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult
from tests.test_framework import SimpleChannel
from tests.test_kit_close_doors import _SlowWords

DOORS = pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
PATHS = pytest.mark.parametrize(
    ("streaming", "share"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["buffered-trace", "stream-trace", "buffered-shared", "stream-shared"],
)
LOOKUP = AITool(name="lookup", description="Look up", parameters={"type": "object"})
ROUND = AIResponse(
    content="Checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


class _Fails(MockAIProvider):
    """Fails at its first generation, or after *rounds* tool rounds."""

    def __init__(self, *, streaming: bool, rounds: int = 0) -> None:
        super().__init__(streaming=streaming)
        self.rounds = rounds

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) <= self.rounds:
            return ROUND
        raise ProviderError("upstream unavailable", provider="mock", status_code=503)


class _AnswersThenFails(MockAIProvider):
    """Answers in text, never calling the result tool; the re-prompt fails."""

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) == 1:
            return AIResponse(content="Draft.", usage={"input_tokens": 3, "output_tokens": 7})
        raise ProviderError("upstream unavailable", provider="mock", status_code=503)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _delegated(
    provider: MockAIProvider, *, wait: bool, share: bool = False, **delegation: Any
) -> DelegatedTaskResult:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    worker = Agent("worker", provider=provider, tools=[LOOKUP], tool_handler=_found)
    kit.register_channel(worker)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    shared = ["sms"] if share else None
    task = await kit.delegate(
        "r", "worker", "Do it.", wait=wait, share_channels=shared, **delegation
    )
    result = await task.wait(timeout=5.0)
    await kit.close()
    return result


@DOORS
@PATHS
async def test_a_task_carries_its_workers_turn(wait: bool, streaming: bool, share: bool) -> None:
    answer = AIResponse(content="Done.", usage={"input_tokens": 4, "output_tokens": 2})
    provider = MockAIProvider(ai_responses=[answer], streaming=streaming)
    result = await _delegated(provider, wait=wait, share=share)

    assert result.status == TaskStatus.COMPLETED
    assert result.metadata["turns"]["worker"]["loop_end_reason"] == "completed"
    assert result.metadata["turns"]["worker"]["ai_usage"]["output_tokens"] == 2
    assert result.exception is None


@DOORS
@PATHS
@pytest.mark.parametrize("rounds", [0, 1], ids=["first-round", "after-a-round"])
async def test_a_failed_task_keeps_the_error_its_turn_raised(
    wait: bool, streaming: bool, share: bool, rounds: int
) -> None:
    result = await _delegated(_Fails(streaming=streaming, rounds=rounds), wait=wait, share=share)

    assert result.status == TaskStatus.FAILED
    assert isinstance(result.exception, ProviderError)
    assert result.exception.status_code == 503
    # Its turn reported it: whoever hands it on reports it no second time.
    assert was_reported(result.exception)
    assert "exception" not in result.model_dump()


async def test_a_re_prompted_worker_carries_its_last_turn_only() -> None:
    """The first turn completed without the result; the re-prompt failed
    before any round: the task's record is the last turn's, which names no
    end, never the first turn's ``completed``."""
    result = await _delegated(
        _AnswersThenFails(), wait=True, require_structured_result=True, max_result_retries=1
    )

    assert result.status == TaskStatus.FAILED
    assert "worker" not in result.metadata.get("turns", {})


async def test_a_cut_tasks_result_copies_and_pickles() -> None:
    result = await _delegated(_Fails(streaming=False, rounds=99), wait=True)
    result = result.model_copy(update={"exception": TaskCutShortError("max_rounds", "Checking.")})

    # A round trip of the test's own object, as a task runner that copies or
    # pickles its results makes: nothing untrusted is loaded.
    for copied in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):  # noqa: S301
        assert isinstance(copied.exception, TaskCutShortError)
        assert (copied.exception.reason, copied.exception.narration) == ("max_rounds", "Checking.")
    assert "exception" not in DelegatedTaskResult.model_json_schema()["properties"]


# -- a task cancelled from outside -------------------------------------------------


async def _cut_kit(started: asyncio.Event) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("worker", provider=_SlowWords(started)))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    return kit


@pytest.mark.parametrize("cut", ["close-inline", "close-background", "cancel-task"])
async def test_a_task_cut_mid_turn_carries_its_turn_cancelled(cut: str) -> None:
    """Read as a room turn's caller reads a read cancelled from outside
    (RFC §23.3, §6.4)."""
    started = asyncio.Event()
    kit = await _cut_kit(started)
    call = asyncio.ensure_future(kit.delegate("r", "worker", "Go.", wait=cut == "close-inline"))
    task = None if cut == "close-inline" else await call
    await started.wait()
    if cut == "cancel-task":
        assert task is not None
        await kit.cancel_task(task.id)
    await kit.close()
    task = task or await asyncio.wait_for(call, timeout=5.0)

    assert task.result is not None
    assert task.result.status == TaskStatus.CANCELLED
    assert task.result.metadata["turns"] == {"worker": {"loop_end_reason": "cancelled"}}


async def test_a_task_cut_before_its_turn_began_carries_none() -> None:
    kit = await _cut_kit(asyncio.Event())
    task = await kit.delegate("r", "worker", "Go.", wait=False)
    await kit.cancel_task(task.id)
    await kit.close()

    assert task.result is not None
    assert task.result.status == TaskStatus.CANCELLED
    assert "turns" not in task.result.metadata

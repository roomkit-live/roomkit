"""A delegated turn that failed after it began keeps its end on the task
(RMK-433, RFC §6.4, §23.3 step 6).

The task fails with the turn's error (the provider's, the ACP agent's), and
carries what the child room's record holds: ``loop_end_reason`` (``error``,
an ACP agent's ``interrupted``) and the worker's last narration as its
output, on ``ON_TASK_COMPLETED`` too, whether the turn streamed or not and
whether a transport is shared. The failure is logged as its error is.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import acp
import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    RoomKit,
    TaskCutShortError,
    TaskTurnFailedError,
)
from roomkit.channels.agent import Agent
from roomkit.core._failure_log import log_failure
from roomkit.core.mixins._child_execution import _turn_failure
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.response_metadata import ResponseMetadata
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_channels.test_acp import _channel
from tests.test_framework import AILikeChannel, SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2 == 1:
            return LOOPING
        raise ProviderError("upstream 400", provider="mock", status_code=400)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _delegate(
    kit: RoomKit, agent_id: str, *, wait: bool = True, **kw: Any
) -> tuple[Any, list[Any]]:
    completed: list[Any] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC, name="done")
    async def on_done(event: Any, ctx: Any) -> None:
        completed.append(event.metadata.get("loop_end_reason"))

    task = await kit.delegate("p", agent_id, "Find it.", wait=wait, **kw)
    if not wait:
        await task.wait(timeout=5)
    for _ in range(50):
        if completed:
            break
        await asyncio.sleep(0.01)
    assert task.result is not None
    return task.result, completed


@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
@pytest.mark.parametrize("shared", [False, True])
async def test_a_worker_turn_failed_after_a_round_keeps_its_end(wait: bool, shared: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("email-out"))
    kit.register_channel(
        Agent(
            "worker",
            provider=_FailsAfterARound(streaming=True),
            tools=[LOOKUP],
            tool_handler=_found,
            tool_search=False,
            max_tool_rounds=3,
        )
    )
    await kit.create_room(room_id="p")
    await kit.attach_channel("p", "email-out")

    result, completed = await _delegate(
        kit, "worker", wait=wait, share_channels=["email-out"] if shared else None
    )

    # The error names the provider that failed (RMK-547).
    assert (result.status, result.error) == ("failed", "mock (400): upstream 400")
    assert result.output == "Still checking."
    assert result.metadata["loop_end_reason"] == "error"
    assert completed == ["error"]
    await kit.close()


async def test_an_acp_worker_whose_prompt_raised_is_interrupted() -> None:
    async def prompt(connection: Any, session_id: str, *args: Any, **kwargs: Any) -> Any:
        await connection.client.session_update(
            session_id, acp.update_agent_message_text("Let me look into that.")
        )
        raise ConnectionError("pipe closed")

    with tempfile.TemporaryDirectory() as tmp:
        kit = RoomKit()
        channel, connection, _ = _channel(Path(tmp), emit_updates=False)
        connection.prompt = lambda *a, **k: prompt(connection, *a, **k)  # type: ignore[method-assign]
        kit.register_channel(channel)
        await kit.create_room(room_id="p")

        result, completed = await _delegate(kit, channel.channel_id)

        assert result.error == "ACP agent prompt failed: pipe closed"
        assert result.output == "Let me look into that."
        assert result.metadata["loop_end_reason"] == "interrupted"
        assert completed == ["interrupted"]
        await kit.close()


def test_a_failed_worker_turn_is_logged_as_its_error(caplog: pytest.LogCaptureFixture) -> None:
    error = ProviderError("upstream 400", provider="mock", status_code=400)
    with caplog.at_level(logging.DEBUG, logger="roomkit.test"):
        log_failure(
            logging.getLogger("roomkit.test"),
            TaskTurnFailedError(error, "error", "Still checking."),
            "Task t1",
        )

    [record] = caplog.records
    assert record.levelno == logging.WARNING and record.exc_info is None
    assert "status=400" in record.getMessage()


class _BufferedFailedWorker(AILikeChannel):
    """A worker that answers buffered: a narration, then its provider failed."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        narration = RoomEvent(
            room_id=event.room_id,
            source=EventSource(channel_id=self.channel_id, channel_type=ChannelType.AI),
            content=TextContent(body="Still checking."),
        )
        return ChannelOutput(
            responded=True,
            response_events=[narration],
            response_metadata=ResponseMetadata({"loop_end_reason": "error"}),
            error=ProviderError("upstream 400", provider="mock", status_code=400),
        )


@pytest.mark.parametrize("shared", [False, True])
async def test_a_buffered_worker_turn_failed_after_it_began_keeps_its_end(shared: bool) -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("email-out"))
    kit.register_channel(_BufferedFailedWorker("worker"))
    await kit.create_room(room_id="p")
    await kit.attach_channel("p", "email-out")

    result, completed = await _delegate(
        kit, "worker", share_channels=["email-out"] if shared else None
    )

    assert (result.error, result.output) == ("mock (400): upstream 400", "Still checking.")
    assert result.metadata["loop_end_reason"] == "error"
    assert completed == ["error"]
    await kit.close()


SUBMITTED = {"status": "completed", "summary": "found it", "data": {}}


class _SubmitsThenFails(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) == 1:
            return AIResponse(
                content="Submitting.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="s1", name="submit_result", arguments=SUBMITTED)],
            )
        raise ProviderError("overloaded 529", provider="mock", status_code=529)


async def test_a_result_submitted_before_the_failure_still_counts() -> None:
    kit = RoomKit()
    kit.register_channel(
        Agent("worker", provider=_SubmitsThenFails(streaming=True), tool_search=False)
    )
    await kit.create_room(room_id="p")

    result, _ = await _delegate(kit, "worker", require_structured_result=True)

    assert result.status == "completed"
    assert json.loads(result.output)["summary"] == "found it"
    await kit.close()


class _FailsBeforeAnyEnd(AILikeChannel):
    """A second agent shared into the child room, failing before any end."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.source.channel_id != "system":
            return ChannelOutput.empty()
        error = ProviderError("B's upstream 503", provider="b", status_code=503)
        return ChannelOutput(responded=True, error=error)


async def test_one_agents_failure_never_carries_anothers_end() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("email-out"))
    kit.register_channel(_FailsBeforeAnyEnd("b"))
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(ai_responses=[LOOPING] * 10, streaming=True),
            tools=[LOOKUP],
            tool_handler=_found,
            tool_search=False,
            max_tool_rounds=1,
        )
    )
    await kit.create_room(room_id="p")
    await kit.attach_channel("p", "email-out")
    await kit.attach_channel("p", "b", category=ChannelCategory.INTELLIGENCE)

    result, _ = await _delegate(kit, "worker", share_channels=["b", "email-out"])

    assert (result.error, result.output) == ("B's upstream 503", None)
    assert "loop_end_reason" not in result.metadata
    await kit.close()


def test_a_failure_with_no_end_or_a_completed_one_or_a_cut_is_raised_as_it_is() -> None:
    error = ProviderError("upstream 400", provider="mock", status_code=400)
    cut = TaskCutShortError("max_rounds", "Still checking.")

    assert _turn_failure(error, "text", None) is error
    assert _turn_failure(error, "text", "completed") is error
    assert _turn_failure(cut, "text", "error") is cut
    wrapped = _turn_failure(error, "text", "error")
    assert isinstance(wrapped, TaskTurnFailedError) and wrapped.__cause__ is error

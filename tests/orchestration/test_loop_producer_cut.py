"""A Loop whose producer's task failed says so (RMK-435, RMK-529, RFC
§19.7.4, §23.3).

The sync Loop publishes no empty producer message: with no output at all
the turn has no answer; with an earlier output, that output goes out,
not approved, with why the loop stopped. Either way the caller reads how the
producer's last turn ended under ``turns``, a cut there and no error, and a
turn that failed as the error it raised, its type kept. The async Loop's
delivered text says the producer's task failed, never with its error. The
voice ``delegate_workers`` of a Supervisor is a strategy's tool, unbound by
the channel's default call timeout.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import TextContent
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.loop import Loop, _async_loop_and_deliver
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


class _Refused(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        raise ProviderError("401 invalid x-api-key sk-live-SECRET", provider="p", status_code=401)


def _producer(responses: list[AIResponse]) -> Agent:
    return Agent(
        "producer",
        provider=MockAIProvider(ai_responses=responses, streaming=True),
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=1,
    )


async def _sync_loop(
    producer: Agent,
    reviews: list[str],
    *,
    max_iterations: int = 3,
    errors: list[Any] | None = None,
) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(producer)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=reviews))
    await kit.create_room(
        room_id="r",
        orchestration=Loop(agent=producer, reviewer=reviewer, max_iterations=max_iterations),
    )
    await kit.attach_channel("r", "sms")
    if errors is not None:

        @kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC, name="errors")
        async def on_error(event: Any, ctx: Any) -> None:
            errors.append(event)

    result = await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Write it."))
    )
    return kit, result


async def _producer_messages(kit: RoomKit) -> list[tuple[str | None, dict[str, Any]]]:
    return [
        (e.content.body if isinstance(e.content, TextContent) else None, e.metadata)
        for e in await kit.get_timeline("r", limit=50)
        if e.source.channel_id == "producer" and e.type == EventType.MESSAGE
    ]


def _producer_turn(result: Any) -> dict[str, Any]:
    return dict(result.response_metadata).get("turns", {}).get("producer", {})


async def test_a_producer_cut_before_any_output_gives_no_answer_and_its_reason() -> None:
    kit, result = await _sync_loop(_producer([LOOPING] * 10), ["APPROVED"])

    assert await _producer_messages(kit) == []
    assert result.error is None
    assert _producer_turn(result)["loop_end_reason"] == "max_rounds"
    await kit.close()


async def test_a_producer_cut_after_an_output_keeps_it_not_approved() -> None:
    first = AIResponse(content="Draft one.")
    errors: list[Any] = []
    kit, result = await _sync_loop(
        _producer([first, *[LOOPING] * 10]), ["Needs work."] * 3, errors=errors
    )

    [(body, metadata)] = await _producer_messages(kit)
    assert body == "Draft one."
    assert (metadata["approved"], metadata["stopped"], metadata["iteration"]) == (
        False,
        "producer_failed",
        1,
    )
    # The draft goes out and the cut reads under turns, to the caller: an
    # expected end is no error and fires no ON_ERROR, as a room turn's cut
    # fires none (RMK-513, RMK-529).
    assert result.error is None
    assert _producer_turn(result)["loop_end_reason"] == "max_rounds"
    await asyncio.sleep(0.1)
    assert errors == []
    state = get_conversation_state(await kit.get_room("r"))
    assert state.context["_loop_stopped"] == "producer_failed"
    await kit.close()


async def test_a_producer_whose_provider_fails_gives_its_error_to_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        kit, result = await _sync_loop(Agent("producer", provider=_Refused()), ["APPROVED"])

    assert await _producer_messages(kit) == []
    assert isinstance(result.error, ProviderError)
    assert result.error.status_code == 401
    assert len([r for r in caplog.records if "invalid x-api-key" in r.getMessage()]) == 1
    await kit.close()


async def test_a_completed_loop_says_how_the_producers_last_turn_ended() -> None:
    drafts = [
        AIResponse(content=f"Draft {n}.", usage={"input_tokens": 3, "output_tokens": n})
        for n in (1, 2)
    ]
    kit, result = await _sync_loop(_producer(drafts), ["Needs work.", "APPROVED"])

    turn = _producer_turn(result)
    assert turn["loop_end_reason"] == "completed"
    assert turn["ai_usage"]["output_tokens"] == 2
    await kit.close()


async def test_a_producer_with_nothing_to_say_is_named_so() -> None:
    kit, result = await _sync_loop(Agent("producer", provider=MockAIProvider(responses=[""])), [])

    assert str(result.error) == "The producer's task gave no output"
    await kit.close()


async def test_a_loop_that_runs_no_iteration_blames_no_producer() -> None:
    kit, result = await _sync_loop(_producer([]), [], max_iterations=0)

    assert await _producer_messages(kit) == []
    assert result.error is None
    await kit.close()


@pytest.mark.parametrize(
    ("reviews", "stopped", "iteration"),
    [(["Needs work.", "APPROVED"], "approved", 2), (["Needs work."] * 3, "max_iterations", 3)],
)
async def test_the_result_says_how_the_loop_stopped(
    reviews: list[str], stopped: str, iteration: int
) -> None:
    drafts = [AIResponse(content=f"Draft {n}.") for n in range(1, 4)]
    kit, result = await _sync_loop(_producer(drafts), reviews)

    [(_, metadata)] = await _producer_messages(kit)
    assert (metadata["stopped"], metadata["iteration"]) == (stopped, iteration)
    assert metadata["approved"] is (stopped == "approved")
    assert result.error is None
    await kit.close()


async def _async_text(producer: Agent) -> str:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(producer)
    reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"]))
    kit.register_channel(reviewer)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    delivered: list[str] = []
    real_deliver = kit.deliver

    async def spy(room_id: str, content: Any, **kw: Any) -> Any:
        delivered.append(str(content))
        return await real_deliver(room_id, content, **kw)

    kit.deliver = spy  # type: ignore[method-assign]
    await _async_loop_and_deliver(
        kit=kit,
        room_id="r",
        notify="sms",
        producer=producer,
        reviewers=[reviewer],
        strategy=None,
        task_desc="Write it.",
        max_iterations=3,
        on_done=lambda: None,
    )
    await kit.close()
    [text] = delivered
    return text


async def test_the_async_loop_names_the_producers_cut() -> None:
    text = await _async_text(_producer([LOOPING] * 10))

    assert text == (
        "[Your background review loop stopped before any output: "
        "the producer's task failed (cut short: max_rounds). Tell the user.]"
    )


async def test_the_async_loop_says_its_producer_failed_without_the_error() -> None:
    text = await _async_text(Agent("producer", provider=_Refused()))

    assert text == (
        "[Your background review loop stopped before any output: "
        "the producer's task failed. Tell the user.]"
    )
    assert "SECRET" not in text


async def test_the_voice_delegate_workers_waits_on_its_workers() -> None:
    kit = RoomKit()
    voice = RealtimeVoiceChannel(
        "voice", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
    )
    kit.register_channel(voice)
    worker = Agent("worker", provider=MockAIProvider())
    supervisor = Agent("sup", provider=MockAIProvider())
    kit.register_channel(worker)
    kit.register_channel(supervisor)
    await kit.create_room(
        room_id="r",
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[worker],
            strategy="parallel",
            auto_delegate=True,
            async_delivery=True,
        ),
    )

    assert voice._call_timeout("delegate_workers", "r") is None
    await kit.close()


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2 == 1:
            return LOOPING
        raise ProviderError("upstream 400", provider="mock", status_code=400)


def _failing_after_a_round() -> Agent:
    return Agent(
        "producer",
        provider=_FailsAfterARound(streaming=True),
        tools=[LOOKUP],
        tool_handler=_found,
        tool_search=False,
        max_tool_rounds=3,
    )


async def test_a_producer_failing_after_a_round_gives_its_error_not_a_cut() -> None:
    kit, result = await _sync_loop(_failing_after_a_round(), ["APPROVED"])

    assert isinstance(result.error, ProviderError)
    assert result.error.status_code == 400
    assert _producer_turn(result)["loop_end_reason"] == "error"
    await kit.close()


async def test_the_async_loop_names_a_producer_failing_after_a_round_as_failed() -> None:
    text = await _async_text(_failing_after_a_round())

    assert text == (
        "[Your background review loop stopped before any output: "
        "the producer's task failed. Tell the user.]"
    )

"""A supervisor's task-formulation pass that hands on no task (RMK-436, RFC §19.7.3).

Stopped short of its answer (its round cap), no worker runs and the message
it answered still gets an answer: the supervisor's fallback, stored and
delivered with the turn's record (``loop_end_reason``, ``ai_usage``, what the
turn wrote). Stopped on purpose (``cancelled``), nothing is said. Failed with
a provider error, the caller reads the error, logged once at its own level.
None of these hands the room's transports an empty stream.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.steering import Cancel
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall, ProviderError
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_response_metadata
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
LOOPING = AIResponse(
    content="Still checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


class _Streamer(SimpleChannel):
    """A transport that takes streamed answers, recording each one."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.streams: list[list[str]] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self, text_stream: Any, event: Any, binding: Any, context: Any
    ) -> Any:
        self.streams.append([d async for d in text_stream if isinstance(d, str)])
        return ChannelOutput.empty()


class _FailsAfterARound(MockAIProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if len(self.calls) % 2 == 1:
            return LOOPING
        raise ProviderError("upstream 400", provider="mock", status_code=400)


async def _supervised(
    provider: MockAIProvider, *, cancel: bool = False, max_rounds: int = 1
) -> tuple[RoomKit, Any, _Streamer, list[str]]:
    kit = RoomKit()
    transport = _Streamer("ws")
    kit.register_channel(transport)
    held: dict[str, Agent] = {}

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        record = current_response_metadata()
        if record is not None:
            record["kb_source"] = "doc-42"
        if cancel:
            held["sup"].steer(Cancel(), room_id="r")
        return "found"

    supervisor = Agent(
        "sup",
        provider=provider,
        tools=[LOOKUP],
        tool_handler=lookup,
        tool_search=False,
        max_tool_rounds=max_rounds,
    )
    held["sup"] = supervisor
    worker = Agent("worker", provider=MockAIProvider(responses=["worker answer"]))
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    delegated: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC, name="delegated")
    async def on_delegated(event: Any, ctx: Any) -> None:
        delegated.append(event.metadata.get("agent_id"))

    orchestration = Supervisor(
        supervisor=supervisor, workers=[worker], strategy="parallel", auto_delegate=True
    )
    await kit.create_room(room_id="r", orchestration=orchestration)
    await kit.attach_channel("r", "ws")
    result = await kit.process_inbound(
        InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="Find it."))
    )
    await asyncio.sleep(0.1)
    return kit, result, transport, delegated


def _delivered_messages(transport: SimpleChannel) -> list[str]:
    return [e.content.body for e in transport.delivered if e.type == EventType.MESSAGE]


async def _supervisor_messages(kit: RoomKit) -> list[RoomEvent]:
    return [
        e
        for e in await kit.get_timeline("r", limit=50)
        if e.source.channel_id == "sup" and e.type == EventType.MESSAGE
    ]


async def test_a_cut_pass_answers_with_the_supervisors_fallback() -> None:
    kit, result, transport, delegated = await _supervised(
        MockAIProvider(ai_responses=[LOOPING] * 50, streaming=True)
    )

    assert delegated == []
    [fallback] = await _supervisor_messages(kit)
    assert fallback.content.body == FALLBACK_FAILED
    # One deeper than the message it answers, as any answer (RFC §8.3).
    assert fallback.chain_depth == 1
    # The turn's whole record, as its last message would carry it.
    assert fallback.metadata["loop_end_reason"] == "max_rounds"
    assert fallback.metadata["kb_source"] == "doc-42"
    assert "ai_usage" in fallback.metadata
    assert result.response_metadata["kb_source"] == "doc-42"
    assert _delivered_messages(transport) == [FALLBACK_FAILED]
    assert transport.streams == []
    assert result.error is None
    await kit.close()


async def test_a_pass_stopped_on_purpose_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="roomkit"):
        kit, _, transport, delegated = await _supervised(
            MockAIProvider(ai_responses=[LOOPING] * 50, streaming=True), cancel=True, max_rounds=3
        )

    assert delegated == []
    assert await _supervisor_messages(kit) == []
    assert _delivered_messages(transport) == []
    assert transport.streams == []
    # A stop someone chose is no failure: nothing logged, as on a room turn.
    assert "no answer to hand on" not in caplog.text
    await kit.close()


async def test_a_failed_pass_gives_its_error_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="roomkit"):
        kit, result, transport, delegated = await _supervised(
            _FailsAfterARound(streaming=True), max_rounds=3
        )

    assert delegated == []
    assert isinstance(result.error, ProviderError)
    assert _delivered_messages(transport) == []
    assert transport.streams == []
    # Named once, at its own level: whoever opened the turn may not read it.
    named = [r for r in caplog.records if "upstream 400" in r.getMessage()]
    assert [r.levelno for r in named] == [logging.WARNING]
    await kit.close()

"""A turn that did not complete reports nothing and runs nothing more (RMK-282).

ON_AI_RESPONSE reports a turn whose loop reached its end; a turn whose
response stream was closed first (a barge-in, a consumer that refused the
answer) or that was cancelled from outside fires nothing, and its span never
ends ``ok`` (RFC §6.4). A Cancel that lands while a round's calls are
announced, before they run, stops them (RFC §21.3).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelOutput
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from roomkit.models.steering import Cancel
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    ProviderError,
    StreamDone,
    StreamEvent,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from tests.test_framework import SimpleChannel

_LOOKUP = AITool(name="lookup", description="Look up")


def _round(text: str = "Let me check that for you.") -> AIResponse:
    return AIResponse(
        content=text,
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="tc1", name="lookup", arguments={"q": "wire-money"})],
    )


class _BargeIn(SimpleChannel):
    """A voice transport: the user speaks over the first streamed chunk."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(self, text_stream, event, binding, context):  # type: ignore[no-untyped-def]
        async for chunk in text_stream:
            if isinstance(chunk, str):
                return ChannelOutput.empty()
        return ChannelOutput.empty()


class _Turn:
    """A room with one AI channel, what ran and what was reported."""

    def __init__(self) -> None:
        self.telemetry = MockTelemetryProvider()
        self.kit = RoomKit(telemetry=self.telemetry)
        self.ran: list[dict[str, Any]] = []
        self.reports: list[Any] = []

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.ran.append(arguments)
        return "ok"

    async def build(self, provider: MockAIProvider, **channel: Any) -> AIChannel:
        ai = AIChannel(
            "ai1", provider=provider, tool_handler=self.handler, tools=[_LOOKUP], **channel
        )
        self.kit.register_channel(SimpleChannel("sms1"))
        self.kit.register_channel(ai)
        await self.kit.create_room(room_id="r1")
        await self.kit.attach_channel("r1", "sms1")
        await self.kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

        @self.kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC, name="spy")
        async def spy(event: Any, ctx: Any) -> None:
            self.reports.append(event)

        return ai

    async def say(self, body: str = "go") -> None:
        await self.kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body=body))
        )
        await self.kit.hook_engine.drain_observers()  # the ASYNC report spy

    def spans(self) -> list[Any]:
        return self.telemetry.get_spans(SpanKind.LLM_GENERATE)

    async def rows(self, kind: EventType) -> list[Any]:
        events = await self.kit.store.list_events("r1")
        return [e for e in events if e.type == kind and e.source.channel_id == "ai1"]


async def test_a_barge_in_on_a_streamed_tool_turn_reports_nothing() -> None:
    turn = _Turn()
    turn.kit.register_channel(_BargeIn("tts1"))
    await turn.build(MockAIProvider(streaming=True, ai_responses=[_round(), _round("Done.")]))
    await turn.kit.attach_channel("r1", "tts1")

    await turn.say()

    assert turn.reports == []
    assert turn.ran == []
    assert [span.status for span in turn.spans()] == ["cancelled"]
    [message] = await turn.rows(EventType.MESSAGE)
    assert message.metadata.get("cancelled") is True


async def test_a_barge_in_keeps_what_the_rounds_used_on_the_span() -> None:
    """Nothing is reported for the turn, so its span keeps its tokens."""
    turn = _Turn()
    turn.kit.register_channel(_BargeIn("tts1"))
    used = {"input_tokens": 1000, "output_tokens": 50}
    first = _round("").model_copy(update={"usage": used})
    await turn.build(MockAIProvider(streaming=True, ai_responses=[first, _round("Here.")]))
    await turn.kit.attach_channel("r1", "tts1")

    await turn.say()

    [span] = turn.spans()
    assert span.status == "cancelled"
    assert span.attributes[Attr.LLM_INPUT_TOKENS] == 1000
    assert span.attributes[Attr.LLM_OUTPUT_TOKENS] == 50
    assert span.attributes[Attr.LLM_TOOL_COUNT] == 1


async def test_a_schema_turn_cut_short_reports_nothing_in_either_loop(streaming: bool) -> None:
    turn = _Turn()
    provider = MockAIProvider(
        streaming=streaming,
        response_schema=True,
        response_schema_with_tools=True,
        ai_responses=[_round("")],
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    await turn.build(provider, response_schema=schema, max_tool_rounds=1)
    errors: list[Any] = []

    @turn.kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC, name="card")
    async def card(event: Any, ctx: Any) -> None:
        errors.append(event)

    await turn.say()

    assert turn.reports == []
    assert len(errors) == 1
    # The refused answer fails the turn: an error, not a cancellation.
    assert [span.status for span in turn.spans()] == ["error"]


async def test_a_stop_while_the_calls_are_announced_runs_none() -> None:
    turn = _Turn()
    ai = await turn.build(MockAIProvider(streaming=True, ai_responses=[_round(""), _round()]))

    @turn.kit.hook(HookTrigger.BEFORE_BROADCAST, name="stop_on_start")
    async def stop(event: Any, ctx: Any) -> HookResult:
        if event.type == EventType.TOOL_CALL_START:
            ai.steer(Cancel(reason="user pressed stop"), room_id="r1")
        return HookResult.allow()

    await turn.say()

    assert turn.ran == []
    [end] = await turn.rows(EventType.TOOL_CALL_END)
    assert end.content.status == "failed"
    assert [(r.loop_end_reason, r.tool_calls_count) for r in turn.reports] == [("cancelled", 0)]


class _StopAfterTheRound(MockAIProvider):
    """The model's round is over when the stop lands, its calls not yet announced."""

    channel: AIChannel | None = None

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        async for event in super().generate_structured_stream(context):
            yield event
            if isinstance(event, StreamDone) and self.channel is not None:
                self.channel.steer(Cancel(reason="user pressed stop"), room_id="r1")


async def test_a_stop_after_the_models_last_event_announces_nothing() -> None:
    turn = _Turn()
    provider = _StopAfterTheRound(streaming=True, ai_responses=[_round(""), _round()])
    provider.channel = await turn.build(provider)

    await turn.say()

    assert turn.ran == []
    assert await turn.rows(EventType.TOOL_CALL_START) == []
    assert [(r.loop_end_reason, r.tool_calls_count) for r in turn.reports] == [("cancelled", 0)]


class _StreamingTransport(SimpleChannel):
    """A transport that reads the whole stream."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(self, text_stream, event, binding, context):  # type: ignore[no-untyped-def]
        async for _ in text_stream:
            pass
        return ChannelOutput.empty()


@pytest.mark.parametrize("streamed_to", [True, False], ids=["streaming-target", "headless"])
async def test_a_turn_that_fails_while_a_call_is_open_closes_it(
    streamed_to: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("executor bug")

    monkeypatch.setattr(AIChannel, "_execute_round_tools", broken)
    turn = _Turn()
    if streamed_to:
        turn.kit.register_channel(_StreamingTransport("ws1"))
    await turn.build(MockAIProvider(streaming=True, ai_responses=[_round(""), _round()]))
    if streamed_to:
        await turn.kit.attach_channel("r1", "ws1")

    await turn.say()

    [end] = await turn.rows(EventType.TOOL_CALL_END)
    assert end.content.status == "failed"
    # The row names the outcome; the exception's text stays in the log.
    assert end.content.error == "turn failed"


class _Hang(MockAIProvider):
    """Never answers; ``asked`` is set once a turn reached it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.asked = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.asked.set()
        await asyncio.sleep(3600)
        raise AssertionError("never answers")


async def test_a_turn_cancelled_from_outside_closes_its_span(streaming: bool) -> None:
    turn = _Turn()
    provider = _Hang(streaming=streaming)
    await turn.build(provider)
    task = asyncio.create_task(turn.say())
    await asyncio.wait_for(provider.asked.wait(), 1.0)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert turn.reports == []
    assert [span.status for span in turn.spans()] == ["cancelled"]


async def test_a_steering_cancel_closes_its_span_cancelled(streaming: bool) -> None:
    """RMK-289: a turn that reached its end on a steering Cancel is reported
    ``cancelled``, and its span is too, never ``ok`` (RFC §6.4)."""
    turn = _Turn()
    ai: AIChannel | None = None

    async def cancelling(name: str, arguments: dict[str, Any]) -> str:
        assert ai is not None
        ai.steer(Cancel(), room_id="r1")
        return "ok"

    turn.handler = cancelling  # type: ignore[method-assign]
    ai = await turn.build(MockAIProvider(streaming=streaming, ai_responses=[_round(), _round()]))

    await turn.say()

    assert [r.loop_end_reason for r in turn.reports] == ["cancelled"]
    assert [span.status for span in turn.spans()] == ["cancelled"]


class _FailsAfterTheRound(MockAIProvider):
    """Answers the first round, then the provider fails, whichever loop draws."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._generations = 0

    def _next_response(self) -> AIResponse:
        self._generations += 1
        if self._generations == 2:
            raise ProviderError("upstream 500", retryable=False, status_code=500)
        return super()._next_response()


async def test_a_schema_turn_the_provider_interrupts_fails_in_either_loop(
    streaming: bool,
) -> None:
    """RMK-289: interrupted after a round, a turn constrained to a schema has no
    document to deliver: it fails ``truncated``, the marker never delivered as
    its answer (RFC A.9)."""
    turn = _Turn()
    provider = _FailsAfterTheRound(
        streaming=streaming,
        response_schema=True,
        response_schema_with_tools=True,
        ai_responses=[_round("Let me check that."), AIResponse(content="never")],
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    await turn.build(provider, response_schema=schema)

    result = await turn.kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await turn.kit.hook_engine.drain_observers()

    assert isinstance(result.error, ResponseSchemaError)
    assert result.error.reason == "truncated"
    messages = [e.content.body for e in await turn.rows(EventType.MESSAGE)]
    assert "[Response interrupted]" not in messages
    assert [span.status for span in turn.spans()] == ["error"]


async def test_a_final_answer_failing_its_schema_after_a_round_is_no_interruption(
    streaming: bool,
) -> None:
    """RMK-289: the provider's own check refusing the final answer fails the
    turn with that error; the turn was not interrupted, so no marker is
    delivered and nothing reports it (RFC A.9)."""
    turn = _Turn()
    provider = MockAIProvider(
        streaming=streaming,
        response_schema=True,
        response_schema_with_tools=True,
        ai_responses=[_round("Let me check that."), AIResponse(content="Sure, it is yes.")],
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    await turn.build(provider, response_schema=schema)

    result = await turn.kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await turn.kit.hook_engine.drain_observers()

    assert isinstance(result.error, ResponseSchemaError)
    assert result.error.reason == "invalid_json"
    assert turn.reports == []
    messages = [e.content.body for e in await turn.rows(EventType.MESSAGE)]
    assert "[Response interrupted]" not in messages

"""Reasoning blocks go back as they came (RFC §6.4, RMK-377).

A vendor whose reasoning comes in blocks (Anthropic) signs each one and refuses
a replayed round whose blocks were merged, split or reordered; a redacted block
comes as opaque data. Each block is kept with its own signature, a redacted one
as its data, and the next round replays them where they came relative to the
text and the calls. A provider without blocks keeps one block per round.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCall,
    AIToolCallPart,
    ServedCall,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    response_stream_events,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.round_parts import RoundTranscript
from roomkit.providers.ai.thinking_blocks import ThinkingBlocks
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.anthropic.request import build_messages
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.realtime.reasoning import (
    AIProviderReasoningBackend,
    ReasoningRequest,
    ToolCallResult,
)
from tests.tool_loop_modes import run_tool_loop

_CTX = AIContext(messages=[AIMessage(role="user", content="hi")])


def _delta(block: int | None, thinking: str = "", **fields: Any) -> StreamThinkingDelta:
    return StreamThinkingDelta(thinking=thinking, block=block, **fields)


class TestThinkingBlocks:
    def test_each_block_keeps_its_text_and_signature(self) -> None:
        blocks = ThinkingBlocks()
        for delta in (
            _delta(0, "first "),
            _delta(0, "thought"),
            _delta(0, signature="S0"),
            _delta(2, redacted="RRR"),
            _delta(3, "second"),
            _delta(3, signature="S3"),
        ):
            blocks.add(delta)

        assert blocks.parts() == [
            AIThinkingPart(thinking="first thought", signature="S0"),
            AIThinkingPart(thinking="", redacted="RRR"),
            AIThinkingPart(thinking="second", signature="S3"),
        ]

    def test_a_named_block_cut_before_its_signature_is_not_replayed(self) -> None:
        blocks = ThinkingBlocks()
        for delta in (_delta(0, "kept"), _delta(0, signature="S0"), _delta(2, "cut")):
            blocks.add(delta)

        assert blocks.parts() == [AIThinkingPart(thinking="kept", signature="S0")]
        assert blocks.text == "keptcut"
        assert blocks.last_signature == "S0"

    def test_reasoning_without_blocks_is_one_block(self) -> None:
        blocks = ThinkingBlocks()
        for delta in (_delta(None, "a"), _delta(None, signature="S1"), _delta(None, "b")):
            blocks.add(delta)

        assert not blocks.keyed
        assert blocks.parts() == [AIThinkingPart(thinking="ab", signature="S1")]


class _AnthropicStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events, self._final = events, final

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for event in self._events:
                yield event

        return gen()

    async def get_final_message(self) -> Any:
        return self._final


def _start(index: int, **block: Any) -> Any:
    return SimpleNamespace(
        type="content_block_start", index=index, content_block=SimpleNamespace(**block)
    )


def _block_delta(index: int, **delta: Any) -> Any:
    return SimpleNamespace(type="content_block_delta", index=index, delta=SimpleNamespace(**delta))


def _stop(index: int) -> Any:
    return SimpleNamespace(type="content_block_stop", index=index)


def _interleaved_events() -> list[Any]:
    """Thinking, a call, a redacted block, thinking again, a second call."""
    return [
        _start(0, type="thinking"),
        _block_delta(0, type="thinking_delta", thinking="Weather first."),
        _block_delta(0, type="signature_delta", signature="S0"),
        _stop(0),
        _start(1, type="tool_use", id="toolu_1", name="get_weather"),
        _block_delta(1, type="input_json_delta", partial_json='{"city": "Paris"}'),
        _stop(1),
        _start(2, type="redacted_thinking", data="RRR"),
        _stop(2),
        _start(3, type="thinking"),
        _block_delta(3, type="thinking_delta", thinking="Now the time."),
        _block_delta(3, type="signature_delta", signature="S3"),
        _stop(3),
        _start(4, type="tool_use", id="toolu_2", name="get_time"),
        _block_delta(4, type="input_json_delta", partial_json='{"city": "Tokyo"}'),
        _stop(4),
    ]


def _anthropic(events: list[Any]) -> AnthropicAIProvider:
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    final = SimpleNamespace(content=[], usage=usage, stop_reason="tool_use", model="claude")
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kw: _AnthropicStream(events, final))
    )
    return provider


class TestAnthropic:
    async def test_each_streamed_delta_names_its_block(self) -> None:
        events = [
            e
            async for e in _anthropic(_interleaved_events()).generate_structured_stream(_CTX)
            if isinstance(e, StreamThinkingDelta)
        ]

        assert [(e.block, e.thinking, e.signature, e.redacted) for e in events] == [
            (0, "Weather first.", None, None),
            (0, "", "S0", None),
            (2, "", None, "RRR"),
            (3, "Now the time.", None, None),
            (3, "", "S3", None),
        ]

    async def test_generate_keeps_every_block(self) -> None:
        response = await _anthropic(_interleaved_events()).generate(_CTX)

        assert response.thinking_parts == [
            AIThinkingPart(thinking="Weather first.", signature="S0"),
            AIThinkingPart(thinking="", redacted="RRR"),
            AIThinkingPart(thinking="Now the time.", signature="S3"),
        ]
        assert response.thinking == "Weather first.Now the time."
        assert response.thinking_signature == "S3"

    def test_each_block_is_rendered_as_its_own(self) -> None:
        message = AIMessage(
            role="assistant",
            content=[
                AIThinkingPart(thinking="a", signature="S0"),
                AIToolCallPart(id="t1", name="get_weather", arguments={}),
                AIThinkingPart(thinking="", redacted="RRR"),
            ],
        )

        [rendered] = build_messages([message])

        assert rendered["content"] == [
            {"type": "thinking", "thinking": "a", "signature": "S0"},
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
            {"type": "redacted_thinking", "data": "RRR"},
        ]

    def test_another_vendors_unsigned_reasoning_is_left_out(self) -> None:
        """A round a fallback receives from a provider whose reasoning carries
        no signature: Anthropic refuses the block, takes the round without it
        (RMK-398, measured 2026-10-03)."""
        message = AIMessage(
            role="assistant",
            content=[
                AIThinkingPart(thinking="I should look it up."),
                AIToolCallPart(id="call_1", name="lookup", arguments={}),
            ],
        )

        [rendered] = build_messages([message])

        assert rendered["content"] == [
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
        ]


class _Scripted(AIProvider):
    """Streams its first round as scripted, then answers."""

    def __init__(self, first: list[StreamEvent]) -> None:
        self._rounds = [first, [StreamTextDelta(text="Done."), StreamDone(finish_reason="stop")]]
        self.calls: list[AIContext] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        raise NotImplementedError

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        self.calls.append(context.model_copy(deep=True))
        for event in self._rounds[len(self.calls) - 1]:
            yield event


def _label(part: Any) -> str:
    if isinstance(part, AIThinkingPart):
        return f"thinking:{part.signature or part.redacted}"
    if isinstance(part, AITextPart):
        return f"text:{part.text}"
    return f"call:{part.id}"


_TOOLS = [
    AITool(name="get_weather", description="d"),
    AITool(name="get_time", description="d"),
]


async def _replayed_round(first: list[StreamEvent]) -> list[Any]:
    provider = _Scripted(first)
    channel = AIChannel(
        "ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"), tool_search=False
    )
    await run_tool_loop(
        channel, AIContext(messages=[AIMessage(role="user", content="go")], tools=_TOOLS)
    )
    return next(m for m in provider.calls[1].messages if m.role == "assistant").content


class TestTheLoopReplaysTheRound:
    async def test_blocks_go_back_where_they_came(self) -> None:
        content = await _replayed_round(
            [
                _delta(0, "Weather first."),
                _delta(0, signature="S0"),
                StreamToolCall(id="c1", name="get_weather", arguments={"city": "Paris"}),
                _delta(2, redacted="RRR"),
                _delta(3, "Now the time."),
                _delta(3, signature="S3"),
                StreamToolCall(id="c2", name="get_time", arguments={"city": "Tokyo"}),
                StreamDone(finish_reason="tool_use"),
            ]
        )

        assert [(type(p).__name__, getattr(p, "id", None)) for p in content] == [
            ("AIThinkingPart", None),
            ("AIToolCallPart", "c1"),
            ("AIThinkingPart", None),
            ("AIThinkingPart", None),
            ("AIToolCallPart", "c2"),
        ]
        thinking = [p for p in content if isinstance(p, AIThinkingPart)]
        assert [(p.signature, p.redacted) for p in thinking] == [
            ("S0", None),
            (None, "RRR"),
            ("S3", None),
        ]

    async def test_text_between_blocks_stays_where_it_came(self) -> None:
        content = await _replayed_round(
            [
                _delta(0, "Weather first."),
                _delta(0, signature="S0"),
                StreamTextDelta(text="Looking it up."),
                StreamToolCall(id="c1", name="get_weather", arguments={}),
                _delta(2, "Now the time."),
                _delta(2, signature="S2"),
                StreamTextDelta(text="And the time."),
                StreamToolCall(id="c2", name="get_time", arguments={}),
                StreamDone(finish_reason="tool_use"),
            ]
        )

        assert [_label(p) for p in content] == [
            "thinking:S0",
            "text:Looking it up.",
            "call:c1",
            "thinking:S2",
            "text:And the time.",
            "call:c2",
        ]

    async def test_a_call_the_provider_ran_keeps_its_place(self) -> None:
        ran = StreamToolCall(
            id="b1", name="Bash", arguments={"cmd": "ls"}, served=ServedCall(result="a.txt")
        )
        content = await _replayed_round(
            [
                _delta(0, "List first."),
                _delta(0, signature="S0"),
                ran,
                _delta(2, "Now the weather."),
                _delta(2, signature="S2"),
                StreamToolCall(id="c2", name="get_weather", arguments={}),
                StreamDone(finish_reason="tool_use"),
            ]
        )

        assert [_label(p) for p in content] == [
            "thinking:S0",
            "call:b1",
            "thinking:S2",
            "call:c2",
        ]

    async def test_a_block_cut_before_its_signature_is_not_replayed(self) -> None:
        content = await _replayed_round(
            [
                _delta(0, "Weather first."),
                _delta(0, signature="S0"),
                StreamToolCall(id="c1", name="get_weather", arguments={}),
                _delta(2, "Then the ti"),
                StreamDone(finish_reason="max_tokens"),
            ]
        )

        assert [_label(p) for p in content] == ["thinking:S0", "call:c1"]

    async def test_reasoning_without_blocks_goes_first_as_one(self) -> None:
        content = await _replayed_round(
            [
                StreamTextDelta(text="Looking."),
                _delta(None, "a"),
                _delta(None, "b"),
                StreamToolCall(id="c1", name="get_weather", arguments={}),
                StreamDone(finish_reason="tool_calls"),
            ]
        )

        assert content[0] == AIThinkingPart(thinking="ab")
        assert content[1] == AITextPart(text="Looking.")
        assert isinstance(content[2], AIToolCallPart)


def test_a_wrapped_response_hands_each_block_on() -> None:
    response = AIResponse(
        content="",
        thinking_parts=[
            AIThinkingPart(thinking="a", signature="S0"),
            AIThinkingPart(thinking="", redacted="RRR"),
        ],
    )

    deltas = [e for e in response_stream_events(response) if isinstance(e, StreamThinkingDelta)]

    assert [(d.block, d.thinking, d.signature, d.redacted) for d in deltas] == [
        (0, "a", "S0", None),
        (1, "", None, "RRR"),
    ]


async def test_the_reasoning_backend_replays_every_block(streaming: bool) -> None:
    """The backend runs on the AI channel's loop (RMK-396): its rounds replay
    their blocks as that loop's do."""
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="",
                thinking_parts=[
                    AIThinkingPart(thinking="a", signature="S0"),
                    AIThinkingPart(thinking="", redacted="RRR"),
                ],
                tool_calls=[AIToolCall(id="c1", name="get_weather", arguments={})],
            ),
            AIResponse(content="Sunny."),
        ],
    )
    backend = AIProviderReasoningBackend(provider)
    request = ReasoningRequest(
        session=SimpleNamespace(id="s1", room_id="r1", state=VoiceSessionState.ACTIVE),  # type: ignore[arg-type]
        delegation_id="d1",
        transcript=[],
        first=True,
        tools=[{"name": "get_weather", "description": "Weather", "parameters": {}}],
        execute_tool_call=AsyncMock(return_value=ToolCallResult("ok")),
    )

    _ = [output async for output in backend.run(request)]

    replayed = next(m for m in provider.calls[1].messages if m.role == "assistant").content
    assert [_label(p) for p in replayed] == ["thinking:S0", "thinking:RRR", "call:c1"]


async def test_a_response_read_through_generate_replays_its_blocks_first(
    streaming: bool,
) -> None:
    """A response does not say where its blocks came: they go first."""
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="Looking.",
                finish_reason="tool_use",
                thinking_parts=[
                    AIThinkingPart(thinking="a", signature="S0"),
                    AIThinkingPart(thinking="", redacted="RRR"),
                ],
                tool_calls=[AIToolCall(id="c1", name="get_weather", arguments={})],
            ),
            AIResponse(content="Done."),
        ],
    )
    channel = AIChannel(
        "ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"), tool_search=False
    )

    await run_tool_loop(
        channel, AIContext(messages=[AIMessage(role="user", content="go")], tools=_TOOLS)
    )

    replayed = next(m for m in provider.calls[1].messages if m.role == "assistant").content
    assert [_label(p) for p in replayed] == [
        "thinking:S0",
        "thinking:RRR",
        "text:Looking.",
        "call:c1",
    ]


def test_replaying_a_call_the_round_never_made_is_an_error() -> None:
    transcript = RoundTranscript()
    transcript.add_thinking(_delta(0, "a", signature="S0"))

    with pytest.raises(RuntimeError, match="never made"):
        transcript.parts([StreamToolCall(id="ghost", name="x", arguments={})])

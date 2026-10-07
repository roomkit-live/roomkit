"""What a provider hands the tool loop for a call, on every provider (RMK-284, RFC §6.4).

Arguments are a mapping and never an error, every call of a response has its
own id, two calls stay two, and a call the output cap cut is marked partial
and never runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolCall,
    StreamToolCall,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.openai_dialect import ToolCallSlots, message_tool_calls
from roomkit.providers.ai.tool_calls import CallIds, arguments_cut, call_cut, tool_arguments
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.mistral.ai import MistralAIProvider
from roomkit.providers.mistral.config import MistralConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from tests.tool_loop_modes import run_tool_loop

_CTX = AIContext(
    messages=[AIMessage(role="user", content="hi")],
    tools=[AITool(name="now", description="current time")],
)


class TestArguments:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, {}),
            ("", {}),
            ("  ", {}),
            ('{"q": "a"}', {"q": "a"}),
            ({"q": "a"}, {"q": "a"}),
            ("null", {}),
            ("[1, 2]", {"raw": "[1, 2]"}),
            ('{"a": ' + "[" * 100_000, {"raw": '{"a": ' + "[" * 100_000}),
            ('{"q": "ab', {"raw": '{"q": "ab'}),
        ],
    )
    def test_arguments_are_a_mapping_never_an_error(self, raw: Any, expected: Any) -> None:
        assert tool_arguments(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "cut"), [('{"q": "ab', True), ("null", False), ("", False), ('{"q": 1}', False)]
    )
    def test_only_text_that_stops_before_its_json_ends_is_cut(self, raw: str, cut: bool) -> None:
        assert arguments_cut(raw) is cut

    @pytest.mark.parametrize(
        ("finish_reason", "partial"),
        [("length", True), ("MAX_TOKENS", True), ("content_filter", True), ("stop", False)],
    )
    def test_a_call_is_partial_when_the_response_cut_it(
        self, finish_reason: str, partial: bool
    ) -> None:
        assert call_cut('{"q": "ab', finish_reason) is partial
        assert call_cut('{"q": "ab"}', finish_reason) is False


class TestCallIds:
    def test_every_call_of_a_response_gets_its_own_id(self) -> None:
        ids = CallIds()

        handed = [ids("c1", "now"), ids(None, "now"), ids("", "now"), ids("c1", "now")]

        assert handed[0] == "c1"
        assert len(set(handed)) == 4
        assert all(i.startswith("call_now_") for i in handed[1:])

    def test_a_server_id_that_is_not_text_becomes_text(self) -> None:
        assert CallIds()(7, "now") == "7"


class TestStreamedSlots:
    def test_whole_calls_on_one_index_stay_two_calls(self) -> None:
        slots = ToolCallSlots()
        slots.fold(0, "a", "now", '{"tz": "A"}')
        slots.fold(0, "b", "later", '{"tz": "B"}')

        calls = slots.calls("tool_calls")

        assert [(c.id, c.name, c.arguments) for c in calls] == [
            ("a", "now", {"tz": "A"}),
            ("b", "later", {"tz": "B"}),
        ]

    def test_whole_calls_without_ids_on_one_index_stay_two_calls(self) -> None:
        slots = ToolCallSlots()
        slots.fold(0, None, "roll_die", '{"sides": 6}')
        slots.fold(0, None, "roll_die", '{"sides": 6}')

        calls = slots.calls("tool_calls")

        assert [c.arguments for c in calls] == [{"sides": 6}, {"sides": 6}]
        assert calls[0].id != calls[1].id

    def test_the_composition_events_carry_the_calls_final_id(self) -> None:
        slots = ToolCallSlots()
        delta = slots.fold(0, None, "now", '{"tz": "A"}')

        [call] = slots.calls("tool_calls")

        assert delta is not None
        assert delta.id == call.id

    def test_fragments_of_one_call_fold_together(self) -> None:
        slots = ToolCallSlots()
        slots.fold(0, "a", "now", '{"tz": ')
        slots.fold(0, None, None, '"A"}')

        [call] = slots.calls("tool_calls")

        assert (call.id, call.arguments, call.partial) == ("a", {"tz": "A"}, False)

    def test_a_call_cut_by_the_output_cap_is_partial(self) -> None:
        slots = ToolCallSlots()
        slots.fold(0, "a", "now", "{}")
        slots.fold(1, "b", "write", '{"path": "/tmp/a", "content": "hel')

        calls = slots.calls("length")

        assert [c.partial for c in calls] == [False, True]
        assert calls[1].arguments == {"raw": '{"path": "/tmp/a", "content": "hel'}


def _openai(response: Any) -> OpenAIAIProvider:
    provider = OpenAIAIProvider(OpenAIConfig(api_key="sk-test", model="gpt-5.4"))
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )
    return provider


def _openai_response(calls: list[tuple[str | None, str]], finish_reason: str) -> Any:
    tool_calls = [
        SimpleNamespace(id=i, function=SimpleNamespace(name="now", arguments=a)) for i, a in calls
    ]
    message = SimpleNamespace(content=None, tool_calls=tool_calls, refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
        model="m",
    )


class TestOpenAIDialect:
    async def test_a_buffered_response_reads_every_call_the_same_way(self) -> None:
        response = _openai_response(
            [(None, ""), (None, "null"), (None, '{"q": "ab')], finish_reason="length"
        )

        calls = (await _openai(response).generate(_CTX)).tool_calls

        assert [c.arguments for c in calls] == [{}, {}, {"raw": '{"q": "ab'}]
        # Only the last call can be cut: the ones another call followed were
        # closed by it, and run when their arguments read (RMK-438).
        assert [c.partial for c in calls] == [False, False, True]
        assert len({c.id for c in calls}) == 3

    async def test_a_stream_with_null_arguments_is_no_error(self) -> None:
        async def chunks() -> Any:
            delta = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        index=0, id="c0", function=SimpleNamespace(name="now", arguments="null")
                    )
                ],
            )
            yield SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)]
            )
            done = SimpleNamespace(content=None, tool_calls=None)
            yield SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=done, finish_reason="tool_calls")]
            )

        provider = _openai(chunks())

        events = [e async for e in provider.generate_structured_stream(_CTX)]

        calls = [e for e in events if isinstance(e, StreamToolCall)]
        assert [c.arguments for c in calls] == [{}]

    async def test_a_stream_keeps_calls_apart_and_marks_the_cut_one(self) -> None:
        def chunk(index: int, call_id: str | None, name: str | None, args: str) -> Any:
            call = SimpleNamespace(
                index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args)
            )
            delta = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)]
            )

        async def chunks() -> Any:
            yield chunk(0, None, "now", '{"tz": "A"}')
            yield chunk(0, None, "now", '{"tz": "B"}')
            yield chunk(1, None, "write", '{"path": "/tmp/a", "content": "hel')
            done = SimpleNamespace(content=None, tool_calls=None)
            yield SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=done, finish_reason="length")]
            )

        events = [e async for e in _openai(chunks()).generate_structured_stream(_CTX)]

        calls = [e for e in events if isinstance(e, StreamToolCall)]
        assert [(c.name, c.partial) for c in calls] == [
            ("now", False),
            ("now", False),
            ("write", True),
        ]
        assert len({c.id for c in calls}) == 3


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


async def test_an_anthropic_tool_use_cut_by_max_tokens_is_partial() -> None:
    fragment = '{"path": "/tmp/a", "content": "hel'
    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="write_file"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json=fragment),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
    ]
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    final = SimpleNamespace(content=[], usage=usage, stop_reason="max_tokens", model="claude")
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kw: _AnthropicStream(events, final))
    )

    [call] = (await provider.generate(_CTX)).tool_calls

    assert call.partial is True
    assert call.arguments == {"raw": fragment}


async def test_an_anthropic_block_the_stream_never_closed_is_partial_when_cut() -> None:
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    block = SimpleNamespace(type="tool_use", id="toolu_9", name="write_file", input={"path": "/a"})
    final = SimpleNamespace(content=[block], usage=usage, stop_reason="max_tokens", model="claude")
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = SimpleNamespace(
        # The stream opened, as every Anthropic stream does, and was cut
        # before the block's stop.
        messages=SimpleNamespace(
            stream=lambda **kw: _AnthropicStream(
                [SimpleNamespace(type="message_start", message=final)], final
            )
        )
    )

    [call] = (await provider.generate(_CTX)).tool_calls

    assert (call.id, call.partial) == ("toolu_9", True)


class TestMistral:
    async def test_id_less_whole_calls_on_index_zero_stay_two(self) -> None:
        def event(args: Any) -> Any:
            call = SimpleNamespace(
                index=0, id="null", function=SimpleNamespace(name="roll_die", arguments=args)
            )
            delta = SimpleNamespace(content=None, tool_calls=[call])
            choice = SimpleNamespace(delta=delta, finish_reason=None)
            return SimpleNamespace(data=SimpleNamespace(choices=[choice], usage=None))

        async def stream() -> Any:
            yield event('{"sides": 6}')
            yield event({"sides": 20})  # the SDK types arguments Dict | str

        provider = MistralAIProvider(MistralConfig(api_key="k", model="mistral-large-latest"))
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(stream_async=AsyncMock(return_value=stream()))
        )

        events = [e async for e in provider.generate_structured_stream(_CTX)]

        calls = [e for e in events if isinstance(e, StreamToolCall)]
        assert [c.arguments for c in calls] == [{"sides": 6}, {"sides": 20}]
        assert len({c.id for c in calls}) == 2
        assert "null" not in {c.id for c in calls}


def test_polargrid_reads_every_buffered_call_the_same_way() -> None:
    # polargrid-sdk's ToolCall requires an id; a server may repeat one.
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call_0", function=SimpleNamespace(name="search", arguments='{"q": 1}')
            ),
            SimpleNamespace(
                id="call_0", function=SimpleNamespace(name="search", arguments="null")
            ),
            SimpleNamespace(
                id="call_0", function=SimpleNamespace(name="search", arguments='{"q": ')
            ),
        ]
    )

    # PolarGrid reads a response's calls through the chat wire's shared reader.
    calls = message_tool_calls(message, "length")

    assert [c.arguments for c in calls] == [{"q": 1}, {}, {"raw": '{"q": '}]
    assert [c.partial for c in calls] == [False, False, True]
    assert len({c.id for c in calls}) == 3


def _gemini_part(sig: bytes | None = None) -> Any:
    call = SimpleNamespace(name="roll_die", args={"sides": 6}, id=None)
    return SimpleNamespace(text=None, thought=False, function_call=call, thought_signature=sig)


async def _gemini_calls(chunks: list[list[Any]]) -> list[StreamToolCall]:
    async def stream() -> Any:
        for parts in chunks:
            candidate = SimpleNamespace(finish_reason=None, content=SimpleNamespace(parts=parts))
            yield SimpleNamespace(
                usage_metadata=None, prompt_feedback=None, candidates=[candidate]
            )

    async def generate(**kwargs: Any) -> Any:
        return stream()

    provider = GeminiAIProvider(GeminiConfig(api_key="k"))
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate))
    )
    events = [e async for e in provider.generate_structured_stream(_CTX)]
    return [e for e in events if isinstance(e, StreamToolCall)]


class TestGeminiCalls:
    async def test_two_identical_calls_of_one_chunk_stay_two(self) -> None:
        calls = await _gemini_calls([[_gemini_part(b"s"), _gemini_part()]])

        assert len(calls) == 2
        assert calls[0].id != calls[1].id

    async def test_a_call_re_emitted_in_a_later_chunk_stays_one(self) -> None:
        calls = await _gemini_calls([[_gemini_part()], [_gemini_part(b"s")]])

        [call] = calls
        assert call.metadata.get("thought_signature")


async def test_a_partial_call_never_runs_and_the_model_reads_why(streaming: bool) -> None:
    handler = AsyncMock(return_value="ok")
    cut = AIToolCall(id="c1", name="now", arguments={"raw": '{"tz": "Eu'}, partial=True)
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="", finish_reason="length", tool_calls=[cut]),
            AIResponse(content="Retrying later."),
        ],
    )
    channel = AIChannel("ai1", provider=provider, tool_handler=handler, tool_search=False)
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=_CTX.tools)

    run = await run_tool_loop(channel, context)

    handler.assert_not_awaited()
    [call] = run.calls
    assert call.failed is True
    assert "cut" in (call.error or "").lower()

"""One rule for a call that cannot run, and the call edges every provider shares
(RFC §6.4, RMK-309, RMK-301).

A call whose arguments do not read as an object is partial and never runs,
whatever the provider and whatever stop reason the response gave; the model
reads whether the response cut it or it was written unreadable. Streamed calls
stay apart, their composition events carry the id they end with, and a Gemini
re-emission folds into its first copy whichever of the two carries an id.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.providers.ai import FinishReason, tool_arguments
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCall,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.ai.openai_dialect import ToolCallSlots, message_tool_calls
from roomkit.providers.ai.tool_calls import (
    call_cut,
    call_garbled,
    call_partial,
    partial_call_error,
    unreadable_arguments,
)
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.gemini.ai import GeminiAIProvider
from roomkit.providers.gemini.config import GeminiConfig
from roomkit.providers.ollama.config import OllamaConfig
from roomkit.providers.openai.ai import OpenAIAIProvider
from roomkit.providers.openai.config import OpenAIConfig
from tests.tool_loop_modes import run_tool_loop

_CTX = AIContext(messages=[AIMessage(role="user", content="hi")])
_FRAGMENT = '{"path": "/tmp/a", "content": "hel'


class TestTheRule:
    @pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e900", "-1e900"])
    def test_nonfinite_text_arguments_preserve_the_wire_input_and_are_classified_partial(
        self, number: str
    ) -> None:
        raw = f'{{"record": [{{"amount": {number}}}]}}'
        assert tool_arguments(raw) == {"raw": raw}
        assert unreadable_arguments(raw)
        assert call_partial(raw, "stop") and call_garbled(raw, "stop")
        assert call_cut(raw, "length")

    def test_finite_numbers_and_number_names_in_strings_remain_readable(self) -> None:
        raw = '{"sample": [1e300, "NaN", "-Infinity"]}'
        assert tool_arguments(raw) == {"sample": [1e300, "NaN", "-Infinity"]}
        assert not call_partial(raw, "stop")
        supplied = {"sdk": float("inf")}
        assert tool_arguments(supplied) is supplied

    def test_arguments_that_do_not_read_as_an_object(self) -> None:
        unreadable = [_FRAGMENT, "[1, 2]", '"text"', "7"]
        readable = ["", "  ", None, "null", '{"a": 1}', {"a": 1}]
        assert all(unreadable_arguments(raw) for raw in unreadable)
        assert not any(unreadable_arguments(raw) for raw in readable)

    def test_a_stream_without_a_stop_reason_cut_its_call(self) -> None:
        assert call_cut(_FRAGMENT, None) and call_cut(_FRAGMENT, "length")
        assert not call_garbled(_FRAGMENT, None)

    @pytest.mark.parametrize(
        "finish",
        [
            FinishReason.LENGTH,
            FinishReason.CONTENT_FILTER,
            FinishReason.REFUSAL,
            FinishReason.ERROR,
        ],
    )
    def test_every_ending_that_stops_a_call_cuts_it(self, finish: FinishReason) -> None:
        """A generation's ``error`` among them (RMK-438)."""
        assert call_partial("", finish) and call_cut(_FRAGMENT, finish)
        assert not call_garbled(_FRAGMENT, finish)

    def test_a_call_another_followed_is_not_cut(self) -> None:
        """Closed by the call after it: whole, it runs; unreadable, the model
        wrote it so (RMK-438)."""
        assert not call_partial("", "length", last=False)
        assert call_garbled(_FRAGMENT, "length", last=False)
        assert not call_cut(_FRAGMENT, "length", last=False)

    def test_unreadable_arguments_on_an_ordinary_stop_were_written_so(self) -> None:
        assert call_garbled("[1, 2]", "tool_calls") and call_garbled(_FRAGMENT, "stop")
        assert not call_cut("[1, 2]", "tool_calls")

    def test_the_model_reads_which(self) -> None:
        assert partial_call_error("x", garbled=False)["error"] == "Tool call cut off"
        assert partial_call_error("x", garbled=True)["error"] == "Tool call arguments unreadable"


def _chunk(index: int, call_id: str | None, name: str | None, args: str) -> Any:
    call = SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args)
    )
    delta = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def _openai(chunks: list[Any], finish_reason: str | None) -> OpenAIAIProvider:
    async def stream() -> Any:
        for chunk in chunks:
            yield chunk
        if finish_reason is not None:
            done = SimpleNamespace(content=None, tool_calls=None)
            yield SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=done, finish_reason=finish_reason)]
            )

    provider = OpenAIAIProvider(OpenAIConfig(api_key="sk-test", model="gpt-5.4"))
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream())))
    )
    return provider


async def _streamed_calls(provider: Any) -> list[StreamToolCall]:
    events = [e async for e in provider.generate_structured_stream(_CTX)]
    return [e for e in events if isinstance(e, StreamToolCall)]


class TestOpenAIStream:
    async def test_a_stream_that_stops_without_a_reason_mid_arguments_cut_its_call(self) -> None:
        provider = _openai([_chunk(0, "c1", "write", _FRAGMENT)], finish_reason=None)

        [call] = await _streamed_calls(provider)

        assert (call.partial, call.garbled) == (True, False)
        assert call.arguments == {"raw": _FRAGMENT}

    async def test_an_array_of_arguments_on_an_ordinary_stop_was_written_unreadable(
        self,
    ) -> None:
        provider = _openai([_chunk(0, "c1", "write", "[1, 2]")], finish_reason="tool_calls")

        [call] = await _streamed_calls(provider)

        assert (call.partial, call.garbled) == (True, True)


def _fold(*fragments: tuple[int, str | None, str | None, str]) -> tuple[list, list]:
    slots = ToolCallSlots()
    deltas = [slots.fold(*fragment) for fragment in fragments]
    return [d for d in deltas if d is not None], slots.calls("tool_calls")


class TestOnlyTheLastCallIsCut:
    def test_an_unreadable_call_another_followed_was_written_so(self) -> None:
        """Under a cut response, the first call was closed by the second: its
        unreadable arguments are the model's, not the cut's (RMK-438)."""
        slots = ToolCallSlots()
        slots.fold(0, "c1", "lookup", '{"q": "pa')
        slots.fold(1, "c2", "now", "{}")

        first, last = slots.calls("length")

        assert (first.partial, first.garbled) == (True, True)
        assert (last.partial, last.garbled) == (False, False)

    def test_polargrid_reads_the_last_call_among_those_with_a_function(self) -> None:
        """A call entry with no function is no call: the cut one before it is
        still the response's last (RMK-438)."""
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(id="c1", function=SimpleNamespace(name="lookup", arguments="")),
                SimpleNamespace(id="c2", function=None),
            ]
        )

        # PolarGrid reads a response's calls through the chat wire's shared reader.
        [call] = message_tool_calls(message, "length")

        assert (call.partial, call.garbled) == (True, False)


class TestSplitting:
    def test_a_call_without_arguments_then_another_stay_two(self) -> None:
        _, calls = _fold((0, None, "now", ""), (0, None, "later", '{"tz": "B"}'))

        assert [(c.name, c.arguments) for c in calls] == [("now", {}), ("later", {"tz": "B"})]

    def test_name_only_first_fragments_keep_each_call_its_arguments(self) -> None:
        _, calls = _fold(
            (0, None, "now", ""),
            (0, None, None, '{"tz": "A"}'),
            (0, None, "later", ""),
            (0, None, None, '{"tz": "B"}'),
        )

        assert [(c.name, c.arguments) for c in calls] == [
            ("now", {"tz": "A"}),
            ("later", {"tz": "B"}),
        ]

    def test_a_name_repeated_on_every_fragment_and_trailing_blanks_make_one_call(self) -> None:
        _, calls = _fold((0, None, "now", '{"a":'), (0, None, "now", "1}"), (0, None, "now", " "))

        assert [(c.name, c.arguments, c.partial) for c in calls] == [("now", {"a": 1}, False)]

    def test_the_same_tool_called_twice_with_name_only_first_fragments_stays_two(
        self,
    ) -> None:
        _, calls = _fold(
            (0, None, "roll", ""),
            (0, None, None, '{"d": 6}'),
            (0, None, "roll", ""),
            (0, None, None, '{"d": 20}'),
        )

        assert [(c.name, c.arguments, c.partial) for c in calls] == [
            ("roll", {"d": 6}, False),
            ("roll", {"d": 20}, False),
        ]

    def test_a_call_opened_by_a_new_id_takes_the_name_that_follows(self) -> None:
        _, calls = _fold((0, "c1", "f", "{}"), (0, "c2", None, ""), (0, None, "g", "{}"))

        assert [(c.id, c.name, c.arguments) for c in calls] == [("c1", "f", {}), ("c2", "g", {})]

    def test_the_same_tool_called_twice_whole_stays_two(self) -> None:
        _, calls = _fold((0, None, "roll", '{"d": 6}'), (0, None, "roll", '{"d": 20}'))

        assert [c.arguments for c in calls] == [{"d": 6}, {"d": 20}]


class TestCompositionIds:
    def test_two_calls_sharing_a_server_id_announce_the_ids_they_end_with(self) -> None:
        deltas, calls = _fold((0, "call_0", "now", "{}"), (1, "call_0", "later", "{}"))

        assert [d.id for d in deltas] == [c.id for c in calls]
        assert len({c.id for c in calls}) == 2

    def test_an_id_that_arrives_after_the_first_fragment_does_not_rename_the_call(self) -> None:
        deltas, calls = _fold((0, None, "now", '{"a"'), (0, "srv_1", None, ":1}"))

        [call] = calls
        assert {d.id for d in deltas} == {call.id}
        assert call.arguments == {"a": 1}


def _gemini_part(call_id: str | None, sig: bytes | None = None, sides: int = 6) -> Any:
    call = SimpleNamespace(name="roll_die", args={"sides": sides}, id=call_id)
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
    return await _streamed_calls(provider)


class TestGeminiReEmission:
    async def test_a_re_emission_without_the_id_folds_into_the_signed_copy(self) -> None:
        [call] = await _gemini_calls([[_gemini_part("f1", b"s")], [_gemini_part(None)]])

        assert call.metadata.get("thought_signature")

    async def test_a_re_emission_with_the_id_folds_into_the_id_less_copy(self) -> None:
        [call] = await _gemini_calls([[_gemini_part(None)], [_gemini_part("f1", b"s")]])

        assert call.metadata.get("thought_signature")

    async def test_two_calls_with_different_ids_stay_two(self) -> None:
        calls = await _gemini_calls([[_gemini_part("f1", b"s")], [_gemini_part("f2")]])

        assert len(calls) == 2


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


def _anthropic_block(index: int, server_id: str, args: str) -> list[Any]:
    return [
        SimpleNamespace(
            type="content_block_start",
            index=index,
            content_block=SimpleNamespace(type="tool_use", id=server_id, name="now"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(type="input_json_delta", partial_json=args),
        ),
        SimpleNamespace(type="content_block_stop", index=index),
    ]


async def test_anthropic_calls_sharing_a_server_id_get_their_own() -> None:
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    final = SimpleNamespace(content=[], usage=usage, stop_reason="tool_use", model="claude")
    events = [*_anthropic_block(0, "toolu_1", "{}"), *_anthropic_block(1, "toolu_1", "{}")]
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kw: _AnthropicStream(events, final))
    )

    items = [e async for e in provider.generate_structured_stream(_CTX)]

    calls = [e for e in items if isinstance(e, StreamToolCall)]
    deltas = [e for e in items if isinstance(e, StreamToolCallDelta)]
    assert len({c.id for c in calls}) == 2
    assert {d.id for d in deltas} == {c.id for c in calls}


async def test_an_anthropic_block_the_stream_never_closed_keeps_its_announced_id() -> None:
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    block = SimpleNamespace(type="tool_use", id="toolu_1", name="now", input={"path": "/a"})
    final = SimpleNamespace(content=[block], usage=usage, stop_reason="max_tokens", model="claude")
    events = _anthropic_block(0, "toolu_1", '{"path": "/a", "content": "hel')[:2]
    provider = AnthropicAIProvider(AnthropicConfig(api_key="k", model="claude-sonnet-5-5"))
    provider._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kw: _AnthropicStream(events, final))
    )

    items = [e async for e in provider.generate_structured_stream(_CTX)]

    [call] = [e for e in items if isinstance(e, StreamToolCall)]
    assert {e.id for e in items if isinstance(e, StreamToolCallDelta)} == {call.id}
    assert (call.partial, call.garbled) == (True, False)
    assert call.arguments == {"raw": '{"path": "/a", "content": "hel'}


def _ollama() -> Any:
    module = MagicMock()
    with patch.dict(sys.modules, {"ollama": module}):
        from roomkit.providers.ollama.ai import OllamaAIProvider

        return OllamaAIProvider(OllamaConfig(host="http://localhost:11434", model="qwen3:8b"))


class TestOllama:
    async def test_calls_of_one_response_never_share_an_id(self) -> None:
        def chunk(done: bool) -> Any:
            call = SimpleNamespace(id="x", function=SimpleNamespace(name="now", arguments={}))
            message = SimpleNamespace(role="assistant", tool_calls=[call])
            return SimpleNamespace(message=message, done=done, done_reason="stop")

        async def stream() -> Any:
            yield chunk(False)
            yield chunk(True)

        provider = _ollama()
        provider._client = SimpleNamespace(chat=AsyncMock(return_value=stream()))

        calls = await _streamed_calls(provider)

        assert len(calls) == 2 and calls[0].id != calls[1].id

    def test_every_reasoning_part_of_a_message_is_replayed(self) -> None:
        message = AIMessage(
            role="assistant",
            content=[
                AIThinkingPart(thinking="first "),
                AIThinkingPart(thinking="second"),
                AITextPart(text="Done."),
            ],
        )

        [rendered] = _ollama()._build_messages([message], None)

        assert rendered["thinking"] == "first second"


async def test_a_call_written_unreadable_never_runs_and_the_model_reads_why(
    streaming: bool,
) -> None:
    handler = AsyncMock(return_value="ok")
    garbled = AIToolCall(
        id="c1", name="now", arguments={"raw": "[1, 2]"}, partial=True, garbled=True
    )
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[garbled]),
            AIResponse(content="Retrying."),
        ],
    )
    channel = AIChannel("ai1", provider=provider, tool_handler=handler, tool_search=False)
    tools = [AITool(name="now", description="d", parameters={"type": "object"})]
    context = AIContext(messages=[AIMessage(role="user", content="go")], tools=tools)

    await run_tool_loop(channel, context)

    handler.assert_not_awaited()
    answer = provider.calls[1].messages[-1].content[0].result
    assert json.loads(answer)["error"] == "Tool call arguments unreadable"

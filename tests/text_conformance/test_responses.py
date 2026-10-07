"""What a provider hands the loop from a response: its calls, whole or not
runnable, its reasoning and its usage (RFC §6.4)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.providers.ai.base import AIContext, AIMessage, AITool
from roomkit.providers.ai.response_schema import ResponseSchemaError
from roomkit.providers.ai.tool_calls import is_malformed_call, is_truncation
from tests.text_conformance.driver import (
    ARGUMENT_TEXT,
    CACHE_USAGE,
    CACHE_WRITE_USAGE,
    CALL_INDEX,
    CALLS_IN_ONE_CHUNK,
    COMPOSITION,
    FILTER_STOP,
    FUNCTIONLESS_CALL,
    MALFORMED_CALL,
    OBJECT_ARGUMENTS_RESPONSE,
    REASONING_USAGE,
    REDACTED_REASONING,
    REPEATED_ID,
    RESPONSE_CALL_WITHOUT_ID,
    SERVER_ID,
    SIGNED_REASONING,
    STREAM_USAGE,
    STREAM_WITHOUT_FINISH,
    THINK_TAGS,
    USAGE_ALONE,
    WRITTEN_UNREADABLE,
    Driver,
)
from tests.text_conformance.scenario import LOOKUP, generation, tool_context
from tests.text_conformance.script import Call, Finish, Reasoning, Script, Usage

NOW = AITool(name="now", description="The current time.")


class TestCalls:
    async def test_a_call_reaches_the_loop_whole(self, driver: Driver, mode: str) -> None:
        script = Script(
            calls=(Call("lookup", '{"q": "paris"}', id="c1", index=0, fragments=3),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.name, call.arguments, call.partial) == ("lookup", {"q": "paris"}, False)
        assert call.id

    async def test_arguments_sent_as_an_object_read_as_their_text_does(
        self, driver: Driver, mode: str
    ) -> None:
        """A server that sends a call's arguments as an object, not as text
        (Mistral's SDK types them ``Dict | str``), hands the loop the same
        call, streamed or not (RMK-484)."""
        if mode == "generate":
            driver.require(OBJECT_ARGUMENTS_RESPONSE)
        script = Script(
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0, as_object=True),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.name, call.arguments, call.partial) == ("lookup", {"q": "a"}, False)

    async def test_two_calls_stay_two_each_with_its_own_id(
        self, driver: Driver, mode: str
    ) -> None:
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0, fragments=2),
                Call("lookup", '{"q": "b"}', id="c2", index=1, fragments=2),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]
        assert len({c.id for c in answer.calls}) == 2

    async def test_a_new_server_id_is_the_calls_own(self, driver: Driver, mode: str) -> None:
        """A provider mints an id only when the server gave none, or gave one
        an earlier call of the response took (RFC §6.4, RMK-510)."""
        driver.require(SERVER_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0),
                Call("lookup", '{"q": "b"}', id="c2", index=1),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [c.id for c in answer.calls] == ["c1", "c2"]

    async def test_a_call_with_no_name_reaches_the_loop(self, driver: Driver, mode: str) -> None:
        """A function with no name and no arguments is a call, streamed or
        not, which the loop refuses for its missing name (RFC §6.4, RMK-510)."""
        script = Script(calls=(Call("", "", id="c1", index=0),), finish="tool")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.name, call.arguments) == ("", {})

    async def test_calls_starting_in_one_chunk_stay_apart(self, driver: Driver) -> None:
        driver.require(CALLS_IN_ONE_CHUNK)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0, fragments=2),
                Call("lookup", '{"q": "b"}', id="c2", index=1, fragments=2),
            ),
            finish="tool",
            calls_in_one_chunk=True,
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]

    async def test_calls_without_ids_get_their_own(self, driver: Driver, mode: str) -> None:
        if mode == "generate":
            driver.require(RESPONSE_CALL_WITHOUT_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', index=0),
                Call("lookup", '{"q": "b"}', index=1),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        ids = [c.id for c in answer.calls]
        assert len(ids) == 2 and all(ids) and len(set(ids)) == 2

    async def test_calls_sharing_a_server_id_get_their_own(
        self, driver: Driver, mode: str
    ) -> None:
        driver.require(REPEATED_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id="dup", index=0),
                Call("lookup", '{"q": "b"}', id="dup", index=1),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [c.arguments for c in answer.calls] == [{"q": "a"}, {"q": "b"}]
        assert len({c.id for c in answer.calls}) == 2

    async def test_calls_without_an_index_stay_apart(self, driver: Driver) -> None:
        driver.require(CALL_INDEX)
        script = Script(
            calls=(Call("lookup", '{"q": "a"}', id="c1"), Call("lookup", '{"q": "b"}', id="c2")),
            finish="tool",
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert [(c.id, c.arguments) for c in answer.calls] == [
            ("c1", {"q": "a"}),
            ("c2", {"q": "b"}),
        ]

    @pytest.mark.parametrize("ids", [("c1", "c2"), ("dup", "dup")], ids=["apart", "shared"])
    async def test_composition_names_each_call_by_the_id_it_ends_with(
        self, driver: Driver, ids: tuple[str, str]
    ) -> None:
        driver.require(COMPOSITION)
        if ids[0] == ids[1]:
            driver.require(REPEATED_ID)
        script = Script(
            calls=(
                Call("lookup", '{"q": "a"}', id=ids[0], index=0, fragments=3),
                Call("lookup", '{"q": "b"}', id=ids[1], index=1, fragments=3),
            ),
            finish="tool",
        )

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        # Each call's composition events, under the id it ends with, spell its
        # arguments: none is announced under another call's id.
        composed: dict[str, str] = {}
        for delta in answer.deltas:
            composed[delta.id] = composed.get(delta.id, "") + delta.arguments_delta
        expected = zip(answer.calls, script.calls, strict=True)
        assert composed == {call.id: written.arguments for call, written in expected}


class TestCallsThatDoNotRun:
    async def test_a_call_the_output_cap_cut_is_partial(self, driver: Driver, mode: str) -> None:
        driver.require(ARGUMENT_TEXT)
        script = Script(calls=(Call("lookup", '{"q": "par', id="c1", index=0),), finish="cut")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, False)

    @pytest.mark.parametrize(
        ("arguments", "partial"),
        [("", True), ("null", False), ('{"q": "a"}', False)],
        ids=["nothing", "null", "whole"],
    )
    async def test_a_cut_call_runs_when_its_arguments_arrived_whole(
        self, driver: Driver, mode: str, arguments: str, partial: bool
    ) -> None:
        """Partial when its arguments do not read, or when nothing arrived
        before the cut: nothing under a cut is no evidence of no arguments
        (RFC §6.4, RMK-398, measured on Anthropic)."""
        driver.require(ARGUMENT_TEXT)
        script = Script(calls=(Call("lookup", arguments, id="c1", index=0),), finish="cut")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (partial, False)

    @pytest.mark.parametrize("finish", ["context", "filtered"])
    @pytest.mark.parametrize("arguments", ["", '{"q": "par'], ids=["nothing", "fragment"])
    async def test_a_call_any_cut_stopped_is_partial(
        self, driver: Driver, mode: str, finish: Finish, arguments: str
    ) -> None:
        """The context window, a content filter or a refusal cuts a call as the
        output cap does, under each provider's word for it: the call does not
        run, and the model reads that it was cut (RFC §6.4)."""
        driver.require(ARGUMENT_TEXT)
        if finish == "filtered":
            driver.require(FILTER_STOP)
        script = Script(calls=(Call("lookup", arguments, id="c1", index=0),), finish=finish)

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, False)

    async def test_a_call_another_followed_runs_though_the_response_was_cut(
        self, driver: Driver, mode: str
    ) -> None:
        """A call another call followed was closed by it: the cut can only
        reach the last one (RFC §6.4)."""
        driver.require(ARGUMENT_TEXT)
        script = Script(
            calls=(
                Call("now", "", id="c1", index=0),
                Call("lookup", '{"q": "par', id="c2", index=1),
            ),
            finish="cut",
        )

        answer = await generation(driver, script, mode, tool_context(NOW, LOOKUP))

        first, last = answer.calls
        assert (first.name, first.arguments, first.partial) == ("now", {}, False)
        assert (last.partial, last.garbled) == (True, False)

    async def test_a_stream_that_stops_without_a_reason_cuts_its_call(
        self, driver: Driver
    ) -> None:
        driver.require(ARGUMENT_TEXT, STREAM_WITHOUT_FINISH)
        script = Script(calls=(Call("lookup", '{"q": "par', id="c1", index=0),), finish="none")

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, False)

    async def test_arguments_written_unreadable_are_garbled(
        self, driver: Driver, mode: str
    ) -> None:
        driver.require(ARGUMENT_TEXT, WRITTEN_UNREADABLE)
        script = Script(calls=(Call("lookup", "[1, 2]", id="c1", index=0),), finish="tool")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        [call] = answer.calls
        assert (call.partial, call.garbled) == (True, True)


class TestEndings:
    @pytest.mark.parametrize("finish", ["cut", "context"])
    async def test_a_response_that_ran_out_of_room_says_so(
        self, driver: Driver, mode: str, finish: Finish
    ) -> None:
        """The output cap and the context window are one ending to the loop
        and to a schema check, under every provider's word (RFC §6.4)."""
        answer = await generation(
            driver, Script(text="Half an ans", finish=finish), mode, tool_context(LOOKUP)
        )

        assert is_truncation(answer.finish_reason), answer.finish_reason

    @pytest.mark.parametrize("finish", ["malformed", "unexpected"])
    async def test_a_call_the_vendor_would_not_hand_over_says_so(
        self, driver: Driver, mode: str, finish: Finish
    ) -> None:
        driver.require(MALFORMED_CALL)
        answer = await generation(driver, Script(finish=finish), mode, tool_context(LOOKUP))

        assert is_malformed_call(answer.finish_reason), answer.finish_reason


class TestReasoningReceived:
    async def test_reasoning_reaches_the_loop(self, driver: Driver, mode: str) -> None:
        # Signed, for a wire that drops an unsigned block; a wire without
        # signatures does not write it.
        script = Script(reasoning=(Reasoning("why", signature="S0"),), text="ok")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [p.thinking for p in answer.reasoning] == ["why"]

    async def test_a_signed_block_keeps_its_signature(self, driver: Driver, mode: str) -> None:
        driver.require(SIGNED_REASONING)
        script = Script(
            reasoning=(Reasoning("why", signature="S0"),),
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [(p.thinking, p.signature) for p in answer.reasoning] == [("why", "S0")]

    async def test_a_redacted_block_keeps_its_data(self, driver: Driver, mode: str) -> None:
        driver.require(REDACTED_REASONING)
        script = Script(
            reasoning=(Reasoning(redacted="RRR"),),
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),),
            finish="tool",
        )

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [p.redacted for p in answer.reasoning] == ["RRR"]

    async def test_a_think_block_the_cap_cut_is_reasoning(self, driver: Driver, mode: str) -> None:
        """A ``<think>`` block the output cap cut before its close is
        reasoning, never answer, streamed or not (RMK-484)."""
        driver.require(THINK_TAGS)
        script = Script(text="<think>still weighing the options", finish="cut")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [p.thinking for p in answer.reasoning] == ["still weighing the options"]
        assert answer.text == ""

    async def test_a_think_block_reads_the_same_streamed_or_not(
        self, driver: Driver, mode: str
    ) -> None:
        """The reasoning and the answer of a response with ``<think>`` tags
        are what the stream reads, spaces included, in both modes (RMK-531)."""
        driver.require(THINK_TAGS)
        script = Script(text="<think> weighing it \n</think>\n\nThe answer.  ")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert [p.thinking for p in answer.reasoning] == [" weighing it \n"]
        assert answer.text == "\n\nThe answer.  "


def _usage_mode(driver: Driver, mode: str) -> None:
    if mode == "stream":
        driver.require(STREAM_USAGE)


class TestUsage:
    async def test_input_and_output_reach_the_loop(self, driver: Driver, mode: str) -> None:
        _usage_mode(driver, mode)
        script = Script(text="ok", usage=Usage(input=11, output=7))

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert (answer.usage["input_tokens"], answer.usage["output_tokens"]) == (11, 7)
        # A cache counter the response reported at zero (every wire sends
        # one) is absent, as one it did not report (RFC §6.7).
        assert set(answer.usage) == {"input_tokens", "output_tokens"}

    async def test_usage_on_a_chunk_of_its_own_reaches_the_loop(self, driver: Driver) -> None:
        """Read wherever it arrives: a chunk with no choice counts too
        (RMK-510)."""
        driver.require(STREAM_USAGE, USAGE_ALONE)
        script = Script(text="ok", usage=Usage(input=11, output=7), usage_alone=True)

        answer = await generation(driver, script, "stream", tool_context(LOOKUP))

        assert (answer.usage["input_tokens"], answer.usage["output_tokens"]) == (11, 7)

    async def test_cache_reads_are_counted_apart(self, driver: Driver, mode: str) -> None:
        _usage_mode(driver, mode)
        driver.require(CACHE_USAGE)
        script = Script(text="ok", usage=Usage(input=11, output=7, cache_read=5))

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert answer.usage["input_tokens"] == 11
        assert answer.usage.get("cache_read_input_tokens") == 5

    async def test_cache_writes_are_counted_apart(self, driver: Driver, mode: str) -> None:
        _usage_mode(driver, mode)
        driver.require(CACHE_WRITE_USAGE)
        script = Script(text="ok", usage=Usage(input=11, output=7, cache_write=4))

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert answer.usage["input_tokens"] == 11
        assert answer.usage.get("cache_creation_input_tokens") == 4

    async def test_reasoning_tokens_are_a_detail_of_output(
        self, driver: Driver, mode: str
    ) -> None:
        _usage_mode(driver, mode)
        driver.require(REASONING_USAGE)
        script = Script(text="ok", usage=Usage(input=11, output=7, reasoning=3))

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert answer.usage["output_tokens"] == 7
        assert answer.usage.get("reasoning_tokens") == 3


class _Recorder:
    """Telemetry that keeps the time-to-first-token metric's labels."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def record_metric(self, name: str, value: float, **kw: Any) -> None:
        if name == "roomkit.llm.ttfb_ms":
            self.labels.append(dict(kw.get("attributes") or {}).get("provider", "?"))


class TestModesAgree:
    """What a response tells the loop beside its content, the same through
    ``generate()`` and the stream (RMK-500)."""

    async def test_the_model_reported_is_the_one_that_answered(
        self, driver: Driver, mode: str
    ) -> None:
        """The response's own model, never the one asked for, when the
        response names it, alike on both modes (RMK-500, RMK-510)."""
        script = Script(text="ok", answered_by="served-model")

        answer = await generation(driver, script, mode, tool_context(LOOKUP))

        assert answer.metadata.get("model") == "served-model"

    async def test_time_to_first_token_is_labelled_alike_on_both_modes(
        self, driver: Driver
    ) -> None:
        context = tool_context(LOOKUP)
        generated, streamed = _Recorder(), _Recorder()
        provider = driver.provider(Script(text="ok"))
        provider._telemetry = generated  # type: ignore[attr-defined]
        await provider.generate(context)
        provider = driver.provider(Script(text="ok"))
        provider._telemetry = streamed  # type: ignore[attr-defined]
        async for _ in provider.generate_structured_stream(context):
            pass

        assert generated.labels == streamed.labels != []

    async def test_a_stream_of_calls_alone_records_no_first_token(self, driver: Driver) -> None:
        script = Script(calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),), finish="tool")
        recorder = _Recorder()
        provider = driver.provider(script)
        provider._telemetry = recorder  # type: ignore[attr-defined]
        async for _ in provider.generate_structured_stream(tool_context(LOOKUP)):
            pass

        assert recorder.labels == []

    async def test_text_held_back_to_the_end_records_its_first_token(self, driver: Driver) -> None:
        """A ``<`` may open a think tag: a parser holds it until the stream
        ends, and its flush is the stream's first output (RMK-500)."""
        recorder = _Recorder()
        provider = driver.provider(Script(text="<"))
        provider._telemetry = recorder  # type: ignore[attr-defined]
        async for _ in provider.generate_structured_stream(tool_context(LOOKUP)):
            pass

        assert len(recorder.labels) == 1


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class TestResponseSchema:
    async def test_entries_with_no_function_are_no_tool_round(
        self, driver: Driver, mode: str
    ) -> None:
        """Under a response schema, a round is a step of the loop only when
        calls reach it: entries with no function hand it none, so the answer
        is due and checked, on both modes (RFC §6.7, RMK-510)."""
        driver.require(FUNCTIONLESS_CALL)
        if not driver.provider(Script()).supports_response_schema:
            pytest.skip(f"{driver.label}: the provider takes no response schema")
        script = Script(calls=(Call("x", "", id="c1", index=0, functionless=True),), finish="tool")
        context = AIContext(
            messages=[AIMessage(role="user", content="go")], response_schema=_SCHEMA
        )

        with pytest.raises(ResponseSchemaError) as raised:
            await generation(driver, script, mode, context)

        assert raised.value.reason == "invalid_json"

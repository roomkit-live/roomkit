"""What a provider sends: its tools declared, and earlier rounds replayed as
its wire needs them (RFC §6.4, §6.7)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from roomkit.providers.ai.base import (
    AIImagePart,
    AIMessage,
    AITextPart,
    AIThinkingPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    thinking_parts_of,
)
from roomkit.providers.ai.round_parts import round_parts
from tests.text_conformance.driver import (
    IMAGE_RESULTS,
    REDACTED_REASONING,
    SCHEMA_AS_GIVEN,
    THINK_TAGS,
    Driver,
)
from tests.text_conformance.scenario import LOOKUP, PNG, generation, tool_context
from tests.text_conformance.script import Call, Reasoning, Script


class TestDeclaration:
    async def test_a_tool_without_parameters_is_declared_as_an_empty_object(
        self, driver: Driver, mode: str
    ) -> None:
        bare = AITool(name="now", description="Current time.")
        await generation(driver, Script(text="ok"), mode, tool_context(bare))

        assert driver.declared(driver.requests[0])["now"] == {
            "type": "object",
            "properties": {},
        }

    async def test_a_root_without_a_type_is_declared_an_object(
        self, driver: Driver, mode: str
    ) -> None:
        """Anthropic and OpenAI refuse an untyped root (RMK-398, measured)."""
        untyped = AITool(
            name="lookup_q",
            description="d",
            parameters={"properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        await generation(driver, Script(text="ok"), mode, tool_context(untyped))

        declared = driver.declared(driver.requests[0])["lookup_q"]
        assert declared["type"] == "object"
        assert declared["properties"] == {"q": {"type": "string"}}

    async def test_a_schema_is_declared_as_given(self, driver: Driver) -> None:
        driver.require(SCHEMA_AS_GIVEN)
        schema = {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "enum": [1, 2, 3]},
                "place": {"$ref": "#/$defs/place"},
            },
            "$defs": {"place": {"type": "string"}},
        }
        tool = AITool(name="set_level", description="d", parameters=schema)
        await generation(driver, Script(text="ok"), "stream", tool_context(tool))

        assert driver.declared(driver.requests[0])["set_level"] == schema

    async def test_a_name_the_vendor_refuses_fails_before_the_request(
        self, driver: Driver
    ) -> None:
        if not driver.refused_names:
            pytest.skip(f"{driver.label}: the server decides its names, the provider checks none")
        for name in driver.refused_names:
            tool = AITool(name=name, description="d")
            with pytest.raises(ProviderError, match=name) as raised:
                await generation(driver, Script(text="ok"), "stream", tool_context(tool))
            assert raised.value.before_request is True
        assert driver.requests == []

    async def test_a_name_the_vendor_accepts_is_declared(self, driver: Driver) -> None:
        for name in driver.accepted_names:
            await generation(
                driver,
                Script(text="ok"),
                "stream",
                tool_context(AITool(name=name, description="d")),
            )
            assert name in driver.declared(driver.requests[-1])


def _round(*parts: Any) -> AIMessage:
    return AIMessage(role="assistant", content=list(parts))


def _results(*parts: AIToolResultPart) -> AIMessage:
    return AIMessage(role="tool", content=list(parts))


# A round the provider itself produced: Gemini 3 signs its calls, which the
# other wires ignore.
_SIGNED = {"thought_signature": base64.b64encode(b"S0").decode()}
_CALL = AIToolCallPart(id="c1", name="lookup", arguments={"q": "a"}, metadata=_SIGNED)


def _sent_no_reasoning_if_dropped(driver: Driver) -> None:
    """On a wire that replays none, the reasoning is nowhere in the request."""
    if driver.reasoning == "dropped":
        assert "why" not in str(driver.requests[-1])


async def _replay(driver: Driver, *history: AIMessage) -> list[Any]:
    messages = [AIMessage(role="user", content="go"), *history]
    await generation(driver, Script(text="ok"), "stream", tool_context(LOOKUP, messages=messages))
    return driver.replayed(driver.requests[-1])


class TestReplay:
    async def test_a_call_and_its_results_go_back_paired(self, driver: Driver) -> None:
        # Two tools, so a wire that pairs by name pairs something.
        second = AIToolCallPart(id="c2", name="fetch", arguments={"q": "b"}, metadata=_SIGNED)
        items = await _replay(
            driver,
            _round(AITextPart(text="Looking."), _CALL, second),
            _results(
                AIToolResultPart(tool_call_id="c1", name="lookup", result="found"),
                AIToolResultPart(tool_call_id="c2", name="fetch", result="denied", is_error=True),
            ),
        )

        calls = [i for i in items if i[0] == "call"]
        results = [i for i in items if i[0] == "result"]
        assert [c[2] for c in calls] == [{"q": "a"}, {"q": "b"}]
        assert [r[2] for r in results] == ["found", "denied"]
        assert [r[1] for r in results] == [c[1] for c in calls]
        flags = [r[3] for r in results]
        assert flags == ([False, True] if driver.error_flag else [None, None])

    async def test_an_image_result_reaches_the_model(self, driver: Driver) -> None:
        driver.require(IMAGE_RESULTS)
        items = await _replay(
            driver,
            _round(_CALL),
            _results(
                AIToolResultPart(
                    tool_call_id="c1",
                    name="lookup",
                    result=[AITextPart(text="shot"), AIImagePart(url=PNG)],
                )
            ),
        )

        assert ("image",) in items

    async def test_reasoning_goes_back_by_the_wires_convention(self, driver: Driver) -> None:
        items = await _replay(
            driver,
            _round(AIThinkingPart(thinking="why", signature="S0"), _CALL),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        reasoning = [i for i in items if i[0] in ("thinking", "redacted", "field")]
        expected = {
            "blocks": [("thinking", "why", "S0")],
            "call_signature": [],
            "field": [("field", "why")],
            "dropped": [],
        }[driver.reasoning]
        assert reasoning == expected
        _sent_no_reasoning_if_dropped(driver)

    async def test_another_vendors_round_goes_back_as_the_wire_takes_it(
        self, driver: Driver
    ) -> None:
        """A round a fallback receives from the primary: reasoning without a
        signature, a call without one. Anthropic refuses the block and Gemini 3
        the call (RMK-398, measured 2026-10-03)."""
        items = await _replay(
            driver,
            _round(
                AIThinkingPart(thinking="why"),
                AIToolCallPart(id="c1", name="lookup", arguments={"q": "a"}),
            ),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        kinds = {item[0] for item in items}
        if driver.reasoning == "call_signature":
            assert ("text", 'I called lookup({"q": "a"}).') in items
            assert not kinds & {"call", "result", "signature"}
        else:
            assert {"call", "result"} <= kinds
            assert "thinking" not in kinds or driver.reasoning != "blocks"

    async def test_a_redacted_block_goes_back_as_received(self, driver: Driver) -> None:
        driver.require(REDACTED_REASONING)
        items = await _replay(
            driver,
            _round(AIThinkingPart(thinking="", redacted="RRR"), _CALL),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        assert ("redacted", "RRR") in items


_REASONING_KINDS = ("thinking", "redacted", "field", "signature")
# What a received round's reasoning ("why", signed "S0") goes back as.
_GOES_BACK_AS: dict[str, list[tuple[str, ...]]] = {
    "blocks": [("thinking", "why", "S0")],
    "field": [("field", "why")],
    "dropped": [],
}


class TestRoundTrip:
    async def test_a_received_round_goes_back_as_the_wire_needs(self, driver: Driver) -> None:
        """A round received, kept as the loop keeps it, then replayed."""
        script = Script(
            reasoning=(Reasoning("why", signature="S0"),),
            calls=(
                Call("lookup", '{"q": "a"}', id="c1", index=0),
                Call("lookup", '{"q": "b"}', id="c2", index=1),
            ),
            finish="tool",
        )
        response = await driver.provider(script).generate(tool_context(LOOKUP))
        kept = round_parts(thinking_parts_of(response), response.content, response.tool_calls)

        items = await _replay(
            driver,
            AIMessage(role="assistant", content=kept),
            _results(
                *(
                    AIToolResultPart(tool_call_id=c.id, name=c.name, result="found")
                    for c in response.tool_calls
                )
            ),
        )

        reasoning = [i for i in items if i[0] in _REASONING_KINDS]
        if driver.reasoning == "call_signature":
            # Gemini signs the first call only; every call goes back signed.
            assert [i[2] for i in reasoning] == ["S0", "S0"]
        else:
            assert reasoning == _GOES_BACK_AS[driver.reasoning]
        _sent_no_reasoning_if_dropped(driver)

    async def test_a_think_tag_read_from_an_answer_goes_back_as_the_wire_needs(
        self, driver: Driver
    ) -> None:
        """A model's ``<think>`` is read out of its text, and the next request
        carries it only in a field its wire reads (RFC §6.4, RMK-1369)."""
        driver.require(THINK_TAGS)
        script = Script(
            text="<think>why</think>Looking.",
            calls=(Call("lookup", '{"q": "a"}', id="c1", index=0),),
            finish="tool",
        )
        response = await driver.provider(script).generate(tool_context(LOOKUP))
        kept = round_parts(thinking_parts_of(response), response.content, response.tool_calls)

        items = await _replay(
            driver,
            AIMessage(role="assistant", content=kept),
            _results(AIToolResultPart(tool_call_id="c1", name="lookup", result="found")),
        )

        assert ("text", "Looking.") in items
        assert [i for i in items if i[0] in _REASONING_KINDS] == _GOES_BACK_AS[driver.reasoning]
        _sent_no_reasoning_if_dropped(driver)

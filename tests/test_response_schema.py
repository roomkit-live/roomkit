"""A response schema: the portable subset, the AIContext field, the shared contract (RFC §6.7)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from roomkit import ResponseSchemaError
from roomkit.providers.ai import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    MockAIProvider,
    ProviderError,
    StreamDone,
    check_portable_schema,
    schema_mismatch,
)

VERDICT: dict[str, Any] = {
    "title": "Verdict",
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["yes", "no"], "description": "The answer."},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "detail": {
            "type": "object",
            "properties": {"count": {"type": "integer"}, "flag": {"type": "boolean"}},
            "required": ["count", "flag"],
            "additionalProperties": False,
        },
    },
    "required": ["label", "confidence", "reasons", "detail"],
    "additionalProperties": False,
}

ANSWER = json.dumps(
    {"label": "yes", "confidence": 0.9, "reasons": [], "detail": {"count": 1, "flag": True}}
)


def _with(path: list[str], **changes: Any) -> dict[str, Any]:
    """A copy of VERDICT with ``changes`` applied to the subschema at ``path``."""
    schema = copy.deepcopy(VERDICT)
    node = schema
    for key in path:
        node = node[key]
    for key, value in changes.items():
        if value is ...:
            node.pop(key)
        else:
            node[key] = value
    return schema


def _context(**overrides: Any) -> AIContext:
    defaults: dict[str, Any] = {"messages": [AIMessage(role="user", content="Is it?")]}
    defaults.update(overrides)
    return AIContext(**defaults)


class TestPortableSubset:
    def test_accepts_every_portable_keyword(self) -> None:
        check_portable_schema(VERDICT)

    def test_accepts_an_object_without_properties(self) -> None:
        check_portable_schema(
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        )

    @pytest.mark.parametrize(
        ("schema", "where"),
        [
            ({"type": "array", "items": {"type": "string"}}, "root type"),
            (_with(["properties", "label"], type=["string", "null"]), "$.label"),
            (_with(["properties", "confidence"], minimum=0), "$.confidence"),
            (_with([], additionalProperties=...), "$"),
            (_with([], additionalProperties=True), "$"),
            (_with([], required=["label", "confidence", "reasons"]), "$"),
            (_with([], required=["label", "label", "confidence", "reasons"]), "$"),
            (_with(["properties", "reasons"], items=...), "$.reasons"),
            (_with(["properties", "confidence"], enum=["1"]), "$.confidence"),
            (_with(["properties", "label"], enum=[]), "$.label"),
            (_with(["properties", "label"], enum=["yes", "yes"]), "$.label"),
            (_with(["properties", "label"], type=..., anyOf=[{"type": "string"}]), "$.label"),
            (_with(["properties", "detail"], type=..., **{"$ref": "#/$defs/x"}), "$.detail"),
            (_with(["properties", "label"], description=3), "$.label"),
            (
                _with(["properties", "detail", "properties", "count"], type="null"),
                "$.detail.count",
            ),
        ],
    )
    def test_refuses_what_one_provider_or_another_breaks_on(
        self, schema: dict[str, Any], where: str
    ) -> None:
        with pytest.raises(ValueError, match=r"\$|root") as exc:
            check_portable_schema(schema)

        assert where in str(exc.value) or where == "root type"

    def test_refuses_property_names_that_are_not_strings(self) -> None:
        schema = _with([], properties={1: {"type": "string"}}, required=[])

        with pytest.raises(ValueError, match="keyed by name"):
            check_portable_schema(schema)


def _answer(**changes: Any) -> dict[str, Any]:
    document = json.loads(ANSWER)
    document.update(changes)
    return document


class TestSchemaMismatch:
    def test_a_fitting_document_has_none(self) -> None:
        assert schema_mismatch(VERDICT, json.loads(ANSWER)) is None

    def test_a_whole_float_counts_as_an_integer(self) -> None:
        assert schema_mismatch(VERDICT, _answer(detail={"count": 2.0, "flag": False})) is None

    @pytest.mark.parametrize(
        ("document", "where"),
        [
            ([], "$: expected an object"),
            ({"label": "yes"}, "missing"),
            (_answer(extra=1), "unexpected ['extra']"),
            (_answer(label="maybe"), "$.label"),
            (_answer(label=1), "$.label: expected string"),
            (_answer(confidence=True), "$.confidence: expected number"),
            (_answer(reasons=["ok", 3]), "$.reasons[1]"),
            (_answer(reasons="ok"), "$.reasons: expected an array"),
            (_answer(detail={"count": 1.5, "flag": True}), "$.detail.count"),
            (_answer(detail={"count": 1, "flag": "yes"}), "$.detail.flag"),
        ],
    )
    def test_names_where_a_document_departs(self, document: Any, where: str) -> None:
        found = schema_mismatch(VERDICT, document)

        assert found is not None
        assert where in found


class TestAIContextField:
    def test_defaults_to_none(self) -> None:
        assert _context().response_schema is None

    def test_keeps_a_portable_schema(self) -> None:
        assert _context(response_schema=VERDICT).response_schema == VERDICT

    def test_refuses_a_non_portable_schema_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="additionalProperties"):
            _context(response_schema=_with([], additionalProperties=True))

    def test_refuses_it_on_assignment_as_a_hook_would_set_it(self) -> None:
        context = _context()

        with pytest.raises(ValidationError, match="items"):
            context.response_schema = _with(["properties", "reasons"], items=...)

    def test_refuses_it_through_model_copy_which_skips_validation(self) -> None:
        with pytest.raises(ValueError, match="enum"):
            _context().model_copy(
                update={"response_schema": _with(["properties", "label"], enum=[])}
            )

    def test_model_copy_can_clear_it(self) -> None:
        copied = _context(response_schema=VERDICT).model_copy(update={"response_schema": None})

        assert copied.response_schema is None


class TestMockHonoursTheContract:
    async def test_generate_returns_the_json_document(self) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True)

        response = await provider.generate(_context(response_schema=VERDICT))

        assert json.loads(response.content)["label"] == "yes"
        assert provider.calls[0].response_schema == VERDICT

    async def test_a_provider_without_support_refuses_before_sending(self) -> None:
        provider = MockAIProvider([ANSWER])

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=VERDICT))

        assert exc.value.reason == "unsupported"
        assert exc.value.before_request is True
        assert provider.calls == []

    async def test_tools_in_the_same_turn_are_refused(self) -> None:
        provider = MockAIProvider([ANSWER], response_schema=True)
        tool = AITool(name="lookup", description="Look it up")

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=VERDICT, tools=[tool]))

        assert exc.value.reason == "unsupported"
        assert "tools" in str(exc.value)

    async def test_streaming_carries_the_schema_and_checks_the_answer(self) -> None:
        provider = MockAIProvider([ANSWER, "Sure."], response_schema=True, streaming=True)
        context = _context(response_schema=VERDICT)

        streamed = [event async for event in provider.generate_structured_stream(context)]
        with pytest.raises(ResponseSchemaError) as exc:
            async for _ in provider.generate_stream(context):
                pass

        assert isinstance(streamed[-1], StreamDone)
        assert exc.value.reason == "invalid_json"

    @pytest.mark.parametrize(
        ("scripted", "reason"),
        [
            (AIResponse(content="", finish_reason="refusal"), "refusal"),
            (AIResponse(content='{"label": "y', finish_reason="length"), "truncated"),
            (AIResponse(content='{"label": "y', finish_reason="model_length"), "truncated"),
            (AIResponse(content="Sure, the answer is yes.", finish_reason="stop"), "invalid_json"),
            (AIResponse(content='{"verdict": "yes"}', finish_reason="stop"), "invalid_json"),
        ],
    )
    async def test_an_answer_without_its_document_raises(
        self, scripted: AIResponse, reason: str
    ) -> None:
        provider = MockAIProvider(ai_responses=[scripted], response_schema=True)

        with pytest.raises(ResponseSchemaError) as exc:
            await provider.generate(_context(response_schema=VERDICT))

        assert exc.value.reason == reason

    async def test_prose_without_a_schema_is_left_alone(self) -> None:
        provider = MockAIProvider(["Sure, the answer is yes."], response_schema=True)

        response = await provider.generate(_context())

        assert response.content == "Sure, the answer is yes."

    def test_the_error_is_a_provider_error_never_retried(self) -> None:
        error = ResponseSchemaError("x", reason="refusal", provider="mock")

        assert isinstance(error, ProviderError)
        assert error.retryable is False
        assert error.provider == "mock"
        # Only an unsupported schema is refused before any request.
        assert error.before_request is False

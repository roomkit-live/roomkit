"""Public extraction preserves definitions and validates raw schemas."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tests.test_tool_protocol import FakeTool

from roomkit.providers.ai import AITool
from roomkit.tools import extract_tools


def test_a_minimal_schema_supplies_the_extraction_defaults() -> None:
    definitions, handler = extract_tools([{"name": "read"}])
    assert [tool.model_dump() for tool in definitions] == [
        {
            "name": "read",
            "description": "",
            "parameters": {},
            "tags": [],
            "changes_state": False,
            "defer_loading": False,
        }
    ]
    assert handler is None


def test_typed_definitions_keep_fields_that_raw_schemas_do_not_extract() -> None:
    class ToolWithExtraFields(FakeTool):
        @property
        def definition(self) -> dict[str, Any]:
            return {**raw, "name": "local"}

    typed = AITool(name="typed", description="typed", defer_loading=True, tags=["typed"])
    raw: dict[str, Any] = {
        "name": "raw",
        "parameters": {"nested": {"value": 1}},
        "tags": ["météo", "forecast"],
        "changes_state": True,
        "defer_loading": True,
        "ignored": "metadata",
    }
    definitions, handler = extract_tools([typed, raw, ToolWithExtraFields("local", "ok")])

    assert [tool.name for tool in definitions] == ["typed", "raw", "local"]
    assert definitions[0] is typed
    assert definitions[0].defer_loading is True
    assert definitions[1].model_dump() == {
        "name": "raw",
        "description": "",
        "parameters": {"nested": {"value": 1}},
        "tags": ["météo", "forecast"],
        "changes_state": True,
        "defer_loading": False,
    }
    assert raw["defer_loading"] is True
    assert definitions[2].model_dump() == {**definitions[1].model_dump(), "name": "local"}
    assert handler is not None


def test_only_typed_definitions_skip_revalidation() -> None:
    typed = AITool(name="typed", description="typed")
    typed.name = "invalid name"

    class InvalidTool(FakeTool):
        @property
        def definition(self) -> dict[str, Any]:
            return typed.model_dump()

    definitions, handler = extract_tools([typed])
    assert definitions[0] is typed
    assert handler is None
    for raw in (typed.model_dump(), InvalidTool("invalid", "ok")):
        with pytest.raises(ValidationError):
            extract_tools([raw])


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (None, []),
        (False, []),
        (0, []),
        ("", []),
        ([], []),
        ("ab", ["a", "b"]),
        ({"z": 1, "a": 2}, ["z", "a"]),
    ],
)
def test_raw_tags_follow_the_extractor_iterable_contract(tags: Any, expected: list[str]) -> None:
    definitions, handler = extract_tools([{"name": "read", "tags": tags}])
    assert definitions[0].tags == expected
    assert handler is None


def test_a_raw_schema_requires_a_name() -> None:
    with pytest.raises(KeyError, match="name"):
        extract_tools([{}])


@pytest.mark.parametrize("tags", [1, True])
def test_truthy_non_iterable_tags_are_rejected(tags: Any) -> None:
    with pytest.raises(TypeError):
        extract_tools([{"name": "read", "tags": tags}])


async def test_one_handler_is_returned_directly_with_its_decline_unchanged() -> None:
    tool = FakeTool("only", "ok")
    _, handler = extract_tools([tool])
    assert handler == tool.handler
    assert await handler("unknown", {}) == '{"error": "Unknown tool: unknown"}'


async def test_duplicate_definitions_keep_order_and_the_first_handler_wins() -> None:
    definitions, handler = extract_tools([FakeTool("same", "first"), FakeTool("same", "second")])
    assert [tool.name for tool in definitions] == ["same", "same"]
    assert handler is not None
    assert await handler("same", {}) == "first"

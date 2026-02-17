"""Tests for compose_tool_handlers."""

from __future__ import annotations

import json

import pytest

from roomkit.tools.compose import compose_tool_handlers


async def _weather_handler(name: str, arguments: dict) -> str:
    if name == "get_weather":
        city = arguments.get("city", "unknown")
        return json.dumps({"temp": 20, "city": city})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def _math_handler(name: str, arguments: dict) -> str:
    if name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return json.dumps({"result": a + b})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def _fallback_handler(name: str, arguments: dict) -> str:
    return json.dumps({"error": f"Unknown tool: {name}"})


async def test_first_handler_match() -> None:
    composed = compose_tool_handlers(_weather_handler, _math_handler)
    result = await composed("get_weather", {"city": "Montreal"})
    parsed = json.loads(result)
    assert parsed == {"temp": 20, "city": "Montreal"}


async def test_second_handler_match() -> None:
    composed = compose_tool_handlers(_weather_handler, _math_handler)
    result = await composed("add", {"a": 3, "b": 4})
    parsed = json.loads(result)
    assert parsed == {"result": 7}


async def test_three_handlers() -> None:
    async def _greet(name: str, arguments: dict) -> str:
        if name == "greet":
            return f"Hello, {arguments.get('who', 'world')}!"
        return json.dumps({"error": f"Unknown tool: {name}"})

    composed = compose_tool_handlers(_greet, _weather_handler, _math_handler)
    # First handler
    assert await composed("greet", {"who": "Alice"}) == "Hello, Alice!"
    # Second handler
    result = json.loads(await composed("get_weather", {"city": "Paris"}))
    assert result["city"] == "Paris"
    # Third handler
    result = json.loads(await composed("add", {"a": 1, "b": 2}))
    assert result["result"] == 3


async def test_no_match_passthrough() -> None:
    composed = compose_tool_handlers(_weather_handler, _fallback_handler)
    result = await composed("unknown_tool", {})
    parsed = json.loads(result)
    assert parsed == {"error": "Unknown tool: unknown_tool"}


async def test_non_json_result_not_treated_as_unknown() -> None:
    async def _plain_handler(name: str, arguments: dict) -> str:
        return "plain text result"

    composed = compose_tool_handlers(_plain_handler, _math_handler)
    # Non-JSON result is not an unknown-tool error, so it should be returned
    result = await composed("anything", {})
    assert result == "plain text result"


async def test_error_without_unknown_tool_prefix_not_skipped() -> None:
    async def _error_handler(name: str, arguments: dict) -> str:
        return json.dumps({"error": "Something went wrong"})

    composed = compose_tool_handlers(_error_handler, _math_handler)
    result = await composed("add", {"a": 1, "b": 2})
    parsed = json.loads(result)
    # The first handler's error is NOT an "unknown tool" error, so it's returned
    assert parsed == {"error": "Something went wrong"}


def test_requires_at_least_two_handlers() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compose_tool_handlers(_weather_handler)

"""The arguments BEFORE_TOOL_USE leaves are the call's, and are validated
again, in one wording, on every door (RMK-482, RFC §9.3, §21.1).

A hook may hand back new arguments, edit the event's own in place, or
answer MODIFY with an edited copy; either way the call runs with them, and
is refused when they no longer fit the schema, before its handler runs, the
model reading that the arguments were rewritten: the model's own call was
validated before the hook ran.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookResult
from tests.tool_doors import DOORS, Hooks, run_door

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
REFUSAL = "Invalid rewritten arguments for 'lookup': missing required argument 'city'"


def _edits(how: str, arguments: dict[str, Any]) -> Any:
    """A BEFORE_TOOL_USE hook leaving the call *arguments*, *how*."""

    async def before(event: Any, ctx: Any) -> HookResult:
        if how == "in-place":
            event.arguments.clear()
            event.arguments.update(arguments)
            return HookResult.allow()
        if how == "modified":
            return HookResult.modify(replace(event, arguments=dict(arguments)))
        return HookResult(action="allow", metadata={"arguments": dict(arguments)})

    return before


@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("how", ["in-place", "returned", "modified"])
async def test_the_call_runs_with_the_arguments_a_hook_left(door: str, how: str) -> None:
    """RMK-1021 item 26: a MODIFY's arguments reached the next hook, never
    the handler, which ran with the model's."""
    served: list[dict[str, Any]] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        served.append(dict(arguments))
        return "sunny"

    await run_door(
        door,
        lookup,
        arguments={"city": "Paris"},
        hooks=Hooks(before=_edits(how, {"city": "Lyon"})),
        schema=SCHEMA,
    )

    assert served == [{"city": "Lyon"}]


@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("how", ["in-place", "returned", "modified"])
async def test_arguments_a_hook_broke_are_refused_as_rewritten(door: str, how: str) -> None:
    served: list[dict[str, Any]] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        served.append(arguments)
        return "sunny"

    seen = await run_door(
        door,
        lookup,
        arguments={"city": "Paris"},
        hooks=Hooks(before=_edits(how, {})),
        schema=SCHEMA,
    )

    # A backend reads (text, is_error, refused); every other door the text.
    read = seen.model_read[0] if isinstance(seen.model_read, tuple) else seen.model_read
    assert REFUSAL in str(read)
    assert served == []

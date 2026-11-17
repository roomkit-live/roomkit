"""A retained executor task never lends its old scope to a declared owner verb."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import TurnHost, current_turn_host
from roomkit.tools import current_response_metadata, current_tool_actor_id, current_tool_call
from tests.test_turns_host import _kit, _say


async def test_a_retained_child_calls_between_turns_without_lending_its_context() -> None:
    ready = asyncio.Event()
    children: list[asyncio.Task[Any]] = []

    async def use(host: TurnHost) -> None:
        record = current_response_metadata()
        assert record is not None
        record["old"] = True

        async def later() -> dict[str, Any]:
            await ready.wait()
            assert current_turn_host() is host
            assert current_response_metadata() is record
            answer = await host.call("app.later")
            assert current_turn_host() is host
            assert current_response_metadata() is record
            assert record["old"] is True
            return answer

        children.append(asyncio.create_task(later()))

    async def owner(arguments: dict[str, Any]) -> dict[str, Any]:
        call = current_tool_call()
        return {
            "host": current_turn_host() is not None,
            "record": current_response_metadata() is not None,
            "actor": current_tool_actor_id(),
            "id": None if call is None else call.tool_call_id,
        }

    kit, _, _ = await _kit(use)
    kit.declare_host_verb("app.later", owner, between_turns=True)
    try:
        await _say(kit)
        ready.set()
        assert await asyncio.wait_for(children[0], 2) == {
            "host": False,
            "record": False,
            "actor": None,
            "id": "",
        }
    finally:
        ready.set()
        await asyncio.gather(*children, return_exceptions=True)
        await kit.close()

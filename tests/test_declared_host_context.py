"""A retained executor task never lends its old scope to a declared owner verb."""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import TurnHost, current_turn_host, current_turn_id
from roomkit.tools import current_response_metadata, current_tool_actor_id, current_tool_call
from roomkit.turns import current_turn_id as executor_turn_id
from tests.test_turns_host import _kit, _say


async def test_a_declared_verb_reads_its_calling_turn_id_without_its_host() -> None:
    assert current_turn_id() is None
    observed: list[str] = []

    async def use(host: TurnHost) -> None:
        turn_id = executor_turn_id()
        assert turn_id is not None
        assert await host.call("app.id") == turn_id
        observed.append(turn_id)

    async def owner(arguments: dict[str, Any]) -> str | None:
        assert current_turn_host() is None
        return current_turn_id()

    kit, _, _ = await _kit(use)
    kit.declare_host_verb("app.id", owner)
    try:
        await _say(kit)
        assert len(observed) == 1
        assert current_turn_id() is None
    finally:
        await kit.close()


async def test_a_retained_child_calls_between_turns_without_lending_its_context() -> None:
    ready = asyncio.Event()
    children: list[asyncio.Task[Any]] = []

    async def use(host: TurnHost) -> None:
        record = current_response_metadata()
        assert record is not None
        record["old"] = True
        turn_id = current_turn_id()
        assert turn_id is not None

        async def later() -> dict[str, Any]:
            await ready.wait()
            assert current_turn_host() is host
            assert current_response_metadata() is record
            assert current_turn_id() == turn_id
            answer = await host.call("app.later")
            assert current_turn_id() == turn_id
            assert current_turn_host() is host
            assert current_response_metadata() is record
            assert record["old"] is True
            return answer

        children.append(asyncio.create_task(later()))

    async def owner(arguments: dict[str, Any]) -> dict[str, Any]:
        call = current_tool_call()
        return {
            "host": current_turn_host() is not None,
            "turn_id": current_turn_id(),
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
            "turn_id": None,
            "record": False,
            "actor": None,
            "id": "",
        }
    finally:
        ready.set()
        await asyncio.gather(*children, return_exceptions=True)
        await kit.close()

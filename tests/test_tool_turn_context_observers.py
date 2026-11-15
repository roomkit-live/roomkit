"""An observer borrows tool depth only while both scopes are running."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.core._reentrance import chain_floor, floors_lent_while_it_runs
from roomkit.tools import current_tool_chain_depth, tool_turn_context


@pytest.mark.parametrize("floor", [0, 2])
async def test_a_live_observer_stops_borrowing_an_ended_tool_scope(floor: int) -> None:
    started, finish = asyncio.Event(), asyncio.Event()

    async def observer() -> tuple[int, int, int]:
        with floors_lent_while_it_runs(floor):
            before = chain_floor()
            started.set()
            await finish.wait()
            return before, current_tool_chain_depth(), chain_floor()

    with tool_turn_context(chain_depth=3):
        child = asyncio.create_task(observer())
        await started.wait()
    finish.set()
    assert await child == (3, 3, floor)


async def test_an_ended_observer_stops_lending_a_live_tool_scope_to_its_child() -> None:
    finish = asyncio.Event()

    async def child() -> tuple[int, int]:
        await finish.wait()
        return current_tool_chain_depth(), chain_floor()

    with tool_turn_context(chain_depth=3):
        with floors_lent_while_it_runs(0):
            task = asyncio.create_task(child())
        finish.set()
        assert await task == (3, 0)
        assert chain_floor() == 3

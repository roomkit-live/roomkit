"""Concurrent channel calls report changes already reported by their peers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit.turns._rooms import room_state_key
from tests.test_turns_channel_control import _kit


@pytest.mark.parametrize("initial", [None, "opus"], ids=["forget", "restore"])
@pytest.mark.parametrize("fails", [False, True], ids=["returned", "raised"])
async def test_a_call_reports_restoring_the_state_a_concurrent_call_changed(
    initial: str | None, fails: bool
) -> None:
    kit, agent = await _kit()
    if initial is not None:
        await kit.channel_call("agent", "room", "set", model=initial)
    entered, release = asyncio.Event(), asyncio.Event()
    verb = agent.channel_call

    async def held(room_id: str, name: str, arguments: dict[str, Any]) -> Any:
        if name != "restore":
            return await verb(room_id, name, arguments)
        # The executor has read this call's initial state before entering here.
        entered.set()
        await release.wait()
        result = await verb(room_id, "forget" if initial is None else "set", arguments)
        if fails:
            raise ValueError("refused after restoring state")
        return result

    agent.channel_call = held  # type: ignore[method-assign]
    async with asyncio.timeout(5):
        restoring = asyncio.create_task(
            kit.channel_call("agent", "room", "restore", model=initial)
        )
        await entered.wait()
        await kit.channel_call("agent", "room", "set", model="sonnet")
        stored = await kit.get_room("room")
        assert stored.metadata[room_state_key("agent")] == {"config": {"model": "sonnet"}}
        release.set()
        if fails:
            with pytest.raises(ValueError, match="refused after restoring state"):
                await restoring
        else:
            await restoring
    stored = await kit.get_room("room")
    if initial is None:
        assert room_state_key("agent") not in stored.metadata
    else:
        assert stored.metadata[room_state_key("agent")] == {"config": {"model": initial}}
    # Lose only the channel's local copy: the next read must restore the right state.
    agent.config.clear()
    expected = {} if initial is None else {"model": initial}
    assert await kit.channel_call("agent", "room", "read") == expected
    await kit.close()

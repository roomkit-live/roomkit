"""A hook reached by an owner host call can close its own kit."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, TurnHost
from roomkit.models.event import RoomEvent, TextContent
from tests.test_turns_host import _kit, _say


@pytest.mark.usefixtures("close_grace", "cut_unwind")
async def test_an_owner_host_call_can_close_its_kit_without_being_cancelled() -> None:
    async def use(host: TurnHost) -> Any:
        return await host.send_event("r1", "agent", TextContent(body="close owner"))

    kit, _, _ = await _kit(use)
    closed = asyncio.Event()

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def closes(event: RoomEvent, context: Any) -> HookResult:
        if isinstance(event.content, TextContent) and event.content.body == "close owner":
            await kit.close()
            closed.set()
        return HookResult.allow()

    try:
        async with asyncio.timeout(10):
            await _say(kit)
            await closed.wait()
    finally:
        await kit.close()

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import HookExecution, HookTrigger, TextContent
from tests.test_kit_close import _kit, _message, _Transport
from tests.test_kit_close_boundaries import HeldCursor


async def test_an_observers_own_answer_held_on_its_cursor_ends_with_its_close(
    close_grace: float,
) -> None:
    """RFC §12.10.4: witness the retained answer before its sender closes."""
    store = HeldCursor()
    store.release.set()
    target = _Transport("ws")
    kit = await _kit(target, store=store)
    returned = asyncio.Event()

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def reply(event: Any, context: Any) -> None:
        if event.content.body != "Hi.":
            return
        store.reading.clear()
        store.release.clear()
        answer = await kit.send_event("r", "sms", TextContent(body="Hello."))
        assert answer.index == event.index + 1
        await asyncio.wait_for(store.reading.wait(), 1)
        await kit.close()
        returned.set()

    await asyncio.wait_for(kit.process_inbound(_message("Hi."), room_id="r"), 1)
    await asyncio.wait_for(returned.wait(), 1)
    await asyncio.wait_for(
        asyncio.gather(*kit._lanes.running(), *kit.hook_engine.running_observers()), 1
    )
    store.release.set()
    await asyncio.wait_for(
        asyncio.gather(*kit._lanes.running(), *kit.hook_engine.running_observers()), 1
    )
    assert target.sent == ["Hi."]
    assert [
        event.content.body
        for event in await store.list_events("r")
        if isinstance(event.content, TextContent)
    ] == ["Hi.", "Hello."]

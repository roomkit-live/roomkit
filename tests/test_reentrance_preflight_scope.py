"""A pending scope read keeps the caller's captured budget and delivery key."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.models import InboundResult, Room, RoomContext, RoomEvent
from roomkit.store import InMemoryStore
from tests.test_framework import SimpleChannel
from tests.test_reentrance import _body, _message, _messages, _observed


class ScopeStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.hold = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def get_room(self, room_id: str) -> Room | None:
        if self.hold:
            self.hold = False
            self.entered.set()
            await self.release.wait()
        return await super().get_room(room_id)


@pytest.mark.parametrize("cancel", [False, True])
async def test_a_scope_read_outliving_its_observer_keeps_its_budget_and_key(cancel: bool) -> None:
    store = ScopeStore()
    kit = RoomKit(store=store, max_chain_depth=1)
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r1", organization_id="org")
    await kit.attach_channel("r1", "sms")
    children: list[asyncio.Task[InboundResult]] = []
    descendants: list[InboundResult] = []

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def enter(event: RoomEvent, context: RoomContext) -> None:
        if _body(event) == "hello":
            for _ in range(9):
                answer = await kit.process_inbound(_message("again"), room_id="r1")
                assert not answer.blocked
            store.hold = True
            message = _message("last").model_copy(update={"delivery_key": "pending-scope"})
            children.append(
                asyncio.create_task(
                    kit.process_inbound(message, room_id="r1", organization_id="org")
                )
            )
            await store.entered.wait()
        elif _body(event) == "last":
            descendants.append(await kit.process_inbound(_message("descendant"), room_id="r1"))

    try:
        await asyncio.wait_for(kit.process_inbound(_message("hello"), room_id="r1"), 5)
        await asyncio.wait_for(store.entered.wait(), 5)
        # The child is still in the scope read, outside any driver task.
        # Draining proves that its originating observer has ended.
        await asyncio.wait_for(_observed(kit), 5)
        assert len(children) == 1 and not children[0].done()
        if cancel:
            assert await kit.cancel_delivery("pending-scope", organization_id="other") == 0
            assert await kit.cancel_delivery("pending-scope", organization_id="org") == 1
        store.release.set()
        answer = await asyncio.wait_for(children[0], 5)
        assert not answer.blocked and answer.event is not None
        await asyncio.wait_for(_observed(kit), 5)
        assert len(descendants) == 1
        assert (descendants[0].blocked, descendants[0].reason, descendants[0].event) == (
            True,
            "reentry_loop_cap",
            None,
        )
        texts = [_body(event) for event in await _messages(kit)]
        assert texts == ["hello"] + ["again"] * 9 + ["last"]
    finally:
        store.release.set()
        await kit.close()

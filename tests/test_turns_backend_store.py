"""A room owner registers one stable executor with its turn backend on every store."""

from __future__ import annotations

import weakref
from uuid import uuid4

from roomkit import (
    AIChannel,
    ChannelCategory,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.base import ConversationStore
from roomkit.turns import InMemoryTurnBackend, RoomOwner, TurnExecutor, TurnLink, TurnRequest


class RecordingBackend(InMemoryTurnBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.executor_ref: weakref.ReferenceType[TurnExecutor] | None = None

    async def attend(self, owner: RoomOwner) -> None:
        self.calls.append("attend")
        assert owner.executor is not None
        self.executor_ref = weakref.ref(owner.executor)
        await super().attend(owner)

    async def serve(self, executor: TurnExecutor) -> None:
        self.calls.append("serve")
        assert self.executor_ref is not None and self.executor_ref() is executor
        await super().serve(executor)

    async def submit(self, request: TurnRequest) -> TurnLink:
        self.calls.append("submit")
        return await super().submit(request)

    async def close(self) -> None:
        self.calls.append("close")
        await super().close()


async def test_two_turns_use_one_registration_and_persist_both_answers(
    contract_store: ConversationStore,
) -> None:
    room_id = uuid4().hex
    backend = RecordingBackend()
    kit = RoomKit(store=contract_store, turn_backend=backend)
    try:
        kit.register_channel(WebSocketChannel("ws"))
        kit.register_channel(
            AIChannel("ai", provider=MockAIProvider(responses=["first", "second"]))
        )
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ws")
        await kit.attach_channel(room_id, "ai", category=ChannelCategory.INTELLIGENCE)
        for text in ("one", "two"):
            await kit.send_event(room_id, "ws", TextContent(body=text))
        timeline = await kit.get_timeline(room_id)
        assert [
            event.content.body
            for event in timeline
            if event.source.channel_id == "ai" and isinstance(event.content, TextContent)
        ] == ["first", "second"]
    finally:
        await kit.close()
    assert backend.calls == ["attend", "serve", "submit", "submit", "close"]

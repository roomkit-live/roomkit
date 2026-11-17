"""The local turn host reads fresh JSON room context (RFC §10.6.5)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import AIChannel, RoomKit, TurnHost, current_turn_host
from roomkit.models import ChannelBinding, ChannelCategory, Participant, Room, RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel


class _ContextStore(InMemoryStore):
    """An injected store boundary whose next context load can fail."""

    fault: str | None = None

    async def load_room_context(
        self, room_id: str
    ) -> tuple[Room | None, list[ChannelBinding], list[Participant]]:
        if self.fault == "missing":
            return None, [], []
        if self.fault == "error":
            raise OSError("context store unavailable")
        return await super().load_room_context(room_id)


async def _kit(
    use: Callable[[TurnHost], Awaitable[Any]],
    *,
    streaming: bool,
    window: int = 50,
    store: InMemoryStore | None = None,
) -> tuple[RoomKit, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def handler(_name: str, _arguments: dict[str, Any]) -> str:
        host = current_turn_host()
        assert host is not None
        seen["host"] = host
        seen["result"] = await use(host)
        return "read"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(
                content="", tool_calls=[AIToolCall(id="read-1", name="read", arguments={})]
            ),
            AIResponse(content="done"),
        ],
    )
    agent = AIChannel(
        "agent",
        provider=provider,
        max_context_events=window,
        tools=[AITool(name="read", description="Read the room", parameters={"type": "object"})],
        tool_handler=handler,
    )
    kit = RoomKit(store=store)
    kit.register_channel(SimpleChannel("ops"))
    kit.register_channel(SimpleChannel("late"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1", metadata={"nested": {"version": "before"}})
    await kit.attach_channel("r1", "ops")
    await kit.send_event("r1", "ops", TextContent(body="earlier"))
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    return kit, seen


async def _say(kit: RoomKit) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="alice", content=TextContent(body="read now")),
        room_id="r1",
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("window", [0, 50])
async def test_the_host_rereads_room_members_bindings_and_history(
    streaming: bool, window: int
) -> None:
    async def use(host: TurnHost) -> dict[str, Any]:
        before = await host.room_context()
        await kit.update_room_metadata("r1", {"nested": {"version": "after"}})
        await kit.attach_channel("r1", "late")
        await kit.add_member("r1", "ops", "bob", display_name="Bob")
        await kit.send_event("r1", "agent", TextContent(body="written during the tool"))
        after = await host.room_context()
        after["room"]["metadata"]["nested"]["version"] = "caller changed its copy"
        fresh = await host.room_context()
        return {"before": before, "mutated_copy": after, "fresh": fresh}

    kit, seen = await _kit(use, streaming=streaming, window=window)
    try:
        await _say(kit)
        result = seen["result"]
    finally:
        await kit.close()
    before, fresh = result["before"], result["fresh"]
    assert before["room"]["metadata"]["nested"] == {"version": "before"}
    assert fresh["room"]["metadata"]["nested"] == {"version": "after"}
    assert "late" not in {b["channel_id"] for b in before["bindings"]}
    assert "late" in {b["channel_id"] for b in fresh["bindings"]}
    assert "bob" not in {p["id"] for p in before["participants"]}
    assert "bob" in {p["id"] for p in fresh["participants"]}
    bodies = [e["content"].get("body") for e in fresh["recent_events"]]
    assert bodies == ["earlier", "read now", "written during the tool"]
    assert fresh["history_start_index"] == 0
    assert isinstance(fresh["room"]["created_at"], str)
    assert RoomContext.model_validate(fresh).room.id == "r1"
    assert set(fresh) == {
        "room",
        "bindings",
        "participants",
        "recent_events",
        "history_start_index",
        "author_register",
    }


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fault", ["missing", "error"])
async def test_a_host_context_read_propagates_its_store_failure(
    streaming: bool, fault: str
) -> None:
    store = _ContextStore()

    async def use(host: TurnHost) -> tuple[str, str]:
        store.fault = fault
        try:
            await host.room_context()
        except Exception as error:
            return type(error).__name__, str(error)
        finally:
            store.fault = None
        raise AssertionError("a failed context read returned a snapshot")

    kit, seen = await _kit(use, streaming=streaming, store=store)
    try:
        await _say(kit)
        result = seen["result"]
    finally:
        await kit.close()
    expected = (
        ("RoomNotFoundError", "Room r1 not found")
        if fault == "missing"
        else ("OSError", "context store unavailable")
    )
    assert result == expected


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_room_context_reader_is_refused_after_its_turn(streaming: bool) -> None:
    async def use(host: TurnHost) -> str:
        return (await host.room_context())["room"]["id"]

    kit, seen = await _kit(use, streaming=streaming)
    try:
        await _say(kit)
        assert seen["result"] == "r1"
        with pytest.raises(RuntimeError) as refused:
            await seen["host"].room_context()
        assert str(refused.value) == (
            "room_context needs a turn: after its turn ended, a host sends "
            "add_observation, post_status, send_event, update_room_metadata only (RFC §10.6.5)"
        )
    finally:
        await kit.close()

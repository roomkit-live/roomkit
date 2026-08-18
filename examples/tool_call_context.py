"""Per-call tool context — which room, and whose turn.

A tool handler receives only ``(name, arguments)``. Everything else it might
close over at construction time — a room, a user, a database handle scoped to
one person — describes whoever attached the channel, because one ``AIChannel``
object is registered per ``channel_id`` and shared by every room and every
speaker it serves.

``roomkit.tools`` exposes the turn instead, through a contextvar the tool loop
sets:

- ``current_tool_room_id()``      — the room this turn belongs to
- ``current_tool_actor_id()``     — whose turn it is
- ``current_tool_allowed_names()`` — the toolset the turn resolved

This example puts two people in one room, both talking to the same agent, and
shows the handler answering each of them correctly — including refusing, twice,
because the actor is not something to trust on sight:

1. Alice is an identified member: the tool resolves her to an identity and
   answers with her rows.
2. Bob is on the roster but never identified: the id reads back just fine, so
   the handler has to check ``identification`` itself and refuse.
3. A system injection has no author at all: ``None``, and the tool refuses
   rather than borrowing whoever spoke last.

Run with:
    uv run python examples/tool_call_context.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallContent,
    WebSocketChannel,
)
from roomkit.models.enums import IdentificationStatus
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_tool_actor_id, current_tool_allowed_names, current_tool_room_id

# The rows the tool guards. Keyed by *identity*, not by participant id: the
# participant is how someone shows up in one room, the identity is who they are.
INVOICES: dict[str, list[str]] = {
    "user-42": ["INV-1001: $240.00 (paid)", "INV-1002: $80.00 (due)"],
    "user-77": ["INV-2001: $1,500.00 (overdue)"],
}


def _tool_turn() -> list[AIResponse]:
    """One tool round then a final answer — the mock's script for one turn."""
    return [
        AIResponse(
            content="Let me pull those up.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[AIToolCall(id="tc", name="my_invoices", arguments={})],
        ),
        AIResponse(
            content="Here is what I found.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        ),
    ]


async def main() -> None:
    kit = RoomKit()

    async def my_invoices(name: str, arguments: dict[str, Any]) -> str:
        """Answer the person whose turn it is — after establishing who that is."""
        room_id = current_tool_room_id()
        actor_id = current_tool_actor_id()
        print(f"  [tool] room={room_id} actor={actor_id} toolset={current_tool_allowed_names()}")

        # No author: a system injection, a webhook, a scheduled run. Refusing is
        # the answer — the alternative is serving the last human who spoke.
        if room_id is None or actor_id is None:
            return json.dumps({"error": "This turn has no author to answer for."})

        # The actor names the turn; it does not authenticate it. Until the room
        # has identified the sender, the id is whatever the channel supplied.
        participant = await kit.store.get_participant(room_id, actor_id)
        if (
            participant is None
            or participant.identification is not IdentificationStatus.IDENTIFIED
        ):
            return json.dumps({"error": f"Sender {actor_id} is not identified."})

        rows = INVOICES.get(participant.identity_id or "", [])
        return json.dumps({"identity": participant.identity_id, "invoices": rows})

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-billing",
        provider=MockAIProvider(ai_responses=_tool_turn() * 3, streaming=False),
        system_prompt="You are a billing assistant.",
        tool_handler=my_invoices,
        tools=[
            AITool(
                name="my_invoices",
                description="List the invoices of the person asking.",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    room = await kit.create_room(room_id="billing-room")
    await kit.attach_channel(room.id, "ws-user")
    await kit.attach_channel(room.id, "ai-billing", category=ChannelCategory.INTELLIGENCE)

    # Alice joined with a known identity; Bob is on the roster but unresolved.
    await kit.add_member(room.id, "ws-user", "alice", identity_id="user-42")
    await kit.add_member(room.id, "ws-user", "bob")

    async def ask(sender_id: str) -> None:
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id=sender_id,
                content=TextContent(body="What do I owe?"),
            )
        )
        await asyncio.sleep(0.1)

    print("=== Alice asks (identified) ===")
    await ask("alice")

    print("\n=== Bob asks (same channel object, same room, not identified) ===")
    await ask("bob")

    print("\n=== A system injection asks (no author) ===")
    await kit.send_event(
        room.id,
        "ws-user",
        TextContent(body="Nightly reconciliation: what is outstanding?"),
        participant_id=None,
    )
    await asyncio.sleep(0.1)

    print("\n=== What the tool returned, in order ===")
    for event in await kit.store.list_events(room.id):
        if isinstance(event.content, ToolCallContent) and event.content.status == "completed":
            print(f"  {event.content.result}")


if __name__ == "__main__":
    asyncio.run(main())

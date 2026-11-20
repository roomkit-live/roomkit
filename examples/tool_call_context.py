"""Per-call tool context — which room, and whose turn.

A tool handler receives only ``(name, arguments)``. Everything else it might
close over at construction time — a room, a user, a database handle scoped to
one person — describes whoever attached the channel, because one ``AIChannel``
object is registered per ``channel_id`` and shared by every room and every
speaker it serves.

``roomkit.tools`` exposes the turn instead, through a contextvar the tool loop
sets:

- ``current_tool_room_id()``      — the room this turn belongs to
- ``current_tool_room()``         — the ``Room`` itself, as the turn loaded it
- ``current_tool_actor_id()``     — whose turn it is
- ``current_tool_allowed_names()`` — the toolset the turn resolved, less what
  its tool policy denies

This example puts two people in one room, both talking to the same agent, and
shows the handler answering each of them correctly — including refusing, twice,
because the actor is not something to trust on sight:

1. Alice is an identified member: the tool resolves her to an identity and
   answers with her rows.
2. Bob is on the roster but never identified: the id reads back just fine, so
   the handler has to check ``identification`` itself and refuse.
3. An event the host sends with no participant has no author at all:
   ``None``, and the tool refuses rather than borrowing whoever spoke last.
4. The same handler, called from a realtime voice session: the realtime
   channel installs the same context around the call, so Alice's voice turn
   is answered like her text turn.
5. The same handler, called directly as a unit test would: no channel runs,
   ``tool_turn_context`` describes Alice's turn around the call and takes it
   away when the block exits.

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
    RealtimeVoiceChannel,
    Room,
    RoomKit,
    TextContent,
    ToolCallContent,
    ToolHandler,
    WebSocketChannel,
)
from roomkit.models import IdentificationStatus
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.tools import (
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_room,
    current_tool_room_id,
    tool_turn_context,
)
from roomkit.voice.realtime import MockRealtimeProvider, MockRealtimeTransport

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


async def _alice_by_voice(kit: RoomKit, handler: ToolHandler, room: Room) -> None:
    """Serve *handler* from a realtime voice session Alice speaks in."""
    print("\n=== Alice asks by voice (same handler, realtime channel) ===")
    rt_provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice-billing",
        provider=rt_provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
    )
    kit.register_channel(voice)
    await kit.attach_channel(room.id, "voice-billing")
    session = await voice.start_session(room.id, "alice", "fake-ws")
    # The realtime channel installs the per-call context around the handler:
    # room, Room and actor read back as on the text path; the toolset and the
    # response record are the AI channel's and read None here.
    await rt_provider.simulate_tool_call(session, "call-voice", "my_invoices", {})
    await asyncio.sleep(0.1)
    _session_id, _call_id, submitted = rt_provider.tool_results[0]
    print(f"  {submitted}")


async def _alice_in_a_test(handler: ToolHandler, room: Room) -> None:
    """Call *handler* directly, as a unit test would, during Alice's turn."""
    print("\n=== Alice's turn, described by a test (no channel runs) ===")
    # The turn a tool loop would give the handler: the Room names the room id,
    # and the block restores the previous context (here: none) on the way out.
    invoices_tool = AITool(name="my_invoices", description="List the invoices of the asker.")
    with tool_turn_context(room=room, actor_id="alice", tools=[invoices_tool]):
        print(f"  {await handler('my_invoices', {})}")
    print(f"  outside the block: actor={current_tool_actor_id()}")


async def main() -> None:
    kit = RoomKit()

    async def my_invoices(name: str, arguments: dict[str, Any]) -> str:
        """Answer the person whose turn it is — after establishing who that is."""
        room_id = current_tool_room_id()
        actor_id = current_tool_actor_id()
        # The Room object of the turn, the same one its RoomContext holds: the
        # tenant a host would otherwise re-read by id on every call is right here.
        room = current_tool_room()
        tenant = room.metadata.get("tenant") if room is not None else None
        print(
            f"  [tool] room={room_id} tenant={tenant} actor={actor_id} "
            f"toolset={current_tool_allowed_names()}"
        )

        # No author: an event that names no participant (one the host sends
        # with none named, another agent's answer). Refusing is the answer —
        # the alternative is serving the last human who spoke.
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

    room = await kit.create_room(room_id="billing-room", metadata={"tenant": "acme"})
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

    print("\n=== An event the host sends with no participant asks (no author) ===")
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

    await _alice_by_voice(kit, my_invoices, room)
    await _alice_in_a_test(my_invoices, room)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

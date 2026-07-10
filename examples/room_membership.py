"""Explicit room membership and "seen by" aggregation.

Demonstrates deliberate join/leave on top of the participant model, distinct
from the lazy `ensure_participant` that materialises a sender the first time
they speak. Shows:
- add_member() — an intentional, idempotent join (safe on every room open)
- list_members() / is_member() — the active roster
- ON_PARTICIPANT_JOINED / ON_PARTICIPANT_LEFT lifecycle hooks
- remove_member() — a soft leave (status flip, history preserved)
- list_read_markers() — per-channel read high-water-marks, mapped back to
  members for a "seen by" receipt

Run with:
    uv run python examples/room_membership.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)

MEMBERS = [("alice", "Alice"), ("bob", "Bob"), ("carol", "Carol")]


async def main() -> None:
    kit = RoomKit()

    for member_id, _ in MEMBERS:
        kit.register_channel(WebSocketChannel(f"ws-{member_id}"))

    # --- Lifecycle hooks fire on every join and leave ---
    @kit.hook(HookTrigger.ON_PARTICIPANT_JOINED, execution=HookExecution.ASYNC)
    async def on_joined(event, ctx) -> None:
        print(f"  [hook] joined: {event.content.data['participant_id']}")

    @kit.hook(HookTrigger.ON_PARTICIPANT_LEFT, execution=HookExecution.ASYNC)
    async def on_left(event, ctx) -> None:
        print(f"  [hook] left: {event.content.data['participant_id']}")

    await kit.create_room(room_id="team")

    # --- Explicit join: deliberate, and idempotent on re-open ---
    print("Members join the room...")
    for member_id, display in MEMBERS:
        await kit.attach_channel("team", f"ws-{member_id}")
        await kit.add_member(
            "team",
            channel_id=f"ws-{member_id}",
            participant_id=member_id,
            identity_id=member_id,
            display_name=display,
        )

    # Re-adding an active member is a no-op (no write, no event, no hook)
    await kit.add_member("team", channel_id="ws-alice", participant_id="alice")

    roster = await kit.list_members("team")
    print(f"\nActive roster ({len(roster)}): {', '.join(p.display_name for p in roster)}")
    print(f"Is Bob a member? {await kit.is_member('team', 'bob')}")

    # --- Alice posts; Bob and Carol read to different points ---
    print("\nAlice posts three messages...")
    msg_ids: list[str] = []
    for text in ["Standup in 5?", "Agenda: releases", "Notes in the doc"]:
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-alice", sender_id="alice", content=TextContent(body=text)
            )
        )
        if result.event:
            msg_ids.append(result.event.id)

    await kit.mark_read("team", "ws-bob", msg_ids[2])  # Bob read everything
    await kit.mark_read("team", "ws-carol", msg_ids[0])  # Carol read only the first

    # --- "Seen by": map channel read-markers back to members ---
    markers = await kit.list_read_markers("team")
    channel_to_name = {p.channel_id: p.display_name for p in await kit.list_members("team")}
    print("\nSeen by (read up to event index):")
    for channel_id, index in sorted(markers.items()):
        print(f"  {channel_to_name.get(channel_id, channel_id)}: index {index}")

    # --- Soft leave: status flips to LEFT, history/read-markers survive ---
    print("\nCarol leaves...")
    await kit.remove_member("team", "carol")
    active = await kit.list_members("team")
    everyone = await kit.list_members("team", include_left=True)
    print(f"Active now ({len(active)}): {', '.join(p.display_name for p in active)}")
    print(
        f"Including those who left ({len(everyone)}): "
        f"{', '.join(f'{p.display_name}={p.status.value}' for p in everyone)}"
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

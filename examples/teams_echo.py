"""Microsoft Teams echo bot — minimal example using MockTeamsProvider.

This example demonstrates the full Teams webhook parsing and RoomKit pipeline
without requiring Azure credentials or botbuilder-core. It simulates inbound
Activities as they would arrive from the Bot Framework.

Run with:
    python examples/teams_echo.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    HookExecution,
    MockTeamsProvider,
    RoomKit,
    parse_teams_webhook,
)
from roomkit.channels import TeamsChannel
from roomkit.models.enums import HookTrigger


async def main() -> None:
    # --- Provider & channel setup --------------------------------------------
    provider = MockTeamsProvider()
    kit = RoomKit()
    kit.register_channel(TeamsChannel("teams-main", provider=provider))

    # Echo hook — send back whatever the user said
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def echo(event, context):  # noqa: ARG001
        body = getattr(event.content, "body", "")
        if body:
            print(f"  Echo would reply: '{body}'")

    # --- Simulate inbound Activities -----------------------------------------

    # 1. Personal (1:1) message
    personal_activity = {
        "type": "message",
        "id": "activity-001",
        "text": "Hello from Teams!",
        "from": {"id": "user-aad-id-123", "name": "Alice"},
        "conversation": {
            "id": "a]conv-personal-001",
            "conversationType": "personal",
        },
        "recipient": {"id": "bot-aad-id", "name": "MyBot"},
        "serviceUrl": "https://smba.trafficmanager.net/amer/",
        "channelData": {"tenant": {"id": "tenant-abc"}},
    }

    # 2. Group chat message with @mention
    group_activity = {
        "type": "message",
        "id": "activity-002",
        "text": "<at>MyBot</at> tell me a joke",
        "from": {"id": "user-aad-id-456", "name": "Bob"},
        "conversation": {
            "id": "19:meeting_abc@thread.v2",
            "conversationType": "groupChat",
        },
        "recipient": {"id": "bot-aad-id", "name": "MyBot"},
        "serviceUrl": "https://smba.trafficmanager.net/amer/",
        "channelData": {"tenant": {"id": "tenant-abc"}},
    }

    # 3. Non-message Activity (should be skipped)
    update_activity = {
        "type": "conversationUpdate",
        "id": "activity-003",
        "membersAdded": [{"id": "bot-aad-id"}],
        "conversation": {"id": "conv-personal-001"},
    }

    for label, payload in [
        ("Personal message", personal_activity),
        ("Group @mention", group_activity),
        ("Conversation update (skip)", update_activity),
    ]:
        print(f"\n--- {label} ---")
        messages = parse_teams_webhook(payload, channel_id="teams-main")

        if not messages:
            print("  (no messages parsed — skipped)")
            continue

        for inbound in messages:
            conv_id = inbound.metadata.get("conversation_id", "") if inbound.metadata else ""
            print(f"  Sender: {inbound.sender_id}")
            print(f"  Text:   {inbound.content.body}")  # type: ignore[union-attr]
            print(f"  Conv:   {conv_id}")
            print(f"  Group:  {inbound.metadata.get('is_group', False)}")  # type: ignore[union-attr]

            # Create room and attach channel per conversation
            room_id = f"teams-{conv_id}" if conv_id else "teams-default"
            rooms = await kit.store.list_rooms()
            if not any(r.id == room_id for r in rooms):
                await kit.create_room(room_id=room_id)
                await kit.attach_channel(
                    room_id,
                    "teams-main",
                    metadata={"teams_conversation_id": conv_id},
                )

            result = await kit.process_inbound(inbound)
            print(f"  Blocked: {result.blocked}")

    # --- Show what the mock provider captured --------------------------------
    print(f"\n--- Mock provider captured {len(provider.sent)} outbound message(s) ---")
    for msg in provider.sent:
        print(f"  To: {msg['to']}, Content: {msg['event'].content}")

    # --- Show conversation history -------------------------------------------
    for room in await kit.store.list_rooms():
        events = await kit.store.list_events(room.id)
        print(f"\nRoom '{room.id}' ({len(events)} events):")
        for ev in events:
            print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

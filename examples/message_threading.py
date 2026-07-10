"""Message threading (flat two-level, Slack/Teams style).

Demonstrates in-app threads built on ``RoomEvent.parent_event_id``:
- Replying to a message with ``InboundMessage.parent_event_id``
- The flat two-level invariant: replying to a reply collapses to the root
- Reading a thread with ``EventFilter(parent_event_id=...)`` and the main
  timeline with ``EventFilter(top_level_only=True)``
- ``store.get_thread_summaries`` for a "N replies · last reply" affordance
- An AI reply inheriting the thread root, so ``@`` mentions answer in-thread

Threading is transport-agnostic: the parent is applied centrally in the
inbound pipeline, so it works for WebSocket, SMS, email, etc. — no per-channel
wiring. It is distinct from ``ChannelData.thread_id`` (a provider-native thread
reference such as a Slack ``thread_ts`` or Discord message id).

Run with:
    uv run python examples/message_threading.py
"""

from __future__ import annotations

import asyncio

from roomkit import InboundMessage, RoomEvent, RoomKit, TextContent, WebSocketChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import ChannelCategory
from roomkit.models.store_filter import EventFilter
from roomkit.providers.ai.mock import MockAIProvider


def _short(event_id: str) -> str:
    return f"{event_id[:8]}…"


async def main() -> None:
    kit = RoomKit()

    ws_alice = WebSocketChannel("ws-alice")
    ws_bob = WebSocketChannel("ws-bob")
    ai = AIChannel("assistant", provider=MockAIProvider(responses=["Tuesday works for me."]))
    kit.register_channel(ws_alice)
    kit.register_channel(ws_bob)
    kit.register_channel(ai)

    # Clients need a connection registered to receive deliveries.
    for ch in (ws_alice, ws_bob):
        ch.register_connection(f"{ch.channel_id}-conn", lambda _c, _e: asyncio.sleep(0))

    await kit.create_room(room_id="team-room")
    await kit.attach_channel("team-room", "ws-alice")
    await kit.attach_channel("team-room", "ws-bob")

    async def say(channel_id: str, sender: str, body: str, parent: str | None = None) -> RoomEvent:
        result = await kit.process_inbound(
            InboundMessage(
                channel_id=channel_id,
                sender_id=sender,
                content=TextContent(body=body),
                parent_event_id=parent,
            )
        )
        assert result.event is not None
        return result.event

    # --- A root message ---
    root = await say("ws-alice", "alice", "Can we move the sync to this week?")
    print(f'1. alice (root {_short(root.id)}): "Can we move the sync to this week?"')

    # --- Bob replies in the thread ---
    reply = await say("ws-bob", "bob", "Sure — which day?", parent=root.id)
    print(f'2. bob   (reply, parent={_short(reply.parent_event_id or "")}): "Sure — which day?"')

    # --- Alice replies to Bob's reply. Flat two-level: it threads under the
    #     ROOT, not under Bob's reply. ---
    nested = await say("ws-alice", "alice", "Tuesday?", parent=reply.id)
    print(
        f'3. alice (reply to a reply, parent={_short(nested.parent_event_id or "")}): "Tuesday?"'
        f"  -> collapsed to root: {nested.parent_event_id == root.id}"
    )

    # --- Attach the AI and @mention it inside the thread; its reply inherits
    #     the thread root and lands in the same thread. ---
    await kit.attach_channel("team-room", "assistant", category=ChannelCategory.INTELLIGENCE)
    await say("ws-alice", "alice", "@assistant does Tuesday work?", parent=root.id)

    # --- Read the main timeline (roots + standalone, replies excluded) ---
    timeline = await kit.store.list_events(
        "team-room", event_filter=EventFilter(top_level_only=True)
    )
    print("\nMain timeline (top level only):")
    for ev in timeline:
        if isinstance(ev.content, TextContent):
            print(f"  - {ev.source.channel_id}: {ev.content.body}")

    # --- Read one thread ---
    thread = await kit.store.list_events(
        "team-room", event_filter=EventFilter(parent_event_id=root.id)
    )
    print(f"\nThread under root {_short(root.id)} ({len(thread)} replies):")
    for ev in thread:
        if isinstance(ev.content, TextContent):
            print(f"  - {ev.source.channel_id}: {ev.content.body}")

    # --- Thread summary (for a "N replies" affordance without fetching them) ---
    summaries = await kit.store.get_thread_summaries("team-room", [root.id])
    summary = summaries[root.id]
    print(
        f"\nSummary for root {_short(root.id)}: "
        f"{summary.reply_count} replies, last at {summary.last_reply_at}"
    )


if __name__ == "__main__":
    asyncio.run(main())

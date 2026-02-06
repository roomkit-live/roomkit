"""Content moderation with BEFORE_BROADCAST hooks.

Demonstrates how to use sync hooks to block, allow, or modify messages
before they reach other channels. Shows:
- HookResult.block() to reject messages
- HookResult.modify() to redact sensitive content
- HookResult.allow() to let messages through
- Hook priority ordering (lower runs first)

Run with:
    uv run python examples/hook_moderation.py
"""

from __future__ import annotations

import asyncio
import re

from roomkit import (
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)


async def main() -> None:
    kit = RoomKit()

    ws_user = WebSocketChannel("ws-user")
    ws_agent = WebSocketChannel("ws-agent")
    kit.register_channel(ws_user)
    kit.register_channel(ws_agent)

    agent_inbox: list[RoomEvent] = []

    async def agent_recv(_conn: str, event: RoomEvent) -> None:
        agent_inbox.append(event)

    ws_agent.register_connection("agent-conn", agent_recv)

    await kit.create_room(room_id="moderated-room")
    await kit.attach_channel("moderated-room", "ws-user")
    await kit.attach_channel("moderated-room", "ws-agent")

    # --- Hook 1: Block profanity (priority=0, runs first) ---
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="profanity_filter", priority=0)
    async def profanity_filter(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent):
            blocked_words = {"badword", "spam", "scam"}
            words = set(event.content.body.lower().split())
            if words & blocked_words:
                return HookResult.block(
                    reason=f"Blocked: contains prohibited words {words & blocked_words}"
                )
        return HookResult.allow()

    # --- Hook 2: Redact phone numbers (priority=1, runs second) ---
    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="pii_redactor", priority=1)
    async def pii_redactor(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if isinstance(event.content, TextContent):
            redacted = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED]", event.content.body)
            if redacted != event.content.body:
                modified_event = event.model_copy(update={"content": TextContent(body=redacted)})
                return HookResult.modify(modified_event)
        return HookResult.allow()

    # --- Test 1: Normal message (allowed) ---
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Hello, how are you?"),
        )
    )
    print(f"1. Normal message -> blocked={result.blocked}")

    # --- Test 2: Profanity (blocked) ---
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="This is badword content"),
        )
    )
    print(f"2. Profanity     -> blocked={result.blocked}, reason={result.reason}")

    # --- Test 3: Phone number (redacted via modify) ---
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Call me at 555-123-4567 please"),
        )
    )
    print(f"3. Phone number  -> blocked={result.blocked}")

    # --- Inspect what the agent received ---
    print(f"\nAgent inbox ({len(agent_inbox)} messages):")
    for ev in agent_inbox:
        print(f"  <- {ev.content.body}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())

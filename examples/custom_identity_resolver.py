"""Custom identity resolver.

Demonstrates how to implement a custom IdentityResolver to identify
inbound message senders. Shows:
- Subclassing IdentityResolver with a custom resolve() method
- MockIdentityResolver for quick testing
- IdentityResult statuses: IDENTIFIED, AMBIGUOUS, UNKNOWN
- Identity hooks: ON_IDENTITY_UNKNOWN, ON_IDENTITY_AMBIGUOUS
- IdentityHookResult for handling unknown senders

Run with:
    uv run python examples/custom_identity_resolver.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    HookTrigger,
    Identity,
    IdentificationStatus,
    IdentityHookResult,
    IdentityResolver,
    IdentityResult,
    InboundMessage,
    MockIdentityResolver,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.delivery import InboundMessage as InboundMessageType


# --- Custom identity resolver: simulates a database lookup ---
class DatabaseIdentityResolver(IdentityResolver):
    """Resolves identity by looking up sender_id in a simulated database."""

    def __init__(self) -> None:
        # Simulated user database
        self._db: dict[str, Identity] = {
            "alice-phone": Identity(
                id="user-001",
                display_name="Alice Martin",
                email="alice@example.com",
                phone="+15551234567",
            ),
            "bob-phone": Identity(
                id="user-002",
                display_name="Bob Smith",
                email="bob@example.com",
                phone="+15559876543",
            ),
        }
        # Ambiguous entries (same name, multiple accounts)
        self._ambiguous: dict[str, list[Identity]] = {
            "john-phone": [
                Identity(id="user-003", display_name="John Doe", email="john1@example.com"),
                Identity(id="user-004", display_name="John Doe", email="john2@example.com"),
            ]
        }

    async def resolve(self, message: InboundMessageType, context: RoomContext) -> IdentityResult:
        # Check exact match
        identity = self._db.get(message.sender_id)
        if identity:
            return IdentityResult(
                status=IdentificationStatus.IDENTIFIED,
                identity=identity,
            )

        # Check ambiguous match
        candidates = self._ambiguous.get(message.sender_id)
        if candidates:
            return IdentityResult(
                status=IdentificationStatus.AMBIGUOUS,
                candidates=candidates,
            )

        # Unknown sender
        return IdentityResult(status=IdentificationStatus.UNKNOWN)


async def main() -> None:
    # --- Setup with custom resolver ---
    resolver = DatabaseIdentityResolver()
    kit = RoomKit(identity_resolver=resolver)

    ws = WebSocketChannel("ws-main")
    kit.register_channel(ws)
    ws.register_connection("conn", lambda _c, _e: asyncio.sleep(0))

    # --- Hook: Handle unknown senders ---
    @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
    async def handle_unknown(
        event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
    ) -> IdentityHookResult | None:
        print(f"  [hook] Unknown sender detected — creating pending participant")
        return IdentityHookResult.pending(display_name="Unknown User")

    # --- Hook: Handle ambiguous senders ---
    @kit.identity_hook(HookTrigger.ON_IDENTITY_AMBIGUOUS)
    async def handle_ambiguous(
        event: RoomEvent, ctx: RoomContext, id_result: IdentityResult
    ) -> IdentityHookResult | None:
        names = [c.display_name for c in id_result.candidates]
        print(f"  [hook] Ambiguous sender — {len(id_result.candidates)} candidates: {names}")
        # For demo, pick the first candidate
        return IdentityHookResult.resolved(id_result.candidates[0])

    await kit.create_room(room_id="identity-room")
    await kit.attach_channel("identity-room", "ws-main")

    # --- Test 1: Known sender (alice) ---
    print("Test 1: Known sender (alice-phone)")
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-main",
            sender_id="alice-phone",
            content=TextContent(body="Hello!"),
        )
    )
    print(f"  Result: blocked={result.blocked}")

    # --- Test 2: Ambiguous sender (john) ---
    print("\nTest 2: Ambiguous sender (john-phone)")
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-main",
            sender_id="john-phone",
            content=TextContent(body="Hi there!"),
        )
    )
    print(f"  Result: blocked={result.blocked}")

    # --- Test 3: Unknown sender ---
    print("\nTest 3: Unknown sender (stranger)")
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-main",
            sender_id="stranger",
            content=TextContent(body="Who am I?"),
        )
    )
    print(f"  Result: blocked={result.blocked}")

    # --- Show participants ---
    participants = await kit.store.list_participants("identity-room")
    print(f"\nParticipants ({len(participants)}):")
    for p in participants:
        print(
            f"  {p.id}: display_name={p.display_name}, "
            f"identification={p.identification}, "
            f"identity_id={p.identity_id}"
        )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

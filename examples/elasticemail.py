"""Elastic Email example — send an email through Elastic Email.

Run with:
    ELASTIC_EMAIL_API_KEY=... uv run python examples/elasticemail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import require_env

from roomkit import EmailChannel, InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.providers.elasticemail import ElasticEmailConfig, ElasticEmailProvider


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    env = require_env("ELASTIC_EMAIL_API_KEY")

    config = ElasticEmailConfig(
        api_key=env["ELASTIC_EMAIL_API_KEY"],
        from_email="noreply@example.com",
        from_name="RoomKit Demo",
    )
    provider = ElasticEmailProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()

    # The WebSocket channel acts as the message source (e.g. a user typing
    # in a web UI).  The email channel is a broadcast target — when a message
    # arrives from ws-system, RoomKit delivers it to email-main via the
    # ElasticEmail provider.
    ws = WebSocketChannel("ws-system")
    email_ch = EmailChannel(
        "email-main",
        provider=provider,
        from_address=config.from_email,
    )
    kit.register_channel(ws)
    kit.register_channel(email_ch)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "ws-system")
    await kit.attach_channel(
        "demo-room",
        "email-main",
        metadata={
            "email_address": "recipient@domain.com",
            "subject": "Hello from RoomKit",
        },
    )

    # --- Send a message ------------------------------------------------------
    # The message enters from ws-system and gets broadcast to email-main,
    # which triggers ElasticEmailProvider.send().
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-system",
            sender_id="system",
            content=TextContent(body="This email was sent via Elastic Email!"),
        )
    )
    print(f"Sent message -> blocked={result.blocked}")

    # --- Show conversation history -------------------------------------------
    events = await kit.store.list_events("demo-room")
    print(f"\nRoom history ({len(events)} events):")
    for ev in events:
        print(f"  [{ev.source.channel_id}] {ev.content.body}")  # type: ignore[union-attr]

    await provider.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

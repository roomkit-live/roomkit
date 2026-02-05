"""WhatsApp Personal example — bidirectional messaging via neonize.

Connects a personal WhatsApp account using the multidevice protocol.
On first run, scan the QR code printed in the terminal with WhatsApp
(Settings > Linked Devices > Link a Device).

Requires:
    pip install roomkit[whatsapp-personal]

Run with:
    uv run python examples/whatsapp_personal.py

Warning:
    This uses the unofficial WhatsApp Web protocol via neonize.
    Personal use and experimentation only.
"""

from __future__ import annotations

import asyncio
import logging
import os

# Quiet whatsmeow internal logs (515 reconnect, websocket teardown, app state, etc.)
logging.getLogger("whatsmeow").setLevel(logging.ERROR)
logging.getLogger("Whatsmeow").setLevel(logging.ERROR)

from roomkit import (
    HookExecution,
    HookResult,
    HookTrigger,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WhatsAppPersonalChannel,
)
from roomkit.models.event import AudioContent, LocationContent, MediaContent
from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider
from roomkit.sources import WhatsAppPersonalSourceProvider


async def main() -> None:
    # --- Configuration -------------------------------------------------------
    db_path = os.environ.get("WA_SESSION_DB", "wa-session.db")
    channel_id = "wa-personal"

    # --- Lifecycle event handler ---------------------------------------------
    async def on_wa_event(event_type: str, data: dict) -> None:
        if event_type == "qr":
            import segno

            qr_data = ",".join(data["codes"])
            print(f"\n{'=' * 50}")
            print("Scan this QR code with WhatsApp:")
            print(f"{'=' * 50}")
            segno.make(qr_data).terminal(compact=True)
            print()
        elif event_type == "authenticated":
            print(f"Paired as +{data['user']} (device {data['device']}), connecting...")
        elif event_type == "connected":
            print("WhatsApp connected and ready!")
        elif event_type == "disconnected":
            print("WhatsApp connection lost, reconnecting...")
        elif event_type == "logged_out":
            print("Logged out — delete session DB and re-pair.")
        elif event_type == "presence":
            sender = data.get("sender_name") or data.get("sender", "?")
            state = data.get("state", "?")
            media = data.get("media", "text")
            if state == "composing":
                action = "recording audio..." if media == "audio" else "typing..."
                print(f"[{sender}] {action}")
            elif state == "paused":
                print(f"[{sender}] stopped typing")
        elif event_type == "receipt":
            sender = data.get("sender_name") or data.get("sender", "?")
            rtype = data.get("type", "?")
            msg_ids = data.get("message_ids", [])
            ids = ", ".join(msg_ids[:3]) + ("..." if len(msg_ids) > 3 else "")
            print(f"Receipt: {rtype} from {sender} [{ids}]")

    # --- Source + Provider + Channel -----------------------------------------
    source = WhatsAppPersonalSourceProvider(
        db=db_path,
        channel_id=channel_id,
        on_event=on_wa_event,
        device_name="RoomKit",
        device_platform="chrome",
        self_chat=True,
    )

    provider = WhatsAppPersonalProvider(source)

    # --- RoomKit wiring ------------------------------------------------------
    kit = RoomKit()
    kit.register_channel(WhatsAppPersonalChannel(channel_id, provider=provider))

    # Echo hook — log every inbound message
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="echo_reply")
    async def echo_reply(event: RoomEvent, ctx: RoomContext) -> HookResult:
        name = event.metadata.get("push_name") or event.source.participant_id or "unknown"
        c = event.content
        if isinstance(c, TextContent):
            print(f"[{name}] {c.body}")
        elif isinstance(c, AudioContent):
            dur = f" ({c.duration_seconds:.0f}s)" if c.duration_seconds else ""
            size = len(c.url) if c.url else 0
            print(f"[{name}] audio{dur} [{c.mime_type}, {size} bytes b64]")
        elif isinstance(c, MediaContent):
            caption = f" — {c.caption}" if getattr(c, "caption", None) else ""
            size = len(c.url) if c.url else 0
            print(f"[{name}] {c.mime_type}{caption} [{size} bytes b64]")
        elif isinstance(c, LocationContent):
            label = c.label or "pin"
            print(f"[{name}] location: {label} ({c.latitude}, {c.longitude})")
        else:
            print(f"[{name}] {type(c).__name__}")
        return HookResult.allow()

    # --- Attach and run ------------------------------------------------------
    await kit.attach_source(channel_id, source, auto_restart=True)

    print(f"WhatsApp personal source attached (db={db_path})")
    print("Waiting for messages... Press Ctrl+C to stop.\n")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await kit.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

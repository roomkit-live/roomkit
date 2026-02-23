"""Microsoft Teams bot example — receive and echo messages via Bot Framework.

Requires:
    pip install roomkit[teams] aiohttp

Password auth:
    TEAMS_APP_ID=... TEAMS_APP_PASSWORD=... python examples/teams_bot.py

Certificate auth:
    TEAMS_APP_ID=... \\
    TEAMS_CERT_THUMBPRINT=AABB... \\
    TEAMS_CERT_PRIVATE_KEY="$(cat key.pem)" \\
    python examples/teams_bot.py

    Optionally set TEAMS_CERT_PUBLIC for the x5c JWT header (cert rotation).

Then expose the server via ngrok or similar and register the endpoint
(e.g. https://<host>/api/messages) in the Azure Bot configuration.
You can also test locally with the Bot Framework Emulator.
"""

from __future__ import annotations

import os

from aiohttp import web

from roomkit import (
    BotFrameworkTeamsProvider,
    HookExecution,
    RoomKit,
    TeamsConfig,
    is_bot_added,
    parse_teams_activity,
    parse_teams_webhook,
)
from roomkit.channels import TeamsChannel
from roomkit.models.enums import HookTrigger


async def main() -> None:
    app_id = os.environ.get("TEAMS_APP_ID", "")
    app_password = os.environ.get("TEAMS_APP_PASSWORD", "")
    tenant_id = os.environ.get("TEAMS_TENANT_ID", "common")
    cert_thumbprint = os.environ.get("TEAMS_CERT_THUMBPRINT", "")
    cert_private_key = os.environ.get("TEAMS_CERT_PRIVATE_KEY", "")
    cert_public = os.environ.get("TEAMS_CERT_PUBLIC", "")

    if not app_id:
        print("Set TEAMS_APP_ID to run this example.")
        return

    if cert_thumbprint and cert_private_key:
        # Certificate-based auth
        config_kwargs: dict[str, str] = {
            "app_id": app_id,
            "tenant_id": tenant_id,
            "certificate_thumbprint": cert_thumbprint,
            "certificate_private_key": cert_private_key,
        }
        if cert_public:
            config_kwargs["certificate_public"] = cert_public
        config = TeamsConfig(**config_kwargs)
        print("Using certificate-based authentication.")
    elif app_password:
        # Password auth
        config = TeamsConfig(app_id=app_id, app_password=app_password, tenant_id=tenant_id)
        print("Using password-based authentication.")
    else:
        print("Set TEAMS_APP_PASSWORD or TEAMS_CERT_THUMBPRINT + TEAMS_CERT_PRIVATE_KEY.")
        return
    provider = BotFrameworkTeamsProvider(config)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()
    teams_ch = TeamsChannel("teams-main", provider=provider)
    kit.register_channel(teams_ch)

    # Echo hook — echo back when the bot is mentioned
    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def on_message(event, context):  # noqa: ARG001
        body = getattr(event.content, "body", "")
        sender = (event.metadata or {}).get("sender_name") or event.source.participant_id
        mentioned = (event.metadata or {}).get("bot_mentioned", False)
        tag = " (@bot)" if mentioned else ""
        print(f"[{event.source.channel_id}] {sender}: {body}{tag}")

        # Echo back only when the bot is mentioned
        if mentioned and body:
            from roomkit.models.enums import ChannelType
            from roomkit.models.event import EventSource, RoomEvent, TextContent

            conv_id = (event.metadata or {}).get("conversation_id", "")
            if conv_id:
                reply = RoomEvent(
                    room_id=event.room_id,
                    source=EventSource(
                        channel_id="teams-main",
                        channel_type=ChannelType.TEAMS,
                    ),
                    content=TextContent(body=f"Echo: {body}"),
                )
                result = await provider.send(reply, to=conv_id)
                print(f"  Echo sent: success={result.success} error={result.error}")

    # --- Helper: ensure a RoomKit room exists for a conversation -------------
    async def ensure_room(conv_id: str) -> str:
        """Create a RoomKit room for a Teams conversation if it doesn't exist."""
        room_id = f"teams-{conv_id}" if conv_id else "teams-default"
        rooms = await kit.store.list_rooms()
        if not any(r.id == room_id for r in rooms):
            await kit.create_room(room_id=room_id)
            await kit.attach_channel(
                room_id,
                "teams-main",
                metadata={"teams_conversation_id": conv_id},
            )
            print(f"  Created room: {room_id}")
        return room_id

    # --- aiohttp webhook handler ---------------------------------------------
    async def handle_messages(request: web.Request) -> web.Response:
        payload = await request.json()

        # Always save the conversation reference (needed for proactive sends)
        await provider.save_conversation_reference(payload)

        activity = parse_teams_activity(payload)
        activity_type = activity["activity_type"]
        conv_id = activity["conversation_id"]

        # --- Bot installed into a team/group/personal chat -------------------
        if is_bot_added(payload):
            room_id = await ensure_room(conv_id)
            conv_type = activity["conversation_type"]
            print(f"Bot added to {conv_type} conversation: {conv_id} -> room {room_id}")
            return web.Response(status=200)

        # --- Regular messages ------------------------------------------------
        if activity_type == "message":
            messages = parse_teams_webhook(payload, channel_id="teams-main")
            for inbound in messages:
                await ensure_room(conv_id)
                result = await kit.process_inbound(inbound)
                print(f"  Processed: blocked={result.blocked}")

        return web.Response(status=200)

    # --- Start server --------------------------------------------------------
    app = web.Application()
    app.router.add_post("/api/messages", handle_messages)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 3978)
    print("Bot listening on http://0.0.0.0:3978/api/messages")
    await site.start()

    # Keep running
    import asyncio

    await asyncio.Event().wait()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

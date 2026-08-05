# Transport Providers

Transport providers handle sending and receiving messages over external protocols. Each provider implements a channel-specific ABC.

## SMS

### Twilio

```python
from roomkit import RoomKit, SMSChannel
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig

sms = SMSChannel("sms-twilio", provider=TwilioSMSProvider(TwilioConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+15551234567",
)))

kit = RoomKit()
kit.register_channel(sms)
```

### Telnyx

```python
from roomkit.providers.telnyx.sms import TelnyxSMSProvider
from roomkit.providers.telnyx.config import TelnyxConfig

sms = SMSChannel("sms-telnyx", provider=TelnyxSMSProvider(TelnyxConfig(
    api_key="KEY...",
    from_number="+15551234567",
)))
```

### Sinch

```python
from roomkit.providers.sinch.sms import SinchSMSProvider
from roomkit.providers.sinch.config import SinchConfig

sms = SMSChannel("sms-sinch", provider=SinchSMSProvider(SinchConfig(
    service_plan_id="...",
    api_token="...",
    from_number="+15551234567",
)))
```

### VoiceMeUp

```python
from roomkit.providers.voicemeup.sms import VoiceMeUpSMSProvider
from roomkit.providers.voicemeup.config import VoiceMeUpConfig

sms = SMSChannel("sms-vmu", provider=VoiceMeUpSMSProvider(VoiceMeUpConfig(
    username="...",
    auth_token="...",
    from_number="+15551234567",
)))
```

### Webhook Parsing

Each SMS provider has a webhook parser:

```python
from roomkit.providers.twilio.sms import parse_twilio_webhook
from roomkit.providers.telnyx.sms import parse_telnyx_webhook
from roomkit.providers.sinch.sms import parse_sinch_webhook
from roomkit.providers.voicemeup.sms import VoiceMeUpSMSProvider  # use provider.parse_inbound()

# Or use the universal webhook parser
message = await kit.process_webhook(meta=request_data, channel_id="sms-twilio")
```

## RCS

```python
from roomkit import RCSChannel
from roomkit.providers.twilio.rcs import TwilioRCSProvider, TwilioRCSConfig

rcs = RCSChannel("rcs-main", provider=TwilioRCSProvider(TwilioRCSConfig(
    account_sid="AC...",
    auth_token="...",
    messaging_service_sid="MG...",  # Required for RCS (must be RCS-enabled)
)))
```

Also available via Telnyx: `TelnyxRCSProvider`, `TelnyxRCSConfig`.

## Email

### Elastic Email

```python
from roomkit import EmailChannel
from roomkit.providers.elasticemail.email import ElasticEmailProvider
from roomkit.providers.elasticemail.config import ElasticEmailConfig

email = EmailChannel("email-main", provider=ElasticEmailProvider(ElasticEmailConfig(
    api_key="...",
    from_email="support@example.com",
    from_name="Support Team",
)))
```

### SendGrid

```python
from roomkit.providers.sendgrid.email import SendGridEmailProvider
from roomkit.providers.sendgrid.config import SendGridConfig

email = EmailChannel("email-sg", provider=SendGridEmailProvider(SendGridConfig(
    api_key="SG...",
    from_email="support@example.com",
)))
```

## WhatsApp

### Business API (Cloud)

```python
from roomkit import WhatsAppChannel
from roomkit.providers.whatsapp.base import WhatsAppProvider

whatsapp = WhatsAppChannel("wa-business", provider=WhatsAppProvider(
    access_token="...",
    phone_number_id="...",
))
```

### Personal (neonize)

```python
from roomkit import WhatsAppPersonalChannel
from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider

whatsapp = WhatsAppPersonalChannel("wa-personal", provider=WhatsAppPersonalProvider())
```

Requires `pip install roomkit[whatsapp-personal]`. Uses the neonize library for multidevice protocol with typing indicators, read receipts, and media handling.

## Facebook Messenger

```python
from roomkit import MessengerChannel
from roomkit.providers.messenger.facebook import FacebookMessengerProvider
from roomkit.providers.messenger.config import MessengerConfig

messenger = MessengerChannel("messenger", provider=FacebookMessengerProvider(MessengerConfig(
    page_access_token="...",
    app_secret="...",
    verify_token="...",
)))
```

Webhook parser: `parse_messenger_webhook(request_data, channel_id="messenger")` — returns `list[InboundMessage]`.

## Telegram

```python
from roomkit import TelegramChannel
from roomkit.providers.telegram.bot import TelegramBotProvider
from roomkit.providers.telegram.config import TelegramConfig

telegram = TelegramChannel("telegram", provider=TelegramBotProvider(TelegramConfig(
    bot_token="123456:ABC-DEF...",
)))
```

Webhook parser: `parse_telegram_webhook(request_data, channel_id="telegram")` — returns `list[InboundMessage]`.

- Inbound media (`photo`, `voice`, `audio`, `video_note`, `video`, `document`) parses to a `TextContent` whose body is the caption — empty when there is none, as on a voice note — plus `metadata["file_id"]` and `metadata["media_type"]`, and whichever of `duration`, `mime_type`, `file_name`, `file_size` Telegram sent.
- `parse_telegram_message(msg)` is the layer below: it reads a Telegram `message` into `TelegramMessageParts` (content, metadata, `message_id`, `sender_id`) and attributes nothing. `parse_telegram_webhook` is that function plus the ordinary attribution — the sender is `message.from.id`. Use the lower one when your identity model is not Telegram's (a one-bot-per-user deployment attributes a DM to the bot's owner), so that reading a `file_id` never costs you your identity model.
- Resolving that `file_id` to bytes belongs to the provider, which holds the bot token: `path = await provider.get_file(file_id)` then `data = await provider.download_file(path)`. Both return `None` on failure and log a warning that never carries the URL (every Bot API URL embeds the token). Telegram caps Bot API downloads at 20 MB and refuses larger files at the `getFile` step; `metadata["file_size"]` tells you before you spend the call.
- RoomKit stops at the bytes — transcription and storage are the application's call.
- `parse_telegram_message` also gives `entities` (a caption's markup comes from `caption_entities`), `reply_to_message_id` and `media_group_id`. None of them reach the `InboundMessage` — `parse_telegram_webhook`'s metadata is unchanged.

`TelegramBotProvider` is `TelegramBotAPI` plus the rendering of a `RoomEvent`. The API half is the Bot API surface an application needs around its sends, so it never writes a second HTTP client for the same token:

```python
me = await provider.get_me()                       # metadata["result"] = the bot object
if not me.success:
    ...                                            # me.error, me.metadata["description"]
await provider.set_webhook(url, secret=secret, allowed_updates=["message", "callback_query"])
await provider.answer_callback_query(cq.id, "Approved.")
await provider.edit_message_text(chat, msg_id, "Approved", reply_markup={"inline_keyboard": []})
```

Also `get_updates`, `delete_webhook`, `leave_chat`, `send_message` (plain text, no Markdown pass), `send_force_reply` (its `provider_message_id` is what a later reply is matched against), `send_chat_action` and `edit_message_reply_markup`. Every call answers with a `ProviderResult`: `telegram_<code>` / `http_<status>` / `timeout`, and Telegram's own words under `metadata["description"]`.

Update reading is two levels. `parse_telegram_update(payload)` says which form arrived — a `message`, an `edited_message` (same shape, `edited=True`), or a `callback_query` parsed into a `TelegramCallback` (`id`, `data`, `sender_id`, `chat_id`, `message_id`, `message_text`). `callback_data` is posted by whoever pressed the button, so treat what it names as a claim to check.

`mentions_bot(msg, bot_username=..., bot_id=...)` says whether a group message addressed the bot — a reply to the bot, `bot_command`, `mention`, `text_mention`, or the handle as plain text. It gives the fact; whether that group answers only when addressed is your policy. `entity_text(text, entity)` slices by an entity's offsets, which count **UTF-16 code units** — a code-point slice returns the wrong substring as soon as an emoji precedes the mention.

## Discord

Discord has no webhook parser — it is a source + provider pair sharing one persistent gateway connection. `DiscordGatewaySource` owns the `discord.Client` (inbound); `DiscordBotProvider` reuses that client for outbound sends.

```python
from roomkit import DiscordChannel, RoomKit
from roomkit.providers.discord import DiscordBotProvider, DiscordConfig
from roomkit.sources.discord import DiscordGatewaySource

config = DiscordConfig(
    bot_token="...",               # SecretStr
    intents_message_content=True,  # privileged intent — enable in the Developer Portal
    ignore_bots=True,              # drop inbound messages authored by other bots
)
source = DiscordGatewaySource(config, channel_id="discord-main")
provider = DiscordBotProvider(source)  # sends through the source's client

kit = RoomKit()
kit.register_channel(DiscordChannel("discord-main", provider=provider))
await kit.attach_source("discord-main", source, auto_restart=True)  # connects the gateway
```

Requires `pip install roomkit[discord]` (discord.py). The Message Content intent is privileged: enable it under Bot > Privileged Gateway Intents in the Discord Developer Portal, or every inbound `message.content` arrives empty.

- Recipient key `discord_channel_id` resolves the target Discord channel snowflake at delivery time.
- Capabilities: text + rich + media, `max_length=2000`, threading and reactions. `RichContent` is sent as an embed; `MediaContent` with an http(s) URL rides in the message content (Discord auto-embeds), a `data:` URI is decoded and uploaded as a file.
- Threading: outbound `channel_data.thread_id` (a message snowflake) becomes a reply reference; inbound reply references become `InboundMessage.thread_id`.
- Inbound parsing: `parse_discord_message(message, channel_id, bot_user_id=..., ignore_bots=True)` returns `InboundMessage | None` — the bot's own messages (and other bots, by default) are dropped, so echo hooks never loop. Metadata carries `guild_id`, `channel_id`, `channel_name`, `author_name`, `author_bot`, `message_id`. Override via `DiscordGatewaySource(..., parser=...)`.
- Reactions: `provider.send_reaction(channel_id, message_id, emoji)` outbound; inbound reaction add/remove events reach the source's `on_event` callback as normalized dicts (`action`, `emoji`, `user_id`, `message_id`, `channel_id`) — outside the message pipeline.
- Testing: `MockDiscordProvider` records `sent` and `reactions` without the `discord` dependency. ABC: `DiscordProvider`.

Runnable example: `examples/discord_bot.py`.

## Buzz (Nostr)

Buzz (Block's Nostr-based team workspace) follows the same source + provider pairing: `BuzzRelaySource` owns a `buzzkit.BuzzClient` — NIP-42 authentication plus a realtime channel subscription — and `BuzzProvider` reuses that client for outbound sends over the relay's HTTP bridge, so one Nostr identity serves both directions.

```python
from roomkit import BuzzChannel, RoomKit
from roomkit.providers.buzz import BuzzConfig, BuzzProvider
from roomkit.sources.buzz import BuzzRelaySource

config = BuzzConfig(
    relay_url="wss://your-community.communities.buzz.xyz",
    private_key="nsec1...",   # agent's Nostr secret (nsec or hex) — signs events, authenticates (NIP-42)
    ignore_own=True,          # drop the agent's own events (echo guard)
    auto_join=True,           # NIP-29 self-join (kind 9000, role=bot) on connect
    announce_presence=True,   # kind 20001 "online" on connect + periodic heartbeat
    auth_tag=None,            # optional NIP-OA owner attestation (buzzkit.compute_auth_tag)
)
source = BuzzRelaySource(config, channel_id="buzz-main", relay_channel_id="<channel-uuid>")
provider = BuzzProvider(source)  # sends through the source's client

kit = RoomKit()
kit.register_channel(BuzzChannel("buzz-main", provider=provider))
await kit.create_room(room_id="buzz-room")
await kit.attach_channel("buzz-room", "buzz-main", metadata={"buzz_channel_id": "<channel-uuid>"})
await kit.attach_source("buzz-main", source, auto_restart=True)  # connects + subscribes
```

Requires `pip install roomkit[buzz]` (installs `buzzkit`, a compiled wheel kept out of the aggregate extras). Hosted Buzz communities are closed relays: the agent's key must be a member first — claim an invite once with `buzzkit`'s `claim_invite`, then copy the channel UUID from the Buzz app.

- Recipient key `buzz_channel_id` resolves the target Buzz relay channel UUID at delivery time. Each source subscribes to a single relay channel — register one source per Buzz channel and bind each to its room.
- Capabilities: text only, `max_length=65536`, threading and reactions advertised.
- Inbound parsing: `parse_buzz_event(event, channel_id, own_pubkey=..., ignore_own=True)` converts a kind-9 Nostr event dict to an `InboundMessage` — sender pubkey becomes `sender_id`, the Nostr event id becomes `external_id` and `idempotency_key`, metadata carries `nostr_event_id`, `nostr_kind`, `buzz_channel_id`. Subscribe to other event kinds with `BuzzRelaySource(..., kinds=[...], parser=...)`.
- `provider.send(event, to=channel_uuid)` publishes a kind-9 channel message signed with the agent's key; the returned `ProviderResult.provider_message_id` is the Nostr event id. HTTP-bridge sends succeed even while the inbound WebSocket is mid-reconnect; the source reconnects with exponential backoff (1 s doubling to a 30 s cap).
- Testing: `MockBuzzProvider` records `sent` without the `buzzkit` dependency. ABC: `BuzzRelayProvider`.
- Presence: with `announce_presence=True` the source publishes kind-20001 `"online"` on connect, heartbeats every 30 s (surviving transient publish failures), and publishes `"offline"` on a deliberate `stop()` so the agent's dot flips immediately instead of lapsing by relay TTL.
- Owner control commands (`buzzkit>=0.3.0`): with `obey_owner_commands=True` (default), a kind-9 message whose trimmed content is exactly `!shutdown` / `!cancel` / `!rotate`, mentioning the agent and authored by the **proven** owner — the NIP-OA auth tag's Schnorr-verified attester, else `BuzzConfig.owner_pubkey` — is consumed before the pipeline (the AI never answers its own stop command). `!shutdown` stops the source gracefully; all commands reach the optional `on_owner_command` callback, which takes over the response when provided. No provable owner, or a non-owner author → the message flows normally (fail-closed). Replay-safe: a command issued before the source started is stale (relays replay recent history on every subscribe) — consumed without action; one issued during a disconnection is honored when the reconnect replays it.
- Inbound metadata carries `nostr_created_at` (unix seconds, the Nostr timestamp) so apps can tell live traffic from relay-history replay — used by `examples/buzz_agent.py`'s echo guard.

Runnable example: `examples/buzz_bot.py`.

### BuzzAgent — the lifecycle runner

`BuzzAgent` (`roomkit.providers.buzz`) turns a configured RoomKit app into a first-class Buzz agent: it attaches the sources (taking over their `on_owner_command`), installs SIGTERM/SIGINT handlers, optionally arms an `exit_after_inactivity` bound (seconds; default off; reaper on its own timer), and exits every stop cause — owner `!shutdown`, signal, inactivity — through one graceful path: `kit.close()` (presence `offline`, sockets closed). `run()` is single-shot, consumes the kit, and returns a `BuzzAgentStopCause` (`owner_shutdown` / `signal` / `inactivity`); exit the process with code 0 so supervisors never restart an intentional stop.

```python
from roomkit.providers.buzz import BuzzAgent, BuzzConfig

config = BuzzConfig.from_env()   # BUZZ_PRIVATE_KEY (or NOSTR_PRIVATE_KEY) / BUZZ_RELAY_URL / BUZZ_AUTH_TAG — fail-closed
agent = BuzzAgent(kit, [source], exit_after_inactivity=7200)
cause = await agent.run()        # blocks until the owner, a signal, or idleness stops it
```

Runnable example: `examples/buzz_agent.py`.

Buzz huddles (live voice calls in ephemeral channels) are handled by the voice subsystem, not this transport: `BuzzHuddleBackend` (`roomkit.voice.backends.buzz_huddle`) is a `VoiceBackend` carrying huddle Opus audio for a `RealtimeVoiceChannel`, and `BuzzHuddleWatcher` owns the announcement-to-call lifecycle, watching the parent channel for kind-48100 announcements (`KIND_HUDDLE_STARTED` / `huddle_announcement_parser` in `roomkit.sources.buzz`) and bridging each huddle. See `examples/buzz_voice_agent.py`.

## Microsoft Teams

```python
from roomkit import TeamsChannel
from roomkit.providers.teams.bot_framework import BotFrameworkTeamsProvider
from roomkit.providers.teams.config import TeamsConfig

teams = TeamsChannel("teams", provider=BotFrameworkTeamsProvider(TeamsConfig(
    app_id="...",
    app_password="...",
)))
```

Features: proactive messaging, bot mention detection, reaction handling, conversation reference storage.

```python
from roomkit.providers.teams.webhook import parse_teams_webhook, is_bot_added

# Parse incoming Teams activity
activity = parse_teams_webhook(request_data)

# Check if bot was added to a conversation
if is_bot_added(activity):
    # Handle bot installation
    pass
```

## HTTP (Generic Webhook)

```python
from roomkit import HTTPChannel
from roomkit.providers.http.provider import WebhookHTTPProvider
from roomkit.providers.http.config import HTTPProviderConfig

http = HTTPChannel("webhook", provider=WebhookHTTPProvider(HTTPProviderConfig(
    url="https://api.example.com/messages",
    headers={"Authorization": "Bearer ..."},
)))
```

## WebSocket

WebSocket channels don't use a provider — they handle connections directly:

```python
from roomkit import WebSocketChannel

ws = WebSocketChannel("ws-client")

# Register a connection — room_id says which conversation this socket is for
ws.register_connection("conn-1", on_receive_callback, room_id="room-1")

# In production, connect to the framework
await kit.connect_websocket("ws-client", "conn-1", send_fn, room_id="room-1")
await kit.disconnect_websocket("ws-client", "conn-1")
```

One channel instance can serve several rooms, so a connection has to say which
one it belongs to: the channel delivers a room's events only to the
connections registered for it. A client holding several conversations open on
one socket subscribes to the extra rooms rather than opening more sockets:

```python
kit.subscribe_websocket("ws-client", "conn-1", "room-2")
kit.unsubscribe_websocket("ws-client", "conn-1", "room-2")
```

## Phone Number Utilities

```python
from roomkit.providers.sms.phone import is_valid_phone, normalize_phone

is_valid_phone("+15551234567")   # True
normalize_phone("555-123-4567")  # "+15551234567"
```

## Delivery Status Tracking

Track delivery status for sent messages:

```python
from roomkit import DeliveryStatus

@kit.on_delivery_status
async def track_delivery(status: DeliveryStatus) -> None:
    if status.status == "failed":
        print(f"Message {status.message_id} failed: {status.error_message}")

# Process status webhooks from providers
await kit.process_delivery_status(status)
```

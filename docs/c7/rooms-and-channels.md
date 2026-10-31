# Rooms and Channels

## Creating Rooms

```python
from roomkit import RoomKit

kit = RoomKit()

# Auto-generated ID
room = await kit.create_room()

# Explicit ID with metadata
room = await kit.create_room(
    room_id="support-123",
    metadata={"topic": "billing", "priority": "high"},
)
```

A room is created once (RFC §14.3): `create_room()` with an id the store
already holds raises `RoomAlreadyExistsError` and leaves the room as it was.
Read the room first (`get_room()`) when the id may exist. Room ids are global,
across organizations: the refusal comes whichever organization holds the id,
and says only that it is taken. A host that lets its tenants choose room ids
namespaces them (`f"{tenant}:{name}"`) or lets RoomKit generate them, since a
refused id is one another tenant may hold.

## Room Lifecycle

```python
from roomkit import RoomKit

kit = RoomKit()

# Create room
room = await kit.create_room(room_id="session-1")

# Update metadata
await kit.update_room_metadata("session-1", {"status": "escalated"})

# Close a room
await kit.close_room("session-1")

# Check timers (call periodically, e.g. every 60s)
transitioned = await kit.check_all_timers()
```

A transition the room has already made does nothing: `close_room()` on a
CLOSED room, and `archive_room()` on an ARCHIVED one, return the room as
stored, writing and firing nothing. No call takes a room out of ARCHIVED:
`close_room()` on an archived room raises `TransitionRefusedError` (`room_id`,
`status`, `transition`), nothing written. Timers (`RoomTimers`, `paused_at`)
are described under Resilience: every operation on a room applies its due
timer transitions first, and a delivered message resumes a PAUSED room.

## Channel Types

The `ChannelType` enum defines 24 values: `SMS`, `MMS`, `RCS`, `EMAIL`, `WHATSAPP`, `WHATSAPP_PERSONAL`, `WEBSOCKET`, `AI`, `VOICE`, `REALTIME_VOICE`, `REALTIME_AUDIO_VIDEO`, `PUSH`, `MESSENGER`, `TELEGRAM`, `TEAMS`, `DISCORD`, `BUZZ`, `WEBHOOK`, `VIDEO`, `AUDIO_VIDEO`, `CONFERENCE`, `CLI`, `HOST`, `SYSTEM`. RoomKit ships with these factories and channel classes:

| Channel | Factory / Class | Category | Use Case |
|---------|----------------|----------|----------|
| SMS | `SMSChannel(id, provider=...)` | TRANSPORT | Text messages via Twilio, Telnyx, Sinch, VoiceMeUp |
| RCS | `RCSChannel(id, provider=..., fallback=True)` | TRANSPORT | Rich messaging via Twilio, Telnyx |
| Email | `EmailChannel(id, provider=...)` | TRANSPORT | Email via ElasticEmail, SendGrid |
| WhatsApp | `WhatsAppChannel(id, provider=...)` | TRANSPORT | WhatsApp Business API |
| WhatsApp Personal | `WhatsAppPersonalChannel(id, provider=...)` | TRANSPORT | WhatsApp via neonize |
| Messenger | `MessengerChannel(id, provider=...)` | TRANSPORT | Facebook Messenger |
| Telegram | `TelegramChannel(id, provider=...)` | TRANSPORT | Telegram Bot API |
| Teams | `TeamsChannel(id, provider=...)` | TRANSPORT | Microsoft Teams Bot Framework |
| Discord | `DiscordChannel(id, provider=...)` | TRANSPORT | Discord bot (recipient key `discord_channel_id`) |
| Buzz | `BuzzChannel(id, provider=...)` | TRANSPORT | Nostr relay channels (recipient key `buzz_channel_id`) |
| Webhook | `HTTPChannel(id, provider=...)` | TRANSPORT | Generic HTTP webhook (`ChannelType.WEBHOOK`) |
| WebSocket | `WebSocketChannel(id)` | TRANSPORT | Real-time bidirectional |
| CLI | `CLIChannel(id)` | TRANSPORT | Interactive terminal (stdin/stdout REPL) |
| Host | `HostChannel(channel_id="host")` | TRANSPORT | The host application's own voice, written with `kit.post()` (INBOUND, never delivered to) |
| AI | `AIChannel(id, provider=...)` | INTELLIGENCE | LLM responses |
| Agent | `Agent(id, provider=...)` | INTELLIGENCE | Agent with tools + greeting |
| ACP | `ACPChannel(id, command, cwd=...)` | INTELLIGENCE | External coding agent via Agent Client Protocol (`channel_type` is `AI`; one session per room) |
| Voice | `VoiceChannel(id, stt=..., tts=..., backend=...)` | TRANSPORT | Real-time audio |
| Realtime Voice | `RealtimeVoiceChannel(id, provider=...)` | TRANSPORT | Speech-to-speech AI |
| Video | `VideoChannel(id, backend=..., vision=...)` | TRANSPORT | Video track with vision analysis |
| Audio+Video | `AudioVideoChannel(id, backend=..., stt=..., tts=...)` | TRANSPORT | Combined voice + video |
| Realtime Audio+Video | `RealtimeAudioVideoChannel(id, provider=..., transport=...)` | TRANSPORT | Speech-to-speech AI with video |
| Conference | `ConferenceChannel(id, backend=...)` | TRANSPORT | Multi-party SFU conference with AI bot (LiveKit or mock backend) |

`MMS`, `PUSH`, and `SYSTEM` have no factory: `MMS` is set automatically when an SMS channel carries media, `SYSTEM` marks framework-generated events, and `PUSH` is reserved.

## Registering and Attaching Channels

Channels are registered globally, then attached to specific rooms:

```python
from roomkit import AIChannel, ChannelCategory, RoomKit, SMSChannel
from roomkit.providers.twilio import TwilioConfig, TwilioSMSProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

kit = RoomKit()

# Create and register channels
sms = SMSChannel("sms-main", provider=TwilioSMSProvider(TwilioConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+1234567890",
)))
ai = AIChannel("ai-agent", provider=AnthropicAIProvider(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-opus-5",
)))

kit.register_channel(sms)
kit.register_channel(ai)

# Create room and attach
await kit.create_room(room_id="support-room")
await kit.attach_channel("support-room", "sms-main")
await kit.attach_channel("support-room", "ai-agent", category=ChannelCategory.INTELLIGENCE)
```

## Channel Access Levels

Control what a channel can do in a room:

```python
from roomkit import Access

# Default: read and write
await kit.attach_channel("room", "channel", access=Access.READ_WRITE)

# Read only: receives messages but cannot send
await kit.attach_channel("room", "channel", access=Access.READ_ONLY)

# Write only: sends messages but doesn't receive broadcasts
await kit.attach_channel("room", "channel", access=Access.WRITE_ONLY)

# None: temporarily disabled
await kit.set_access("room", "channel", Access.NONE)
```

A message from a channel that cannot write (read-only, `NONE`, muted) is
stored `BLOCKED`, as is one a hook blocks, with the visibility it would have
been delivered with: its own intersected with its channel's binding (RFC
§7.3). A host reading the room's refused records sees no more of one than
its channels would have seen of the message.

## Where Answers Go

`InboundMessage(response_visibility=...)` and `send_event(...,
response_visibility=...)` say where the answers to an event may go, in the
visibility vocabulary (RFC §5.12): `None` restricts nothing and any other value
is a scope. Both doors raise `ValueError` for `""` and for `internal`; an empty
scope that reaches an event another way (a hook's modify) names nobody. Every
answer the message's cascade derives from it, an agent's answer to another
agent's answer, a stream's rows and the record stored in place of an answer
included, is what its channel gave it intersected with the scope; a response's own
`response_visibility` holds within it. A `BEFORE_BROADCAST` hook may narrow an
event's `response_visibility`, never widen or clear it (`internal` reads as
`none`), and a response its hooks widen is narrowed back within the scope.

The scope binds the cascade only. Work that leaves it starts a new cascade the
scope does not follow: a background delegation's result handed back with
`deliver()` or `send_event()`, a tool's own `send_event()`, a child room's
answer to the channels it shares. Give such a call its own
`response_visibility`.

## Writing as the Host

What the application says to a room's people — a notice, a workflow's
question, a reminder — is the host's own words, written with `kit.post()`
(RFC §10.5):

```python
await kit.post(room_id, "Your order shipped.")          # texted to the correspondent
await kit.post(room_id, "Summarise the order.", addressed_to=["assistant"])
await kit.post(room_id, "Back at 3 pm.", participant_id="desk", metadata={"job": "hours"})
await kit.post(room_id, "Refund issued.", idempotency_key=webhook_delivery_id)
```

- The post comes from a `HostChannel`, a transport of type `HOST` whose
  direction is `INBOUND`: the room reads it as a party's message and
  broadcasts it to its transports. It is never delivered to, and it narrows
  nothing an agent's answer may carry: it declares every media type, and the
  room's other transports decide what an answer uses.
- It solicits no agent unless `addressed_to` names one. To let the room's
  routing decide, use `send_event()` from the host channel instead.
- `participant_id` names who speaks: once two people take part, an agent
  reads the turn as `Acme Support: "..."`. The speaker is a participant the
  host channel the post comes from reaches (their `channel_id`, or
  `connected_via`, names it), by their id or their identity's: an id names
  its participant only on the channels they are reached through. Add a
  member on the host channel with its id, `"host"` for the kit's own:

  ```python
  await kit.add_member(room_id, "host", "desk", display_name="Acme Support")
  ```

  Any other id (nobody's, a member reached through SMS only, an agent
  channel's) raises `ParticipantNotFoundError` before anything is written.
  `send_event()` takes any id, and shows a name only on its channel.
- `chain_depth` names the chain the post continues, as `send_event()`'s
  does: 0, the default, opens one. Work that follows an agent's answer from a
  task (a hook's, say) passes the answer's depth plus one, so its round trips
  with the agent end at `max_chain_depth`.
- `metadata` says where it comes from (a workflow, its step, a job).
- `idempotency_key`: a key the host channel holds in the room is a replay. The
  post returns the event the key committed and texts nothing again, and from
  the room's own work it draws no pass. A blank key is no key. A webhook
  delivered twice posts once. A replay into a room closed since, or after the
  host channel was detached, raises `RoomClosedError` or
  `ChannelNotFoundError`, as `send_event()` does: the room's status and
  binding are read before the key (RFC §10.1).
- The kit registers its host channel under `"host"` on first use. `"host"`
  taken by a channel of another type raises `ValueError`; `channel_id=` then
  speaks from another `HostChannel` you registered, and a channel of another
  type is refused before anything is written.
- A room that never bound the host channel gets it attached on its first
  post, decided under the room lock: a binding you set up, before or
  concurrently, keeps its settings. A room you detached the host channel from
  is refused with `ChannelNotFoundError`, and a closed or archived room with
  `RoomClosedError`; nothing is written. `attach_channel()` grants it again.
  One `process_timeout` bounds the whole call, the wait for the room lock
  included (`ProcessTimeoutError`); once the lock is held, the attach runs to
  its end even if the caller is cancelled, and stays when the deadline passes
  during it, the message then not written. A first post a hook or a turn
  makes past its reentry budget is stored `BLOCKED` with `reentry_loop_cap`
  and binds nothing: the post draws its pass before the attach.
- A host channel is attached as a transport only: `category=INTELLIGENCE`
  raises `ValueError`. `process_inbound()` on it raises `ValueError` before
  anything is read: it receives no external payload.
- `deliver()` is the other way in: content for the room's agents, never shown
  to the correspondent.

The host's words come from neither an intelligence channel, which would make
them an agent's output (a Discussion refuses to bind one that is not its
agent, and `ADDRESSED_ONLY` lets an unaddressed one solicit no agent), nor a
`SYSTEM` source, which marks what the framework writes: no author, and
changes reserved to the system.

## Muting Channels

Muting suppresses a channel's output (AI responses) without detaching it. Side effects (tasks, observations) still fire.

```python
# Mute AI output
await kit.mute("room", "ai-agent")

# Unmute
await kit.unmute("room", "ai-agent")

# Mute only the output direction (AI won't respond, but still sees messages)
await kit.mute_output("room", "ai-agent")
await kit.unmute_output("room", "ai-agent")
```

A binding change that changes nothing writes and fires nothing (RFC §7.2):
muting a muted channel, or a setter given the values the binding holds,
returns the binding as stored with no record and no `ON_CHANNEL_MUTED` /
`ON_CHANNEL_UNMUTED`. A gate change (`mute`, `unmute`, `mute_output`,
`unmute_output`, `set_access`) still calls `Channel.update_binding()` with the
binding as stored, so a channel whose cached state drifted is set right.

An intelligence channel unregistered while still bound is reported, when the
event solicits it, as `delivery_failed` with the error
`channel_not_registered` (not retryable, no `ON_ERROR`); a transport not
registered in this process is skipped silently.

## Per-Room Channel Configuration

Pass metadata when attaching to customize per-room behavior:

```python
await kit.attach_channel(
    "weather-room",
    "ai-agent",
    category=ChannelCategory.INTELLIGENCE,
    metadata={
        "system_prompt": "You are a weather assistant.",
        "temperature": 0.3,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        ],
    },
)
```

## Content Types

Events carry typed content. RoomKit supports 11 content types, all discriminated unions on the `type` field:

| Type | `type` Literal | Key Fields | Use Case |
|------|---------------|-----------|----------|
| `TextContent` | `"text"` | `body`, `language` | Plain text messages |
| `RichContent` | `"rich"` | `body`, `format` (html/markdown), `plain_text`, `buttons`, `cards`, `quick_replies` | Formatted messages with UI elements |
| `MediaContent` | `"media"` | `url`, `mime_type`, `filename`, `size_bytes`, `caption` | Images, documents, files |
| `AudioContent` | `"audio"` | `url`, `mime_type`, `duration_seconds`, `transcript` | Voice messages |
| `VideoContent` | `"video"` | `url`, `mime_type`, `duration_seconds`, `thumbnail_url` | Video messages |
| `LocationContent` | `"location"` | `latitude`, `longitude`, `label`, `address` | Geographic coordinates |
| `CompositeContent` | `"composite"` | `parts` (list of EventContent, max depth 5, a sixth level refused wherever content enters the kit) | Multi-part messages |
| `TemplateContent` | `"template"` | `template_id`, `language`, `parameters`, `body` | WhatsApp Business / RCS templates |
| `SystemContent` | `"system"` | `body`, `code`, `data` (dict) | System-generated events |
| `EditContent` | `"edit"` | `target_event_id`, `new_content`, `edit_source` | Message edits |
| `DeleteContent` | `"delete"` | `target_event_id`, `delete_type`, `reason` | Message deletion |

```python
from roomkit.models import (
    AudioContent,
    CompositeContent,
    DeleteContent,
    EditContent,
    LocationContent,
    MediaContent,
    RichContent,
    SystemContent,
    TemplateContent,
    TextContent,
    VideoContent,
)

# Plain text
TextContent(body="Hello!")

# Rich content with buttons and Markdown
RichContent(body="Choose:", format="markdown", plain_text="Choose:", buttons=[{"text": "A", "payload": "a"}])

# Media, Audio, Location, Template
MediaContent(url="https://example.com/image.png", mime_type="image/png", caption="Photo")
AudioContent(url="https://example.com/voice.ogg", mime_type="audio/ogg", transcript="Hello there")
LocationContent(latitude=40.7128, longitude=-74.0060, label="NYC Office")
TemplateContent(template_id="order_confirm", language="en", parameters={"1": "ORD-123"})

# Multi-part (max depth 5: a message, a parsed event, a hook's result, an
# answer or a send_event() holding a sixth level is refused, nothing stored)
CompositeContent(parts=[
    TextContent(body="Here's the photo:"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg"),
])

# Edit and Delete
EditContent(target_event_id="evt-abc", new_content=TextContent(body="Corrected text"))
DeleteContent(target_event_id="evt-abc", delete_type="sender", reason="User retracted")
```

## Inbound Routing and Addresses

A channel's type decides how its addresses are written and where its
recipient and its chat are (RFC §5.7, Appendix A).
`roomkit.channels.addressing.addressing_of(channel_type)` gives the defaults a
`TransportChannel` takes unless it declares its own
(`TransportChannel(..., addressing=ChannelAddressing(...), country_code=...)`):

| Type | Address | Recipient key | Chat key |
|---|---|---|---|
| SMS, MMS, RCS, WHATSAPP | E.164 | `phone_number` | — |
| WHATSAPP_PERSONAL | E.164 | `phone_number` | `group_jid` (a group message's group) |
| VOICE | E.164 | — | — |
| EMAIL | lower-case, no display name | `email_address` | — |
| MESSENGER | as written | `facebook_user_id` | — |
| TELEGRAM | its chat | `telegram_chat_id` | `chat_id` |
| TEAMS | its chat | `teams_conversation_id` | `conversation_id` |
| any other | as written | — | — |

Discord (`discord_channel_id` / `channel_id`) and Buzz (`buzz_channel_id`)
declare chat keys of their own; a webhook forwards a binding's `recipient_id`
to its provider, which names no one. A recipient, and a chat, is a string
that is not blank or an integer (a chat id). A binding with the key absent,
null or a blank string holds no recipient: routing reads no correspondent in
it, so a room opened for no one keeps its first routed sender, whose address
is written there. Any other value (`true`, a float, a list, an object) is not
a recipient, and is not "no recipient" either: routing reads it as someone
else's address. On a binding naming no participant and not `group`, no
routed sender is admitted through it or recorded on it; a binding naming its
correspondent still routes them there, and a `group` binding admits everyone.
The value is never written over, and routing logs a warning naming the
binding and the value's type. A delivery to either is refused
`no_recipient`. Under a message's chat key, either is no chat. On a channel
whose replies go to a chat only, a message that does not carry its chat has no
address: it records no one, and a room opened for it names neither
correspondent nor recipient (deliveries refused `no_recipient` until the host
names one).

A message without a `room_id` is routed by the default strategy (RFC §10.4),
scoped to the caller's organization:

1. The newest ACTIVE or PAUSED room whose binding of the channel names the
   sender: its correspondent (`participant_id`) is the sender's address or
   the identity the store resolves it to, or, naming none and not `group`,
   its recipient is the address. Otherwise the newest ACTIVE or PAUSED room
   holding a participant who is the sender reached through the channel
   (`connected_via`). The newest room either name finds wins, whatever its
   status: a paused conversation keeps its correspondent, and their next
   delivered message resumes it.
2. Otherwise the one room step 3 reads, when it admits the sender: the
   ACTIVE rooms bound to the channel and the PAUSED ones whose binding is
   `group`, "exactly one" counting both.
3. Otherwise a new room.

The store answers step 1 with `find_room_id_by_binding(channel_id,
participant_id, identity_id, room_filter, *, recipient_key=None)` and
`find_room_id_by_participant(channel_id, participant_id, identity_id,
room_filter)`, step 3 with `find_room_ids_by_channel(channel_id, room_filter,
limit, *, group=None)`; `RoomFilter(statuses, organization_id)` names the
rooms read. A delivery status that names no room goes to the ACTIVE or PAUSED
room whose binding names its recipient (in the channel's form), else the one
ACTIVE room unless its binding names another correspondent (its participant,
else its recipient), and resumes nothing; `process_delivery_status(status, *,
organization_id=None)` and `process_webhook(..., organization_id=...)` read
that organization's rooms only.

Step 2's attach of a room in hand that does not bind the channel runs under
the room lock once the room's due timers apply, at the channel's defaults,
never to a CLOSED or ARCHIVED room, never over a binding made by then (a
creation hook's included) and never of a detached channel.

A keyed `kit.deliver()` names one publication in its room, whatever transport
carries it (RFC §22.1): the key is looked up within the room before a
transport is resolved, and the lookup, the resolution and the commit run in
one section under the room lock; a replay returns the original outcome. The
publication is stored under `roomkit:publication:<key>`, a namespace only the
delivery path writes: a message's key in it is dropped, `send_event()`
refuses one, and a sender's message keyed like a coming publication never
stands in for it.

`kit.deliver()`, `send_greeting()`, `regenerate_target()`,
`regenerate_response()` and `delegate()` take `organization_id=` (RFC
§17.2): another organization's room raises `RoomNotFoundError` before any
hook, queue, turn or child room. Unset, no room is read for the scope.

## Content Transcoding

When broadcasting events, the EventRouter automatically **transcodes** content to match each target channel's capabilities. A channel that only supports TEXT will receive a text fallback of a RichContent message.

### How Transcoding Works

1. Event broadcast starts → EventRouter iterates over all target channel bindings
2. For each target, call `ContentTranscoder.transcode(content, source_binding, target_binding)`
3. Transcoder checks `target_binding.capabilities.media_types` to decide if content passes through or needs conversion; a file (`MediaContent`) also needs a MIME type a non-empty `supported_media_types` lists (compared without parameters, whatever the case) and, when its `size_bytes` is known, a size within `max_media_size_bytes`, else it is delivered as its text form (RFC §5.8, §6.6): never dropped, never sent as is for the provider to refuse. An empty list takes every type; an unknown size, any size
4. If transcode returns `None` → delivery is skipped for that channel

### Channel Capabilities

Each channel binding declares what content types it supports via `ChannelCapabilities`:

```python
from roomkit.models import ChannelCapabilities, ChannelMediaType

# SMS channel: text only, 160 char limit
sms_caps = ChannelCapabilities(
    media_types=[ChannelMediaType.TEXT],
    max_length=160,
)

# WebSocket channel: rich content, media, audio, video
ws_caps = ChannelCapabilities(
    media_types=[
        ChannelMediaType.TEXT, ChannelMediaType.RICH, ChannelMediaType.MEDIA,
        ChannelMediaType.AUDIO, ChannelMediaType.VIDEO, ChannelMediaType.LOCATION,
    ],
    supports_edit=True,
    supports_delete=True,
)
```

`ChannelMediaType` enum: `TEXT`, `RICH`, `MEDIA`, `AUDIO`, `VIDEO`, `LOCATION`, `TEMPLATE`.

### Default Fallback Chain

| Content Type | If Target Supports It | Fallback |
|-------------|----------------------|----------|
| `TextContent` | Always passes through | — |
| `RichContent` | RICH in media_types → pass | `TextContent(plain_text or body)` |
| `MediaContent` | MEDIA in media_types, its type listed (or none listed) and its known size within the limit → pass | `TextContent("[Media: {caption or filename or url}]")`, the URL only when a link, else the media type |
| `AudioContent` | AUDIO in media_types → pass | `TextContent(transcript)` or `"[Voice message: {url}]"`, the URL only when a link, else the media type |
| `VideoContent` | VIDEO in media_types → pass | `TextContent("[Video: {url}]")`, the URL only when a link, else the media type |
| `LocationContent` | LOCATION in media_types → pass | `TextContent("[Location: {label} ({lat}, {lon})]")`, each coordinate with one decimal at least and never an exponent (`45.0`, `0.00005`) |
| `CompositeContent` | Recursive transcode of all parts, none dropped | If all parts become text → flatten to single TextContent, joined by newlines, an empty part skipped |
| `TemplateContent` | TEMPLATE in media_types → pass | `TextContent(body or "[Template: {id}]")` |
| `EditContent` | `supports_edit` → pass, a file in `new_content` the target does not take as its text form | `TextContent("Correction: {text form of new_content}")`, whatever the target takes of `new_content` |
| `DeleteContent` | `supports_delete` → pass | `TextContent("[Message deleted]")` |

### Multichannel Transcoding Example

A single event gets adapted differently for each channel:

```python
# User sends a rich message with an image from WebSocket
content = CompositeContent(parts=[
    RichContent(body="**Check this out!**", format="markdown", plain_text="Check this out!"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg", caption="Sunset"),
])

# WebSocket channel: receives CompositeContent as-is (supports RICH + MEDIA)
# SMS channel: receives TextContent("Check this out!\n[Media: Sunset]")
# Voice channel (TTS): receives TextContent("Check this out!\n[Media: Sunset]")
```

## Edits and Deletes

An event is an edit when its type is `EDIT` and its content an `EditContent`,
a delete when `DELETE` and `DeleteContent` (RFC §10.3). The pipeline checks it
under the room lock on every pass that commits an event: an inbound message, an
agent's response, a hook's injected event, a `send_event()`. A refusal stores
nothing: the inbound path returns `InboundResult(blocked=True, reason=...)`,
`send_event()` raises `ChangeRefusedError` (its `reason`), a refused response or
injected change is logged at warning level.

| Rule | Refused with |
|------|--------------|
| A type and a content that disagree, or a change held elsewhere (a composite's part, an edit's new content) | `target_event_not_found` |
| The target is not an event of the room, is `BLOCKED`, deleted, or itself an edit or a delete (an edited event may be edited again) | `target_event_not_found` |
| A sender's change (`edit_source` `"sender"` or none, `DeleteType.SENDER`) of an event they did not author | `not_original_author`; `identity_required_for_edit` when either has no author |
| `DeleteType.ADMIN` from anyone but an OWNER or AGENT, neither LEFT nor BANNED, on the channel they write on | `not_authorized`; `identity_required_for_edit` with no sender id |
| A SYSTEM change (`DeleteType.SYSTEM`, `edit_source="system"`) from anything but a SYSTEM channel or a hook | `not_authorized` |
| An edit or a delete a hook injects that is not a SYSTEM change | `not_authorized` |

An event that travelled outbound (a channel's own output, an agent's response)
is its channel's, whatever participant it names: an agent edits its own
responses and no one else's. `edit_source` is an `EditSource`, `"sender"` or
`"system"`, or none: an `EditContent` naming another is refused as it is built.
A hook's `modify` keeps the target, the `edit_source` or
`delete_type` and the kind of content; once a hook ran under the lock, the
target is read and checked again, so a delete committed during an edit's hooks
refuses the edit.

An accepted change commits with its target in one store transaction: an edit
writes `new_content` and `metadata.edited=True` on the target, a delete
`metadata.deleted=True`. The change reaches no one who could not see its target
(its visibility is narrowed to the target's), it is delivered with the target as
the commit left it, and `ON_EVENT_UPDATED` / `ON_EVENT_DELETED` fire once the
lock is released. Whatever pass committed the change (an inbound message, an
agent's response, a hook's injected event), their context is the room as the
commit left it: the target as rewritten, what its hooks committed ahead of it,
and the change as the newest event, at its committed index and as its hooks
left it, with `ctx.room.event_count` and `latest_index` past it.

```python
await kit.process_inbound(
    InboundMessage(
        channel_id="ws-user",
        sender_id="alice",
        event_type=EventType.EDIT,
        content=EditContent(target_event_id=message.id, new_content=TextContent(body="50000$")),
    )
)
```

A soft-deleted event stays in every history a channel is handed, its content
withheld: an empty text in place of what was said and no `source.raw_payload`,
its id, index, thread, metadata and `channel_data` kept, so the delete that
follows names an event the reader holds. An edit of a deleted event has its new
content withheld alike: the delete's commit marks each edit of its target
`metadata.deleted` too, so it is withheld even where the history no longer holds
the message. A DELETE row alone (one the host's `commit_event()` stored) hides
nothing. `visible_events()`, the context a channel parses a
message with and the one a broadcast delivers with all hold it so; the stored
record and the contexts hooks receive keep the content.

`kit.delete_event(room_id, event_id, cascade_replies=True)` is the host's hard
delete: it removes the event, its thread replies unless told not to (the whole
thread: every path that commits an event, an agent's answer, a hook's injected
event and a `commit_event()` record included, roots its `parent_event_id` to
the thread's root), and the edits and deletes of every event it removes (an
edit holds the corrected text),
returns their ids, and fires `ON_EVENT_DELETED` once per removed event.
`kit.update_event()` rewrites an event's content, source or metadata, but for
the record of its author (`metadata["author"]`), which the room fixed at the
commit: the update keeps it, and writes none on an event committed with none,
whatever the metadata it is given holds. Neither
runs the checks above: the host owns their authorization. Neither fails once
its change is stored: when the room cannot be read after the lock is
released, a warning is logged and the trigger fires with the context the call
held under the lock. Both take
`organization_id=` (RFC §17.2): another organization's room is a missing one,
`None` or `[]` with nothing written, its due timers included.

## Detaching Channels

```python
await kit.detach_channel("room", "channel-id")
```

## Binding Metadata Updates

```python
await kit.update_binding_metadata("room", "ai-agent", {"temperature": 0.9})
```

## Querying Rooms

```python
# Get a room
room = await kit.get_room("room-id")

# List bindings
bindings = await kit.store.list_bindings("room-id")

# List participants
participants = await kit.store.list_participants("room-id")

# Query event timeline
events = await kit.get_timeline("room-id", offset=0, limit=50)
```

A participant is one shared record per `(room_id, participant_id)`, even when
`connected_via` names several channels. Its `status` is therefore global to the
record, not per-channel presence. When two channel memberships have independent
connect/disconnect lifecycles, give them distinct participant ids and correlate
them through `identity_id`.

An inbound sender id names its participant only on a channel in their
`connected_via` (RFC §5.5): the same id arriving through another channel is
another sender, with none of the participant's name, role, administrative
authority, authorship, identification or verification: routed without a room
id it gets a room of its own, not the participant's, and
`current_tool_actor_id()` reads `None` for it. Link one person's
channels with `add_member` (or `join`) on each, or let the identity pipeline
resolve them, which records the channel it resolved them on:

```python
await kit.add_member("room-id", "console", "u-1", role=ParticipantRole.OWNER)
# A message on "web" from sender id "u-1": labelled "@web", no role, no ADMIN delete.
await kit.add_member("room-id", "web", "u-1")  # records "web"; the role stays OWNER
# Now the same message on "web" is the owner's.
```

A host serving several organizations passes `organization_id=` to
`add_member`, `remove_member`, `rename_member`, `list_members` and
`is_member` (RFC §17.2): another organization's room raises
`RoomNotFoundError`, its roster neither read nor written. Unset, the call is
unscoped, as before.

## Cancelling a turn

Cancelling a message's delivery stops its agents' turns and only those: a turn
still waiting in the room's lane (or behind the room's ACP prompt, or behind
the agent's previous turn when it takes a room's turns one at a time) never
starts, a running one is cut with its tools, and nothing answers after. The
message itself is still delivered to the room's other channels and fires its
`AFTER_BROADCAST` hooks. The room's other turns go on.

Cancelling a task awaiting `kit.process_inbound()` cancels and drains that
call's tools, generation and stream consumers before propagating
`asyncio.CancelledError`. Other rooms and shared providers remain usable.
Committed events and already streamed text remain in the timeline.

With `defer_delivery=True`, explicitly call
`await result.delivery.cancel(reason="hangup", timeout=5.0)` to stop that turn.
The returned result's `cancellation_reason` names its terminal cancellation;
`await result.delivery.wait()` joins the same cleanup. Cancelling a `wait()`
alone leaves deferred work running. A completed handle is unchanged by cancel.

The default cleanup budget is five seconds on both paths. `TimeoutError`
means cleanup is incomplete: keep resources still in use, and for a deferred
call await its handle later. Repeated cancellation does not interrupt
finalizers. A handle cannot drain its own room lock or lane (`RuntimeError`).

Code without the handle names the delivery by a key the caller set on the
message, `InboundMessage(..., delivery_key="task-42")`, and cancels it from
anywhere in the process, a tool handler included:
`await kit.cancel_delivery("task-42", reason="lead_cancelled")`. It answers how
many deliveries the key named and cancelled (0 once they are over),
returns at once, and each call's caller reads `cancellation_reason` on its
result. The key names the delivery from the start of `process_inbound`: a
message cancelled before its commit (while it is routed, parsed, or its sender
resolved) is committed and delivered all the same, and its turns never start.
A message held for verification stays named until its release is answered: a
cancel while it waits counts 1, and its release gets no answer.
Several messages may share a key. The key is never stored and grants
nothing: check who may cancel before calling, and pass `organization_id=` so a
key two tenants both chose only reaches the caller's rooms. Pass it to
`process_inbound` as well: a call without it belongs to no organization until
it reads its room (right after routing), and a scoped cancel reaching it
before is applied then, uncounted. A call refused before its commit reports
its refusal, not the cancellation. A `kit.steer(Cancel(), ...)` reaches only
turns that have started; use the delivery to stop one message's turns.
External effects already committed and tasks detached by application handlers
are outside this cancellation contract. See `examples/cancel_delivery.py`.

## Closing the kit

`await kit.close()` (or leaving `async with RoomKit()`) closes the kit in one
shutdown, whoever calls it and however often:

1. **Every new call is refused at once, but the reads and the teardown
   calls.** From the moment `close()` starts, a call raises `KitClosedError`,
   a `RoomKitError`, whoever makes it: from outside, from a hook or a tool
   still running, or from a channel's own `close()`. That holds for the calls
   that write or deliver and for every other one, synchronous ones included
   (`steer`, `transcribe`, `synthesize`, `register_channel`,
   `unregister_channel`, `hook()`, `on()`, `identity_hook()`,
   `on_delivery_status()`, `subscribe_websocket`, `setup_background`,
   `setup_task_tools`…), for a channel's own doors
   (`RealtimeVoiceChannel.start_session`, a participant arriving in a
   conference), and for `async with` on a closed kit. The reads (`get_room`,
   `get_timeline`, `get_binding`, `list_*`, `is_member`, `regenerate_target`,
   `room_recordings`, `room_strategy_is_current`, `sender_verification`,
   `source_health`, `verification_of`, and the synchronous `get_channel`,
   `hooks_for`, `identity_enabled_for`, `list_channels`, `list_sources`,
   `room_strategy`, `speak_queue`) answer during the close, and after it
   while the store is open (an `InMemoryStore`, a Postgres pool you handed
   in); once the kit closed the store (SQLite, a Postgres pool the kit
   opened), a read raises `KitClosedError`, not the store's `RuntimeError`.
   Work a read starts (a source's health check) is no read. The teardown
   calls (`cancel_delivery`, `cancel_task`, `disconnect_websocket`,
   `disconnect_voice`, `disconnect_video`, `leave`, `unsubscribe_room`,
   `unsubscribe_websocket`) return doing nothing: the close tears down what
   they would.
2. **What has not started is cut at once**: a delivery queued in a room's lane
   behind another is dropped and never sent (one placed in the lane while it
   was idle runs from its placement, and gets the grace of rule 4, whatever
   the store; the lane is idle once the sends ahead have answered, their
   responses still running, and no event's ASYNC hooks hold it: an ASYNC
   `BEFORE_BROADCAST` observer that closes the kit finds its event's
   delivery placed, an answer's included), the kit's background runs and its verification
   timers are cut, and a call waiting for its room's lock is cut, nothing of
   it stored (`process_inbound` and `regenerate_response` return
   `InboundResult(blocked=True, reason="kit_closed")`; `send_event`,
   `commit_event` and a room change such as `attach_channel` raise
   `KitClosedError`).
3. **A model turn is cut at once**, its stream and its tool calls. The text it
   already streamed is stored marked `cancelled`; an open tool call gets its
   end row, `failed`, error `kit_closed`. The turn a tool closing the kit
   runs in is cut the same way, but the call that awaits `close()` itself
   runs on: its end row is stored before the store is released, and what the
   turn does after the close is refused quietly. A call that awaits a close
   it started on a task of its own is cut with its turn, and the close runs
   on.
4. **Other running work gets one grace of five seconds**: a hook, a store
   write, a room change, a transport send, the kit's own steps that wait on
   your code (a source's `stop()`, a delegation's `on_complete`, run side by
   side, every source's task cut whatever its `stop()` does, no source
   restarted), a voice or video channel's session handlers. A call that
   commits within the grace is stored and delivered to no one (its
   `delivery_results` empty). Then what is left is cut: a call cut before
   its commit stores nothing; one cut as its commit landed reads its event
   back once its writes unwound, and `process_inbound` returns it with
   `kit_closed` (a BLOCKED record keeping its hooks' reason), `send_event`,
   `commit_event` and `post` return it. A store that completes the write
   later, or a read-back that fails (logged), leaves the call reported as
   not stored. The turn backend closes (its turns and their calls waited
   for on its own budget: a second in process, its `TurnTimeouts.cancel`
   across Redis), then the channels, and the store is released (a Postgres
   pool the store or the lock manager opened closed within a second, its
   connections still in use terminated; their own `close()`, called by you,
   waits for them). `close()` takes the grace, plus the turn backend's budget,
   plus the channels' and the kit's own backends' closes, each on its own
   budget, plus at most a second more, twice, for cut work that ignores its
   cancellation. Nothing is
   written or delivered after `close()` returns by work that honours its
   cut. Work that ignores it past the unwind is abandoned and named in one
   line of the log, each task by the work it runs: a store the kit closed then refuses its writes, a store
   that stays open (an `InMemoryStore`, a pool you handed in) takes them, and
   a send it still makes is its channel's own.
5. **Every call the close cuts returns one result**, its
   `cancellation_reason` `"kit_closed"` (never `"lane_cancelled"`), with the
   answers already stored (`response_events`) and each target's delivery: a
   target already delivered keeps `sent`, a send cut mid-way is `failed` with
   code `kit_closed`, retryable. A deferred call's `result.delivery.wait()`
   returns it once its delivery set ended or was cut. A call under way when
   the close starts reads `kit_closed` too, even when its sends end within
   the grace.

To have a delivery go out, wait for `result.delivery.wait()` before
`close()`; to have a turn's answer, let the turn end first.

A `close()` from the kit's own work (a tool, a hook, a conference handler, a
session's end) does not wait for that work nor cut it, but for the model turn
it runs in (step 3), whether it started the shutdown or joined one under way;
a delivery lane it runs in drops the delivery it kept, which could only run
once the store is released. An ASYNC observer is not the work that fired it:
its `close()` waits for that work as for any other. A delivery the closing
work itself placed, which its room's lane still holds (an `AFTER_BROADCAST`
hook that sends, then closes the kit), is dropped as the close starts. Concurrent `close()` calls share one
shutdown and its result: a channel that failed to close raises the same `ExceptionGroup` to
every caller. A shutdown cut itself (every task cancelled, as `asyncio.run`
does as it ends) raises `ShutdownCancelledError` to a caller nobody
cancelled. See `examples/close_with_work_in_flight.py`.

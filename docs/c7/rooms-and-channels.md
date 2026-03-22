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

## Room Lifecycle

```python
from roomkit import RoomKit, RoomTimers

kit = RoomKit()

# Create room with auto-pause and auto-close timers
room = await kit.create_room(room_id="session-1")

# Update metadata
await kit.update_room_metadata("session-1", {"status": "escalated"})

# Close a room
await kit.close_room("session-1")

# Check timers (call periodically, e.g. every 60s)
transitioned = await kit.check_all_timers()
```

## Channel Types

RoomKit ships with these channel types:

| Channel | Factory / Class | Category | Use Case |
|---------|----------------|----------|----------|
| SMS | `SMSChannel(id, provider=...)` | TRANSPORT | Text messages via Twilio, Telnyx, Sinch |
| RCS | `RCSChannel(id, provider=...)` | TRANSPORT | Rich messaging via Twilio, Telnyx |
| Email | `EmailChannel(id, provider=...)` | TRANSPORT | Email via ElasticEmail, SendGrid |
| WhatsApp | `WhatsAppChannel(id, provider=...)` | TRANSPORT | WhatsApp Business API |
| WhatsApp Personal | `WhatsAppPersonalChannel(id, provider=...)` | TRANSPORT | WhatsApp via neonize |
| Messenger | `MessengerChannel(id, provider=...)` | TRANSPORT | Facebook Messenger |
| Telegram | `TelegramChannel(id, provider=...)` | TRANSPORT | Telegram Bot API |
| Teams | `TeamsChannel(id, provider=...)` | TRANSPORT | Microsoft Teams Bot Framework |
| HTTP | `HTTPChannel(id, provider=...)` | TRANSPORT | Generic webhook |
| WebSocket | `WebSocketChannel(id)` | TRANSPORT | Real-time bidirectional |
| AI | `AIChannel(id, provider=...)` | INTELLIGENCE | LLM responses |
| Agent | `Agent(id, provider=...)` | INTELLIGENCE | Agent with tools + greeting |
| Voice | `VoiceChannel(id, stt=..., tts=..., backend=...)` | TRANSPORT | Real-time audio |
| Realtime Voice | `RealtimeVoiceChannel(id, provider=...)` | INTELLIGENCE | Speech-to-speech AI |

## Registering and Attaching Channels

Channels are registered globally, then attached to specific rooms:

```python
from roomkit import RoomKit, SMSChannel, ChannelCategory
from roomkit.channels.ai import AIChannel
from roomkit.providers.twilio.sms import TwilioSMSProvider
from roomkit.providers.twilio.config import TwilioConfig
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig

kit = RoomKit()

# Create and register channels
sms = SMSChannel("sms-main", provider=TwilioSMSProvider(TwilioConfig(
    account_sid="AC...",
    auth_token="...",
    from_number="+1234567890",
)))
ai = AIChannel("ai-agent", provider=AnthropicAIProvider(AnthropicConfig(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514",
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
| `CompositeContent` | `"composite"` | `parts` (list of EventContent, max depth 5) | Multi-part messages |
| `TemplateContent` | `"template"` | `template_id`, `language`, `parameters`, `body` | WhatsApp Business / RCS templates |
| `SystemContent` | `"system"` | `body`, `code`, `data` (dict) | System-generated events |
| `EditContent` | `"edit"` | `target_event_id`, `new_content`, `edit_source` | Message edits |
| `DeleteContent` | `"delete"` | `target_event_id`, `delete_type`, `reason` | Message deletion |

```python
from roomkit.models.event import (
    TextContent, RichContent, MediaContent, AudioContent, VideoContent,
    LocationContent, CompositeContent, TemplateContent, SystemContent,
    EditContent, DeleteContent,
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

# Multi-part (max depth 5)
CompositeContent(parts=[
    TextContent(body="Here's the photo:"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg"),
])

# Edit and Delete
EditContent(target_event_id="evt-abc", new_content=TextContent(body="Corrected text"))
DeleteContent(target_event_id="evt-abc", delete_type="sender", reason="User retracted")
```

## Content Transcoding

When broadcasting events, the EventRouter automatically **transcodes** content to match each target channel's capabilities. A channel that only supports TEXT will receive a text fallback of a RichContent message.

### How Transcoding Works

1. Event broadcast starts → EventRouter iterates over all target channel bindings
2. For each target, call `ContentTranscoder.transcode(content, source_binding, target_binding)`
3. Transcoder checks `target_binding.capabilities.media_types` to decide if content passes through or needs conversion
4. If transcode returns `None` → delivery is skipped for that channel

### Channel Capabilities

Each channel binding declares what content types it supports via `ChannelCapabilities`:

```python
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.enums import ChannelMediaType

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
| `MediaContent` | MEDIA in media_types → pass | `TextContent("[Media: {caption}]")` |
| `AudioContent` | AUDIO in media_types → pass | `TextContent(transcript)` or `"[Voice message: {url}]"` |
| `VideoContent` | VIDEO in media_types → pass | `TextContent("[Video: {url}]")` |
| `LocationContent` | LOCATION in media_types → pass | `TextContent("[Location: {label} ({lat}, {lon})]")` |
| `CompositeContent` | Recursive transcode of all parts | If all parts become text → flatten to single TextContent |
| `TemplateContent` | TEMPLATE in media_types → pass | `TextContent(body or "[Template: {id}]")` |
| `EditContent` | `supports_edit` → pass | Transcode `new_content` + prefix "Correction:" |
| `DeleteContent` | `supports_delete` → pass | `TextContent("[Message deleted]")` |

### Custom Content Transcoder

Implement the `ContentTranscoder` ABC to customize how content is adapted:

```python
from roomkit.core.router import ContentTranscoder
from roomkit.models.channel import ChannelBinding
from roomkit.models.event import EventContent, TextContent, MediaContent
from roomkit.models.enums import ChannelMediaType

class MyTranscoder(ContentTranscoder):
    async def transcode(
        self,
        content: EventContent,
        source_binding: ChannelBinding,
        target_binding: ChannelBinding,
    ) -> EventContent | None:
        target_types = target_binding.capabilities.media_types

        # Custom: convert images to descriptive text for voice channels
        if isinstance(content, MediaContent) and content.mime_type.startswith("image/"):
            if ChannelMediaType.MEDIA not in target_types:
                caption = content.caption or "an image"
                return TextContent(body=f"[Image received: {caption}]")

        # Custom: truncate long text for SMS
        if isinstance(content, TextContent):
            max_len = target_binding.capabilities.max_length
            if max_len and len(content.body) > max_len:
                return TextContent(body=content.body[: max_len - 3] + "...")

        return content

# Override the default transcoder on the RoomKit instance
kit = RoomKit()
kit._transcoder = MyTranscoder()
```

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

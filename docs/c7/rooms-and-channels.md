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

Events carry typed content. The event router transcodes content to match each channel's capabilities:

```python
from roomkit.models.event import (
    TextContent,
    RichContent,
    MediaContent,
    AudioContent,
    VideoContent,
    LocationContent,
    CompositeContent,
    TemplateContent,
    SystemContent,
    EditContent,
    DeleteContent,
)

# Plain text
TextContent(body="Hello!")

# Rich content with buttons
RichContent(
    body="Choose an option:",
    buttons=[{"text": "Option A", "payload": "a"}],
)

# Media attachment
MediaContent(url="https://example.com/image.png", mime_type="image/png")

# Multi-part message
CompositeContent(parts=[
    TextContent(body="Here's the photo:"),
    MediaContent(url="https://example.com/photo.jpg", mime_type="image/jpeg"),
])
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

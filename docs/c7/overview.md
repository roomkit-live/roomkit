# Overview

RoomKit is a pure async Python library for building multi-channel conversation systems. It provides a **room-based abstraction** where conversations happen in rooms, participants communicate through channels, and hooks let you intercept and modify the flow.

## Key Concepts

- **Room** — A container for a conversation. Holds participants, channel bindings, and an ordered event timeline.
- **Channel** — A communication endpoint (SMS, Email, Voice, AI, WebSocket, etc.). Registered once, attached to many rooms.
- **Hook** — A function that intercepts events at specific points in the pipeline. Can block, modify, or observe messages.
- **Event Router** — Broadcasts events to all channels attached to a room, with content transcoding per channel capabilities.
- **Identity Pipeline** — Maps external sender IDs to known participants with challenge/response flows.
- **Realtime Events** — Ephemeral events (typing, presence, reactions) that are not stored in history.

## Architecture at a Glance

RoomKit uses a hub-and-spoke model. The `RoomKit` orchestrator sits at the center. Channels connect on the edges. Messages flow inbound through a defined pipeline, get stored, then broadcast outbound to all attached channels.

```
Inbound Message
  -> InboundRoomRouter.route()       # Find target room
  -> Channel.handle_inbound()        # Parse -> RoomEvent
  -> IdentityResolver.resolve()      # Identify sender
  -> BEFORE_BROADCAST hooks          # Can block/modify
  -> Store event
  -> EventRouter.broadcast()         # Deliver to all channels
    -> Content transcoding           # Adapt per channel capabilities
    -> Rate limiting + retry
  -> AFTER_BROADCAST hooks           # Async side effects
```

## What You Can Build

- **AI-powered support agents** across SMS, WhatsApp, and web chat
- **Voice assistants** with real-time STT/TTS and interruption handling
- **Multi-agent pipelines** where specialized agents hand off conversations
- **Notification systems** that bridge channels (SMS + Email + push)
- **Speech-to-speech AI** with Gemini Live, OpenAI Realtime, or Grok

## Design Principles

- **Async-first** — All I/O is async. No synchronous blocking.
- **Pluggable everything** — Storage, identity, routing, AI providers, voice backends — all swappable via ABCs with in-memory defaults.
- **Zero required deps** — Only `pydantic>=2.9`. Everything else is optional extras.
- **Type-safe** — Strict mypy, full type hints, Pydantic models throughout.
- **Python 3.12+** — Uses modern syntax (`X | None`, not `Optional[X]`).

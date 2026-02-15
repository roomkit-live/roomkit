"""Telemetry with OpenTelemetryProvider — bridges to the OTel SDK.

Demonstrates how to use RoomKit telemetry with OpenTelemetry's
TracerProvider and ConsoleSpanExporter.

Run with:
    pip install opentelemetry-api opentelemetry-sdk
    uv run python examples/telemetry_otel.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    HookResult,
    HookTrigger,
    InboundMessage,
    MockAIProvider,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel


async def main() -> None:
    # Set up OpenTelemetry SDK
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        print("Install OpenTelemetry SDK: pip install opentelemetry-api opentelemetry-sdk")
        return

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Create RoomKit with OpenTelemetry
    from roomkit.telemetry.opentelemetry import OpenTelemetryProvider

    telemetry = OpenTelemetryProvider(tracer_provider=provider)
    kit = RoomKit(telemetry=telemetry)

    # Channels
    ws = WebSocketChannel("ws-user")
    ai = AIChannel("ai-bot", provider=MockAIProvider(responses=["Hello from AI!"]))

    kit.register_channel(ws)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="logger_hook")
    async def log_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
        return HookResult.allow()

    # Create room and attach channels
    await kit.create_room(room_id="demo")
    await kit.attach_channel("demo", "ws-user")
    await kit.attach_channel("demo", "ai-bot", category=ChannelCategory.INTELLIGENCE)

    # Send a message — OTel spans will be exported to console
    print("\n--- Sending message (watch for OTel span output) ---\n")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user1",
            content=TextContent(body="Tell me a joke"),
        )
    )

    telemetry.close()
    provider.shutdown()
    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())

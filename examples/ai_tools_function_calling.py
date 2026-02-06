"""AI tools and function calling with per-room config.

Demonstrates how to configure AI channels with custom tools for
function calling, and how to set per-room AI configuration via
binding metadata. Shows:
- AITool definitions with JSON schema parameters
- Per-room system_prompt, temperature, and tools via binding metadata
- MockAIProvider for testing without API keys

Run with:
    uv run python examples/ai_tools_function_calling.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=MockAIProvider(
            responses=[
                "The weather in Montreal is -5C and snowy.",
                "I found 3 nearby restaurants. The top pick is Le Bouillon.",
            ]
        ),
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    # --- Room 1: Weather assistant with tools ---
    print("=== Room 1: Weather Assistant ===")
    await kit.create_room(room_id="weather-room")
    await kit.attach_channel("weather-room", "ws-user")
    await kit.attach_channel(
        "weather-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            # Per-room AI configuration
            "system_prompt": (
                "You are a weather assistant. Use the get_weather tool to check conditions."
            ),
            "temperature": 0.3,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                },
            ],
        },
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="What's the weather in Montreal?"),
        )
    )

    print("  User asked about weather")
    for ev in inbox:
        if ev.source.channel_id == "ai-assistant":
            print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]

    # --- Verify per-room config was applied ---
    # Check that the AI context was built with per-room settings
    mock_provider: MockAIProvider = ai._provider  # type: ignore[assignment]
    if mock_provider.calls:
        last_call = mock_provider.calls[-1]
        print("\n  AI Context:")
        print(f"    System prompt: {last_call.system_prompt[:60]}...")
        print(f"    Temperature: {last_call.temperature}")
        print(f"    Tools: {[t.name for t in last_call.tools]}")

    # --- Room 2: Restaurant finder (different per-room config) ---
    print("\n=== Room 2: Restaurant Finder ===")
    inbox.clear()

    # Detach from previous room first
    await kit.detach_channel("weather-room", "ws-user")
    await kit.detach_channel("weather-room", "ai-assistant")

    await kit.create_room(room_id="restaurant-room")
    await kit.attach_channel("restaurant-room", "ws-user")
    await kit.attach_channel(
        "restaurant-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "system_prompt": (
                "You are a restaurant finder. Help users discover great places to eat."
            ),
            "temperature": 0.9,
            "tools": [
                {
                    "name": "search_restaurants",
                    "description": "Search for restaurants near a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "cuisine": {"type": "string"},
                            "max_results": {"type": "integer", "default": 5},
                        },
                        "required": ["location"],
                    },
                },
                {
                    "name": "get_restaurant_details",
                    "description": "Get details about a specific restaurant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant_id": {"type": "string"},
                        },
                        "required": ["restaurant_id"],
                    },
                },
            ],
        },
    )

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Find Italian restaurants near downtown Montreal"),
        )
    )

    print("  User asked about restaurants")
    for ev in inbox:
        if ev.source.channel_id == "ai-assistant":
            print(f"  AI replied: {ev.content.body}")  # type: ignore[union-attr]

    if len(mock_provider.calls) > 1:
        last_call = mock_provider.calls[-1]
        print("\n  AI Context:")
        print(f"    System prompt: {last_call.system_prompt[:60]}...")
        print(f"    Temperature: {last_call.temperature}")
        print(f"    Tools: {[t.name for t in last_call.tools]}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

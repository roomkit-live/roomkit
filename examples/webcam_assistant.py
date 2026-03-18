"""Webcam assistant — chat with Claude about what your camera sees.

A simple terminal chat loop where you can ask an AI to look through
your webcam and describe documents, objects, or anything you show it.

The agent has two tools:
- **list_webcams** — discover available camera devices
- **describe_webcam** — capture a frame and analyze it with vision

Requirements:
    pip install roomkit[anthropic,local-video,openai]

Run with:
    ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... \
        uv run python examples/webcam_assistant.py

Environment variables:
    ANTHROPIC_API_KEY    (required) Anthropic API key for Claude chat
    OPENAI_API_KEY       (required) OpenAI API key for vision analysis
    WEBCAM_DEVICE        (optional) Camera device index, default 0
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from roomkit import (
    AITool,
    AnthropicAIProvider,
    AnthropicConfig,
    ChannelCategory,
    DescribeWebcamTool,
    InboundMessage,
    ListWebcamsTool,
    OpenAIVisionConfig,
    OpenAIVisionProvider,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant with access to the user's webcam.

When the user asks you to look at something, read a document, or identify
an object, use the describe_webcam tool with a precise query.

If the user asks which cameras are available, use list_webcams first.

Be concise and direct in your answers.\
"""


# ---------------------------------------------------------------------------
# Tool setup
# ---------------------------------------------------------------------------


def build_tools(
    device: int,
    openai_api_key: str,
) -> tuple[list[AITool], Any]:
    """Build webcam tool definitions and a unified handler."""
    vision = OpenAIVisionProvider(
        OpenAIVisionConfig(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            max_tokens=1024,
        ),
    )
    webcam = DescribeWebcamTool(vision, device=device)
    lister = ListWebcamsTool()

    tools = [
        AITool(
            name=webcam.definition["name"],
            description=webcam.definition["description"],
            parameters=webcam.definition["parameters"],
        ),
        AITool(
            name=lister.definition["name"],
            description=lister.definition["description"],
            parameters=lister.definition["parameters"],
        ),
    ]

    async def tool_handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "describe_webcam":
            query = arguments.get("query", "Describe what you see.")
            raw_device = arguments.get("device")
            dev = int(raw_device) if raw_device is not None else None
            raw_path = arguments.get("save_path")
            path = str(raw_path) if raw_path is not None else None
            return await webcam.analyze(query, device=dev, save_path=path)
        if name == "list_webcams":
            return lister.list()
        return json.dumps({"error": f"Unknown tool: {name}"})

    return tools, tool_handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    device = int(os.environ.get("WEBCAM_DEVICE", "0"))

    if not anthropic_key:
        print("Set ANTHROPIC_API_KEY to run this example.")
        return
    if not openai_key:
        print("Set OPENAI_API_KEY for vision analysis.")
        return

    # --- Tools ---------------------------------------------------------------
    tools, tool_handler = build_tools(device, openai_key)

    # --- RoomKit setup -------------------------------------------------------
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(
            AnthropicConfig(api_key=anthropic_key, max_tokens=2048),
        ),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        tool_handler=tool_handler,
    )

    kit.register_channel(ws)
    kit.register_channel(ai)

    # Collect AI replies for display.
    reply_text: list[str] = []

    async def on_receive(_conn: str, event: RoomEvent) -> None:
        if isinstance(event.content, TextContent):
            reply_text.append(event.content.body)

    ws.register_connection("user-conn", on_receive)

    await kit.create_room(room_id="webcam-room")
    await kit.attach_channel("webcam-room", "ws-user")
    await kit.attach_channel(
        "webcam-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
    )

    # --- Chat loop -----------------------------------------------------------
    print("Webcam Assistant (type 'quit' to exit)")
    print(f"Using camera device {device}\n")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break
        if not user_input.strip():
            continue

        reply_text.clear()

        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="user",
                content=TextContent(body=user_input),
            ),
        )

        for text in reply_text:
            print(f"\nAssistant: {text}\n")

    await kit.close()
    print("Bye!")


if __name__ == "__main__":
    asyncio.run(main())

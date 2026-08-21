"""RoomKit — an agent that draws, whatever provider it converses with.

The point of the ``ImageProvider`` surface (RFC §25) is that drawing is *not*
a mode of the conversation. The agent below holds the conversation through one
provider and draws through another; swap either and the other is unaffected::

    "dessine-moi un renard en origami"
        → AIChannel (any AI provider)  → calls the generate_image tool
        → ImageProvider (OpenAI / Gemini / mock)
        → ImageResult (a data URI)
        → back into the room as MediaContent, and onto disk as a PNG

``ImageResult.data`` is always a ``data:`` URI, which is exactly what
``MediaContent.url`` accepts — so the generated image enters the room with no
conversion step, and every channel bound to that room receives it.

Runs with no key at all: the mock provider draws a real (tiny) PNG, so the
whole path — tool call, room message, file on disk — is exercised offline.
Set a key to draw for real.

Requires:
    pip install roomkit            # mock path, no key
    pip install roomkit[openai]    # OPENAI_API_KEY
    pip install roomkit[gemini]    # GEMINI_API_KEY
    pip install roomkit[xai]       # XAI_API_KEY
    pip install roomkit[openrouter]  # OPENROUTER_API_KEY
    pip install roomkit[azure]     # AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT

Environment variables (first configured provider wins):
    OPENAI_API_KEY — draws with OpenAI (gpt-image-2 by default)
    GEMINI_API_KEY — draws with Gemini (gemini-3.1-flash-image by default)
    XAI_API_KEY    — draws with xAI (grok-imagine-image-2.0 by default)
    OPENROUTER_API_KEY — draws through OpenRouter's Image API
                         (google/gemini-3.1-flash-image by default)
    AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT — draws with an Azure OpenAI
                         deployment (IMAGE_MODEL names the deployment)
    IMAGE_MODEL    — override the model id for whichever provider is selected
    ANTHROPIC_API_KEY — optional: hold the conversation with Claude instead of
                        the mock AI, to see the decoupling for real

Run with:
    uv run python examples/image_generation.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    ImageProvider,
    MockImageProvider,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.models.event import MediaContent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

ROOM_ID = "atelier"
OUTPUT_DIR = Path(__file__).parent / "recordings"

TOOL_DEFINITION: dict[str, Any] = {
    "name": "generate_image",
    "description": "Draw an image from a description and post it in the room.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to draw."},
            "size": {"type": "string", "description": "Geometry, e.g. 1024x1024."},
        },
        "required": ["prompt"],
    },
}


class DrawTool:
    """A tool the agent can call to draw — and to put the drawing in the room.

    Satisfies :class:`roomkit.tools.base.Tool`: a ``definition`` and an async
    ``handler``. The handler returns *text* to the model (a model cannot see
    the picture it just drew, and does not need to) while the image itself
    enters the room as a first-class message.
    """

    def __init__(self, kit: RoomKit, images: ImageProvider, *, room_id: str) -> None:
        self._kit = kit
        self._images = images
        self._room_id = room_id
        self.saved: list[Path] = []

    @property
    def definition(self) -> dict[str, Any]:
        return TOOL_DEFINITION

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        if name != TOOL_DEFINITION["name"]:
            return f"Unknown tool: {name}"

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return "generate_image needs a prompt describing what to draw."
        size = arguments.get("size")

        results = await self._images.generate(prompt, size=str(size) if size else None)
        for index, result in enumerate(results):
            await self._kit.send_event(
                room_id=self._room_id,
                channel_id="atelier-bot",
                content=MediaContent(
                    url=result.data,  # already a data: URI — nothing to convert
                    mime_type=result.mime_type,
                    filename=f"{_slug(prompt)}-{index}.{result.mime_type.rpartition('/')[2]}",
                    caption=result.revised_prompt or prompt,
                ),
                # Solicits nobody (RFC §19.3): the picture is the answer to the
                # turn already in flight, not a new question. Left unaddressed
                # it would re-enter as a fresh prompt, the agent would draw
                # again, and the room would fill with foxes.
                addressed_to=[],
            )
            self.saved.append(_write(result.decoded(), prompt, index, result.mime_type))

        drawn = ", ".join(str(path.name) for path in self.saved[-len(results) :])
        return f"Drew {len(results)} image(s) and posted them in the room: {drawn}"


def _slug(prompt: str) -> str:
    return "-".join(prompt.lower().split())[:40] or "image"


def _write(data: bytes, prompt: str, index: int, mime_type: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{_slug(prompt)}-{index}.{mime_type.rpartition('/')[2]}"
    path.write_bytes(data)
    return path


def build_image_provider() -> ImageProvider:
    """The first configured provider wins; without a key, the mock draws.

    The conversation's provider has no say in this choice — that is the whole
    point of the surface (RFC §25.1).
    """
    model = os.environ.get("IMAGE_MODEL")
    if api_key := os.environ.get("OPENAI_API_KEY"):
        from roomkit.providers.openai import OpenAIImageConfig, OpenAIImageProvider

        return OpenAIImageProvider(
            OpenAIImageConfig(api_key=api_key, model=model or "gpt-image-2")
        )
    if api_key := os.environ.get("GEMINI_API_KEY"):
        from roomkit.providers.gemini import GeminiImageConfig, GeminiImageProvider

        return GeminiImageProvider(
            GeminiImageConfig(api_key=api_key, model=model or "gemini-3.1-flash-image")
        )
    if api_key := os.environ.get("XAI_API_KEY"):
        from roomkit.providers.xai import XAIImageConfig, XAIImageProvider

        return XAIImageProvider(
            XAIImageConfig(api_key=api_key, model=model or "grok-imagine-image-2.0")
        )
    if api_key := os.environ.get("OPENROUTER_API_KEY"):
        from roomkit.providers.openrouter import OpenRouterImageConfig, OpenRouterImageProvider

        return OpenRouterImageProvider(
            OpenRouterImageConfig(api_key=api_key, model=model or "google/gemini-3.1-flash-image")
        )
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if api_key and endpoint:
        from roomkit.providers.azure import AzureImageConfig, AzureImageProvider

        # Azure deployments are user-named; IMAGE_MODEL is the deployment name.
        return AzureImageProvider(
            AzureImageConfig(
                api_key=api_key, azure_endpoint=endpoint, model=model or "gpt-image-1"
            )
        )
    print("No image-provider key in the environment — drawing with MockImageProvider.\n")
    return MockImageProvider()


def build_ai_provider() -> Any:
    """Whoever holds the conversation. Never the one that draws."""
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig

        return AnthropicAIProvider(AnthropicConfig(api_key=api_key))
    return MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Je m'en occupe.",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="tc-1",
                        name="generate_image",
                        arguments={"prompt": "un renard en origami", "size": "1024x1024"},
                    )
                ],
            ),
            AIResponse(content="Voilà le renard — il est dans la conversation."),
        ]
    )


async def main() -> None:
    setup_logging("image_generation", level=logging.WARNING)

    kit = RoomKit()
    images = build_image_provider()

    ws = WebSocketChannel("atelier-bot")
    kit.register_channel(ws)

    draw = DrawTool(kit, images, room_id=ROOM_ID)
    ai = AIChannel(
        "artiste",
        provider=build_ai_provider(),
        system_prompt=(
            "Tu es un illustrateur. Quand on te demande une image, appelle "
            "l'outil generate_image avec une description riche et précise."
        ),
        tools=[draw],
    )
    kit.register_channel(ai)

    received: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        received.append(event)

    ws.register_connection("viewer", on_recv, room_id=ROOM_ID)

    await kit.create_room(room_id=ROOM_ID)
    await kit.attach_channel(ROOM_ID, "atelier-bot")
    await kit.attach_channel(ROOM_ID, "artiste", category=ChannelCategory.INTELLIGENCE)

    print(f"Drawing with {images.name} ({images.model_name})\n")
    await kit.send_event(
        room_id=ROOM_ID,
        channel_id="atelier-bot",
        content=TextContent(body="Dessine-moi un renard en origami, carré."),
    )
    await asyncio.sleep(0.5)

    print("--- room timeline ---")
    for event in await kit.get_timeline(ROOM_ID):
        content = event.content
        if isinstance(content, MediaContent):
            print(f"  [{event.source.channel_id}] 🖼  {content.filename} ({content.mime_type})")
        elif isinstance(content, TextContent):
            print(f"  [{event.source.channel_id}] {content.body}")

    if draw.saved:
        print("\n--- written to disk ---")
        for path in draw.saved:
            print(f"  {path}  ({path.stat().st_size} bytes)")

    await images.close()
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

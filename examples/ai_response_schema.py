"""Response schema — an AI provider answers in a JSON shape you declare.

A support message is triaged into a department, an urgency flag and a one-line
summary. ``AIContext.response_schema`` carries the shape; the provider
constrains its output natively (OpenAI ``response_format``, Anthropic
``output_config``, Gemini ``response_json_schema``, Mistral, Ollama, PolarGrid)
and ``generate()`` returns one JSON document, or raises ``ResponseSchemaError``
saying why it could not: a refusal, a truncated answer, text that is not JSON,
or a provider that cannot take a schema at all.

The same shape reaches three places, one section each:

1. A provider call: ``AIContext(response_schema=...)``.
2. An ``AIChannel`` turn: the channel's ``response_schema`` is every turn's
   default, and a room overrides it through its binding metadata (a
   ``config_provider`` can too, per turn). The room receives the answer only
   once it is checked, streamed or not.
3. A vision frame: ``VisionProvider.analyze_frame(..., response_schema=...)``.

The schema stays within the portable subset (RFC §6.7): every object lists all
its properties in ``required`` and sets ``additionalProperties`` to false, and
only strings carry ``enum``. That subset runs on every provider that supports
response schemas.

Run with:
    uv run python examples/ai_response_schema.py                     # offline, mock
    GEMINI_API_KEY=... uv run python examples/ai_response_schema.py gemini
    ANTHROPIC_API_KEY=... uv run python examples/ai_response_schema.py anthropic
    OPENAI_API_KEY=... uv run python examples/ai_response_schema.py openai
    uv run python examples/ai_response_schema.py ollama              # local server

A real vision provider encodes the frame as JPEG, which takes Pillow or
opencv-python-headless. Anthropic has no vision provider, so that run skips
the third section.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import asyncio
import json

from shared import require_env

from roomkit import (
    AIChannel,
    ChannelCategory,
    InboundMessage,
    ResponseSchemaError,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.providers.ai import AIContext, AIMessage, AIProvider, MockAIProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.providers.gemini import GeminiAIProvider, GeminiConfig
from roomkit.providers.ollama import OllamaAIProvider, OllamaConfig
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig
from roomkit.video import (
    GeminiVisionConfig,
    GeminiVisionProvider,
    MockVisionProvider,
    OpenAIVisionConfig,
    OpenAIVisionProvider,
    VideoFrame,
    VisionProvider,
)

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {
            "type": "string",
            "enum": ["billing", "technical", "sales", "other"],
            "description": "The team that should take the message.",
        },
        "urgent": {"type": "boolean", "description": "Whether the customer threatens to leave."},
        "summary": {"type": "string", "description": "One short sentence."},
    },
    "required": ["department", "urgent", "summary"],
    "additionalProperties": False,
}

# What the escalations room asks of the same channel instead.
URGENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "urgent": {"type": "boolean"},
        "reason": {"type": "string", "description": "Why, in a few words."},
    },
    "required": ["urgent", "reason"],
    "additionalProperties": False,
}

COLOR_SCHEMA = {
    "type": "object",
    "properties": {"color": {"type": "string", "enum": ["red", "green", "blue", "other"]}},
    "required": ["color"],
    "additionalProperties": False,
}

MESSAGE = (
    "I was charged twice for March. This is the third time I'm calling. "
    "If this isn't fixed today I'm cancelling."
)
SYSTEM_PROMPT = "You triage customer support messages."

_TRIAGE = {
    "department": "billing",
    "urgent": True,
    "summary": "Double charge for March, third call, threatens to cancel.",
}
_URGENCY = {"urgent": True, "reason": "Threatens to cancel today."}


def build_provider(name: str) -> AIProvider:
    """The provider named on the command line; the mock needs no key."""
    if name == "gemini":
        env = require_env("GEMINI_API_KEY")
        return GeminiAIProvider(GeminiConfig(api_key=env["GEMINI_API_KEY"]))
    if name == "anthropic":
        env = require_env("ANTHROPIC_API_KEY")
        return AnthropicAIProvider(
            AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"], model="claude-opus-5")
        )
    if name == "openai":
        env = require_env("OPENAI_API_KEY")
        return OpenAIAIProvider(OpenAIConfig(api_key=env["OPENAI_API_KEY"], model="gpt-6.2-sol"))
    if name == "ollama":
        return OllamaAIProvider(OllamaConfig())
    # One scripted answer per call, in the order the sections make them.
    answers = [_TRIAGE, _TRIAGE, _URGENCY]
    return MockAIProvider([json.dumps(a) for a in answers], response_schema=True)


def build_vision(name: str) -> VisionProvider | None:
    """The vision provider for the same vendor, if it has one."""
    if name == "gemini":
        env = require_env("GEMINI_API_KEY")
        return GeminiVisionProvider(GeminiVisionConfig(api_key=env["GEMINI_API_KEY"]))
    if name == "openai":
        env = require_env("OPENAI_API_KEY")
        return OpenAIVisionProvider(
            OpenAIVisionConfig(
                api_key=env["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
                model="gpt-6.2-sol",
                max_tokens=25000,
            )
        )
    if name == "ollama":
        return OpenAIVisionProvider()
    if name == "anthropic":
        return None
    return MockVisionProvider([json.dumps({"color": "red"})], response_schema=True)


async def triage_with_the_provider(provider: AIProvider) -> None:
    """Section 1: one call, one JSON document."""
    print("=== Provider call ===")
    context = AIContext(
        system_prompt=SYSTEM_PROMPT,
        messages=[AIMessage(role="user", content=MESSAGE)],
        response_schema=TRIAGE_SCHEMA,
        max_tokens=1024,
    )
    try:
        response = await provider.generate(context)
    except ResponseSchemaError as exc:
        print(f"  No triage ({exc.reason}): {exc}")
        return
    triage = json.loads(response.content)
    print(f"  Provider:   {provider.name} ({provider.model_name})")
    print(f"  Department: {triage['department']}")
    print(f"  Urgent:     {triage['urgent']}")
    print(f"  Summary:    {triage['summary']}")


async def triage_in_rooms(provider: AIProvider) -> None:
    """Section 2: the channel's schema by default, the room's where it sets one."""
    print("\n=== AIChannel turns ===")
    kit = RoomKit()
    ws = WebSocketChannel("ws-user")
    ai = AIChannel(
        "ai-triage",
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        response_schema=TRIAGE_SCHEMA,
        max_tokens=1024,
    )
    kit.register_channel(ws)
    kit.register_channel(ai)
    answers: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        if event.source.channel_id == "ai-triage":
            answers.append(event)

    ws.register_connection("agent-desk", on_recv, room_id="support")
    ws.subscribe("agent-desk", "escalations")
    rooms = {"support": {}, "escalations": {"response_schema": URGENCY_SCHEMA}}
    for room_id, metadata in rooms.items():
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ws-user")
        await kit.attach_channel(
            room_id, "ai-triage", category=ChannelCategory.INTELLIGENCE, metadata=metadata
        )
        delivered = len(answers)
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user", sender_id="customer", content=TextContent(body=MESSAGE)
            ),
            room_id=room_id,
        )
        await asyncio.sleep(0.1)  # a streamed answer is delivered as the stream ends
        if isinstance(result.error, ResponseSchemaError):
            print(f"  {room_id}: no answer ({result.error.reason}): {result.error}")
            continue
        if len(answers) == delivered:
            print(f"  {room_id}: no answer delivered ({result.error})")
            continue
        body = answers[-1].content.body  # type: ignore[union-attr]
        print(f"  {room_id}: {json.loads(body)}")
    await kit.close()


async def read_a_frame(vision: VisionProvider | None) -> None:
    """Section 3: a frame described in the schema's words."""
    print("\n=== Vision frame ===")
    if vision is None or not vision.supports_response_schema:
        print("  No vision provider here that constrains its answer.")
        return
    width, height = 64, 48
    red = VideoFrame(
        data=bytes([220, 30, 30]) * (width * height),
        codec="raw_rgb24",
        width=width,
        height=height,
    )
    try:
        result = await vision.analyze_frame(
            red, prompt="What colour fills this image?", response_schema=COLOR_SCHEMA
        )
    except ResponseSchemaError as exc:
        print(f"  No colour ({exc.reason}): {exc}")
        return
    finally:
        await vision.close()
    print(f"  {vision.name}: {json.loads(result.description)}")


async def main(provider_name: str) -> None:
    provider = build_provider(provider_name)
    if not provider.supports_response_schema:
        # A consumer meant for any provider falls back to asking for JSON in
        # the prompt and parsing it itself; this example just says so.
        print(f"{provider.name} does not constrain its output to a schema.")
        return
    try:
        await triage_with_the_provider(provider)
        await triage_in_rooms(provider)
    finally:
        await provider.close()
    await read_a_frame(build_vision(provider_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "provider",
        nargs="?",
        default="mock",
        choices=["mock", "gemini", "anthropic", "openai", "ollama"],
    )
    asyncio.run(main(parser.parse_args().provider))

"""PolarGrid AI example — Canadian-hosted chat completions.

Wires :class:`roomkit.providers.polargrid.PolarGridAIProvider` into a
:class:`roomkit.CLIChannel`. PolarGrid serves OpenAI-shaped chat
completions from regional edges in Toronto, Vancouver, and Montreal —
useful when data residency on Canadian soil is a requirement.

PolarGrid's chat endpoint supports tool / function calling as of
polargrid-sdk 0.8.5. This example gives the assistant a ``web_search``
tool (``shared/tools.py``) and lets PolarGrid call it: ask a factual
question ("What is the speed of light?") and the model runs the search,
then answers from the result. Forcing a specific tool is steered, not
hard-guaranteed, on PolarGrid's backend.

The search works with no extra setup (key-free Wikipedia lookup). Set
``TAVILY_API_KEY`` for real web search that also finds niche companies
and current info — get a free key at https://tavily.com.

Set ``POLARGRID_THINKING=1`` to activate qwen's reasoning (sets the
``enable_thinking`` request flag). The model then emits reasoning as
inline ``<think>...</think>`` tags; the provider splits it out and the
CLI shows it as dimmed "thinking" (💭) separate from the answer. Thinking
responses are larger and slower.

**Image input** (polargrid-sdk 0.9.0 multimodal chat): type
``/image <path> [question]`` to attach a local image, e.g.
``/image ~/Pictures/receipt.jpg what's the total?``. The file is embedded
as a base64 ``data:`` URI and sent as an ``image_url`` part; with no
question it defaults to "Analyse this image." Vision needs a vision-capable
model — run with ``POLARGRID_MODEL=qwen-3.6-35b-a3b POLARGRID_REGION=yul-02``
(the default ``qwen-3.5-27b`` is text-only; the ``vision=`` flag in the
welcome banner reflects the configured model).

Run with:
    POLARGRID_API_KEY=pg_... uv run python examples/polargrid_ai.py

Optional overrides (defaults come from PolarGridConfig):
    POLARGRID_MODEL=qwen-3.5-27b
    POLARGRID_REGION=toronto    # pin a region (toronto/vancouver/montreal
                                #   or yto-01/yvr-02/yul-01).
                                # Unset to auto-route to the fastest edge.
    POLARGRID_THINKING=1        # activate qwen reasoning (enable_thinking).
    POLARGRID_DEBUG=1           # log the exact request sent + SDK HTTP debug.
    TAVILY_API_KEY=tvly-...     # enable real web search (else Wikipedia).
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio

from shared import WebSearchTool, env_bool, require_env, setup_logging

from roomkit import (
    ChannelCategory,
    CLIChannel,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.event import (
    CompositeContent,
    EventContent,
    MediaContent,
    TextContent,
)
from roomkit.providers.polargrid import PolarGridAIProvider, PolarGridConfig

_IMAGE_COMMAND = "/image"


def _build_content(line: str) -> EventContent | None:
    """Map a CLI line to inbound content.

    ``/image <path> [question]`` attaches a local image as a base64 ``data:``
    URI (with an optional question); anything else is plain text. Returns
    ``None`` to skip a malformed ``/image`` command — a usage hint is printed
    instead of sending a junk turn to the model.
    """
    if not line.startswith(_IMAGE_COMMAND):
        return TextContent(body=line)

    path_str, _, question = line[len(_IMAGE_COMMAND) :].strip().partition(" ")
    path = Path(path_str).expanduser()
    if not path_str or not path.is_file():
        print(f"  ⚠ usage: {_IMAGE_COMMAND} <path> [question] — no readable file at {path_str!r}")
        return None

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data_uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return CompositeContent(
        parts=[
            TextContent(body=question.strip() or "Analyse this image."),
            MediaContent(url=data_uri, mime_type=mime, filename=path.name),
        ]
    )


# POLARGRID_DEBUG=1 logs the exact request we send (incl. enable_thinking)
# plus the SDK's HTTP debug — handy to share with PolarGrid. Scoped to the
# polargrid logger so asyncio/httpcore/httpx noise stays out.
_DEBUG = env_bool("POLARGRID_DEBUG", default=False)
setup_logging("polargrid_ai")
if _DEBUG:
    logging.getLogger("roomkit.providers.polargrid").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("POLARGRID_API_KEY")
    config_kwargs: dict[str, str] = {"api_key": env["POLARGRID_API_KEY"]}
    if model := os.environ.get("POLARGRID_MODEL"):
        config_kwargs["model"] = model
    if region := os.environ.get("POLARGRID_REGION"):
        config_kwargs["region"] = region

    # POLARGRID_THINKING=1 sets the enable_thinking request flag to activate
    # reasoning; the CLI then renders it (see show_thinking below). Unset
    # leaves it at the model default (None); =0 explicitly disables it.
    thinking = (
        env_bool("POLARGRID_THINKING", default=False)
        if "POLARGRID_THINKING" in os.environ
        else None
    )
    provider = PolarGridAIProvider(
        PolarGridConfig(**config_kwargs, thinking=thinking, debug=_DEBUG)
    )

    kit = RoomKit()

    cli = CLIChannel("cli", show_thinking=True)
    ai = AIChannel(
        "ai-assistant",
        provider=provider,
        system_prompt=(
            "You are a helpful assistant. Keep answers concise. "
            "When a question needs current or factual information, use the "
            "web_search tool and base your answer strictly on its results. "
            "If the results don't contain the answer, say the search didn't "
            "find it rather than guessing from prior knowledge."
        ),
        # A Tool object carries both its schema and its handler, so the
        # AIChannel runs the whole tool loop: model -> web_search -> answer.
        tools=[WebSearchTool()],
        # Reasoning eats into the token budget; give the post-tool answer
        # room when thinking is on, or it gets truncated to an empty
        # response and the channel has to re-prompt for the final answer.
        max_tokens=4096 if thinking else 1024,
    )
    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="demo-room")
    await kit.attach_channel("demo-room", "cli")
    await kit.attach_channel("demo-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="demo-room",
            content_factory=_build_content,
            welcome=(
                f"\nPolarGrid AI demo — model={provider.model_name} "
                f"region={config_kwargs.get('region', 'auto')} "
                f"thinking={'on' if thinking else 'off'} "
                f"vision={'on' if provider.supports_vision else 'off'}\n"
                "Tool: web_search (Wikipedia; Tavily if TAVILY_API_KEY set). "
                "Try 'What is the speed of light?'\n"
                "Image: '/image <path> [question]' to analyse a local image.\n"
                "Type a message and press Enter. Use 'quit' or Ctrl+D to exit.\n"
            ),
        )
    finally:
        await provider.close()
        await kit.close()


if __name__ == "__main__":
    asyncio.run(main())

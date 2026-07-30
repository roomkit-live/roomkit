"""Interactive xAI (Grok) CLI — chat with a Grok model through the full pipeline.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by xAI, so you can
talk to Grok end to end: your input → room → AIChannel → api.x.ai → streamed
answer → terminal.

xAI serves the OpenAI Chat Completions API, and every current Grok text model is
multimodal and tool-capable. Reasoning depth rides ``reasoning_effort``
(``low`` / ``medium`` / ``high``) — Grok reasons unconditionally, so this tunes
how long it thinks rather than whether it does. The reasoning trace itself is not
returned by Chat Completions. List the catalog with ``examples/list_models.py``.

For Grok *voice* (speech-to-speech), see ``XAIRealtimeProvider`` instead.

Requires:
    pip install roomkit[xai]   (and roomkit[console] for colored output)

Environment:
    XAI_API_KEY  — your xAI API key (https://console.x.ai)
    XAI_MODEL    — model id (default: grok-4.5)

Run with:
    XAI_API_KEY=xai-... uv run python examples/xai_ai.py
    XAI_API_KEY=xai-... uv run python examples/xai_ai.py --model grok-4.3
    XAI_API_KEY=xai-... uv run python examples/xai_ai.py --effort low

Type a message at the prompt. Type ``quit`` (or Ctrl+D) to exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging

from roomkit import CLIChannel, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import ChannelCategory
from roomkit.providers.xai import XAIAIProvider, XAIConfig


async def main(args: argparse.Namespace) -> None:
    env = require_env("XAI_API_KEY")

    provider = XAIAIProvider(
        XAIConfig(
            api_key=env["XAI_API_KEY"],
            model=args.model,
            reasoning_effort=args.effort,
        )
    )

    kit = RoomKit()

    cli = CLIChannel("you")
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="xai-cli")
    await kit.attach_channel("xai-cli", "you")
    await kit.attach_channel("xai-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="xai-cli",
            welcome=f"xAI · {args.model} (effort: {args.effort})\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive xAI (Grok) CLI.")
    p.add_argument(
        "--model",
        default=os.environ.get("XAI_MODEL", "grok-4.5"),
        help="Grok model id (e.g. grok-4.3, grok-build-0.1). Env: XAI_MODEL.",
    )
    p.add_argument(
        "--effort",
        default="high",
        choices=("low", "medium", "high"),
        help="Reasoning effort. Grok 4.5 defaults to high and cannot disable reasoning.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("xai_ai")
    asyncio.run(main(_parse_args()))

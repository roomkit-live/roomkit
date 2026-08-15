"""Interactive DeepSeek CLI — chat with a DeepSeek model through the full pipeline.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by DeepSeek, so you
can talk to it end to end: your input → room → AIChannel → api.deepseek.com →
streamed answer → terminal.

DeepSeek serves the OpenAI Chat Completions API. Both V4 models are
tool-capable, text-only, and think by default; thinking is a request parameter
rather than a separate model id, and its depth rides ``--effort``
(``low`` / ``high`` / ``max``). Unlike most providers here, the reasoning trace
*is* returned — it arrives in a dedicated field and shows up as thinking in the
terminal. Token budgets are ignored by this API, so ``--effort`` is the only
lever. List the catalog with ``examples/list_models.py``.

DeepSeek also fronts an Anthropic-compatible endpoint at ``/anthropic``. This
example uses the OpenAI-shaped one, which is the richer of the two: the
Anthropic path drops prompt caching markers, images and the models listing.

Requires:
    pip install roomkit[deepseek]   (and roomkit[console] for colored output)

Environment:
    DEEPSEEK_API_KEY  — your DeepSeek API key (https://platform.deepseek.com)
    DEEPSEEK_MODEL    — model id (default: deepseek-v4-pro)

Run with:
    DEEPSEEK_API_KEY=sk-... uv run python examples/deepseek_ai.py
    DEEPSEEK_API_KEY=sk-... uv run python examples/deepseek_ai.py --model deepseek-v4-flash
    DEEPSEEK_API_KEY=sk-... uv run python examples/deepseek_ai.py --no-thinking

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
from roomkit.providers.deepseek import DeepSeekAIProvider, DeepSeekConfig


async def main(args: argparse.Namespace) -> None:
    env = require_env("DEEPSEEK_API_KEY")

    provider = DeepSeekAIProvider(
        DeepSeekConfig(
            api_key=env["DEEPSEEK_API_KEY"],
            model=args.model,
            reasoning_effort=None if args.no_thinking else args.effort,
            enable_thinking=not args.no_thinking,
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

    await kit.create_room(room_id="deepseek-cli")
    await kit.attach_channel("deepseek-cli", "you")
    await kit.attach_channel("deepseek-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    mode = "thinking off" if args.no_thinking else f"effort: {args.effort}"
    try:
        await cli.run(
            kit,
            room_id="deepseek-cli",
            welcome=f"DeepSeek · {args.model} ({mode})\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive DeepSeek CLI.")
    p.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        help="DeepSeek model id (deepseek-v4-pro, deepseek-v4-flash). Env: DEEPSEEK_MODEL.",
    )
    p.add_argument(
        "--effort",
        default="high",
        choices=("low", "high", "max"),
        help="Reasoning depth. DeepSeek ignores token budgets, so this is the only lever.",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Switch thinking off. Both V4 models think by default.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("deepseek_ai")
    asyncio.run(main(_parse_args()))

"""Interactive OpenRouter CLI — chat through any of 300+ models, one key.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by OpenRouter, so
you can talk to any model on OpenRouter (Anthropic, OpenAI, Google, DeepSeek,
xAI, Qwen, …) by passing its slug. This is the full RoomKit pipeline: your
input → room → AIChannel → OpenRouter → streamed answer → terminal.

OpenRouter speaks the OpenAI Chat Completions API, so reasoning models that
emit a thinking trace render inline (💭) above the answer when ``show_thinking``
is on. List every available slug with ``examples/list_models.py``.

Requires:
    pip install roomkit[openrouter]   (and roomkit[console] for colored output)

Environment:
    OPENROUTER_API_KEY  — your OpenRouter API key (https://openrouter.ai/keys)
    OPENROUTER_MODEL    — model slug (default: anthropic/claude-sonnet-4.5)
    OPENROUTER_SITE_URL — optional; sent as HTTP-Referer for app attribution
    OPENROUTER_APP_NAME — optional; sent as X-Title for app attribution

Run with:
    OPENROUTER_API_KEY=sk-or-... uv run python examples/openrouter_ai.py
    OPENROUTER_API_KEY=sk-or-... uv run python examples/openrouter_ai.py --model openai/gpt-5.5
    OPENROUTER_API_KEY=sk-or-... uv run python examples/openrouter_ai.py --no-think

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
from roomkit.providers.openrouter import OpenRouterAIProvider, OpenRouterConfig


async def main(args: argparse.Namespace) -> None:
    env = require_env("OPENROUTER_API_KEY")

    provider = OpenRouterAIProvider(
        OpenRouterConfig(
            api_key=env["OPENROUTER_API_KEY"],
            model=args.model,
            # Optional app attribution — appear on OpenRouter's leaderboards.
            site_url=os.environ.get("OPENROUTER_SITE_URL"),
            app_name=os.environ.get("OPENROUTER_APP_NAME"),
        )
    )

    kit = RoomKit()

    cli = CLIChannel("you", show_thinking=not args.no_think)
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
        # >0 requests OpenRouter reasoning (mapped to a max_tokens cap); 0 turns
        # it off. Works on any reasoning-capable slug (Claude, GPT, DeepSeek, …).
        thinking_budget=0 if args.no_think else 4096,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="openrouter-cli")
    await kit.attach_channel("openrouter-cli", "you")
    await kit.attach_channel("openrouter-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    try:
        await cli.run(
            kit,
            room_id="openrouter-cli",
            welcome=f"OpenRouter · {args.model}\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive OpenRouter CLI (300+ models, one key).")
    p.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        help="OpenRouter model slug (e.g. openai/gpt-5.5). Env: OPENROUTER_MODEL.",
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Hide the thinking trace for models that stream reasoning.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("openrouter_ai")
    asyncio.run(main(_parse_args()))

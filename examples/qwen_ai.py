"""Interactive Qwen CLI — chat with a Qwen model through the full pipeline.

Wires a :class:`CLIChannel` to an :class:`AIChannel` backed by Alibaba Cloud
Model Studio, so you can talk to Qwen end to end: your input → room →
AIChannel → dashscope → streamed answer → terminal.

Model Studio serves the OpenAI Chat Completions API. Thinking is a boolean
switch plus a token cap — ``--thinking-budget`` maps straight onto Qwen's own
``thinking_budget``, which is unusual: most providers only take an effort tier.
The reasoning trace comes back in a dedicated field and shows up as thinking in
the terminal.

Two things to know about the endpoint. It publishes no models listing, so
``list_models()`` returns roomkit's offline catalog rather than the account's
own set. And the base URL depends on where your key lives — the default here is
the international deployment; pass ``--base-url`` for Beijing
(``https://dashscope.aliyuncs.com/compatible-mode/v1``), US Virginia, or the
workspace-scoped ``https://{WorkspaceId}.{region}.maas.aliyuncs.com/
compatible-mode/v1`` form.

Requires:
    pip install roomkit[qwen-ai]   (and roomkit[console] for colored output)

Environment:
    DASHSCOPE_API_KEY  — your Model Studio API key (https://bailian.console.aliyun.com)
    QWEN_MODEL         — model id (default: qwen3.7-max)
    QWEN_BASE_URL      — endpoint override for your region/workspace

Run with:
    DASHSCOPE_API_KEY=sk-... uv run python examples/qwen_ai.py
    DASHSCOPE_API_KEY=sk-... uv run python examples/qwen_ai.py --model qwen3.7-plus
    DASHSCOPE_API_KEY=sk-... uv run python examples/qwen_ai.py --thinking-budget 2048

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
from roomkit.providers.qwen import QwenAIProvider, QwenConfig

_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


async def main(args: argparse.Namespace) -> None:
    env = require_env("DASHSCOPE_API_KEY")

    provider = QwenAIProvider(
        QwenConfig(
            api_key=env["DASHSCOPE_API_KEY"],
            model=args.model,
            base_url=args.base_url,
            enable_thinking=args.thinking_budget is None or args.thinking_budget > 0,
        )
    )

    kit = RoomKit()

    cli = CLIChannel("you")
    ai = AIChannel(
        "assistant",
        provider=provider,
        system_prompt="You are a helpful assistant. Think step by step, then answer concisely.",
        thinking_budget=args.thinking_budget,
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    await kit.create_room(room_id="qwen-cli")
    await kit.attach_channel("qwen-cli", "you")
    await kit.attach_channel("qwen-cli", "assistant", category=ChannelCategory.INTELLIGENCE)

    budget = "model default" if args.thinking_budget is None else f"{args.thinking_budget} tokens"
    try:
        await cli.run(
            kit,
            room_id="qwen-cli",
            welcome=f"Qwen · {args.model} (thinking: {budget})\nType 'quit' to exit.",
        )
    finally:
        await provider.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Qwen CLI.")
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "qwen3.7-max"),
        help="Qwen model id (qwen3.7-plus, qwen3.6-flash, qwen3-coder-plus). Env: QWEN_MODEL.",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", _DEFAULT_BASE_URL),
        help="Model Studio endpoint for your region/workspace. Env: QWEN_BASE_URL.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Cap the reasoning trace, in tokens. 0 switches thinking off.",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging("qwen_ai")
    asyncio.run(main(_parse_args()))

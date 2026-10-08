"""Reasoning effort on a tool turn, on an OpenAI-compatible server.

Behind a ``base_url`` the OpenAI provider cannot know which model answers, so
on a turn that declares tools it leaves the configured ``reasoning_effort``
out unless the config says the server takes it there
(``supports_reasoning_effort_with_tools=True``, RFC §6.7). Left out, a
reasoning model thinks at its own default on every tool turn (for an agent
with tools, nearly every turn), and the provider logs one warning naming the
field.

This script sends the same tool turn with the field unset, then set, and
reports per run the reasoning and output tokens, the wall time, and whether
the model called the tool. Use it to check that your server takes the effort
beside tools before turning the field on: a server that does not answers the
second half with an error.

Run with:
    OPENAI_COMPAT_BASE_URL=https://api.inceptionlabs.ai/v1 \\
    OPENAI_COMPAT_API_KEY=... OPENAI_COMPAT_MODEL=mercury-2.5 \\
        uv run python examples/openai_compatible_tool_effort.py

``OPENAI_COMPAT_EFFORT`` picks the effort (default ``low``), ``RUNS`` the runs
per half (default 3).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging  # noqa: E402

from roomkit.providers.ai.base import AIContext, AIMessage, AITool  # noqa: E402
from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig  # noqa: E402

setup_logging("openai_compatible_tool_effort")

_WEATHER = AITool(
    name="get_weather",
    description="Current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)
_PROMPT = "Should I take an umbrella in Montreal this afternoon? Check before answering."


async def _run_one(provider: OpenAIAIProvider) -> dict[str, Any]:
    """One tool turn: its token counts, wall time and the tool it called."""
    started = time.monotonic()
    response = await provider.generate(
        AIContext(messages=[AIMessage(role="user", content=_PROMPT)], tools=[_WEATHER])
    )
    return {
        "reasoning": response.usage.get("reasoning_tokens", 0),
        "output": response.usage.get("output_tokens", 0),
        "ms": int((time.monotonic() - started) * 1000),
        "tool": ", ".join(call.name for call in response.tool_calls) or "-",
    }


async def _run_half(label: str, config: OpenAIConfig, runs: int) -> None:
    """Every run of one half on one provider, printed as it lands."""
    print(f"\n{label}")
    print(f"  {'reasoning tok':>13} {'output tok':>10} {'ms':>6}  tool")
    provider = OpenAIAIProvider(config)
    try:
        for _ in range(runs):
            try:
                row = await _run_one(provider)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue
            print(f"  {row['reasoning']:>13} {row['output']:>10} {row['ms']:>6}  {row['tool']}")
    finally:
        await provider.close()


async def main() -> None:
    env = require_env("OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_MODEL")
    effort = os.environ.get("OPENAI_COMPAT_EFFORT", "low")
    runs = int(os.environ.get("RUNS", "3"))
    base = OpenAIConfig(
        api_key=env["OPENAI_COMPAT_API_KEY"],
        base_url=env["OPENAI_COMPAT_BASE_URL"],
        model=env["OPENAI_COMPAT_MODEL"],
        reasoning_effort=effort,
        max_tokens=1024,
    )
    print(f"Model {base.model} at {base.base_url}, reasoning_effort={effort!r}")

    await _run_half("Field unset: the effort is left out of the tool turn", base, runs)
    takes_it = base.model_copy(update={"supports_reasoning_effort_with_tools": True})
    await _run_half("supports_reasoning_effort_with_tools=True: it is sent", takes_it, runs)


if __name__ == "__main__":
    asyncio.run(main())

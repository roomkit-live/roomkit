"""Reasoning effort on a tool turn, on an OpenAI-compatible server.

Behind a ``base_url`` the OpenAI provider cannot know which model answers, so
on a turn that declares tools it leaves the configured ``reasoning_effort``
out unless the config says the server takes it there
(``supports_reasoning_effort_with_tools=True``, RFC §6.7). Left out, a
reasoning model thinks at its own default on every tool turn (for an agent
with tools, nearly every turn), and the provider logs one warning naming the
field.

This script runs the same tool loop with the field unset, then set: a first
round that should call the tool, then a second round that reads the tool's
result, carrying the first round's reasoning back as RoomKit's tool loop
does, and should answer. Per run it reports each round's reasoning tokens and
wall time, the tool called and whether the second round answered. Use it to
check that your server takes the effort beside tools before turning the field
on: a server that does not answers the second half with an error, and one
that cannot read its own reasoning back fails on the second round.

Run with:
    OPENAI_COMPAT_BASE_URL=https://api.inceptionlabs.ai/v1 \\
    OPENAI_COMPAT_API_KEY=... OPENAI_COMPAT_MODEL=mercury-2.5 \\
        uv run python examples/openai_compatible_tool_effort.py

``OPENAI_COMPAT_EFFORT`` picks the effort (default ``low``), ``RUNS`` the runs
per half (default 3).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import require_env, setup_logging  # noqa: E402

from roomkit.providers.ai import round_parts, thinking_parts_of  # noqa: E402
from roomkit.providers.ai.base import (  # noqa: E402
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolResultPart,
)
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
_FORECAST = json.dumps({"conditions": "light rain from 2 pm", "temp_c": 11})


def _next_round(question: AIMessage, called: AIResponse) -> list[AIMessage]:
    """The conversation the second round reads: the question, the first round
    replayed with its reasoning and calls, and each call's result."""
    replayed = round_parts(thinking_parts_of(called), called.content, called.tool_calls)
    results = [
        AIToolResultPart(tool_call_id=call.id, name=call.name, result=_FORECAST)
        for call in called.tool_calls
    ]
    return [
        question,
        AIMessage(role="assistant", content=replayed),
        AIMessage(role="tool", content=list(results)),
    ]


async def _timed(provider: OpenAIAIProvider, messages: list[AIMessage]) -> tuple[AIResponse, int]:
    """One round with the weather tool declared, and its wall time in ms."""
    started = time.monotonic()
    response = await provider.generate(AIContext(messages=messages, tools=[_WEATHER]))
    return response, int((time.monotonic() - started) * 1000)


async def _run_loop(provider: OpenAIAIProvider) -> dict[str, Any]:
    """One tool loop: each round's reasoning tokens and time, the tool called,
    and whether the second round answered."""
    question = AIMessage(role="user", content=_PROMPT)
    first, first_ms = await _timed(provider, [question])
    row: dict[str, Any] = {
        "r1": first.usage.get("reasoning_tokens", 0),
        "ms1": first_ms,
        "tool": ", ".join(call.name for call in first.tool_calls) or "-",
        "r2": "-",
        "ms2": "-",
        "answered": "-",
    }
    if not first.tool_calls:
        return row
    second, second_ms = await _timed(provider, _next_round(question, first))
    row.update(
        r2=second.usage.get("reasoning_tokens", 0),
        ms2=second_ms,
        answered="yes" if second.content.strip() else "no",
    )
    return row


async def _run_half(label: str, config: OpenAIConfig, runs: int) -> None:
    """Every run of one half on one provider, printed as it lands."""
    print(f"\n{label}")
    print(f"  {'round 1 tok':>11} {'ms':>6}  {'tool':<12} {'round 2 tok':>11} {'ms':>6}  answered")
    provider = OpenAIAIProvider(config)
    try:
        for _ in range(runs):
            try:
                row = await _run_loop(provider)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue
            print(
                f"  {row['r1']:>11} {row['ms1']:>6}  {row['tool']:<12} "
                f"{row['r2']:>11} {row['ms2']:>6}  {row['answered']}"
            )
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

    await _run_half("Field unset: the effort is left out of the tool rounds", base, runs)
    takes_it = base.model_copy(update={"supports_reasoning_effort_with_tools": True})
    await _run_half("supports_reasoning_effort_with_tools=True: it is sent", takes_it, runs)


if __name__ == "__main__":
    asyncio.run(main())

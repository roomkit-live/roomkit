"""Ollama thinking-effort benchmark.

Sends the same multi-step reasoning prompt at each ``think`` value
(``False``, ``True``, ``"low"``, ``"medium"``, ``"high"``) and reports
output token count, thinking trace length, and wall time per run. Use
this to check whether your local model actually honors the string
effort levels added in Ollama 0.7+ — not every model does, and the
ones that don't will silently treat ``"high"`` and ``"low"`` as plain
``think=True``.

Read the output table:

* ``thinking_chars`` rises monotonically low → medium → high
  → your model honors effort levels.
* ``thinking_chars`` is roughly flat across low/medium/high
  (and equal to ``default`` row) → model downgraded strings to bool;
  the effort knob has no effect.

The first call cold-starts the model, so the ``off`` row is usually
slower than its small token count would suggest. Compare across the
*other* rows for meaningful numbers.

Run with:
    OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=qwen3:8b \\
        uv run python examples/ollama_think_levels.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging  # noqa: E402

from roomkit.providers.ai.base import AIContext, AIMessage  # noqa: E402
from roomkit.providers.ollama import (  # noqa: E402
    OllamaAIProvider,
    OllamaConfig,
    ThinkEffort,
)

setup_logging("ollama_think_levels")


# Multi-step constraint problem — enough surface for the model to
# reason through, so an honest effort knob should change trace length.
_PROMPT = (
    "I have 3 boxes. Box A weighs twice as much as Box B. Box C weighs "
    "5 kg more than Box A. The total weight is 50 kg. How much does "
    "each box weigh? Solve step by step, then verify your answer."
)

_LEVELS: list[tuple[str, bool | ThinkEffort | None]] = [
    ("off", False),
    ("default", True),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
]


async def _run_one(host: str, model: str, label: str, think: Any) -> dict[str, Any]:
    provider = OllamaAIProvider(OllamaConfig(host=host, model=model, think=think))
    try:
        t0 = time.monotonic()
        response = await provider.generate(
            AIContext(
                messages=[AIMessage(role="user", content=_PROMPT)],
                max_tokens=4096,
            )
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        print(response.thinking)
        return {
            "label": label,
            "eval_count": response.usage.get("output_tokens", 0),
            "prompt_eval_count": response.usage.get("input_tokens", 0),
            "thinking_chars": len(response.thinking or ""),
            "content_chars": len(response.content),
            "elapsed_ms": elapsed_ms,
        }
    finally:
        await provider.close()


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'level':<10} {'out tok':>8} {'think chars':>12} {'content chars':>14} {'ms':>8}")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['label']:<10} {r['eval_count']:>8} "
            f"{r['thinking_chars']:>12} {r['content_chars']:>14} "
            f"{r['elapsed_ms']:>8}"
        )


async def main() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

    print(f"\nThink-level benchmark — model={model} host={host}")
    print(f"Prompt: {_PROMPT}\n")

    rows: list[dict[str, Any]] = []
    for label, think in _LEVELS:
        print(f"  • {label:<8}", end=" ", flush=True)
        try:
            row = await _run_one(host, model, label, think)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue
        rows.append(row)
        print(
            f"out={row['eval_count']:>4} tok | "
            f"think={row['thinking_chars']:>5} chars | "
            f"{row['elapsed_ms']:>5}ms"
        )

    if rows:
        _print_table(rows)
        print(
            "\nIf thinking_chars rises monotonically low → medium → high your\n"
            "model honors effort levels. If it's flat, the model silently\n"
            "treats string effort as plain True — try gpt-oss or deepseek-r1.\n"
        )


if __name__ == "__main__":
    asyncio.run(main())

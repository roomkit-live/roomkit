"""Run with ``uv run python -m benchmarks.media <kind>`` from the repository."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.media import close, resample, taps, turn
from benchmarks.media.machine import environment, run_pass
from benchmarks.media.report import markdown

KINDS: dict[str, str] = {
    "close": "the kit's close against its bound (RK-12.10.4-056)",
    "turn": "the turn detector's fast-path latency (RK-12.3.12-022)",
    "taps": "the debug taps' cost on the pipeline's path (RK-12.3.15-020)",
    "resample": "the default resampler's artifacts (RK-12.3.3-007)",
}


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.media",
        description="Media durations and signal quality the RFC states, measured on mocks.",
    )
    parser.add_argument("kind", choices=sorted(KINDS), help="what to measure")
    parser.add_argument("--passes", type=_positive, default=3)
    parser.add_argument(
        "--output", type=Path, help="a fresh directory (default: benchmark-results/media-...)"
    )
    return parser.parse_args(argv)


def _measure(kind: str, output: Path) -> Callable[[], dict[str, list[float]]]:
    if kind == "close":
        return close.measure()
    if kind == "turn":
        return turn.measure()
    if kind == "taps":
        return taps.measure(disk=output)
    return resample.measure()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The measured paths warn of what they are expected to (the close naming
    # the work it abandoned, each tap session's start): kept out of the output.
    logging.getLogger("roomkit").setLevel(logging.ERROR)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmark-results") / f"media-{args.kind}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    measure = _measure(args.kind, output)
    passes = [run_pass(measure) for _ in range(args.passes)]
    text, verdict = markdown(args.kind, env, passes)
    results = {
        "kind": args.kind,
        "what": KINDS[args.kind],
        "environment": env,
        "verdict": verdict,
        "passes": [p.to_json() for p in passes],
    }
    (output / "results.json").write_text(json.dumps(results, indent=1))
    (output / "summary.md").write_text(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\nresults: {output}\n")
    return 1 if verdict is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

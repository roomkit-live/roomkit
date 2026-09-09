"""Run with ``uv run python -m benchmarks.chat --help`` from the repository."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess  # nosec B404
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.chat.harness import Harness
from benchmarks.chat.measurement import MeasuredProvider
from benchmarks.chat.report import append_sample, sanitize, summarize, write_report
from benchmarks.chat.scenarios import PROMPT, Scenario, scenarios
from roomkit import __version__
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider

logger = logging.getLogger("roomkit.benchmark")


def package_versions() -> dict[str, str | None]:
    """Record optional SDKs without requiring them for an offline mock run."""
    versions: dict[str, str | None] = {}
    for name in ("openai", "httpx", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def read_key(path: Path | None, provider: str) -> str:
    variable = f"{provider.upper()}_API_KEY"
    if path:
        raw = path.expanduser().read_text().strip()
        for line in raw.splitlines():
            if line.startswith(variable + "="):
                return line.split("=", 1)[1].strip().strip("\"'")
        if raw and "\n" not in raw:
            return raw
        raise ValueError(f"Expected a raw key or {variable}=... in key file")
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"Set {variable} or pass --key-file")
    return value


def make_provider(args: argparse.Namespace, key: str) -> AIProvider:
    if args.provider == "mock":
        from roomkit.providers.ai.mock import MockAIProvider

        return MockAIProvider(responses=["pong"], streaming=True)
    common = {
        "api_key": key,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "timeout": 30.0,
        "max_retries": 0,
    }
    if args.base_url:
        common["base_url"] = args.base_url
    if args.provider == "cerebras":
        from roomkit import CerebrasAIProvider, CerebrasConfig

        return CerebrasAIProvider(CerebrasConfig(**common))
    from roomkit.providers.openai import OpenAIAIProvider, OpenAIConfig

    return OpenAIAIProvider(OpenAIConfig(**common))


async def run_suite(
    args: argparse.Namespace,
    provider: AIProvider,
    selected: list[Scenario],
    workdir: Path,
    *,
    secret: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warm = MeasuredProvider(provider)
    for _ in range(args.warmups):
        await warm.generate(
            AIContext(messages=[AIMessage(role="user", content=PROMPT)], temperature=0)
        )
    warmups = [{"elapsed_ms": (c.end - c.start) * 1000, "usage": c.usage} for c in warm.calls]
    samples: list[dict[str, Any]] = []
    # Reproducible scenario ordering, with no security-sensitive randomness.
    rng = random.Random(args.seed)  # nosec B311
    for iteration in range(1, args.repetitions + 1):
        order = list(selected)
        rng.shuffle(order)
        for scenario in order:
            sample: dict[str, Any] = {
                "scenario": scenario.name,
                "iteration": iteration,
                "status": "failed",
                "metrics": {},
            }
            if args.provider == "mock" and not scenario.mock_supported:
                sample.update(status="skipped", error="This scenario requires a live model")
                samples.append(sample)
                continue
            sqlite = (
                workdir / f"{scenario.name}-{iteration}.db" if args.store == "sqlite" else None
            )
            options = await asyncio.to_thread(scenario.make_options)
            h = Harness(provider, streaming=scenario.streaming, sqlite=sqlite, **options)
            h.start = time.perf_counter()
            try:
                await h.add_room("main")
                h.start = time.perf_counter()
                try:
                    async with asyncio.timeout(args.deadline):
                        await scenario.run(h)
                finally:
                    h.end = time.perf_counter()
                await h.validate()
                sample["status"] = "passed" if h.checks and all(h.checks.values()) else "failed"
            except Exception as exc:
                sample["error"] = type(exc).__name__ + ": " + str(exc)[:1000]
                if not h.end:
                    h.end = time.perf_counter()
            finally:
                sample["metrics"] = h.measurements()
                if sample["status"] == "failed":
                    sample["metrics"]["details"]["synthetic_answer"] = h.answer()[:2000]
                await h.close()
            sample = sanitize(sample, secret)
            samples.append(sample)
            await asyncio.to_thread(append_sample, args.output / "samples.jsonl", sample)
            failures = [name for name, value in h.checks.items() if not value]
            logger.warning(
                "%s #%s %s %.1fms %s",
                scenario.name,
                iteration,
                sample["status"],
                sample["metrics"]["elapsed_ms"],
                sample.get("error") or ", ".join(failures),
            )
    return samples, warmups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["cerebras", "openai", "mock"], default="cerebras")
    parser.add_argument("--suite", choices=["chat", "quality"], default="chat")
    parser.add_argument(
        "--variants", type=int, default=3, help="Variants per model-quality family"
    )
    parser.add_argument("--model", default="qwen-3.8-27b")
    parser.add_argument("--base-url")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline", type=float, default=90)
    parser.add_argument("--store", choices=["memory", "sqlite"], default="memory")
    parser.add_argument("--scenarios", default="all", help="Comma-separated scenario ids, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/chat"))
    parser.add_argument("--compare", type=Path, help="Prior results.json")
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("variants must be positive")
    if args.suite == "quality":
        from benchmarks.chat.quality import quality_scenarios

        catalog = quality_scenarios(args.seed, args.variants)
    else:
        catalog = scenarios()
    if args.list:
        for scenario in catalog:
            sys.stdout.write(f"{scenario.name:20} {scenario.description}\n")
        return 0
    if args.repetitions < 1 or args.warmups < 0 or args.deadline <= 0 or args.max_tokens < 1:
        parser.error("repetitions/deadline must be positive; warmups cannot be negative")
    if (args.output / "results.json").exists() or (args.output / "samples.jsonl").exists():
        parser.error("Output already contains a run; choose a new --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        {s.name for s in catalog} if args.scenarios == "all" else set(args.scenarios.split(","))
    )
    unknown = names - {s.name for s in catalog}
    if unknown:
        parser.error(f"Unknown scenarios: {sorted(unknown)}")
    selected = [s for s in catalog if s.name in names]
    key = "" if args.provider == "mock" else read_key(args.key_file, args.provider)
    logging.basicConfig(level=logging.ERROR)
    logger.setLevel(logging.WARNING)
    provider = make_provider(args, key)

    async def execute(workdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            return await run_suite(args, provider, selected, workdir, secret=key)
        finally:
            await provider.close()

    started = datetime.now(UTC).isoformat()
    fingerprint = hashlib.sha256()
    for source in sorted([*Path("src/roomkit").rglob("*.py"), *Path("benchmarks").rglob("*")]):
        if source.is_file() and source.suffix in {".py", ".md"}:
            fingerprint.update(str(source).encode())
            fingerprint.update(source.read_bytes())
    with tempfile.TemporaryDirectory(prefix="roomkit-chat-bench-") as folder:
        samples, warmups = asyncio.run(execute(Path(folder)))
    # Use the developer's Git on PATH with fixed arguments and no shell/input.
    commit = subprocess.check_output(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True)  # nosec B603 B607
    )
    document = {
        "schema_version": 1,
        "environment": {
            "date": started,
            "suite": args.suite,
            "variants": args.variants if args.suite == "quality" else None,
            "provider": args.provider,
            "model": provider.model_name,
            "roomkit": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit": commit,
            "source_sha256": fingerprint.hexdigest(),
            "packages": package_versions(),
            "deadline_seconds": args.deadline,
            "sdk_timeout_seconds": 30,
            "sdk_retries": 0,
            "dirty": dirty,
            "store": args.store,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "warmups": warmups,
        "coverage": [
            {
                "name": s.name,
                "features": s.features,
                "description": s.description,
                "fault_injection": s.fault_injection,
            }
            for s in selected
        ],
        "samples": samples,
        "summary": summarize(samples),
    }
    # Defense in depth: neither key paths nor keys belong in persisted results.
    document = sanitize(document, key)
    previous = json.loads(args.compare.read_text()) if args.compare else None
    write_report(args.output, document, previous)
    sys.stdout.write(f"Results: {args.output / 'report.md'}\n")
    return int(any(sample["status"] == "failed" for sample in samples))


if __name__ == "__main__":
    raise SystemExit(main())

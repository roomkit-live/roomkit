"""Portable benchmark results: raw samples, aggregate CSV and Markdown."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def sanitize(document: dict[str, Any], secret: str) -> dict[str, Any]:
    """Remove a credential even if an upstream exception unexpectedly echoes it."""
    if not secret:
        return document
    return json.loads(json.dumps(document).replace(secret, "[REDACTED]"))


def append_sample(path: Path, sample: dict[str, Any]) -> None:
    """Checkpoint completed samples so an interrupted run retains its evidence."""
    with path.open("a") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in dict.fromkeys(s["scenario"] for s in samples):
        group = [s for s in samples if s["scenario"] == name and s["status"] != "skipped"]
        if not group:
            continue
        passed = [s for s in group if s["status"] == "passed"]
        row: dict[str, Any] = {"scenario": name, "passed": len(passed), "runs": len(group)}
        for metric in ("elapsed_ms", "first_text_ms", "provider_wall_ms", "residual_ms"):
            values = [s["metrics"][metric] for s in passed if s["metrics"].get(metric) is not None]
            row[f"{metric}_median"] = statistics.median(values) if values else None
            row[f"{metric}_p95"] = percentile(values, 0.95)
        row["provider_calls"] = sum(s["metrics"].get("provider_calls", 0) for s in group)
        row["input_tokens"] = sum(
            s["metrics"].get("usage", {}).get("input_tokens", 0) for s in group
        )
        row["output_tokens"] = sum(
            s["metrics"].get("usage", {}).get("output_tokens", 0) for s in group
        )
        row["cache_read_input_tokens"] = sum(
            s["metrics"].get("usage", {}).get("cache_read_input_tokens", 0) for s in group
        )
        rows.append(row)
    return rows


def markdown(document: dict[str, Any], previous: dict[str, Any] | None = None) -> str:
    meta = document["environment"]
    lines = [
        "# RoomKit chat E2E benchmark",
        "",
        f"Date: {meta['date']} · Provider: `{meta['provider']}` · Model: `{meta['model']}`",
        f"Commit: `{meta['commit']}` · Dirty tree: `{meta['dirty']}` · Store: `{meta['store']}`",
        f"Python: `{meta['python']}` · Platform: `{meta['platform']}`",
        "",
        "## Method",
        "",
        "Synthetic prompts only. One reused provider client, fresh RoomKit and rooms per sample. "
        "Warm-ups are recorded separately. Scenario order is shuffled with a recorded seed. "
        "E2E time covers the scenario after initial room setup, including model/tool calls; "
        "the in-process WebSocket callback is the delivery boundary. "
        "A real browser or network socket is not measured. "
        "Scenario assertions are timed; subscriber drain and common validation are excluded.",
        "",
        "`residual_ms` subtracts the union of provider and tool waits from elapsed time. "
        "Streaming provider waits measure awaiting the next item, excluding consumer processing. "
        "The residual includes RoomKit, store, hooks, scheduling and instrumentation; "
        "it is not a pure CPU measurement. Provider waits include SDK, network and queueing. "
        "First text means visible text, not reasoning. Quantiles describe this sample only. "
        "Failed runs stay in the raw data and success denominator, "
        "but are excluded from latency aggregates.",
        "",
        "## Results",
        "",
        "| Scenario | Passed | Median total ms | p95 total ms | "
        "Median first text ms | Median residual ms | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any) -> str:
        return "—" if value is None else f"{value:.2f}"

    for row in document["summary"]:
        lines.append(
            f"| {row['scenario']} | {row['passed']}/{row['runs']} | "
            f"{fmt(row['elapsed_ms_median'])} | {fmt(row['elapsed_ms_p95'])} | "
            f"{fmt(row['first_text_ms_median'])} | {fmt(row['residual_ms_median'])} | "
            f"{row['provider_calls']} |"
        )
    lines += ["", "## Coverage", "", "| Scenario | Features | Fault injection |", "|---|---|---|"]
    for scenario in document["coverage"]:
        lines.append(
            f"| {scenario['name']} | {', '.join(scenario['features'])} | "
            f"{scenario['fault_injection']} |"
        )
    lines += [
        "",
        "Not yet covered: real network WebSocket/server load, external MCP servers, "
        "human approval flows, "
        "orchestration/handoffs, long-context compaction/eviction, rich media, Redis/Postgres, "
        "distributed processes, transport retry/rate limits and voice. "
        "This is a growing chat suite, not a claim of complete RFC conformance.",
        "",
        "## Failed or skipped samples",
        "",
    ]
    for sample in document["samples"]:
        if sample["status"] != "passed":
            failed = [
                k
                for k, passed in sample.get("metrics", {}).get("checks", {}).items()
                if not passed
            ]
            lines.append(
                f"- `{sample['scenario']}` #{sample['iteration']}: {sample['status']}; "
                f"{sample.get('error') or ', '.join(failed)}"
            )
    if previous:
        lines += [
            "",
            "## Comparison to prior run",
            "",
            "Descriptive changes only; compare equivalent environments and configurations.",
            "",
            "| Scenario | Previous median ms | Current median ms | Change |",
            "|---|---:|---:|---:|",
        ]
        old = {r["scenario"]: r for r in previous["summary"]}
        for row in document["summary"]:
            before = old.get(row["scenario"], {}).get("elapsed_ms_median")
            after = row["elapsed_ms_median"]
            if before and after is not None:
                lines.append(
                    f"| {row['scenario']} | {before:.2f} | {after:.2f} | "
                    f"{(after / before - 1) * 100:+.1f}% |"
                )
    return "\n".join(lines) + "\n"


def write_report(
    directory: Path, document: dict[str, Any], previous: dict[str, Any] | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    quality_text = ""
    if document["environment"].get("suite") == "quality":
        from benchmarks.chat.quality_report import quality_markdown, quality_summary

        document["quality_summary"] = quality_summary(document["samples"])
        quality_text = "\n" + quality_markdown(document)
        quality_rows = document["quality_summary"]
        if quality_rows:
            with (directory / "quality.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
                writer.writeheader()
                writer.writerows(quality_rows)
    (directory / "results.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    )
    (directory / "report.md").write_text(markdown(document, previous) + quality_text)
    rows = document["summary"]
    if rows:
        with (directory / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

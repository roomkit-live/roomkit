"""Accuracy metrics kept separate from infrastructure checks and pass-only timing."""

from __future__ import annotations

import statistics
from typing import Any

from benchmarks.chat.report import percentile


def quality_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    families = sorted({s["scenario"].rsplit("_v", 1)[0] for s in samples})
    for family in families:
        group = [
            s
            for s in samples
            if s["scenario"].rsplit("_v", 1)[0] == family and s["status"] != "skipped"
        ]
        if not group:
            continue
        graded = [
            s for s in group if s["metrics"].get("details", {}).get("quality", {}).get("criteria")
        ]
        criteria = [s["metrics"]["details"]["quality"]["criteria"] for s in graded]
        scores = [sum(c.values()) / len(c) for c in criteria]
        latencies = [s["metrics"]["elapsed_ms"] for s in graded]
        infrastructure = sum(
            bool(s.get("error"))
            or any(
                not ok
                for k, ok in s["metrics"].get("checks", {}).items()
                if not k.startswith("quality_")
            )
            for s in group
        )
        rows.append(
            {
                "family": family,
                "attempts": len(group),
                "graded": len(graded),
                "variants": len({s["scenario"] for s in group}),
                "fully_correct": sum(all(c.values()) for c in criteria),
                "task_correct": sum(
                    all(ok for key, ok in c.items() if key not in {"json_only", "schema_keys"})
                    for c in criteria
                ),
                "format_compliant": sum(
                    c.get("json_only", False) and c.get("schema_keys", False) for c in criteria
                ),
                "truncated": sum(
                    any(c.get("finish_reason") == "length" for c in s["metrics"].get("calls", []))
                    for s in group
                ),
                "empty_answers": sum(
                    not s["metrics"]["details"]["quality"].get("answer", "").strip()
                    for s in graded
                ),
                "mean_score": statistics.mean(scores) if scores else None,
                "json_only": sum(c.get("json_only", False) for c in criteria),
                "infrastructure_failures": infrastructure,
                "elapsed_ms_median_all_graded": statistics.median(latencies)
                if latencies
                else None,
                "elapsed_ms_p95_all_graded": percentile(latencies, 0.95),
            }
        )
    return rows


def quality_markdown(document: dict[str, Any]) -> str:
    lines = [
        "## Model quality",
        "",
        "Synthetic, deterministic cases with verifiable answers; no LLM judge. "
        "Variants are generated from a fixed seed and remain identical across reasoning settings. "
        "Repeated trials are not independent questions. This small custom suite is not a general "
        "model ranking, nor a production reliability estimate.",
        "",
        "Fully correct means every rubric criterion passed, including output format. "
        "Task correct excludes the JSON-only and exact-key-set criteria; "
        "format compliance requires both. Truncated counts cases with a length-limited call. "
        "Empty counts graded turns without visible text, even if the provider reports stop. "
        "Partial score is the mean fraction of passed criteria per graded answer. "
        "Infrastructure failures are separate. Unlike the generic timing table, the quality "
        "table includes incorrect graded answers in its latency statistics. "
        "Unfinished runs have no quality score and remain in the attempts denominator.",
        "",
        "| Family | Task correct | Fully correct | Format OK | Truncated | Empty | "
        "Infra errors | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in document["quality_summary"]:
        latency = row["elapsed_ms_median_all_graded"]
        timing = f"{latency:.1f}" if latency is not None else "—"
        lines.append(
            f"| {row['family']} | {row['task_correct']}/{row['attempts']} | "
            f"{row['fully_correct']}/{row['attempts']} | {row['format_compliant']} | "
            f"{row['truncated']} | {row['empty_answers']} | "
            f"{row['infrastructure_failures']} | {timing} |"
        )
    lines += ["", "### Incorrect answers", ""]
    for sample in document["samples"]:
        quality = sample.get("metrics", {}).get("details", {}).get("quality", {})
        failed = [name for name, ok in quality.get("criteria", {}).items() if not ok]
        if failed:
            lines += [
                f"- `{sample['scenario']}` #{sample['iteration']}: {', '.join(failed)}. "
                "See the full prompt, expected answer and model output in results.json."
            ]
    return "\n".join(lines) + "\n"

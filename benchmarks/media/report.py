"""The summary of a media measurement, per kind, against the RFC's figure where
it gives one."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from benchmarks.media import close, turn
from benchmarks.media.machine import Pass, summary


def _pooled(passes: Sequence[Pass]) -> dict[str, list[float]]:
    pooled: dict[str, list[float]] = {}
    for one in passes:
        for key, values in one.samples.items():
            pooled.setdefault(key, []).extend(values)
    return pooled


def _close(passes: Sequence[Pass]) -> tuple[list[str], bool | None]:
    lines = [
        "| Case | n | p50 (s) | p99 (s) | max (s) | Past the grace (s) | Bound (s) | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    ok = True
    for case, values in _pooled(passes).items():
        s, bound = summary(values), close.bound_s(case)
        past = max(0.0, s["max"] - close.GRACE_S)
        # Within the RFC's sum of waits, the close's own work besides, and
        # work that honours its cut spends no unwind.
        within = s["max"] <= bound + close.OWN_WORK_S
        passed = within and (case.endswith("-ignores") or past < close.CUT_UNWIND_S)
        ok = ok and passed
        lines.append(
            f"| {case} | {s['n']} | {s['p50']:.3f} | {s['p99']:.3f} | {s['max']:.3f} "
            f"| {past:.3f} | {bound:.2f} | {'pass' if passed else 'FAIL'} |"
        )
    lines += [
        "",
        f"A case passes within its bound plus {1000 * close.OWN_WORK_S:.0f} ms for the "
        "close's own work between its waits, and, for work that honours its cut, "
        "with no unwind spent.",
    ]
    return lines, ok


def _turn(passes: Sequence[Pass]) -> tuple[list[str], bool | None]:
    pooled = _pooled(passes)
    with_, without = summary(pooled["with-detector"]), summary(pooled["without-detector"])
    lines = [
        "| Case | n | p50 (ms) | p99 (ms) | max (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in (("with-detector", with_), ("without-detector", without)):
        lines.append(f"| {name} | {s['n']} | {s['p50']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |")
    # Two runs' tails do not subtract: the medians do.
    lines.append(f"| added (p50) | | {with_['p50'] - without['p50']:.3f} | | |")
    # The detector path's whole latency bounds what the detector adds.
    passed = with_["max"] <= turn.FIGURE_MS
    lines.append("")
    lines.append(
        f"Figure: {turn.FIGURE_MS:.0f} ms at most added; the detector path's max, "
        f"{with_['max']:.3f} ms, bounds it: {'pass' if passed else 'FAIL'}."
    )
    return lines, passed


def _taps(passes: Sequence[Pass]) -> tuple[list[str], bool | None]:
    pooled = _pooled(passes)
    lines = [
        "| Rate | Taps | n | p50 (µs) | p99 (µs) | max (µs) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, values in pooled.items():
        rate, taps = key.split("-", 1)
        s = summary(values)
        lines.append(
            f"| {rate} | {taps} | {s['n']} | {s['p50']:.1f} | {s['p99']:.1f} | {s['max']:.1f} |"
        )
    lines += ["", "The RFC gives no figure."]
    return lines, None


def _resample(passes: Sequence[Pass]) -> tuple[list[str], bool | None]:
    pooled = _pooled(passes)
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in pooled.items():
        provider, pair, figure = key.split(" ")
        s = summary(values)
        rows.setdefault((provider, pair), {})[figure] = s["p50"]
    lines = [
        "| Resampler | Pair (Hz) | THD+N (dB) | Alias (dB) | Image (dB) | p50 µs/frame |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (provider, pair), figures in rows.items():
        alias = figures.get("alias_db")
        image = figures.get("image_db")
        lines.append(
            f"| {provider} | {pair} | {figures['thdn_db']:.1f} "
            f"| {'' if alias is None else f'{alias:.1f}'} "
            f"| {'' if image is None else f'{image:.1f}'} | {figures['us_per_frame']:.1f} |"
        )
    lines += ["", "The RFC gives no figure, and no threshold is decided."]
    return lines, None


REPORTS: dict[str, Callable[[Sequence[Pass]], tuple[list[str], bool | None]]] = {
    "close": _close,
    "turn": _turn,
    "taps": _taps,
    "resample": _resample,
}


def markdown(kind: str, env: dict[str, Any], passes: Sequence[Pass]) -> tuple[str, bool | None]:
    """The summary of a run, and its verdict (None where the RFC gives no figure)."""
    lines, verdict = REPORTS[kind](passes)
    head = [
        f"# Media measurement: {kind}",
        "",
        f"- date: {env['date']}",
        f"- machine: {env['cpu_model']}, kernel {env['kernel']}, CPUs {env['cpus']}",
        f"- commit: {env['commit']}{' (dirty)' if env['dirty'] else ''}",
        f"- Python: {env['python']}",
        "- passes: "
        + ", ".join(
            f"{p.seconds:.1f} s, other load {100 * p.other_load:.1f} % of a core"
            + (f", {p.attempts} attempts" if p.attempts > 1 else "")
            for p in passes
        ),
        "",
    ]
    return "\n".join([*head, *lines, ""]), verdict

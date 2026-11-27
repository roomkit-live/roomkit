"""What every media measurement records: the machine, the commit, and the
other load on the measured cores, pass by pass (as ``benchmarks.rtp`` does)."""

from __future__ import annotations

import math
import os
import platform
import resource
import subprocess  # nosec B404
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.rtp.processes import cpu_busy_seconds

# Other processes may take this share of one measured core during a pass;
# past it the pass is run again (benchmarks/rtp voids a point at 10 %).
MAX_OTHER_LOAD = 0.10
# How many times a pass spoiled by other load is run again.
RETRIES = 2


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # nosec B603 B607
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment() -> dict[str, Any]:
    """The date, the machine, the commit and the Python of a measurement."""
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.splitlines()
            if ln.startswith("model name")
        ),
        None,
    )
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": model,
        "kernel": platform.release(),
        "cpus": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def _own_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


@dataclass(frozen=True)
class Pass:
    """One pass: its samples, the wall time it took, and the CPU time other
    processes spent on the measured cores meanwhile, as a share of one core."""

    samples: dict[str, list[float]]
    seconds: float
    other_load: float
    attempts: int

    def to_json(self) -> dict[str, Any]:
        return {
            "seconds": round(self.seconds, 3),
            "other_load": round(self.other_load, 4),
            "attempts": self.attempts,
            "samples": self.samples,
        }


def run_pass(measure: Callable[[], dict[str, list[float]]]) -> Pass:
    """Run *measure* once, again while other processes spoil it (up to
    :data:`RETRIES` more times); the last attempt is kept either way."""
    cpus = sorted(os.sched_getaffinity(0))
    attempts = 0
    while True:
        attempts += 1
        busy, own = cpu_busy_seconds(cpus), _own_cpu_seconds()
        started = time.monotonic()
        samples = measure()
        seconds = time.monotonic() - started
        others = cpu_busy_seconds(cpus) - busy - (_own_cpu_seconds() - own)
        load = max(0.0, others) / seconds if seconds > 0 else 0.0
        if load <= MAX_OTHER_LOAD or attempts > RETRIES:
            return Pass(samples, seconds, load, attempts)


def quantile(values: Sequence[float], q: float) -> float:
    """The *q* quantile, nearest rank: p50, p99 and max (q=1) of the samples."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": quantile(values, 0.5),
        "p99": quantile(values, 0.99),
        "max": quantile(values, 1.0),
    }

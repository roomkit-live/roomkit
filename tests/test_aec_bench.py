"""Tests for the offline AEC bench (``scripts/aec_bench.py``).

The bench is what will judge the AEC fixes, so its own math has to be
trustworthy: a scorer that misclassifies doubletalk as a leak would send
the fixes chasing ghosts.  Everything runs against a synthetic dump with
known geometry — a sine reference, a delayed/attenuated echo, a quiet
stretch and a doubletalk stretch.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aec_bench.py"

SAMPLE_RATE = 24000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms


def _load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("aec_bench", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["aec_bench"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


def _sine_frame(freq: float, amplitude: int, index: int) -> bytes:
    samples = bytearray()
    for i in range(FRAME_SAMPLES):
        t = (index * FRAME_SAMPLES + i) / SAMPLE_RATE
        value = int(amplitude * math.sin(2 * math.pi * freq * t))
        samples += value.to_bytes(2, "little", signed=True)
    return bytes(samples)


def _silence() -> bytes:
    return b"\x00\x00" * FRAME_SAMPLES


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _write_dump(path: Path, events: list[dict]) -> Path:
    header = {
        "format": "aec-dump/1",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
        "stream": "s1",
        "provider": "synthetic",
    }
    dump_file = path / "events.jsonl"
    with dump_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for event in events:
            f.write(json.dumps(event) + "\n")
    return dump_file


def _synthetic_events() -> list[dict]:
    """60 frames echo (out == in: a do-nothing AEC), 30 quiet, 30 doubletalk."""
    events: list[dict] = []
    ns = 0

    def ref(data: bytes) -> None:
        nonlocal ns
        ns += 10_000_000
        events.append({"t": "ref", "ns": ns, "d": _b64(data)})

    def cap(in_data: bytes, out_data: bytes) -> None:
        nonlocal ns
        ns += 10_000_000
        events.append({"t": "cap", "ns": ns, "i": _b64(in_data), "o": _b64(out_data)})

    # Echo-active: reference playing, capture is 25% echo, output unchanged.
    for i in range(60):
        playback = _sine_frame(440, 8000, i)
        echo = _sine_frame(440, 2000, i)
        ref(playback)
        cap(echo, echo)
    # Quiet: no reference, faint room noise.
    for i in range(30):
        noise = _sine_frame(60, 40, i)
        cap(noise, noise)
    # Doubletalk: reference playing, user much louder than the echo.
    for i in range(30):
        ref(_sine_frame(440, 8000, i))
        cap(_sine_frame(220, 12000, i), _sine_frame(220, 12000, i))
    return events


def test_measure_classifies_the_regimes(tmp_path):
    dump = bench.load_dump(_write_dump(tmp_path, _synthetic_events()))
    c = bench.classify(bench.measure(dump))

    assert len(c.active) == 6  # 60 echo frames → six 100 ms windows
    assert len(c.quiet) == 2  # tail decays over the first quiet window
    assert len(c.faint) == 1  # decaying tail window: nothing measurable
    assert len(c.doubletalk) == 3


def test_do_nothing_aec_scores_all_leaks(tmp_path):
    dump = bench.load_dump(_write_dump(tmp_path, _synthetic_events()))
    s = bench.score(bench.measure(dump))

    assert s.active == 6
    assert s.median_db == pytest.approx(0.0, abs=0.5)
    assert s.leaks == s.active  # out == in: nothing was cancelled
    assert s.doubletalk == 3


def test_replay_outputs_override_recorded(tmp_path):
    dump = bench.load_dump(_write_dump(tmp_path, _synthetic_events()))
    n_cap = sum(1 for e in dump.events if e.kind == "cap")
    # A perfect canceller: silence out whenever echo was the only input.
    perfect = [_silence()] * n_cap
    s = bench.score(bench.measure(dump, perfect))
    assert s.leaks == 0
    assert s.median_db == bench.ATTENUATION_CAP_DB


def test_activation_toggles_parse_and_are_ignored_by_measure(tmp_path):
    events = _synthetic_events()
    events.insert(0, {"t": "act", "ns": 1, "a": True})
    dump = bench.load_dump(_write_dump(tmp_path, events))
    assert dump.events[0].kind == "act"
    assert dump.events[0].active is True
    assert bench.score(bench.measure(dump)).active == 6


def test_report_cli_smoke(tmp_path, capsys):
    _write_dump(tmp_path, _synthetic_events())
    assert bench.main(["report", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "recorded" in out
    assert "median attenuation" in out


def test_rejects_foreign_files(tmp_path):
    bad = tmp_path / "events.jsonl"
    bad.write_text('{"format": "something-else"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        bench.load_dump(tmp_path)

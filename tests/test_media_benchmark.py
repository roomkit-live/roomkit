"""The media measurements measure what they say (RMK-2160)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from benchmarks.media import __main__ as cli
from benchmarks.media import close, resample, taps, turn
from benchmarks.media.machine import Pass, quantile
from benchmarks.media.report import markdown

from roomkit.voice.pipeline.resampler import NumpyResamplerProvider, SincResamplerProvider

_ENV = {
    "date": "2026-11-27T00:00:00+00:00",
    "cpu_model": "cpu",
    "kernel": "k",
    "cpus": [2],
    "python": "3.13",
    "commit": "c",
    "dirty": False,
}


@pytest.mark.wallclock
async def test_the_close_of_work_that_honours_its_cut_takes_the_grace_alone(
    close_grace: float, cut_unwind: float
) -> None:
    for case in ("hook-honours", "tool-honours"):
        assert await close.sample(case, grace=close_grace) < close_grace + cut_unwind


@pytest.mark.wallclock
async def test_the_close_of_work_that_ignores_its_cut_spends_its_two_unwinds(
    close_grace: float, cut_unwind: float
) -> None:
    assert await close.sample("hook-ignores", grace=close_grace) >= close_grace + 2 * cut_unwind


def test_a_close_past_its_bound_or_an_unwind_spent_by_honouring_work_fails() -> None:
    within = {
        "hook-honours": [close.GRACE_S + 0.01],
        "hook-ignores": [close.GRACE_S + 2.01],
        "tool-honours": [0.001],  # a turn is cut at once: no grace to wait out
    }
    assert markdown("close", _ENV, [Pass(within, 3.0, 0.0, 1)])[1] is True
    unwound = {"hook-honours": [close.GRACE_S + 1.01]}
    assert markdown("close", _ENV, [Pass(unwound, 1.1, 0.0, 1)])[1] is False
    own_work = {"tool-ignores": [close.bound_s("tool-ignores") + 0.006]}
    assert markdown("close", _ENV, [Pass(own_work, 8.1, 0.0, 1)])[1] is True
    past = {"hook-ignores": [close.bound_s("hook-ignores") + close.OWN_WORK_S + 0.001]}
    assert markdown("close", _ENV, [Pass(past, 3.2, 0.0, 1)])[1] is False


async def test_each_turn_is_timed_from_its_transcript_to_its_routed_message() -> None:
    for detector in (True, False):
        latencies = await turn._turn_latencies(2, detector=detector)
        assert len(latencies) == 2 and all(ms > 0 for ms in latencies)


def test_the_taps_cost_is_timed_per_frame_with_and_without_taps(tmp_path: Path) -> None:
    level = logging.getLogger("roomkit").level
    samples = taps.measure(frames=5, disk=tmp_path)()

    assert logging.getLogger("roomkit").level == level  # what the tests capture stays

    assert {key: len(values) for key, values in samples.items()} == dict.fromkeys(taps.CASES, 5)
    assert list(tmp_path.iterdir()) == []  # the taps' files are removed


def test_the_quality_figures_tell_a_clean_resampler_from_a_lossy_one() -> None:
    tone = resample._tone(16000, resample.TONE_HZ)
    passed, _ = resample._resample(NumpyResamplerProvider(), tone, 16000, 16000)
    assert resample.thdn_db(passed, 16000, resample.TONE_HZ) < -90
    high = resample._tone(16000, 6000.0)
    filtered, _ = resample._resample(SincResamplerProvider(), high, 16000, 8000)
    assert resample.level_db(filtered, high) < -30
    assert set(resample._pair(NumpyResamplerProvider(), 16000, 8000)) == {
        "thdn_db",
        "alias_db",
        "us_per_frame",
    }


@pytest.mark.parametrize(("src", "dst"), [p for p in resample.PAIRS if p[1] < p[0]])
def test_the_alias_tone_folds_inside_the_band_never_onto_0_hz(src: int, dst: int) -> None:
    hz = resample.alias_tone_hz(src, dst)
    folded = abs(hz - dst * round(hz / dst))
    assert dst / 2 < hz < src / 2
    assert dst / 8 < folded < dst / 2


def test_quantiles_are_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert (quantile(values, 0.5), quantile(values, 0.99), quantile(values, 1.0)) == (
        50.0,
        99.0,
        100.0,
    )


@pytest.mark.parametrize("argv", [["voice"], ["close", "--passes", "0"]])
def test_cli_refuses_what_it_cannot_measure(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)

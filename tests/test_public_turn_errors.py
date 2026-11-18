"""Errors an application constructs or inspects on the text tool and task paths."""

from __future__ import annotations

import pytest

from roomkit import (
    ChannelRefusalError,
    TaskCutShortError,
    TaskTurnFailedError,
    ToolRefusedError,
    ToolTimeoutError,
    TurnCutShortError,
)


def test_the_channels_refusal_keeps_its_words_and_family() -> None:
    refusal = ChannelRefusalError("outside the turn")
    assert isinstance(refusal, ToolRefusedError)
    assert refusal.message == str(refusal) == "outside the turn"


@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [
        (0.012345678, "0.0123457"),
        (1e-5, "1e-05"),
        (1e6, "1e+06"),
        (-0.0, "-0"),
        (float("inf"), "inf"),
        (float("nan"), "nan"),
    ],
)
def test_the_tool_timeout_keeps_the_bound_and_formats_the_quoted_name(
    seconds: float, rendered: str
) -> None:
    error = ToolTimeoutError("wait's turn", seconds)
    assert error.name == "wait's turn" and error.timeout is seconds
    assert str(error) == f'tool "wait\'s turn" did not answer within {rendered} s'


def test_the_base_turn_cut_allows_an_unspecified_reason() -> None:
    error = TurnCutShortError("operator stopped the turn", None)
    assert error.reason is None and str(error) == "operator stopped the turn"


@pytest.mark.parametrize("narration", [None, "", "Searching"])
def test_a_task_cut_carries_its_open_reason_and_optional_narration(narration: str | None) -> None:
    error = TaskCutShortError("custom stop", narration)
    assert isinstance(error, TurnCutShortError)
    assert (error.reason, error.narration) == ("custom stop", narration)
    assert str(error) == TaskCutShortError.message_for("custom stop")
    assert str(error) == "The worker's turn ended custom stop before its answer"


def test_a_failed_task_turn_keeps_the_original_exception_as_its_cause() -> None:
    error = RuntimeError("upstream failed")
    failed = TaskTurnFailedError(error, "interrupted", "Searching")
    assert (failed.reason, failed.narration) == ("interrupted", "Searching")
    assert failed.__cause__ is error and str(failed) == "upstream failed"
    error.args = ("updated",)
    assert str(failed) == "upstream failed" and str(failed.__cause__) == "updated"

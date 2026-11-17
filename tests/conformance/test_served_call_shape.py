"""A provider's served mark is explicit and strictly read by both fixture paths."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "served", [False, "result", [], {"extra": True}, {"result": 3}, {"is_error": 1}]
)
def test_malformed_served_mark_is_rejected(served: Any, stream: bool) -> None:
    call = {"id": "c", "name": "lookup", "served": served}
    round_ = {"stream": [{"call": call}]} if stream else {"calls": [call]}
    scenario = {"setup": {"channels": [{"id": "ai", "type": "ai", "model": {"rounds": [round_]}}]}}
    assert any("served" in problem for problem in unknown_members(scenario))

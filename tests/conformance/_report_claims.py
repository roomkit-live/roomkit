"""Scripted handlers exercising the public report-claim callback."""

from __future__ import annotations

from collections.abc import Callable

from roomkit.tools import current_tool_call, turn_report_claim


class ReportClaims:
    """Keep a callback on one scripted handler, across its calls and turns."""

    def __init__(self) -> None:
        self._first: Callable[[], bool] | None = None

    def answer(self, mode: str) -> list[bool | None]:
        call = current_tool_call()
        assert call is not None
        claim = turn_report_claim(call.tool_call_id, call.channel_id)
        assert claim is not None
        if mode == "matching":
            second = turn_report_claim(call.tool_call_id, call.channel_id)
            assert second is not None
            return [
                turn_report_claim("missing", call.channel_id) is not None,
                turn_report_claim(call.tool_call_id, "other") is not None,
                claim(),
                second(),
                claim(),
            ]
        if mode == "previous_turn":
            previous = None if self._first is None else self._first()
            if self._first is None:
                self._first = claim
            return [previous]
        assert mode == "reused_id"
        if self._first is None:
            self._first = claim
        return [self._first(), claim()]

"""Cost-tracking telemetry for the screen assistant example."""

from __future__ import annotations

import logging

from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.console import ConsoleTelemetryProvider


class CostTrackingTelemetry(ConsoleTelemetryProvider):
    """Extends console telemetry to accumulate token costs."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.totals: dict[str, int] = {
            "vision_calls": 0,
            "vision_prompt_tokens": 0,
            "vision_completion_tokens": 0,
            "realtime_input_tokens": 0,
            "realtime_output_tokens": 0,
        }

    def end_span(self, span_id: str, **kwargs: object) -> None:
        span = self._spans.get(span_id)
        if span is not None:
            attrs = {**span.attributes, **(kwargs.get("attributes") or {})}  # type: ignore[arg-type]
            inp = attrs.get(Attr.LLM_INPUT_TOKENS, 0)
            out = attrs.get(Attr.LLM_OUTPUT_TOKENS, 0)
            if inp or out:
                if span.kind == SpanKind.REALTIME_TURN:
                    self.totals["realtime_input_tokens"] += int(inp)
                    self.totals["realtime_output_tokens"] += int(out)
        super().end_span(span_id, **kwargs)  # type: ignore[arg-type]

    def record_metric(self, name: str, value: float, **kwargs: object) -> None:
        if name == "roomkit.realtime.input_tokens":
            self.totals["realtime_input_tokens"] += int(value)
        elif name == "roomkit.realtime.output_tokens":
            self.totals["realtime_output_tokens"] += int(value)
        super().record_metric(name, value, **kwargs)  # type: ignore[arg-type]

    def print_summary(self) -> None:
        v_total = self.totals["vision_prompt_tokens"] + self.totals["vision_completion_tokens"]
        r_total = self.totals["realtime_input_tokens"] + self.totals["realtime_output_tokens"]
        print()
        print("Session Cost Summary")
        print("-" * 40)
        print(f"  Vision API calls:       {self.totals['vision_calls']}")
        print(
            f"  Vision tokens:          {v_total:,} "
            f"({self.totals['vision_prompt_tokens']:,} in / "
            f"{self.totals['vision_completion_tokens']:,} out)"
        )
        print(
            f"  Realtime voice tokens:  {r_total:,} "
            f"({self.totals['realtime_input_tokens']:,} in / "
            f"{self.totals['realtime_output_tokens']:,} out)"
        )
        print(f"  Total tokens:           {v_total + r_total:,}")

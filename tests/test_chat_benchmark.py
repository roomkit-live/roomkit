"""Offline contracts for the benchmark: routing, isolation and honest measurements."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from benchmarks.chat.harness import Harness
from benchmarks.chat.measurement import MeasuredProvider, covered_seconds
from benchmarks.chat.report import append_sample, percentile, sanitize, summarize
from benchmarks.chat.scenarios import Scenario, scenarios

from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider


@pytest.mark.parametrize("store", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "scenario", [s for s in scenarios() if s.mock_supported], ids=lambda s: s.name
)
async def test_offline_scenarios(scenario: Scenario, store: str, tmp_path: Path) -> None:
    h = Harness(
        MockAIProvider(responses=["pong"], streaming=True),
        streaming=scenario.streaming,
        sqlite=tmp_path / "chat.db" if store == "sqlite" else None,
        **scenario.make_options(),
    )
    try:
        await h.add_room("main")
        await scenario.run(h)
        await h.validate()
        assert h.checks and all(h.checks.values()), h.checks
    finally:
        await h.close()


@pytest.mark.parametrize("streaming", [True, False])
async def test_tool_measurements_cover_the_public_pipeline(streaming: bool) -> None:
    scenario = next(s for s in scenarios() if s.name == "parallel_tools")
    provider = MockAIProvider(
        streaming=streaming,
        tool_call_delta_chunks=3,
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(id="north", name="stock", arguments={"warehouse": "north"}),
                    AIToolCall(id="south", name="stock", arguments={"warehouse": "south"}),
                ],
            ),
            AIResponse(content="42", finish_reason="stop"),
        ],
    )
    h = Harness(provider, streaming=streaming, **scenario.make_options())
    try:
        await h.add_room("main")
        await scenario.run(h)
        await h.validate()
        assert all(h.checks.values()), h.checks
        assert h.responses[0].tool_calls_count == 2
        assert h.responses[0].round_count == 1
        assert h.answer() == "42"
    finally:
        await h.close()


async def test_provider_wait_excludes_suspended_consumer() -> None:
    provider = MeasuredProvider(MockAIProvider(responses=["pong"], streaming=True))
    start = time.perf_counter()
    stream = provider.generate_structured_stream(AIContext())
    await anext(stream)
    consumer_start = time.perf_counter()
    await asyncio.sleep(0.02)
    consumer_end = time.perf_counter()
    async for _ in stream:
        pass
    call = provider.calls[0]
    assert all(right <= consumer_start or left >= consumer_end for left, right in call.waits)
    assert covered_seconds(call.waits) < time.perf_counter() - start


def test_concurrent_duration_is_a_union() -> None:
    assert covered_seconds([(2, 5), (1, 3), (6, 7), (2, 4)]) == 5
    assert covered_seconds([]) == 0
    assert percentile([1, 2, 3, 4, 5], 0.95) == 4.8


def test_failure_is_in_denominator_and_usage_but_not_latency(tmp_path: Path) -> None:
    passed = {
        "scenario": "chat",
        "status": "passed",
        "metrics": {
            "elapsed_ms": 10,
            "usage": {"input_tokens": 100},
            "provider_calls": 1,
        },
    }
    failed = {
        "scenario": "chat",
        "status": "failed",
        "metrics": {
            "elapsed_ms": 999,
            "usage": {"input_tokens": 200},
            "provider_calls": 2,
        },
        "error": "upstream echoed temporary-secret",
    }
    rows = summarize([passed, failed])
    assert rows[0]["runs"] == 2 and rows[0]["passed"] == 1
    assert rows[0]["elapsed_ms_median"] == 10
    assert rows[0]["input_tokens"] == 300
    assert rows[0]["provider_calls"] == 3
    path = tmp_path / "samples.jsonl"
    append_sample(path, sanitize(failed, "temporary-secret"))
    assert "temporary-secret" not in path.read_text()
    assert json.loads(path.read_text())["status"] == "failed"


def test_skill_state_is_fresh_for_each_sample() -> None:
    scenario = next(s for s in scenarios() if s.name == "skills")
    first, second = scenario.make_options(), scenario.make_options()
    assert first["skills"] is not second["skills"]
    assert first["script_executor"] is not second["script_executor"]
    skill = first["skills"].get_skill("quote-policy")
    assert skill is not None and skill.metadata.gated_tool_names == ["inventory"]

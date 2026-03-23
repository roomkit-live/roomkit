"""Tests for Supervisor orchestration (orchestration/strategies/supervisor.py)."""

from __future__ import annotations

import pytest

from roomkit.channels.agent import Agent
from roomkit.orchestration.strategies.supervisor import Supervisor, WorkerStrategy
from roomkit.providers.ai.mock import MockAIProvider


def _make_agent(agent_id: str, role: str = "worker") -> Agent:
    return Agent(agent_id, provider=MockAIProvider(), role=role)


class TestSupervisor:
    def test_constructor_defaults(self) -> None:
        supervisor = _make_agent("sup", role="supervisor")
        workers = [_make_agent("w1"), _make_agent("w2")]
        s = Supervisor(supervisor=supervisor, workers=workers)
        assert s._supervisor is supervisor
        assert len(s._workers) == 2
        assert s._strategy is None
        assert s._auto_delegate is False

    def test_constructor_with_strategy(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(supervisor=supervisor, workers=workers, strategy="sequential")
        assert s._strategy == WorkerStrategy.SEQUENTIAL

    def test_constructor_parallel_strategy(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(supervisor=supervisor, workers=workers, strategy="parallel")
        assert s._strategy == WorkerStrategy.PARALLEL

    def test_auto_delegate_without_strategy_raises(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        with pytest.raises(ValueError, match="auto_delegate"):
            Supervisor(supervisor=supervisor, workers=workers, auto_delegate=True)

    def test_agents_returns_supervisor(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(supervisor=supervisor, workers=workers)
        result = s.agents()
        assert len(result) == 1
        assert result[0] is supervisor

    def test_agents_empty_when_async_delivery(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(
            supervisor=supervisor,
            workers=workers,
            strategy="parallel",
            async_delivery=True,
        )
        result = s.agents()
        assert result == []

    def test_worker_strategy_enum_values(self) -> None:
        assert WorkerStrategy.SEQUENTIAL == "sequential"
        assert WorkerStrategy.PARALLEL == "parallel"

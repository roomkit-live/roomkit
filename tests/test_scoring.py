"""Tests for roomkit.scoring — scorer ABC, mock, hook, and tracker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from roomkit.models.task import Observation
from roomkit.models.tool_call import AIResponseEvent
from roomkit.scoring import (
    ConversationScorer,
    DimensionReport,
    MockScorer,
    QualityReport,
    QualityTracker,
    Score,
    ScoringHook,
)
from roomkit.store.memory import InMemoryStore

# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------


class TestScore:
    def test_defaults(self):
        s = Score(value=0.8, dimension="relevance")
        assert s.value == 0.8
        assert s.dimension == "relevance"
        assert s.reason == ""
        assert s.metadata == {}

    def test_with_reason_and_metadata(self):
        s = Score(value=0.5, dimension="safety", reason="mild concern", metadata={"model": "gpt"})
        assert s.reason == "mild concern"
        assert s.metadata["model"] == "gpt"


# ---------------------------------------------------------------------------
# ConversationScorer ABC
# ---------------------------------------------------------------------------


class TestConversationScorer:
    def test_name_default(self):
        class MyScorer(ConversationScorer):
            async def score(self, **kwargs: Any) -> list[Score]:
                return []

        assert MyScorer().name == "MyScorer"

    async def test_close_is_noop_by_default(self):
        class Noop(ConversationScorer):
            async def score(self, **kwargs: Any) -> list[Score]:
                return []

        await Noop().close()


# ---------------------------------------------------------------------------
# MockScorer
# ---------------------------------------------------------------------------


class TestMockScorer:
    async def test_returns_configured_scores(self):
        scores = [Score(value=0.9, dimension="relevance"), Score(value=0.7, dimension="safety")]
        scorer = MockScorer(scores=scores)
        result = await scorer.score(
            response_content="hi", query="hello", room_id="r1", channel_id="ai"
        )
        assert len(result) == 2
        assert result[0].value == 0.9
        assert result[1].dimension == "safety"

    async def test_records_calls(self):
        scorer = MockScorer()
        await scorer.score(
            response_content="answer",
            query="question",
            room_id="room-1",
            channel_id="ai-1",
        )
        assert len(scorer.calls) == 1
        assert scorer.calls[0].query == "question"
        assert scorer.calls[0].room_id == "room-1"

    async def test_default_scores(self):
        scorer = MockScorer()
        result = await scorer.score(response_content="x", query="y", room_id="r", channel_id="c")
        assert len(result) == 1
        assert result[0].value == 1.0
        assert result[0].dimension == "default"

    def test_name(self):
        assert MockScorer().name == "MockScorer"


# ---------------------------------------------------------------------------
# ScoringHook
# ---------------------------------------------------------------------------


def _make_event(
    *,
    response_content: str = "Hello!",
    room_id: str = "room-1",
    channel_id: str = "ai-1",
    latency_ms: int = 100,
    streaming: bool = False,
) -> AIResponseEvent:
    return AIResponseEvent(
        channel_id=channel_id,
        response_content=response_content,
        room_id=room_id,
        latency_ms=latency_ms,
        streaming=streaming,
    )


class TestScoringHook:
    async def test_attach_registers_hook(self):
        scorer = MockScorer()
        hook = ScoringHook(scorers=[scorer])
        kit = MagicMock()
        kit._store = InMemoryStore()
        kit.hook.return_value = lambda fn: fn
        hook.attach(kit)
        kit.hook.assert_called_once()

    async def test_on_response_calls_scorers(self):
        store = InMemoryStore()
        await store.create_room(
            MagicMock(id="room-1", metadata={}, participants=[], channel_bindings=[])
        )
        scorer = MockScorer(scores=[Score(value=0.85, dimension="quality")])
        hook = ScoringHook(scorers=[scorer], store=store)
        event = _make_event()
        ctx = MagicMock(recent_events=[])

        await hook._on_response(event, ctx)

        assert len(scorer.calls) == 1
        assert len(hook.recent_scores) == 1
        assert hook.recent_scores[0]["value"] == 0.85

    async def test_on_response_ignores_non_ai_event(self):
        scorer = MockScorer()
        hook = ScoringHook(scorers=[scorer])
        await hook._on_response("not-an-event", MagicMock())
        assert len(scorer.calls) == 0

    async def test_on_response_handles_scorer_exception(self):
        class FailScorer(ConversationScorer):
            async def score(self, **kwargs: Any) -> list[Score]:
                raise RuntimeError("boom")

        hook = ScoringHook(scorers=[FailScorer()], store=InMemoryStore())
        event = _make_event()
        # Should not raise — exceptions are caught and logged
        await hook._on_response(event, MagicMock(recent_events=[]))
        assert len(hook.recent_scores) == 0

    async def test_on_response_persists_scores(self):
        store = InMemoryStore()
        await store.create_room(
            MagicMock(id="room-1", metadata={}, participants=[], channel_bindings=[])
        )
        scorer = MockScorer(scores=[Score(value=0.9, dimension="relevance", reason="good")])
        hook = ScoringHook(scorers=[scorer], store=store)
        event = _make_event()

        await hook._on_response(event, MagicMock(recent_events=[]))

        obs = await store.list_observations("room-1")
        assert len(obs) == 1
        assert obs[0].category == "score:relevance"
        assert obs[0].confidence == 0.9

    async def test_on_response_no_persist_without_room(self):
        store = InMemoryStore()
        scorer = MockScorer(scores=[Score(value=0.5, dimension="x")])
        hook = ScoringHook(scorers=[scorer], store=store)
        event = _make_event(room_id=None)

        await hook._on_response(event, MagicMock(recent_events=[]))

        # No observations should be persisted when room_id is None
        assert len(hook.recent_scores) == 1

    async def test_buffer_respects_max_recent(self):
        scorer = MockScorer()
        hook = ScoringHook(scorers=[scorer], store=InMemoryStore(), max_recent=2)
        event = _make_event()
        ctx = MagicMock(recent_events=[])

        for _ in range(5):
            await hook._on_response(event, ctx)

        assert len(hook.recent_scores) == 2

    async def test_extract_last_user_query(self):
        @dataclass
        class FakeEvent:
            content: Any
            source: Any

        @dataclass
        class FakeContent:
            body: str

        @dataclass
        class FakeSource:
            channel_id: str

        events = [
            FakeEvent(content=FakeContent("user says hello"), source=FakeSource("sms")),
            FakeEvent(content=FakeContent("ai response"), source=FakeSource("ai-1")),
        ]
        ctx = MagicMock(recent_events=events)
        event = _make_event(channel_id="ai-1")

        query = ScoringHook._extract_last_user_query(event, ctx)
        assert query == "user says hello"

    async def test_extract_last_user_query_empty(self):
        ctx = MagicMock(recent_events=[])
        event = _make_event()
        assert ScoringHook._extract_last_user_query(event, ctx) == ""

    async def test_close_closes_all_scorers(self):
        scorer = MockScorer()
        scorer.close = AsyncMock()
        hook = ScoringHook(scorers=[scorer])
        await hook.close()
        scorer.close.assert_awaited_once()

    async def test_close_handles_scorer_close_error(self):
        scorer = MockScorer()
        scorer.close = AsyncMock(side_effect=RuntimeError("close failed"))
        hook = ScoringHook(scorers=[scorer])
        # Should not raise
        await hook.close()

    async def test_multiple_scorers(self):
        s1 = MockScorer(scores=[Score(value=0.9, dimension="relevance")])
        s2 = MockScorer(scores=[Score(value=0.7, dimension="safety")])
        hook = ScoringHook(scorers=[s1, s2], store=InMemoryStore())
        event = _make_event()

        await hook._on_response(event, MagicMock(recent_events=[]))

        assert len(hook.recent_scores) == 2
        dims = {s["dimension"] for s in hook.recent_scores}
        assert dims == {"relevance", "safety"}


# ---------------------------------------------------------------------------
# QualityTracker
# ---------------------------------------------------------------------------


_obs_counter = 0


async def _seed_observations(
    store: InMemoryStore,
    room_id: str,
    scores: list[tuple[str, float]],
    *,
    feedback: list[tuple[str, float]] | None = None,
    base_time: datetime | None = None,
) -> None:
    """Seed observations into the store for testing."""
    global _obs_counter  # noqa: PLW0603
    base = base_time or datetime.now(UTC)
    for i, (dim, val) in enumerate(scores):
        _obs_counter += 1
        obs = Observation(
            id=f"score-{_obs_counter}",
            room_id=room_id,
            channel_id="ai-1",
            content=f"[{dim}] {val:.2f}",
            category=f"score:{dim}",
            confidence=val,
            created_at=base + timedelta(minutes=i),
            metadata={"dimension": dim},
        )
        await store.add_observation(obs)
    for i, (cat, val) in enumerate(feedback or []):
        _obs_counter += 1
        obs = Observation(
            id=f"feedback-{_obs_counter}",
            room_id=room_id,
            channel_id="sms-1",
            content=f"feedback: {val}",
            category=f"feedback:{cat}",
            confidence=val,
            created_at=base + timedelta(minutes=len(scores) + i),
            metadata={},
        )
        await store.add_observation(obs)


class TestQualityTracker:
    async def test_empty_report(self):
        store = InMemoryStore()
        tracker = QualityTracker(store)
        report = await tracker.report("room-1")
        assert report.avg == 0.0
        assert report.count == 0
        assert report.trend == 0.0

    async def test_basic_report(self):
        store = InMemoryStore()
        await _seed_observations(
            store,
            "room-1",
            [("relevance", 0.8), ("relevance", 0.9), ("safety", 0.7), ("safety", 0.6)],
        )
        tracker = QualityTracker(store)
        report = await tracker.report("room-1")
        assert report.room_id == "room-1"
        assert report.count == 4
        assert 0.74 <= report.avg <= 0.76  # (0.8+0.9+0.7+0.6)/4 = 0.75
        assert "relevance" in report.dimensions
        assert "safety" in report.dimensions

    async def test_dimension_breakdown(self):
        store = InMemoryStore()
        await _seed_observations(
            store, "r1", [("accuracy", 0.9), ("accuracy", 0.7), ("accuracy", 0.8)]
        )
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        dim = report.dimensions["accuracy"]
        assert isinstance(dim, DimensionReport)
        assert dim.count == 3
        assert 0.799 <= dim.avg <= 0.801
        assert dim.min == 0.7
        assert dim.max == 0.9

    async def test_best_worst_dimension(self):
        store = InMemoryStore()
        await _seed_observations(
            store,
            "r1",
            [("good", 0.95), ("good", 0.90), ("bad", 0.3), ("bad", 0.4)],
        )
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        assert report.best_dimension == "good"
        assert report.worst_dimension == "bad"

    async def test_trend_improving(self):
        store = InMemoryStore()
        # First half: low scores, second half: high scores
        await _seed_observations(
            store,
            "r1",
            [("x", 0.3), ("x", 0.4), ("x", 0.8), ("x", 0.9)],
        )
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        assert report.trend > 0  # improving

    async def test_trend_declining(self):
        store = InMemoryStore()
        await _seed_observations(
            store,
            "r1",
            [("x", 0.9), ("x", 0.8), ("x", 0.3), ("x", 0.4)],
        )
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        assert report.trend < 0  # declining

    async def test_trend_requires_min_4(self):
        store = InMemoryStore()
        await _seed_observations(store, "r1", [("x", 0.5), ("x", 0.9)])
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        assert report.trend == 0.0

    async def test_window_hours_filter(self):
        store = InMemoryStore()
        now = datetime.now(UTC)
        old = now - timedelta(hours=48)
        # Old observations
        for i in range(3):
            await store.add_observation(
                Observation(
                    id=f"old-{i}",
                    room_id="r1",
                    channel_id="ai",
                    content="old",
                    category="score:x",
                    confidence=0.3,
                    created_at=old + timedelta(minutes=i),
                    metadata={"dimension": "x"},
                )
            )
        # Recent observations
        for i in range(3):
            await store.add_observation(
                Observation(
                    id=f"new-{i}",
                    room_id="r1",
                    channel_id="ai",
                    content="new",
                    category="score:x",
                    confidence=0.9,
                    created_at=now - timedelta(minutes=i),
                    metadata={"dimension": "x"},
                )
            )
        tracker = QualityTracker(store)
        report = await tracker.report("r1", window_hours=24)
        assert report.count == 3
        assert report.avg == 0.9

    async def test_feedback_stats(self):
        store = InMemoryStore()
        await _seed_observations(
            store,
            "r1",
            [("rel", 0.8)],
            feedback=[("thumbs", 1.0), ("thumbs", 0.0)],
        )
        tracker = QualityTracker(store)
        report = await tracker.report("r1")
        assert report.feedback_count == 2
        assert report.feedback_avg == 0.5

    async def test_report_multi(self):
        store = InMemoryStore()
        await _seed_observations(store, "r1", [("x", 0.8)])
        await _seed_observations(store, "r2", [("x", 0.6)])
        tracker = QualityTracker(store)
        reports = await tracker.report_multi(["r1", "r2"])
        assert len(reports) == 2
        assert isinstance(reports["r1"], QualityReport)
        assert isinstance(reports["r2"], QualityReport)
        assert reports["r1"].avg == 0.8
        assert reports["r2"].avg == 0.6

    async def test_no_window_includes_all(self):
        store = InMemoryStore()
        await _seed_observations(store, "r1", [("x", 0.5), ("x", 0.7)])
        tracker = QualityTracker(store)
        report = await tracker.report("r1", window_hours=None)
        assert report.count == 2

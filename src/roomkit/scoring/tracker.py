"""Quality regression tracker.

Aggregates scores and feedback stored as :class:`~roomkit.models.task.Observation`
objects by :class:`ScoringHook` and :func:`kit.submit_feedback`, producing
quality reports with trends and dimension breakdowns.

Usage::

    from roomkit.scoring import QualityTracker

    tracker = QualityTracker(kit._store)
    report = await tracker.report("room-1")
    print(report)
    # QualityReport(avg=0.82, count=45, trend=-0.03, dimensions={...})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from roomkit.models.task import Observation
from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.scoring.tracker")


@dataclass
class DimensionReport:
    """Quality stats for a single scoring dimension."""

    dimension: str
    avg: float
    count: int
    min: float
    max: float


@dataclass
class QualityReport:
    """Aggregated quality report for a room or agent."""

    room_id: str
    avg: float
    count: int
    trend: float
    dimensions: dict[str, DimensionReport] = field(default_factory=dict)
    worst_dimension: str = ""
    best_dimension: str = ""
    feedback_count: int = 0
    feedback_avg: float = 0.0


class QualityTracker:
    """Aggregates quality scores and feedback into actionable reports.

    Reads :class:`~roomkit.models.task.Observation` objects from the
    store (written by :class:`ScoringHook` with ``category="score:*"``
    and by :func:`submit_feedback` with ``category="feedback:*"``).

    Parameters:
        store: The conversation store to read observations from.
    """

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def report(
        self,
        room_id: str,
        *,
        window_hours: float | None = None,
    ) -> QualityReport:
        """Generate a quality report for a room.

        Args:
            room_id: Room to report on.
            window_hours: Only include observations from the last N hours.
                If None, includes all observations.

        Returns:
            A :class:`QualityReport` with averages, trends, and per-dimension
            breakdowns.
        """
        all_obs = await self._store.list_observations(room_id)

        cutoff = datetime.now(UTC) - timedelta(hours=window_hours) if window_hours else None
        if cutoff:
            all_obs = [o for o in all_obs if o.created_at >= cutoff]

        scores = [o for o in all_obs if o.category and o.category.startswith("score:")]
        feedback = [o for o in all_obs if o.category and o.category.startswith("feedback:")]

        if not scores and not feedback:
            return QualityReport(room_id=room_id, avg=0.0, count=0, trend=0.0)

        # Aggregate scores by dimension
        dimensions = self._aggregate_dimensions(scores)

        # Overall score average
        all_values = [o.confidence for o in scores]
        avg = sum(all_values) / len(all_values) if all_values else 0.0

        # Trend: compare first half vs second half
        trend = self._compute_trend(all_values)

        # Find worst/best dimensions
        worst = min(dimensions.values(), key=lambda d: d.avg).dimension if dimensions else ""
        best = max(dimensions.values(), key=lambda d: d.avg).dimension if dimensions else ""

        # Feedback stats
        fb_values = [o.confidence for o in feedback]
        fb_avg = sum(fb_values) / len(fb_values) if fb_values else 0.0

        return QualityReport(
            room_id=room_id,
            avg=round(avg, 3),
            count=len(scores),
            trend=round(trend, 3),
            dimensions=dimensions,
            worst_dimension=worst,
            best_dimension=best,
            feedback_count=len(feedback),
            feedback_avg=round(fb_avg, 3),
        )

    async def report_multi(
        self,
        room_ids: list[str],
        *,
        window_hours: float | None = None,
    ) -> dict[str, QualityReport]:
        """Generate reports for multiple rooms."""
        return {rid: await self.report(rid, window_hours=window_hours) for rid in room_ids}

    @staticmethod
    def _aggregate_dimensions(
        observations: list[Observation],
    ) -> dict[str, DimensionReport]:
        """Group observations by dimension and compute stats."""
        buckets: dict[str, list[float]] = {}
        for obs in observations:
            dim = obs.metadata.get("dimension", obs.category or "unknown")
            if dim.startswith("score:"):
                dim = dim[6:]
            buckets.setdefault(dim, []).append(obs.confidence)

        return {
            dim: DimensionReport(
                dimension=dim,
                avg=round(sum(vals) / len(vals), 3),
                count=len(vals),
                min=round(min(vals), 3),
                max=round(max(vals), 3),
            )
            for dim, vals in buckets.items()
        }

    @staticmethod
    def _compute_trend(values: list[float]) -> float:
        """Compute trend as difference between second-half and first-half averages."""
        if len(values) < 4:
            return 0.0
        mid = len(values) // 2
        first_half = sum(values[:mid]) / mid
        second_half = sum(values[mid:]) / (len(values) - mid)
        return second_half - first_half

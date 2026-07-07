"""Tests for DriftAnalyzer — allocation drift computation and trend analysis."""
from __future__ import annotations


from trading_assistant.schemas.portfolio_allocation import (
    BotAllocationRecommendation,
    PortfolioAllocationReport,
)
from trading_assistant.skills.drift_analyzer import DriftAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alloc_report(
    recommendations: list[tuple[str, float]],
    week_start: str = "2026-03-01",
    week_end: str = "2026-03-07",
) -> PortfolioAllocationReport:
    """Create a PortfolioAllocationReport from (bot_id, suggested_pct) tuples."""
    recs = [
        BotAllocationRecommendation(
            bot_id=bot_id,
            suggested_allocation_pct=pct,
        )
        for bot_id, pct in recommendations
    ]
    return PortfolioAllocationReport(
        week_start=week_start,
        week_end=week_end,
        recommendations=recs,
    )


def _make_snapshot_dict(
    date: str,
    bot_drifts: list[tuple[str, float]],
    total_drift: float | None = None,
    max_drift: float | None = None,
) -> dict:
    """Create a snapshot dict as returned by AllocationTracker.load_snapshots()."""
    bots = [
        {"bot_id": bid, "recommended_pct": 50.0, "actual_pct": 50.0 - d,
         "drift_pct": d, "abs_drift_pct": abs(d)}
        for bid, d in bot_drifts
    ]
    if total_drift is None:
        total_drift = sum(abs(d) for _, d in bot_drifts) / 2
    if max_drift is None:
        max_drift = max((abs(d) for _, d in bot_drifts), default=0.0)
    return {
        "date": date,
        "week_start": date,
        "week_end": date,
        "bot_allocations": bots,
        "total_drift_pct": total_drift,
        "max_single_drift_pct": max_drift,
    }


# ---------------------------------------------------------------------------
# compute_snapshot — basic cases
# ---------------------------------------------------------------------------

class TestComputeSnapshot:
    def test_no_drift_when_perfect_match(self):
        report = _make_alloc_report([("bot_a", 50.0), ("bot_b", 50.0)])
        actuals = {"bot_a": 50.0, "bot_b": 50.0}

        snap = DriftAnalyzer.compute_snapshot(report, actuals)

        assert snap.total_drift_pct == 0.0
        assert snap.max_single_drift_pct == 0.0
        for ba in snap.bot_allocations:
            assert ba.drift_pct == 0.0
            assert ba.abs_drift_pct == 0.0

    def test_positive_drift(self):
        """Recommended > actual → positive drift_pct."""
        report = _make_alloc_report([("bot_a", 60.0)])
        actuals = {"bot_a": 50.0}

        snap = DriftAnalyzer.compute_snapshot(report, actuals)

        assert len(snap.bot_allocations) == 1
        ba = snap.bot_allocations[0]
        assert ba.drift_pct == 10.0
        assert ba.abs_drift_pct == 10.0

    def test_negative_drift(self):
        """Recommended < actual → negative drift_pct."""
        report = _make_alloc_report([("bot_a", 30.0)])
        actuals = {"bot_a": 50.0}

        snap = DriftAnalyzer.compute_snapshot(report, actuals)

        ba = snap.bot_allocations[0]
        assert ba.drift_pct == -20.0
        assert ba.abs_drift_pct == 20.0

    def test_missing_actual_defaults_to_zero(self):
        """Bot not in actuals dict → actual = 0%."""
        report = _make_alloc_report([("bot_a", 40.0)])
        actuals = {}

        snap = DriftAnalyzer.compute_snapshot(report, actuals)

        ba = snap.bot_allocations[0]
        assert ba.actual_pct == 0.0
        assert ba.drift_pct == 40.0

    def test_multiple_bots_total_drift(self):
        """Total drift = sum(|drift|) / 2."""
        report = _make_alloc_report([("bot_a", 60.0), ("bot_b", 40.0)])
        actuals = {"bot_a": 50.0, "bot_b": 50.0}

        snap = DriftAnalyzer.compute_snapshot(report, actuals)

        # bot_a: drift = +10, bot_b: drift = -10
        # total = (10 + 10) / 2 = 10
        assert snap.total_drift_pct == 10.0
        assert snap.max_single_drift_pct == 10.0

    def test_snapshot_date_from_report(self):
        report = _make_alloc_report(
            [("bot_a", 50.0)],
            week_start="2026-02-24",
            week_end="2026-03-02",
        )
        snap = DriftAnalyzer.compute_snapshot(report, {"bot_a": 50.0})

        assert snap.date == "2026-02-24"
        assert snap.week_start == "2026-02-24"
        assert snap.week_end == "2026-03-02"

    def test_empty_recommendations(self):
        report = _make_alloc_report([])
        snap = DriftAnalyzer.compute_snapshot(report, {})

        assert snap.bot_allocations == []
        assert snap.total_drift_pct == 0.0
        assert snap.max_single_drift_pct == 0.0


# ---------------------------------------------------------------------------
# compute_drift_trend — empty / single snapshot
# ---------------------------------------------------------------------------

class TestComputeDriftTrendEmpty:
    def test_empty_snapshots(self):
        result = DriftAnalyzer.compute_drift_trend([])

        assert result["weekly_drift"] == []
        assert result["trend_direction"] == "stable"
        assert result["avg_drift_pct"] == 0.0
        assert result["persistent_drifters"] == []

    def test_single_snapshot(self):
        snap = _make_snapshot_dict("2026-03-01", [("bot_a", 5.0)])
        result = DriftAnalyzer.compute_drift_trend([snap])

        assert len(result["weekly_drift"]) == 1
        assert result["trend_direction"] == "stable"
        assert result["avg_drift_pct"] == snap["total_drift_pct"]


# ---------------------------------------------------------------------------
# compute_drift_trend — trend detection
# ---------------------------------------------------------------------------

class TestComputeDriftTrendDirection:
    def test_increasing_trend(self):
        """Second half avg drift > first half avg drift by >2% → increasing."""
        snapshots = [
            _make_snapshot_dict(f"2026-0{i+1}-01", [("bot_a", float(i * 2))], total_drift=float(i * 2))
            for i in range(6)
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)
        assert result["trend_direction"] == "increasing"

    def test_decreasing_trend(self):
        """Second half avg drift < first half avg drift by >2% → decreasing."""
        snapshots = [
            _make_snapshot_dict(f"2026-0{i+1}-01", [("bot_a", float(10 - i * 2))], total_drift=float(10 - i * 2))
            for i in range(6)
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)
        assert result["trend_direction"] == "decreasing"

    def test_stable_trend(self):
        """Drift stays roughly the same → stable."""
        snapshots = [
            _make_snapshot_dict(f"2026-0{i+1}-01", [("bot_a", 5.0)], total_drift=5.0)
            for i in range(4)
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)
        assert result["trend_direction"] == "stable"

    def test_weeks_parameter_limits_window(self):
        """Only the last N weeks should be considered."""
        snapshots = [
            _make_snapshot_dict(f"2026-01-{i+1:02d}", [("bot_a", 20.0)], total_drift=20.0)
            for i in range(10)
        ]
        # Last 3 only
        result = DriftAnalyzer.compute_drift_trend(snapshots, weeks=3)
        assert len(result["weekly_drift"]) == 3


# ---------------------------------------------------------------------------
# compute_drift_trend — persistent drifters
# ---------------------------------------------------------------------------

class TestComputeDriftTrendPersistentDrifters:
    def test_persistent_drifter_detected(self):
        """Bot with >5% drift in >50% of weeks → persistent drifter."""
        snapshots = [
            _make_snapshot_dict(f"2026-01-{i+1:02d}", [("bot_a", 8.0), ("bot_b", 2.0)])
            for i in range(4)
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)

        assert "bot_a" in result["persistent_drifters"]
        assert "bot_b" not in result["persistent_drifters"]

    def test_no_persistent_drifter_when_below_threshold(self):
        """Drift <=5% → no persistent drifter."""
        snapshots = [
            _make_snapshot_dict(f"2026-01-{i+1:02d}", [("bot_a", 3.0)])
            for i in range(4)
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)
        assert result["persistent_drifters"] == []

    def test_intermittent_drift_not_persistent(self):
        """Bot drifts only some weeks → depends on threshold."""
        snapshots = [
            _make_snapshot_dict("2026-01-01", [("bot_a", 8.0)]),
            _make_snapshot_dict("2026-01-08", [("bot_a", 2.0)]),
            _make_snapshot_dict("2026-01-15", [("bot_a", 8.0)]),
            _make_snapshot_dict("2026-01-22", [("bot_a", 2.0)]),
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)

        # 2 out of 4 weeks = 50%, threshold is >50% (strict), so NOT persistent
        assert "bot_a" not in result["persistent_drifters"]

    def test_avg_drift_pct(self):
        snapshots = [
            _make_snapshot_dict("2026-01-01", [("bot_a", 4.0)], total_drift=4.0),
            _make_snapshot_dict("2026-01-08", [("bot_a", 6.0)], total_drift=6.0),
        ]
        result = DriftAnalyzer.compute_drift_trend(snapshots)
        assert result["avg_drift_pct"] == 5.0

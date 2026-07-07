# skills/convergence_tracker.py
"""ConvergenceTracker — synthesises learning loop health into a convergence signal.

Reads learning_ledger.jsonl, forecast_history.jsonl, and outcomes.jsonl to
determine whether the system is improving, degrading, oscillating, or stable.
Purely deterministic — no LLM calls.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.convergence import (
    ConvergenceDimension,
    ConvergenceReport,
    DimensionStatus,
    LoopHealthMetrics,
)
from trading_assistant.skills._outcome_utils import is_conclusive_outcome, is_positive_outcome

_MIN_WEEKS = 4  # minimum data points for any trend assessment


class ConvergenceTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir

    def compute_report(self, weeks: int = 12) -> ConvergenceReport:
        """Compute multi-dimensional convergence report."""
        dims = [
            self._check_composite_scores(weeks),
            self._check_prediction_accuracy(weeks),
            self._check_outcome_ratio(weeks),
            self._check_scorecard_evolution(weeks),
            self._check_loop_balance(weeks),
        ]
        overall = self._synthesize_overall(dims)
        oscillation = any(d.status == DimensionStatus.OSCILLATING for d in dims)
        recommendation = self._generate_recommendation(overall, dims)
        health_metrics = self._compute_health_metrics(dims, weeks)
        return ConvergenceReport(
            overall_status=overall,
            dimensions=dims,
            oscillation_detected=oscillation,
            weeks_analyzed=weeks,
            recommendation=recommendation,
            health_metrics=health_metrics,
        )

    # ------------------------------------------------------------------
    # Dimension checkers
    # ------------------------------------------------------------------

    def _check_composite_scores(self, weeks: int) -> ConvergenceDimension:
        """Check composite score trend from learning_ledger.jsonl."""
        records = self._read_jsonl(self._findings_dir / "learning_ledger.jsonl")
        deltas: list[float] = []
        for r in records:
            cd = r.get("composite_delta")
            if cd is None:
                continue
            if isinstance(cd, dict):
                # composite_delta is {bot_id: delta} — aggregate to mean
                vals = [v for v in cd.values() if isinstance(v, (int, float))]
                if vals:
                    deltas.append(sum(vals) / len(vals))
            elif isinstance(cd, (int, float)):
                deltas.append(float(cd))
        # Take last N weeks
        deltas = deltas[-weeks:] if len(deltas) > weeks else deltas

        if len(deltas) < _MIN_WEEKS:
            return ConvergenceDimension(
                name="composite_scores",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=len(deltas),
                detail=f"Only {len(deltas)} weeks of data (need {_MIN_WEEKS}+)",
            )

        status, slope = self._classify_trend(deltas)
        return ConvergenceDimension(
            name="composite_scores",
            status=status,
            trend_value=round(slope, 4),
            window_weeks=len(deltas),
            detail=self._trend_detail("composite score deltas", deltas, status),
        )

    def _check_prediction_accuracy(self, weeks: int) -> ConvergenceDimension:
        """Check prediction accuracy trend from forecast_history.jsonl."""
        records = self._read_jsonl(self._findings_dir / "forecast_history.jsonl")
        accuracies = [
            r.get("accuracy", 0.0)
            for r in records
            if r.get("accuracy") is not None
        ]
        accuracies = accuracies[-weeks:] if len(accuracies) > weeks else accuracies

        if len(accuracies) < _MIN_WEEKS:
            return ConvergenceDimension(
                name="prediction_accuracy",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=len(accuracies),
                detail=f"Only {len(accuracies)} weeks of data (need {_MIN_WEEKS}+)",
            )

        status, slope = self._classify_trend(accuracies, threshold=0.01)
        return ConvergenceDimension(
            name="prediction_accuracy",
            status=status,
            trend_value=round(slope, 4),
            window_weeks=len(accuracies),
            detail=self._trend_detail("prediction accuracy", accuracies, status),
        )

    def _check_outcome_ratio(self, weeks: int) -> ConvergenceDimension:
        """Check positive outcome ratio trend from outcomes.jsonl."""
        records = self._read_jsonl(self._findings_dir / "outcomes.jsonl")
        if not records:
            return ConvergenceDimension(
                name="outcome_ratio",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=0,
                detail="No outcome data available",
            )

        # Group outcomes by week
        weekly_ratios = self._group_outcomes_by_week(records, weeks)

        if len(weekly_ratios) < _MIN_WEEKS:
            return ConvergenceDimension(
                name="outcome_ratio",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=len(weekly_ratios),
                detail=f"Only {len(weekly_ratios)} weeks of data (need {_MIN_WEEKS}+)",
            )

        status, slope = self._classify_trend(weekly_ratios)
        return ConvergenceDimension(
            name="outcome_ratio",
            status=status,
            trend_value=round(slope, 4),
            window_weeks=len(weekly_ratios),
            detail=self._trend_detail("positive outcome ratio", weekly_ratios, status),
        )

    def _check_scorecard_evolution(self, weeks: int) -> ConvergenceDimension:
        """Check if category win rates are trending up over time.

        Uses outcomes.jsonl timestamps to compute rolling win rates at
        4-week intervals.
        """
        records = self._read_jsonl(self._findings_dir / "outcomes.jsonl")
        if not records:
            return ConvergenceDimension(
                name="scorecard_evolution",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=0,
                detail="No outcome data available",
            )

        # Compute rolling win rates at 4-week intervals
        now = datetime.now(timezone.utc)
        window_size_days = 28  # 4 weeks
        step_days = 28
        max_intervals = weeks // 4

        interval_win_rates: list[float] = []
        for i in range(max_intervals):
            end = now - timedelta(days=i * step_days)
            start = end - timedelta(days=window_size_days)
            window_outcomes = [
                r for r in records
                if self._in_time_range(r, start, end) and is_conclusive_outcome(r)
            ]
            if len(window_outcomes) >= 3:
                positive = sum(1 for o in window_outcomes if is_positive_outcome(o))
                interval_win_rates.append(positive / len(window_outcomes))

        # Reverse so chronological order (oldest first)
        interval_win_rates.reverse()

        if len(interval_win_rates) < 2:
            return ConvergenceDimension(
                name="scorecard_evolution",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=len(interval_win_rates) * 4,
                detail="Too few intervals for scorecard trend analysis",
            )

        status, slope = self._classify_trend(interval_win_rates)
        return ConvergenceDimension(
            name="scorecard_evolution",
            status=status,
            trend_value=round(slope, 4),
            window_weeks=len(interval_win_rates) * 4,
            detail=self._trend_detail(
                "category win rates (4-week rolling)", interval_win_rates, status,
            ),
        )

    def _check_loop_balance(self, weeks: int) -> ConvergenceDimension:
        """Check balance between inner-loop and outer-loop positive outcome rates.

        Computes balance_score = 1.0 - abs(inner_rate - outer_rate) per week.
        1.0 = perfect balance, 0.0 = maximum imbalance.
        Skips weeks where only one loop has data.
        """
        records = self._read_jsonl(self._findings_dir / "learning_ledger.jsonl")
        records = records[-weeks:] if len(records) > weeks else records

        balance_scores: list[float] = []
        for r in records:
            inner_total = r.get("inner_total_outcomes", 0)
            outer_total = r.get("outer_total_outcomes", 0)
            inner_pos = r.get("inner_positive_outcomes", 0)
            outer_pos = r.get("outer_positive_outcomes", 0)
            # Skip weeks where only one loop has data
            if inner_total <= 0 or outer_total <= 0:
                continue
            inner_rate = inner_pos / inner_total
            outer_rate = outer_pos / outer_total
            balance_scores.append(1.0 - abs(inner_rate - outer_rate))

        if len(balance_scores) < _MIN_WEEKS:
            return ConvergenceDimension(
                name="loop_balance",
                status=DimensionStatus.INSUFFICIENT_DATA,
                trend_value=0.0,
                window_weeks=len(balance_scores),
                detail=f"Only {len(balance_scores)} weeks with both loop sources (need {_MIN_WEEKS}+)",
            )

        status, slope = self._classify_trend(balance_scores)
        return ConvergenceDimension(
            name="loop_balance",
            status=status,
            trend_value=round(slope, 4),
            window_weeks=len(balance_scores),
            detail=self._trend_detail("loop balance score", balance_scores, status),
        )

    # ------------------------------------------------------------------
    # Loop health metrics
    # ------------------------------------------------------------------

    def _compute_health_metrics(
        self, dims: list[ConvergenceDimension], weeks: int,
    ) -> LoopHealthMetrics:
        """Compute quantitative loop health metrics from JSONL data."""
        outcomes = self._read_jsonl(self._findings_dir / "outcomes.jsonl")
        suggestions = self._read_jsonl(self._findings_dir / "suggestions.jsonl")
        ledger = self._read_jsonl(self._findings_dir / "learning_ledger.jsonl")
        transfers = self._read_jsonl(self._findings_dir / "transfer_outcomes.jsonl")

        return LoopHealthMetrics(
            proposal_to_measurement_days=self._avg_proposal_to_measurement(
                suggestions, outcomes,
            ),
            oscillation_severity=self._compute_oscillation_severity(dims),
            transfer_success_rate=self._compute_transfer_success_rate(transfers),
            recalibration_effectiveness=self._compute_recalibration_effectiveness(
                outcomes,
            ),
            suggestions_per_cycle=self._avg_suggestions_per_cycle(ledger),
            measurement_coverage=self._compute_measurement_coverage(
                suggestions, outcomes,
            ),
        )

    @staticmethod
    def _avg_proposal_to_measurement(
        suggestions: list[dict], outcomes: list[dict],
    ) -> float | None:
        """Average days between suggestion proposal and outcome measurement."""
        # Build index of suggestion_id → proposed_at
        proposed_dates: dict[str, datetime] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            ts = s.get("proposed_at") or s.get("timestamp", "")
            if sid and ts:
                try:
                    proposed_dates[sid] = datetime.fromisoformat(
                        ts.replace("Z", "+00:00"),
                    )
                except (ValueError, TypeError):
                    pass

        latencies: list[float] = []
        for o in outcomes:
            sid = o.get("suggestion_id", "")
            ts = o.get("measurement_date") or o.get("measured_at") or o.get("timestamp", "")
            if sid in proposed_dates and ts:
                try:
                    measured = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    delta = (measured - proposed_dates[sid]).total_seconds() / 86400
                    if delta >= 0:
                        latencies.append(delta)
                except (ValueError, TypeError):
                    pass

        if not latencies:
            return None
        return round(sum(latencies) / len(latencies), 1)

    @staticmethod
    def _compute_oscillation_severity(
        dims: list[ConvergenceDimension],
    ) -> float:
        """Quantify oscillation severity as 0.0-1.0 score.

        Combines fraction of oscillating dimensions with the magnitude
        of their trend reversals.
        """
        if not dims:
            return 0.0
        osc_dims = [
            d for d in dims if d.status == DimensionStatus.OSCILLATING
        ]
        if not osc_dims:
            return 0.0
        # Fraction of dimensions oscillating
        fraction = len(osc_dims) / len(dims)
        # Average absolute trend value of oscillating dimensions (higher = worse)
        avg_magnitude = sum(abs(d.trend_value) for d in osc_dims) / len(osc_dims)
        # Normalize magnitude: clip at 0.1 slope for maximum severity
        magnitude_score = min(1.0, avg_magnitude / 0.1)
        # Combine: 60% fraction, 40% magnitude
        return round(0.6 * fraction + 0.4 * magnitude_score, 3)

    @staticmethod
    def _compute_transfer_success_rate(
        transfers: list[dict],
    ) -> float | None:
        """Positive outcome ratio for cross-bot transfers."""
        conclusive = [
            t for t in transfers
            if t.get("verdict") in ("positive", "negative", "neutral")
        ]
        if not conclusive:
            return None
        positive = sum(
            1 for t in conclusive if t.get("verdict") == "positive"
        )
        return round(positive / len(conclusive), 3)

    @staticmethod
    def _compute_recalibration_effectiveness(
        outcomes: list[dict],
    ) -> float | None:
        """Measure whether recalibrated categories improve subsequent outcomes.

        Compares win rate of outcomes with recalibrated=True vs False.
        """
        recalibrated: list[bool] = []
        non_recalibrated: list[bool] = []
        for o in outcomes:
            if not is_conclusive_outcome(o):
                continue
            positive = is_positive_outcome(o)
            if o.get("recalibrated"):
                recalibrated.append(positive)
            else:
                non_recalibrated.append(positive)

        if len(recalibrated) < 3 or len(non_recalibrated) < 3:
            return None

        recalib_rate = sum(recalibrated) / len(recalibrated)
        non_rate = sum(non_recalibrated) / len(non_recalibrated)
        # Positive = recalibration helps, negative = hurts
        return round(recalib_rate - non_rate, 3)

    @staticmethod
    def _avg_suggestions_per_cycle(ledger: list[dict]) -> float:
        """Average suggestions_proposed per learning cycle."""
        counts = [
            r.get("suggestions_proposed", 0)
            for r in ledger
            if r.get("suggestions_proposed") is not None
        ]
        if not counts:
            return 0.0
        return round(sum(counts) / len(counts), 1)

    @staticmethod
    def _compute_measurement_coverage(
        suggestions: list[dict], outcomes: list[dict],
    ) -> float:
        """Fraction of implemented suggestions that received outcome measurement."""
        implemented = {
            s.get("suggestion_id")
            for s in suggestions
            if s.get("status") in ("implemented", "IMPLEMENTED", "measured")
            and s.get("suggestion_id")
        }
        if not implemented:
            return 0.0
        measured = {
            o.get("suggestion_id")
            for o in outcomes
            if o.get("suggestion_id") in implemented
        }
        return round(len(measured) / len(implemented), 3)

    # ------------------------------------------------------------------
    # Trend analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_trend(
        values: list[float],
        threshold: float = 0.01,
    ) -> tuple[DimensionStatus, float]:
        """Classify a time series as improving, degrading, oscillating, or stable.

        Returns (status, slope).
        """
        if len(values) < 2:
            return DimensionStatus.INSUFFICIENT_DATA, 0.0

        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0

        # Check for oscillation: ≥4 of last 6 values alternate sign in deltas
        if len(values) >= 4:
            deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            recent_deltas = deltas[-6:] if len(deltas) >= 6 else deltas
            if len(recent_deltas) >= 3:
                sign_changes = sum(
                    1 for i in range(len(recent_deltas) - 1)
                    if recent_deltas[i] * recent_deltas[i + 1] < 0
                )
                # Also check if std deviation of deltas > 2× mean absolute delta
                abs_deltas = [abs(d) for d in recent_deltas]
                mean_abs = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0
                std_dev = (
                    statistics.stdev(recent_deltas) if len(recent_deltas) >= 2 else 0
                )
                if sign_changes >= 3 or (mean_abs > 0 and std_dev > 2 * mean_abs):
                    return DimensionStatus.OSCILLATING, slope

        if slope > threshold:
            return DimensionStatus.IMPROVING, slope
        elif slope < -threshold:
            return DimensionStatus.DEGRADING, slope
        else:
            return DimensionStatus.STABLE, slope

    @staticmethod
    def _synthesize_overall(dims: list[ConvergenceDimension]) -> DimensionStatus:
        """Synthesize overall status from individual dimensions."""
        statuses = [d.status for d in dims if d.status != DimensionStatus.INSUFFICIENT_DATA]
        if not statuses:
            return DimensionStatus.INSUFFICIENT_DATA
        if any(s == DimensionStatus.OSCILLATING for s in statuses):
            return DimensionStatus.OSCILLATING
        if all(s == DimensionStatus.IMPROVING for s in statuses):
            return DimensionStatus.IMPROVING
        if all(s == DimensionStatus.DEGRADING for s in statuses):
            return DimensionStatus.DEGRADING
        # Mixed — count
        improving = sum(1 for s in statuses if s == DimensionStatus.IMPROVING)
        degrading = sum(1 for s in statuses if s == DimensionStatus.DEGRADING)
        if improving > degrading:
            return DimensionStatus.IMPROVING
        elif degrading > improving:
            return DimensionStatus.DEGRADING
        return DimensionStatus.STABLE

    @staticmethod
    def _generate_recommendation(
        overall: DimensionStatus, dims: list[ConvergenceDimension],
    ) -> str:
        """Generate actionable recommendation from convergence state."""
        if overall == DimensionStatus.IMPROVING:
            return "System converging — maintain current approach"
        if overall == DimensionStatus.STABLE:
            return "System stable — consider targeted experiments to improve"
        if overall == DimensionStatus.DEGRADING:
            degrading = [d.name for d in dims if d.status == DimensionStatus.DEGRADING]
            return f"System degrading in {', '.join(degrading)} — review recent changes"
        if overall == DimensionStatus.OSCILLATING:
            osc = [d.name for d in dims if d.status == DimensionStatus.OSCILLATING]
            return (
                f"Oscillation detected in {', '.join(osc)} — "
                "avoid reversing last week's suggestions, let changes settle"
            )
        return "Insufficient data — continue collecting outcomes"

    @staticmethod
    def _trend_detail(
        label: str, values: list[float], status: DimensionStatus,
    ) -> str:
        """Build human-readable detail string."""
        if not values:
            return f"No {label} data"
        first = values[0]
        last = values[-1]
        mean = sum(values) / len(values)
        return (
            f"{label}: {first:.3f} → {last:.3f} "
            f"(mean={mean:.3f}, n={len(values)}, trend={status.value})"
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _group_outcomes_by_week(
        self, records: list[dict], weeks: int,
    ) -> list[float]:
        """Group outcomes by ISO week, return chronological positive ratios."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(weeks=weeks)

        weekly: dict[str, list[bool]] = {}
        for r in records:
            ts = r.get("measurement_date") or r.get("measured_at") or r.get("timestamp", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if dt < cutoff:
                continue
            if not is_conclusive_outcome(r):
                continue
            week_key = dt.strftime("%G-W%V")
            weekly.setdefault(week_key, []).append(is_positive_outcome(r))

        # Sort by week key and compute ratios
        ratios: list[float] = []
        for key in sorted(weekly.keys()):
            outcomes = weekly[key]
            if outcomes:
                ratios.append(sum(outcomes) / len(outcomes))
        return ratios

    @staticmethod
    def _in_time_range(record: dict, start: datetime, end: datetime) -> bool:
        """Check if a record's timestamp falls within [start, end)."""
        ts = record.get("measurement_date") or record.get("measured_at") or record.get("timestamp", "")
        if not ts:
            return False
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return start <= dt < end
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

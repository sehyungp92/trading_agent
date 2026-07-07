# skills/prediction_tracker.py
"""PredictionTracker — records and evaluates structured predictions.

Storage: memory/findings/predictions.jsonl
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from trading_assistant.schemas.agent_response import AgentPrediction
from trading_assistant.schemas.prediction_tracking import (
    PredictionEvaluation,
    PredictionRecord,
    PredictionVerdict,
)
from trading_assistant.orchestrator.jsonl_store import append_jsonl

logger = logging.getLogger(__name__)


_NOISE_THRESHOLDS: dict[str, float] = {
    "pnl": 50.0,
    "win_rate": 0.03,
    "drawdown": 0.005,
    "sharpe": 0.1,
}


class PredictionTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "predictions.jsonl"

    def record_predictions(
        self, week: str, predictions: list[AgentPrediction],
    ) -> None:
        """Append structured predictions for a given week/date."""
        if not predictions:
            return
        records = []
        for p in predictions:
            record = PredictionRecord(
                bot_id=p.bot_id,
                strategy_id=getattr(p, "strategy_id", None) or None,
                metric=p.metric,
                direction=p.direction,
                confidence=p.confidence,
                timeframe_days=p.timeframe_days,
                reasoning=p.reasoning,
                week=week,
            )
            records.append(record.model_dump(mode="json"))
        append_jsonl(self._path, records)

    def load_predictions(self, week: str | None = None) -> list[PredictionRecord]:
        """Load predictions, optionally filtered by week.

        P2-1: malformed lines are logged and skipped rather than crashing.
        """
        if not self._path.exists():
            return []
        records: list[PredictionRecord] = []
        for n, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), 1,
        ):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed predictions.jsonl line %d", n)
                continue
            if week and data.get("week") != week:
                continue
            try:
                records.append(PredictionRecord(**data))
            except Exception:
                logger.warning("Skipping invalid PredictionRecord at line %d", n)
        return records

    def evaluate_predictions(
        self, week: str, curated_dir: Path, baseline_date: str = "",
    ) -> PredictionEvaluation:
        """Evaluate predictions for a given week against actual curated data.

        Compares predicted direction with the realized metric change between the
        prediction's baseline date and its target horizon.
        """
        predictions = self.load_predictions(week)
        if not predictions:
            return PredictionEvaluation(week=week)

        verdicts: list[PredictionVerdict] = []
        for pred in predictions:
            anchor_date = self._parse_date(baseline_date or pred.week)
            actual = self._load_actual_metric_change(
                pred.bot_id,
                pred.metric,
                curated_dir,
                anchor_date,
                pred.timeframe_days,
            )
            if actual is None:
                verdicts.append(PredictionVerdict(
                    bot_id=pred.bot_id,
                    strategy_id=pred.strategy_id,
                    metric=pred.metric,
                    predicted_direction=pred.direction,
                    confidence=pred.confidence,
                    status="insufficient_data",
                ))
                continue

            actual_direction = self._classify_direction(actual, pred.metric)
            mag_score = self._compute_magnitude_score(pred.direction, actual, pred.metric)
            correct = pred.direction == actual_direction
            verdicts.append(PredictionVerdict(
                bot_id=pred.bot_id,
                strategy_id=pred.strategy_id,
                metric=pred.metric,
                predicted_direction=pred.direction,
                actual_direction=actual_direction,
                correct=correct,
                confidence=pred.confidence,
                status="correct" if correct else "incorrect",
                magnitude_score=round(mag_score, 3),
            ))

        # Compute aggregates
        evaluated = [v for v in verdicts if v.status != "insufficient_data"]
        total = len(evaluated)
        correct_count = sum(1 for v in evaluated if v.correct)
        accuracy = correct_count / total if total > 0 else 0.0

        # Confidence-weighted accuracy
        conf_sum = sum(v.confidence for v in evaluated) if evaluated else 0.0
        cw_accuracy = (
            sum(v.confidence * (1 if v.correct else 0) for v in evaluated) / conf_sum
            if conf_sum > 0 else 0.0
        )

        # Per-metric accuracy
        by_metric: dict[str, list[bool]] = defaultdict(list)
        for v in evaluated:
            by_metric[v.metric].append(v.correct)
        accuracy_by_metric = {
            m: sum(vals) / len(vals) for m, vals in by_metric.items()
        }

        # Per-strategy accuracy (only for verdicts that carry a strategy_id)
        by_strategy: dict[str, list[bool]] = defaultdict(list)
        for v in evaluated:
            if v.strategy_id:
                by_strategy[v.strategy_id].append(v.correct)
        accuracy_by_strategy = {
            sid: sum(vals) / len(vals) for sid, vals in by_strategy.items()
        }

        # Magnitude-weighted accuracy
        mag_sum = sum(v.magnitude_score for v in evaluated) if evaluated else 0.0
        mw_accuracy = mag_sum / total if total > 0 else 0.0

        return PredictionEvaluation(
            week=week,
            verdicts=verdicts,
            total=total,
            correct=correct_count,
            accuracy=round(accuracy, 3),
            confidence_weighted_accuracy=round(cw_accuracy, 3),
            accuracy_by_metric={m: round(v, 3) for m, v in accuracy_by_metric.items()},
            accuracy_by_strategy={sid: round(v, 3) for sid, v in accuracy_by_strategy.items()},
            magnitude_weighted_accuracy=round(mw_accuracy, 3),
        )

    def compute_directional_bias(
        self, curated_dir: Path, lookback_weeks: int = 12,
    ) -> dict[str, dict]:
        """Compute directional bias per metric over recent predictions.

        For each metric, compares the fraction of predictions predicting
        "improve" vs the fraction of actual "improve" outcomes.

        Returns:
            {metric: {predicted_improve_pct, actual_improve_pct, bias,
            bias_magnitude, sample_size}}
        """
        if not self._path.exists():
            return {}

        predictions = self.load_predictions()
        if not predictions:
            return {}

        # Filter to recent predictions within lookback window
        weeks = sorted({p.week for p in predictions})
        if len(weeks) > lookback_weeks:
            cutoff_week = weeks[-lookback_weeks]
            predictions = [p for p in predictions if p.week >= cutoff_week]

        # Group by metric, evaluate each
        by_metric: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for pred in predictions:
            anchor = self._parse_date(pred.week)
            actual = self._load_actual_metric_change(
                pred.bot_id, pred.metric, curated_dir,
                anchor, pred.timeframe_days,
            )
            if actual is None:
                continue
            actual_dir = self._classify_direction(actual, pred.metric)
            by_metric[pred.metric].append((pred.direction, actual_dir))

        result: dict[str, dict] = {}
        for metric, pairs in by_metric.items():
            if len(pairs) < 5:
                continue
            pred_improve = sum(1 for d, _ in pairs if d == "improve") / len(pairs)
            actual_improve = sum(1 for _, a in pairs if a == "improve") / len(pairs)
            gap = pred_improve - actual_improve
            if gap > 0.15:
                bias = "optimistic"
            elif gap < -0.15:
                bias = "pessimistic"
            else:
                bias = "balanced"
            result[metric] = {
                "predicted_improve_pct": round(pred_improve, 3),
                "actual_improve_pct": round(actual_improve, 3),
                "bias": bias,
                "bias_magnitude": round(abs(gap), 3),
                "sample_size": len(pairs),
            }
        return result

    def get_accuracy_by_metric(self, curated_dir: Path) -> dict[str, float]:
        """Compute rolling accuracy per metric type across all evaluated weeks."""
        if not self._path.exists():
            return {}

        # Get all unique weeks
        weeks: set[str] = set()
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                data = json.loads(line)
                w = data.get("week", "")
                if w:
                    weeks.add(w)

        # Aggregate across weeks
        all_by_metric: dict[str, list[bool]] = defaultdict(list)
        for week in sorted(weeks):
            evaluation = self.evaluate_predictions(week, curated_dir)
            for v in evaluation.verdicts:
                if v.status != "insufficient_data":
                    all_by_metric[v.metric].append(v.correct)

        return {
            m: round(sum(vals) / len(vals), 3)
            for m, vals in all_by_metric.items()
            if vals
        }

    @staticmethod
    def _classify_direction(actual: float, metric: str) -> str:
        """Classify direction with per-metric noise thresholds.

        Changes below the noise threshold are classified as "stable"
        instead of "improve"/"decline".
        """
        threshold = _NOISE_THRESHOLDS.get(metric, 0.0)
        if abs(actual) < threshold:
            return "stable"
        return "improve" if actual > 0 else "decline"

    @staticmethod
    def _compute_magnitude_score(
        predicted_direction: str, actual: float, metric: str,
    ) -> float:
        """Compute alignment strength between prediction and actual change.

        Returns 0-1: 1.0 = perfect alignment, 0.0 = opposite direction.
        """
        threshold = _NOISE_THRESHOLDS.get(metric, 0.0)
        if threshold <= 0:
            return 1.0 if predicted_direction == "stable" else 0.5

        # Normalize actual change relative to threshold
        normalized = actual / threshold  # >1 means above noise

        if predicted_direction == "improve":
            # Higher positive actual → better score
            return max(0.0, min(1.0, 0.5 + normalized * 0.25))
        elif predicted_direction == "decline":
            # Higher negative actual → better score
            return max(0.0, min(1.0, 0.5 - normalized * 0.25))
        else:  # stable
            # Closer to zero → better score
            return max(0.0, 1.0 - abs(normalized) * 0.25)

    def _load_actual_metric_change(
        self,
        bot_id: str,
        metric: str,
        curated_dir: Path,
        baseline_date: date | None,
        timeframe_days: int,
    ) -> float | None:
        """Load realized change between the baseline and target dates."""
        if baseline_date is None:
            return None

        baseline_value = self._load_metric_for_date(
            bot_id, metric, curated_dir, baseline_date,
        )
        target_value = self._load_metric_for_date(
            bot_id, metric, curated_dir, baseline_date + timedelta(days=timeframe_days),
        )
        if baseline_value is None or target_value is None:
            return None
        return target_value - baseline_value

    def _load_metric_for_date(
        self,
        bot_id: str,
        metric: str,
        curated_dir: Path,
        target_date: date,
        max_offset_days: int = 3,
    ) -> float | None:
        """Load the closest available metric near a target date.

        We prefer exact matches, then later dates, then earlier dates. This
        tolerates weekends/market holidays without making evaluations depend on
        the current date.
        """
        for candidate in self._candidate_dates(target_date, max_offset_days):
            summary_path = curated_dir / candidate.strftime("%Y-%m-%d") / bot_id / "summary.json"
            if not summary_path.exists():
                continue
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            metric_value = self._extract_metric(data, metric)
            if metric_value is not None:
                return metric_value
        return None

    @staticmethod
    def _candidate_dates(target_date: date, max_offset_days: int) -> list[date]:
        dates = [target_date]
        for offset in range(1, max_offset_days + 1):
            dates.append(target_date + timedelta(days=offset))
            dates.append(target_date - timedelta(days=offset))
        return dates

    @staticmethod
    def _parse_date(raw: str) -> date | None:
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _extract_metric(summary: dict, metric: str) -> float | None:
        """Extract a metric value from a daily summary."""
        metric_keys = {
            "pnl": ["total_pnl", "pnl", "net_pnl"],
            "win_rate": ["win_rate", "win_pct"],
            "drawdown": ["max_drawdown_pct", "max_drawdown", "drawdown"],
            "sharpe": ["sharpe_rolling_30d", "sharpe", "sharpe_ratio"],
        }
        for key in metric_keys.get(metric, [metric]):
            val = summary.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

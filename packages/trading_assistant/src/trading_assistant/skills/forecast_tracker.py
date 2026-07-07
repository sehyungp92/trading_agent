# skills/forecast_tracker.py
"""ForecastTracker — records weekly accuracy and computes rolling meta-analysis.

Storage: memory/findings/forecast_history.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.forecast_tracking import (
    AccuracyTrend,
    CalibrationBucket,
    ForecastMetaAnalysis,
    ForecastRecord,
)
from trading_assistant.schemas.prediction_tracking import PredictionVerdict


class ForecastTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "forecast_history.jsonl"

    def record_week(self, record: ForecastRecord) -> None:
        """Append a weekly forecast record."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[ForecastRecord]:
        """Load all forecast records from disk."""
        if not self._path.exists():
            return []
        records: list[ForecastRecord] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                records.append(ForecastRecord(**json.loads(line)))
        return records

    def compute_meta_analysis(
        self,
        prediction_verdicts: list[PredictionVerdict] | None = None,
        directional_bias: dict[str, dict] | None = None,
    ) -> ForecastMetaAnalysis:
        """Compute rolling accuracy, trend, and calibration from all records.

        Args:
            prediction_verdicts: Optional list of prediction verdicts for
                empirical calibration computation. When provided, populates
                calibration_buckets, expected_calibration_error, and brier_score.
        """
        records = self.load_all()
        if not records:
            return ForecastMetaAnalysis()

        # Sort by week_start descending
        records.sort(key=lambda r: r.week_start, reverse=True)

        weeks_analyzed = len(records)

        # Rolling accuracy: last 4 weeks
        last_4 = records[:4]
        acc_4w = self._weighted_accuracy(last_4)

        # Rolling accuracy: last 12 weeks
        last_12 = records[:12]
        acc_12w = self._weighted_accuracy(last_12)

        # Trend: compare first half vs second half of available data
        trend = self._compute_trend(records)

        # Per-bot accuracy: aggregate across all weeks
        bot_totals: dict[str, list[float]] = {}
        for rec in records:
            for bot_id, acc in rec.by_bot.items():
                bot_totals.setdefault(bot_id, []).append(acc)
        accuracy_by_bot = {
            bot: sum(accs) / len(accs) for bot, accs in bot_totals.items()
        }

        # Per-metric accuracy: aggregate by_type across all weeks
        metric_totals: dict[str, list[float]] = {}
        for rec in records:
            for metric, acc in rec.by_type.items():
                metric_totals.setdefault(metric, []).append(acc)
        accuracy_by_metric = {
            m: round(sum(accs) / len(accs), 3) for m, accs in metric_totals.items()
        }

        # Calibration adjustment: compare average confidence vs average accuracy
        # Positive = under-confident, Negative = over-confident
        avg_accuracy = acc_4w if weeks_analyzed >= 4 else acc_12w if weeks_analyzed >= 1 else 0.0
        # Assume 0.5 is the "neutral" confidence baseline
        calibration = avg_accuracy - 0.5  # clamped to [-1, 1]
        calibration = max(-1.0, min(1.0, calibration))

        # Empirical calibration from prediction verdicts
        cal_buckets: list[CalibrationBucket] = []
        ece: float | None = None
        brier: float | None = None
        cal_sample_size = 0
        accuracy_by_strategy: dict[str, float] = {}
        if prediction_verdicts:
            cal_buckets, ece, brier = self.compute_calibration(prediction_verdicts)
            cal_sample_size = sum(b.prediction_count for b in cal_buckets)
            # Per-strategy accuracy from verdicts that carry a strategy_id
            strat_totals: dict[str, list[bool]] = {}
            for v in prediction_verdicts:
                if v.status == "insufficient_data" or not v.strategy_id:
                    continue
                strat_totals.setdefault(v.strategy_id, []).append(v.correct)
            accuracy_by_strategy = {
                sid: round(sum(vals) / len(vals), 3)
                for sid, vals in strat_totals.items()
            }

        return ForecastMetaAnalysis(
            rolling_accuracy_4w=round(acc_4w, 3),
            rolling_accuracy_12w=round(acc_12w, 3),
            trend=trend,
            accuracy_by_bot=accuracy_by_bot,
            accuracy_by_metric=accuracy_by_metric,
            accuracy_by_strategy=accuracy_by_strategy,
            calibration_adjustment=round(calibration, 3),
            weeks_analyzed=weeks_analyzed,
            calibration_buckets=cal_buckets,
            expected_calibration_error=ece,
            brier_score=brier,
            calibration_sample_size=cal_sample_size,
            directional_bias=directional_bias or {},
        )

    @staticmethod
    def compute_calibration(
        verdicts: list[PredictionVerdict],
        min_bucket_count: int = 5,
    ) -> tuple[list[CalibrationBucket], float | None, float | None]:
        """Compute calibration buckets, ECE, and Brier score from prediction verdicts.

        Returns:
            (buckets, ece, brier_score). ECE and Brier are None when total
            samples across reliable buckets < min_bucket_count.
        """
        boundaries = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        bucket_data: list[list[PredictionVerdict]] = [[] for _ in boundaries]

        for v in verdicts:
            if v.status == "insufficient_data":
                continue
            conf = max(0.0, min(1.0, v.confidence))
            for i, (lo, hi) in enumerate(boundaries):
                if i == len(boundaries) - 1:
                    if lo <= conf <= hi:
                        bucket_data[i].append(v)
                        break
                else:
                    if lo <= conf < hi:
                        bucket_data[i].append(v)
                        break

        buckets: list[CalibrationBucket] = []
        for i, (lo, hi) in enumerate(boundaries):
            items = bucket_data[i]
            count = len(items)
            correct = sum(1 for v in items if v.correct)
            mean_conf = sum(v.confidence for v in items) / count if count else 0.0
            obs_acc = correct / count if count else 0.0
            gap = mean_conf - obs_acc
            buckets.append(CalibrationBucket(
                bucket_lower=lo,
                bucket_upper=hi,
                prediction_count=count,
                correct_count=correct,
                mean_confidence=round(mean_conf, 4),
                observed_accuracy=round(obs_acc, 4),
                gap=round(gap, 4),
            ))

        # ECE = weighted average of |gap| across reliable buckets
        total_reliable = sum(b.prediction_count for b in buckets if b.is_reliable)
        if total_reliable < min_bucket_count:
            return buckets, None, None

        ece = sum(
            b.prediction_count * abs(b.gap) for b in buckets if b.is_reliable
        ) / total_reliable

        # Brier score = mean of (confidence - correct)^2
        all_items = [v for vs in bucket_data for v in vs]
        if not all_items:
            return buckets, round(ece, 4), None
        brier = sum(
            (v.confidence - (1.0 if v.correct else 0.0)) ** 2 for v in all_items
        ) / len(all_items)

        return buckets, round(ece, 4), round(brier, 4)

    @staticmethod
    def _weighted_accuracy(records: list[ForecastRecord]) -> float:
        """Compute weighted accuracy from a list of records."""
        total_reviewed = sum(r.predictions_reviewed for r in records)
        if total_reviewed == 0:
            return 0.0
        total_correct = sum(r.correct_predictions for r in records)
        return total_correct / total_reviewed

    @staticmethod
    def _compute_trend(records: list[ForecastRecord]) -> AccuracyTrend:
        """Determine accuracy trend from record history."""
        if len(records) < 3:
            return AccuracyTrend.STABLE

        # Compare recent half vs older half
        mid = len(records) // 2
        recent = records[:mid]
        older = records[mid:]

        recent_acc = ForecastTracker._weighted_accuracy(recent)
        older_acc = ForecastTracker._weighted_accuracy(older)

        diff = recent_acc - older_acc
        if diff > 0.05:
            return AccuracyTrend.IMPROVING
        elif diff < -0.05:
            return AccuracyTrend.DEGRADING
        return AccuracyTrend.STABLE

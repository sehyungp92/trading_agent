# tests/test_calibration.py
"""Tests for measured forecast calibration (Section A)."""
from __future__ import annotations



from trading_assistant.schemas.forecast_tracking import CalibrationBucket, ForecastMetaAnalysis
from trading_assistant.schemas.prediction_tracking import PredictionRecord, PredictionVerdict
from trading_assistant.skills.forecast_tracker import ForecastTracker


class TestCalibrationBucket:
    def test_is_reliable_above_threshold(self):
        b = CalibrationBucket(
            bucket_lower=0.6, bucket_upper=0.8,
            prediction_count=10, correct_count=7,
            mean_confidence=0.7, observed_accuracy=0.7, gap=0.0,
        )
        assert b.is_reliable is True

    def test_is_reliable_below_threshold(self):
        b = CalibrationBucket(
            bucket_lower=0.6, bucket_upper=0.8,
            prediction_count=3, correct_count=2,
            mean_confidence=0.7, observed_accuracy=0.667, gap=0.033,
        )
        assert b.is_reliable is False


class TestForecastMetaAnalysisExtended:
    def test_new_fields_default(self):
        meta = ForecastMetaAnalysis()
        assert meta.calibration_buckets == []
        assert meta.expected_calibration_error is None
        assert meta.brier_score is None
        assert meta.calibration_sample_size == 0

    def test_with_calibration_data(self):
        bucket = CalibrationBucket(
            bucket_lower=0.6, bucket_upper=0.8,
            prediction_count=10, correct_count=7,
            mean_confidence=0.7, observed_accuracy=0.7, gap=0.0,
        )
        meta = ForecastMetaAnalysis(
            calibration_buckets=[bucket],
            expected_calibration_error=0.05,
            brier_score=0.2,
            calibration_sample_size=10,
        )
        assert len(meta.calibration_buckets) == 1
        assert meta.expected_calibration_error == 0.05
        assert meta.brier_score == 0.2


class TestPredictionRecordExtended:
    def test_new_optional_fields(self):
        rec = PredictionRecord(
            bot_id="test", metric="pnl", direction="improve", confidence=0.8,
        )
        assert rec.predicted_magnitude is None
        assert rec.actual_magnitude is None
        assert rec.regime_tag is None
        assert rec.prediction_family is None

    def test_with_new_fields(self):
        rec = PredictionRecord(
            bot_id="test", metric="pnl", direction="improve", confidence=0.8,
            predicted_magnitude=0.05, regime_tag="trending",
            prediction_family="performance",
        )
        assert rec.predicted_magnitude == 0.05
        assert rec.regime_tag == "trending"
        assert rec.prediction_family == "performance"


class TestComputeCalibration:
    def _make_verdicts(self, data: list[tuple[float, bool]]) -> list[PredictionVerdict]:
        """Create verdicts from (confidence, correct) tuples."""
        return [
            PredictionVerdict(
                bot_id="test", metric="pnl",
                predicted_direction="improve",
                correct=correct, confidence=conf,
                status="correct" if correct else "incorrect",
            )
            for conf, correct in data
        ]

    def test_empty_verdicts(self):
        buckets, ece, brier = ForecastTracker.compute_calibration([])
        assert len(buckets) == 5
        assert all(b.prediction_count == 0 for b in buckets)
        assert ece is None
        assert brier is None

    def test_insufficient_data_verdicts(self):
        verdicts = self._make_verdicts([(0.7, True), (0.8, False)])
        buckets, ece, brier = ForecastTracker.compute_calibration(verdicts)
        assert ece is None  # Not enough data in any bucket

    def test_well_calibrated(self):
        # 10 predictions at 0.7 confidence, 7 correct = well calibrated
        verdicts = self._make_verdicts(
            [(0.7, True)] * 7 + [(0.7, False)] * 3
        )
        buckets, ece, brier = ForecastTracker.compute_calibration(verdicts)
        assert ece is not None
        assert ece < 0.05  # Well calibrated, small ECE
        assert brier is not None

    def test_overconfident(self):
        # 10 predictions at 0.9 confidence, only 5 correct = overconfident
        verdicts = self._make_verdicts(
            [(0.9, True)] * 5 + [(0.9, False)] * 5
        )
        buckets, ece, brier = ForecastTracker.compute_calibration(verdicts)
        assert ece is not None
        high_bucket = [b for b in buckets if b.bucket_lower == 0.8][0]
        assert high_bucket.prediction_count == 10
        assert high_bucket.gap > 0.3  # Significantly overconfident

    def test_skips_insufficient_data_verdicts(self):
        verdicts = self._make_verdicts([(0.5, True)])
        # Add a verdict with insufficient_data status
        verdicts.append(PredictionVerdict(
            bot_id="test", metric="pnl",
            predicted_direction="improve",
            correct=False, confidence=0.5,
            status="insufficient_data",
        ))
        buckets, _, _ = ForecastTracker.compute_calibration(verdicts)
        total = sum(b.prediction_count for b in buckets)
        assert total == 1  # Only the non-insufficient one counted

    def test_bucket_boundaries(self):
        # Confidence of exactly 1.0 should go in [0.8, 1.0] bucket
        verdicts = self._make_verdicts([(1.0, True)] * 5)
        buckets, _, _ = ForecastTracker.compute_calibration(verdicts)
        last_bucket = buckets[-1]
        assert last_bucket.bucket_upper == 1.0
        assert last_bucket.prediction_count == 5


class TestComputeMetaAnalysisWithVerdicts:
    def test_without_verdicts_preserves_behavior(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        from trading_assistant.schemas.forecast_tracking import ForecastRecord
        tracker.record_week(ForecastRecord(
            week_start="2026-03-01", week_end="2026-03-07",
            predictions_reviewed=10, correct_predictions=7, accuracy=0.7,
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.calibration_buckets == []
        assert meta.expected_calibration_error is None

    def test_with_verdicts_populates_calibration(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        from trading_assistant.schemas.forecast_tracking import ForecastRecord
        tracker.record_week(ForecastRecord(
            week_start="2026-03-01", week_end="2026-03-07",
            predictions_reviewed=10, correct_predictions=7, accuracy=0.7,
        ))
        verdicts = [
            PredictionVerdict(
                bot_id="test", metric="pnl",
                predicted_direction="improve",
                correct=i < 7, confidence=0.7,
                status="correct" if i < 7 else "incorrect",
            )
            for i in range(10)
        ]
        meta = tracker.compute_meta_analysis(prediction_verdicts=verdicts)
        assert len(meta.calibration_buckets) == 5
        assert meta.calibration_sample_size == 10


class TestResponseValidatorCalibration:
    def test_bucket_adjustment_applied(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis

        validator = ResponseValidator(
            forecast_meta={
                "calibration_buckets": [
                    {
                        "bucket_lower": 0.6, "bucket_upper": 0.8,
                        "prediction_count": 10, "correct_count": 5,
                        "mean_confidence": 0.7, "observed_accuracy": 0.5,
                        "gap": 0.2,  # Overconfident
                    },
                ],
            },
        )
        suggestion = AgentSuggestion(
            bot_id="test", title="Test", confidence=0.7,
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        result = validator.validate(parsed)
        # Confidence should be adjusted down from 0.7
        assert result.approved_suggestions[0].confidence < 0.7

    def test_no_adjustment_when_gap_small(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis

        validator = ResponseValidator(
            forecast_meta={
                "calibration_buckets": [
                    {
                        "bucket_lower": 0.6, "bucket_upper": 0.8,
                        "prediction_count": 10, "correct_count": 7,
                        "mean_confidence": 0.7, "observed_accuracy": 0.65,
                        "gap": 0.05,  # Small gap
                    },
                ],
            },
        )
        suggestion = AgentSuggestion(
            bot_id="test", title="Test", confidence=0.7,
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        result = validator.validate(parsed)
        # No adjustment needed, confidence should be preserved
        assert result.approved_suggestions[0].confidence == 0.7

    def test_fallback_to_heuristic_when_no_buckets(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis

        validator = ResponseValidator(
            forecast_meta={"rolling_accuracy_4w": 0.4},
        )
        suggestion = AgentSuggestion(
            bot_id="test", title="Test", confidence=0.8,
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        result = validator.validate(parsed)
        # Should use heuristic fallback: 0.8 * 0.4 = 0.32
        assert result.approved_suggestions[0].confidence < 0.5

    def test_prediction_bucket_adjustment(self):
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import AgentPrediction, ParsedAnalysis

        validator = ResponseValidator(
            forecast_meta={
                "calibration_buckets": [
                    {
                        "bucket_lower": 0.6, "bucket_upper": 0.8,
                        "prediction_count": 10, "correct_count": 4,
                        "mean_confidence": 0.7, "observed_accuracy": 0.4,
                        "gap": 0.3,
                    },
                ],
            },
        )
        prediction = AgentPrediction(
            bot_id="test", metric="pnl", direction="improve", confidence=0.7,
        )
        parsed = ParsedAnalysis(predictions=[prediction])
        result = validator.validate(parsed)
        assert result.approved_predictions[0].confidence < 0.7

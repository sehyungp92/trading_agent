# tests/test_detection_context.py
"""Tests for detection context schemas."""
from datetime import datetime, timezone

import pytest

from trading_assistant.schemas.detection_context import (
    DetectionContext,
    ThresholdProfile,
    ThresholdRecord,
)


class TestDetectionContext:
    def test_margin_observed_above_threshold(self):
        ctx = DetectionContext(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            threshold_value=0.3,
            observed_value=0.5,
        )
        assert ctx.margin == pytest.approx(0.2)

    def test_margin_observed_below_threshold(self):
        ctx = DetectionContext(
            detector_name="filter_cost",
            bot_id="bot2",
            threshold_name="filter_cost_threshold",
            threshold_value=0.4,
            observed_value=0.1,
        )
        assert ctx.margin == pytest.approx(0.3)

    def test_serialization_round_trip(self):
        now = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)
        ctx = DetectionContext(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            threshold_value=0.3,
            observed_value=0.5,
            detected_at=now,
        )
        data = ctx.model_dump()
        restored = DetectionContext.model_validate(data)
        assert restored.detector_name == ctx.detector_name
        assert restored.bot_id == ctx.bot_id
        assert restored.threshold_value == ctx.threshold_value
        assert restored.observed_value == ctx.observed_value
        assert restored.detected_at == now
        assert restored.margin == ctx.margin


class TestThresholdRecord:
    def test_effective_value_returns_default_when_no_learned(self):
        rec = ThresholdRecord(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            default_value=0.3,
        )
        assert rec.effective_value == 0.3

    def test_effective_value_returns_default_when_zero_confidence(self):
        rec = ThresholdRecord(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            default_value=0.3,
            learned_value=0.25,
            confidence=0.0,
        )
        assert rec.effective_value == 0.3

    def test_effective_value_returns_learned_when_confident(self):
        rec = ThresholdRecord(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            default_value=0.3,
            learned_value=0.25,
            confidence=0.8,
            sample_count=20,
        )
        assert rec.effective_value == 0.25

    def test_confidence_accepts_boundary_values(self):
        rec = ThresholdRecord(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            default_value=0.3,
            learned_value=0.25,
            confidence=1.0,
            sample_count=50,
        )
        assert rec.confidence == 1.0
        assert rec.effective_value == 0.25


class TestThresholdProfile:
    def test_serialization_round_trip(self):
        rec = ThresholdRecord(
            detector_name="alpha_decay",
            bot_id="bot1",
            threshold_name="decay_threshold",
            default_value=0.3,
            learned_value=0.25,
            confidence=0.8,
            sample_count=20,
        )
        profile = ThresholdProfile(
            bot_id="bot1",
            thresholds={"alpha_decay:decay_threshold": rec},
            total_outcomes_used=42,
        )
        data = profile.model_dump()
        restored = ThresholdProfile.model_validate(data)
        assert restored.bot_id == "bot1"
        assert restored.total_outcomes_used == 42
        assert "alpha_decay:decay_threshold" in restored.thresholds
        restored_rec = restored.thresholds["alpha_decay:decay_threshold"]
        assert restored_rec.effective_value == 0.25
        assert restored_rec.sample_count == 20

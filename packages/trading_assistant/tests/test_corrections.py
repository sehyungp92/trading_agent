# tests/test_corrections.py
"""Tests for human correction schema."""
from datetime import datetime

from trading_assistant.schemas.corrections import HumanCorrection, CorrectionType


class TestCorrectionType:
    def test_all_types_exist(self):
        assert CorrectionType.TRADE_RECLASSIFY
        assert CorrectionType.REGIME_OVERRIDE
        assert CorrectionType.POSITIVE_REINFORCEMENT
        assert CorrectionType.FREE_TEXT


class TestHumanCorrection:
    def test_trade_reclassify(self):
        c = HumanCorrection(
            correction_type=CorrectionType.TRADE_RECLASSIFY,
            original_report_id="daily-2026-03-01",
            target_id="trade_xyz",
            raw_text="Trade #xyz wasn't bad — it was a planned hedge against spot",
            structured_correction={"new_root_cause": "normal_win", "reason": "planned hedge"},
        )
        assert c.correction_type == CorrectionType.TRADE_RECLASSIFY
        assert c.target_id == "trade_xyz"

    def test_regime_override(self):
        c = HumanCorrection(
            correction_type=CorrectionType.REGIME_OVERRIDE,
            original_report_id="daily-2026-03-01",
            target_id="bot2",
            raw_text="Bot2's regime classification was wrong today, slow trend not ranging",
            structured_correction={"old_regime": "ranging", "new_regime": "trending_up"},
        )
        assert c.structured_correction["new_regime"] == "trending_up"

    def test_has_timestamp(self):
        c = HumanCorrection(
            correction_type=CorrectionType.FREE_TEXT,
            original_report_id="daily-2026-03-01",
            raw_text="Good catch on the volume filter",
        )
        assert c.timestamp is not None
        assert isinstance(c.timestamp, datetime)

    def test_serializes_to_jsonl_format(self):
        c = HumanCorrection(
            correction_type=CorrectionType.POSITIVE_REINFORCEMENT,
            original_report_id="daily-2026-03-01",
            raw_text="Good catch on the volume filter, that's the third time this week",
        )
        d = c.model_dump(mode="json")
        assert "correction_type" in d
        assert "timestamp" in d
        assert "raw_text" in d

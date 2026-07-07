import pytest
from strategies.swing.instrumentation.src.overnight_gap_tracker import OvernightGapTracker

class TestOvernightGapTracker:
    def test_record_close_and_compute_gap(self):
        tracker = OvernightGapTracker()
        tracker.record_close("SPY", 500.0)
        result = tracker.compute_gap("SPY", 505.0)
        assert result["overnight_gap_pct"] == pytest.approx(1.0)
        assert result["prev_close_price"] == 500.0

    def test_no_previous_close_returns_none(self):
        tracker = OvernightGapTracker()
        result = tracker.compute_gap("SPY", 500.0)
        assert result["overnight_gap_pct"] is None
        assert result["prev_close_price"] is None

    def test_negative_gap(self):
        tracker = OvernightGapTracker()
        tracker.record_close("QQQ", 400.0)
        result = tracker.compute_gap("QQQ", 392.0)
        assert result["overnight_gap_pct"] == pytest.approx(-2.0)

    def test_multiple_symbols_independent(self):
        tracker = OvernightGapTracker()
        tracker.record_close("SPY", 500.0)
        tracker.record_close("QQQ", 400.0)
        spy_gap = tracker.compute_gap("SPY", 510.0)
        qqq_gap = tracker.compute_gap("QQQ", 396.0)
        assert spy_gap["overnight_gap_pct"] == pytest.approx(2.0)
        assert qqq_gap["overnight_gap_pct"] == pytest.approx(-1.0)

import pytest
from strategies.swing.instrumentation.src.drawdown_tracker import DrawdownTracker


class TestDrawdownTracker:
    def test_initial_state_is_normal(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        assert tracker.current_tier == "NORMAL"
        assert tracker.drawdown_pct == 0.0
        assert tracker.position_size_multiplier == 1.0

    def test_update_equity_computes_drawdown(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(95_000)
        assert tracker.drawdown_pct == pytest.approx(5.0)

    def test_caution_tier_at_5pct(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(94_999)
        assert tracker.current_tier == "CAUTION"
        assert tracker.position_size_multiplier == 0.5

    def test_danger_tier_at_10pct(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(89_999)
        assert tracker.current_tier == "DANGER"
        assert tracker.position_size_multiplier == 0.25

    def test_halt_tier_at_15pct(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(84_999)
        assert tracker.current_tier == "HALT"
        assert tracker.position_size_multiplier == 0.0

    def test_peak_equity_tracks_high_water_mark(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(110_000)
        assert tracker.peak_equity == 110_000
        tracker.update_equity(105_000)
        assert tracker.drawdown_pct == pytest.approx(100 * 5_000 / 110_000, rel=1e-3)

    def test_get_entry_context_returns_dict(self):
        tracker = DrawdownTracker(initial_equity=100_000)
        tracker.update_equity(96_000)
        ctx = tracker.get_entry_context()
        assert ctx["drawdown_pct_at_entry"] == pytest.approx(4.0)
        assert ctx["drawdown_tier_at_entry"] == "NORMAL"
        assert ctx["position_size_multiplier"] == 1.0

"""Tests for the signal pipeline funnel tracker."""

from datetime import datetime, timezone

from crypto_trader.instrumentation.pipeline_tracker import PipelineFunnel, PipelineTracker


class TestPipelineTracker:
    def _make_tracker(self) -> PipelineTracker:
        return PipelineTracker("test_strategy")

    def test_record_bar_increments(self):
        t = self._make_tracker()
        t.record_bar("BTC")
        t.record_bar("BTC")
        t.record_bar("ETH")
        funnel = t.snapshot_and_reset()
        assert funnel.bars_received["BTC"] == 2
        assert funnel.bars_received["ETH"] == 1

    def test_record_gate_pass_maps_to_stage(self):
        t = self._make_tracker()
        t.record_gate("BTC", "setup", True)
        t.record_gate("BTC", "confirmation", True)
        t.record_gate("BTC", "indicators", True)
        funnel = t.snapshot_and_reset()
        assert funnel.setups_detected["BTC"] == 1
        assert funnel.confirmations["BTC"] == 1
        assert funnel.indicators_ready["BTC"] == 1

    def test_record_gate_fail_tracks_rejection(self):
        t = self._make_tracker()
        t.record_gate("BTC", "risk_check", False)
        t.record_gate("BTC", "risk_check", False)
        t.record_gate("BTC", "setup", False)
        funnel = t.snapshot_and_reset()
        assert funnel.gate_rejections["BTC"]["risk_check"] == 2
        assert funnel.gate_rejections["BTC"]["setup"] == 1
        # Failed gates should NOT count in the pass stage
        assert funnel.setups_detected.get("BTC", 0) == 0

    def test_record_fill_and_trade_closed(self):
        t = self._make_tracker()
        t.record_fill("BTC")
        t.record_fill("BTC")
        t.record_trade_closed("BTC")
        funnel = t.snapshot_and_reset()
        assert funnel.fills["BTC"] == 2
        assert funnel.trades_closed["BTC"] == 1

    def test_snapshot_resets_counters(self):
        t = self._make_tracker()
        t.record_bar("BTC")
        t.record_gate("BTC", "setup", True)
        f1 = t.snapshot_and_reset()
        assert f1.bars_received["BTC"] == 1
        assert f1.setups_detected["BTC"] == 1

        f2 = t.snapshot_and_reset()
        assert f2.bars_received.get("BTC", 0) == 0
        assert f2.setups_detected.get("BTC", 0) == 0

    def test_snapshot_period_timestamps(self):
        t = self._make_tracker()
        f = t.snapshot_and_reset()
        assert f.period_start <= f.period_end
        assert f.strategy_id == "test_strategy"

    def test_unknown_gate_does_not_map_to_stage(self):
        t = self._make_tracker()
        t.record_gate("BTC", "unknown_gate", True)
        funnel = t.snapshot_and_reset()
        # Should not increment any known stage
        assert funnel.setups_detected.get("BTC", 0) == 0
        assert funnel.indicators_ready.get("BTC", 0) == 0


class TestPipelineFunnelAssess:
    def _make_funnel(self, **overrides) -> PipelineFunnel:
        defaults = {
            "strategy_id": "test",
            "period_start": datetime.now(timezone.utc),
            "period_end": datetime.now(timezone.utc),
        }
        defaults.update(overrides)
        return PipelineFunnel(**defaults)

    def test_pipeline_broken_no_bars(self):
        f = self._make_funnel()
        assert PipelineTracker.assess(f) == "pipeline_broken"

    def test_stalled_bars_no_indicators(self):
        f = self._make_funnel(bars_received={"BTC": 10})
        assert PipelineTracker.assess(f) == "stalled"

    def test_no_signals_legitimate(self):
        f = self._make_funnel(
            bars_received={"BTC": 10},
            indicators_ready={"BTC": 10},
        )
        assert PipelineTracker.assess(f) == "no_signals"

    def test_normal_with_entries(self):
        f = self._make_funnel(
            bars_received={"BTC": 10},
            indicators_ready={"BTC": 10},
            setups_detected={"BTC": 2},
            entries_attempted={"BTC": 1},
        )
        assert PipelineTracker.assess(f) == "normal"

    def test_gate_blocked(self):
        f = self._make_funnel(
            bars_received={"BTC": 10},
            indicators_ready={"BTC": 10},
            setups_detected={"BTC": 5},
            gate_rejections={"BTC": {"sizing": 5}},
        )
        assert PipelineTracker.assess(f) == "gate_blocked"

    def test_normal_when_setups_but_partial_blocks(self):
        """Not gate_blocked if gate doesn't block ALL setups."""
        f = self._make_funnel(
            bars_received={"BTC": 10},
            indicators_ready={"BTC": 10},
            setups_detected={"BTC": 5},
            entries_attempted={"BTC": 2},
            gate_rejections={"BTC": {"sizing": 3}},
        )
        assert PipelineTracker.assess(f) == "normal"


class TestPipelineFunnelToDict:
    def test_to_dict_roundtrip(self):
        f = PipelineFunnel(
            strategy_id="test",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            bars_received={"BTC": 5},
            setups_detected={"BTC": 2},
            gate_rejections={"BTC": {"risk_check": 3}},
        )
        d = f.to_dict()
        assert d["strategy_id"] == "test"
        assert d["bars_received"]["BTC"] == 5
        assert d["gate_rejections"]["BTC"]["risk_check"] == 3

    def test_total_helper(self):
        f = PipelineFunnel(
            strategy_id="test",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            bars_received={"BTC": 10, "ETH": 5},
        )
        assert f.total("bars_received") == 15
        assert f.total("nonexistent") == 0

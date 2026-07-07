"""Tests for SlippageAnalyzer → CostModel wiring."""
import json
from datetime import datetime, timezone

from trading_assistant.schemas.cost_model import CostModelConfig, SlippageModel
from trading_assistant.skills.cost_model import CostModel


class TestCostModelFromSlippageExport:
    def test_from_slippage_export_creates_empirical_model(self):
        regime_bps = {"trending": 5.0, "ranging": 8.0, "default": 6.0}
        config = CostModelConfig(
            slippage_model=SlippageModel.EMPIRICAL,
            fees_per_trade_bps=5.0,
            fixed_slippage_bps=3.0,
        )
        model = CostModel.from_slippage_export(regime_bps, config)
        costs = model.compute_costs(100.0, 1.0, regime="trending")
        assert costs.slippage > 0

    def test_from_slippage_export_uses_regime_lookup(self):
        regime_bps = {"trending": 3.0, "ranging": 10.0, "default": 6.0}
        config = CostModelConfig(
            slippage_model=SlippageModel.EMPIRICAL,
            fees_per_trade_bps=5.0,
            fixed_slippage_bps=3.0,
        )
        model = CostModel.from_slippage_export(regime_bps, config)
        trending_costs = model.compute_costs(1000.0, 1.0, regime="trending")
        ranging_costs = model.compute_costs(1000.0, 1.0, regime="ranging")
        assert ranging_costs.slippage > trending_costs.slippage

    def test_from_slippage_export_fallback_to_default(self):
        regime_bps = {"trending": 5.0, "default": 7.0}
        config = CostModelConfig(
            slippage_model=SlippageModel.EMPIRICAL,
            fees_per_trade_bps=5.0,
            fixed_slippage_bps=3.0,
        )
        model = CostModel.from_slippage_export(regime_bps, config)
        costs = model.compute_costs(1000.0, 1.0, regime="unknown_regime")
        assert costs.slippage > 0


class TestRegimeBpsWritten:
    def test_write_curated_writes_regime_bps(self, tmp_path):
        from trading_assistant.schemas.events import TradeEvent
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            TradeEvent(
                trade_id="t1", bot_id="bot1", pair="BTC/USDT",
                side="LONG",
                entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
                exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
                entry_price=100, exit_price=105, position_size=1,
                pnl=5, pnl_pct=5.0, spread_at_entry=3.5, market_regime="trending",
            ),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, [], tmp_path)
        regime_bps_path = output_dir / "regime_bps.json"
        assert regime_bps_path.exists()
        data = json.loads(regime_bps_path.read_text())
        assert "trending" in data

    def test_regime_bps_empty_when_no_spread_data(self, tmp_path):
        from trading_assistant.schemas.events import TradeEvent
        from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder

        trades = [
            TradeEvent(
                trade_id="t1", bot_id="bot1", pair="BTC/USDT",
                side="LONG",
                entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
                exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
                entry_price=100, exit_price=105, position_size=1,
                pnl=5, pnl_pct=5.0, spread_at_entry=0,
            ),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, [], tmp_path)
        data = json.loads((output_dir / "regime_bps.json").read_text())
        assert data == {}

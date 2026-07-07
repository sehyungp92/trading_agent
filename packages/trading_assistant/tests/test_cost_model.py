# tests/test_cost_model.py
"""Tests for transaction cost model."""
import json
from pathlib import Path

from trading_assistant.schemas.cost_model import CostModelConfig, SlippageModel
from trading_assistant.skills.cost_model import CostModel, TradeCosts


class TestTradeCosts:
    def test_total_is_sum(self):
        tc = TradeCosts(fees=10.0, slippage=5.0)
        assert tc.total == 15.0

    def test_zero_costs(self):
        tc = TradeCosts()
        assert tc.total == 0.0


class TestCostModelFixed:
    def test_round_trip_fees(self):
        cfg = CostModelConfig(fees_per_trade_bps=7.0, slippage_model=SlippageModel.FIXED, fixed_slippage_bps=0.0)
        model = CostModel(cfg)
        # Entry at $40000, 0.1 BTC → notional = $4000
        # Round-trip fees: 4000 * 7/10000 * 2 = $5.60
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1)
        assert abs(costs.fees - 5.60) < 0.01
        assert costs.slippage == 0.0

    def test_fixed_slippage(self):
        cfg = CostModelConfig(fees_per_trade_bps=0.0, slippage_model=SlippageModel.FIXED, fixed_slippage_bps=5.0)
        model = CostModel(cfg)
        # Notional = $4000, round-trip slippage: 4000 * 5/10000 * 2 = $4.00
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1)
        assert costs.fees == 0.0
        assert abs(costs.slippage - 4.00) < 0.01

    def test_combined_costs(self):
        cfg = CostModelConfig(fees_per_trade_bps=7.0, slippage_model=SlippageModel.FIXED, fixed_slippage_bps=5.0)
        model = CostModel(cfg)
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1)
        assert abs(costs.total - 9.60) < 0.01  # 5.60 + 4.00

    def test_zero_position(self):
        cfg = CostModelConfig()
        model = CostModel(cfg)
        costs = model.compute_costs(entry_price=40000.0, position_size=0.0)
        assert costs.total == 0.0


class TestCostModelSpreadProportional:
    def test_spread_slippage(self):
        cfg = CostModelConfig(
            fees_per_trade_bps=0.0,
            slippage_model=SlippageModel.SPREAD_PROPORTIONAL,
            fixed_slippage_bps=0.0,
        )
        model = CostModel(cfg)
        # Spread = 10 bps on $4000 notional, round trip: 4000 * 10/10000 * 2 = $8.00
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1, spread_bps=10.0)
        assert abs(costs.slippage - 8.00) < 0.01

    def test_zero_spread(self):
        cfg = CostModelConfig(
            fees_per_trade_bps=0.0,
            slippage_model=SlippageModel.SPREAD_PROPORTIONAL,
        )
        model = CostModel(cfg)
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1, spread_bps=0.0)
        assert costs.slippage == 0.0


class TestCostModelEmpirical:
    def test_loads_slippage_stats(self, tmp_path: Path):
        stats = {"default": 6.0, "trending_up": 4.0, "volatile": 12.0}
        stats_path = tmp_path / "slippage_stats.json"
        stats_path.write_text(json.dumps(stats))
        cfg = CostModelConfig(
            fees_per_trade_bps=0.0,
            slippage_model=SlippageModel.EMPIRICAL,
            slippage_source=str(stats_path),
        )
        model = CostModel(cfg)
        # Volatile regime: 4000 * 12/10000 * 2 = $9.60
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1, regime="volatile")
        assert abs(costs.slippage - 9.60) < 0.01

    def test_falls_back_to_default(self, tmp_path: Path):
        stats = {"default": 6.0}
        stats_path = tmp_path / "slippage_stats.json"
        stats_path.write_text(json.dumps(stats))
        cfg = CostModelConfig(
            fees_per_trade_bps=0.0,
            slippage_model=SlippageModel.EMPIRICAL,
            slippage_source=str(stats_path),
        )
        model = CostModel(cfg)
        # Unknown regime → default: 4000 * 6/10000 * 2 = $4.80
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1, regime="unknown_regime")
        assert abs(costs.slippage - 4.80) < 0.01

    def test_missing_stats_uses_fixed_fallback(self):
        cfg = CostModelConfig(
            fees_per_trade_bps=0.0,
            slippage_model=SlippageModel.EMPIRICAL,
            slippage_source="nonexistent_path.json",
            fixed_slippage_bps=5.0,
        )
        model = CostModel(cfg)
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1)
        assert abs(costs.slippage - 4.00) < 0.01  # falls back to fixed


class TestCostModelMultiplier:
    def test_multiplier_scales_costs(self):
        cfg = CostModelConfig(fees_per_trade_bps=7.0, slippage_model=SlippageModel.FIXED, fixed_slippage_bps=5.0)
        model = CostModel(cfg)
        base = model.compute_costs(entry_price=40000.0, position_size=0.1, cost_multiplier=1.0)
        scaled = model.compute_costs(entry_price=40000.0, position_size=0.1, cost_multiplier=2.0)
        assert abs(scaled.total - base.total * 2) < 0.01

    def test_zero_multiplier(self):
        cfg = CostModelConfig(fees_per_trade_bps=7.0, slippage_model=SlippageModel.FIXED, fixed_slippage_bps=5.0)
        model = CostModel(cfg)
        costs = model.compute_costs(entry_price=40000.0, position_size=0.1, cost_multiplier=0.0)
        assert costs.total == 0.0

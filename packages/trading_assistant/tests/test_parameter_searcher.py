# tests/test_parameter_searcher.py
"""Tests for ParameterSearcher — autoresearch-style parameter neighborhood search."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock


from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.parameter_search import (
    CandidateResult,
    ParameterSearchReport,
    SearchRouting,
)
from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.simulation_metrics import SimulationMetrics
from trading_assistant.skills.backtest_simulator import BacktestSimulator
from trading_assistant.skills.config_registry import ConfigRegistry
from trading_assistant.skills.cost_model import CostModel
from trading_assistant.schemas.regime_conditional import RegimeParameterAnalysis
from trading_assistant.skills.parameter_searcher import (
    ParameterSearcher,
    _MAX_CANDIDATES,
    _REGIME_SENSITIVITY_THRESHOLD,
)
from tests.factories import make_trade


# ─── Helpers ─────────────────────────────────────────────────────────

def _make_param(
    name: str = "signal_strength_min",
    current: float = 0.5,
    valid_range: tuple[float, float] | None = (0.1, 1.0),
    valid_values: list | None = None,
    value_type: str = "float",
    is_safety_critical: bool = False,
    bot_id: str = "bot1",
) -> ParameterDefinition:
    return ParameterDefinition(
        param_name=name,
        bot_id=bot_id,
        param_type=ParameterType.YAML_FIELD,
        file_path="config/params.yaml",
        yaml_key=f"params.{name}",
        current_value=current,
        valid_range=valid_range,
        valid_values=valid_values,
        value_type=value_type,
        category="signal",
        is_safety_critical=is_safety_critical,
    )


def _make_trades(n: int = 20, pnl_base: float = 10.0) -> list[TradeEvent]:
    trades = []
    for i in range(n):
        pnl = pnl_base if i % 3 != 0 else -pnl_base * 0.5
        trades.append(make_trade(
            trade_id=f"t{i}",
            pair="BTC/USD",
            entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            entry_price=50000.0,
            position_size=0.1,
            pnl=pnl,
            pnl_pct=pnl / 50000.0,
            entry_signal_strength=0.8,
            market_regime="trending",
        ))
    return trades


def _make_searcher(
    simulate_fn=None,
    config_dir: Path | None = None,
) -> ParameterSearcher:
    """Create a ParameterSearcher with mock dependencies."""
    registry = ConfigRegistry(config_dir or Path("/nonexistent"))

    cost_config = MagicMock()
    cost_config.fees_per_trade_bps = 7.0
    cost_config.slippage_model = "FIXED"
    cost_config.fixed_slippage_bps = 5.0
    cost_config.spread_impact = False
    cost_config.reject_if_only_profitable_at_zero_cost = False
    cost_config.cost_sensitivity_test = False
    cost_config.cost_multipliers = [1.0]
    cost_model = CostModel(cost_config)

    simulator = BacktestSimulator(cost_model)
    if simulate_fn:
        simulator.simulate = simulate_fn

    return ParameterSearcher(
        config_registry=registry,
        simulator=simulator,
        cost_model=cost_model,
    )


# ─── Grid Generation Tests ──────────────────────────────────────────

class TestGridGeneration:
    def test_grid_from_valid_range(self):
        """Grid uses linspace within valid_range, capped at MAX_CANDIDATES."""
        searcher = _make_searcher()
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        grid = searcher._build_grid(param, proposed_value=0.7)
        assert len(grid) <= _MAX_CANDIDATES + 2  # +2 for proposed + current
        assert 0.7 in grid or any(abs(v - 0.7) < 0.01 for v in grid)
        assert 0.5 in grid or any(abs(v - 0.5) < 0.01 for v in grid)
        for v in grid:
            assert 0.1 <= v <= 1.0

    def test_grid_from_valid_values(self):
        """Categorical params: test all valid_values."""
        searcher = _make_searcher()
        param = _make_param(
            current="mode_a",
            valid_range=None,
            valid_values=["mode_a", "mode_b", "mode_c"],
            value_type="str",
        )
        grid = searcher._build_grid(param, proposed_value="mode_b")
        assert "mode_a" in grid
        assert "mode_b" in grid
        assert "mode_c" in grid

    def test_proposed_always_in_grid(self):
        """The originally proposed value is always included."""
        searcher = _make_searcher()
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        grid = searcher._build_grid(param, proposed_value=0.85)
        assert any(abs(v - 0.85) < 0.01 for v in grid)

    def test_integer_params_rounded_deduped(self):
        """Integer params are rounded and deduplicated."""
        searcher = _make_searcher()
        param = _make_param(
            name="lookback_period",
            current=20,
            valid_range=(5.0, 50.0),
            value_type="int",
        )
        grid = searcher._build_grid(param, proposed_value=25)
        for v in grid:
            assert isinstance(v, int), f"Expected int, got {type(v)}: {v}"
        assert len(grid) == len(set(grid)), "Duplicates in grid"


# ─── Search + Routing Tests ──────────────────────────────────────────

class TestSearchRouting:
    def test_empty_trades_returns_discard(self):
        """No trades → DISCARD gracefully."""
        searcher = _make_searcher()
        param = _make_param()
        report = searcher.search("s1", "bot1", param, 0.7, [], [])
        assert report.routing == SearchRouting.DISCARD
        assert "No trade data" in report.discard_reason

    def test_all_fail_safety_returns_discard(self):
        """When no candidate passes safety → DISCARD."""

        def bad_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=3,  # below minimum
                sharpe_ratio=-0.5,
                win_count=1,
                loss_count=2,
                net_pnl=-100,
            )

        searcher = _make_searcher(simulate_fn=bad_simulate)
        param = _make_param()
        trades = _make_trades(3)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        assert report.routing == SearchRouting.DISCARD
        assert report.candidates_passing == 0

    def test_strong_candidate_routes_approve(self):
        """Strong candidate → APPROVE routing."""

        def good_simulate(trades, missed, params, cost_multiplier=1.0):
            val = params.get("signal_strength_min", 0.5)
            # Proposed value 0.7 performs much better than baseline 0.5
            sharpe = 2.0 if val >= 0.65 else 1.0
            calmar = 3.0 if val >= 0.65 else 1.5
            pf = 3.0 if val >= 0.65 else 1.5
            wr = 0.7 if val >= 0.65 else 0.5
            return SimulationMetrics(
                total_trades=50,
                win_count=int(50 * wr),
                loss_count=50 - int(50 * wr),
                net_pnl=sharpe * 100,
                sharpe_ratio=sharpe,
                calmar_ratio=calmar,
                profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=good_simulate)
        param = _make_param(current=0.5)
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        assert report.routing == SearchRouting.APPROVE
        assert report.best_value is not None
        assert report.candidates_passing > 0

    def test_marginal_candidate_routes_experiment(self):
        """Marginal improvement → EXPERIMENT routing."""
        def marginal_simulate(trades, missed, params, cost_multiplier=1.0):
            val = params.get("signal_strength_min", 0.5)
            # Marginal improvement: within 5% of baseline
            sharpe = 1.02 if val != 0.5 else 1.0
            calmar = 1.52 if val != 0.5 else 1.5
            pf = 1.52 if val != 0.5 else 1.5
            return SimulationMetrics(
                total_trades=50,
                win_count=25,
                loss_count=25,
                net_pnl=100,
                sharpe_ratio=sharpe,
                calmar_ratio=calmar,
                profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=marginal_simulate)
        param = _make_param(current=0.5)
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        assert report.routing == SearchRouting.EXPERIMENT

    def test_cost_sensitivity_blocks_fragile(self):
        """Candidate that collapses at 1.5x cost → fails safety."""
        call_count = [0]

        def fragile_simulate(trades, missed, params, cost_multiplier=1.0):
            call_count[0] += 1
            # Good at normal cost, terrible at 1.5x
            if cost_multiplier > 1.0:
                return SimulationMetrics(
                    total_trades=50,
                    win_count=20,
                    loss_count=30,
                    net_pnl=-50,
                    sharpe_ratio=-0.5,
                    calmar_ratio=-0.3,
                    profit_factor=0.5,
                )
            return SimulationMetrics(
                total_trades=50,
                win_count=30,
                loss_count=20,
                net_pnl=200,
                sharpe_ratio=2.0,
                calmar_ratio=2.5,
                profit_factor=2.0,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=fragile_simulate)
        param = _make_param(current=0.5)
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        # All candidates fail cost sensitivity → DISCARD
        assert report.routing == SearchRouting.DISCARD

    def test_safety_critical_uses_higher_threshold(self):
        """Safety-critical params need 10% improvement, not 5%."""
        def simulate_6pct(trades, missed, params, cost_multiplier=1.0):
            val = params.get("stop_loss_atr", 2.0)
            # 6% improvement: passes normal threshold but not safety-critical
            sharpe = 1.06 if val != 2.0 else 1.0
            calmar = 1.59 if val != 2.0 else 1.5
            pf = 1.908 if val != 2.0 else 1.8
            return SimulationMetrics(
                total_trades=50,
                win_count=30,
                loss_count=20,
                net_pnl=100 * sharpe,
                sharpe_ratio=sharpe,
                calmar_ratio=calmar,
                profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=simulate_6pct)
        param = _make_param(
            name="stop_loss_atr",
            current=2.0,
            valid_range=(0.5, 5.0),
            is_safety_critical=True,
        )
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 2.5, trades, [])
        # 6% improvement insufficient for safety-critical (needs 10%)
        assert report.routing != SearchRouting.APPROVE

    def test_robustness_instability_blocks(self):
        """Unstable neighborhood → low robustness score → blocks APPROVE."""
        call_idx = [0]

        def unstable_simulate(trades, missed, params, cost_multiplier=1.0):
            call_idx[0] += 1
            val = params.get("signal_strength_min", 0.5)
            # Spiky: best value is great, neighbors collapse
            if 0.69 < val < 0.71:
                sharpe = 3.0
                calmar = 4.0
                pf = 3.0
            else:
                sharpe = 0.1  # neighbors collapse
                calmar = 0.05
                pf = 0.8
            return SimulationMetrics(
                total_trades=50,
                win_count=30,
                loss_count=20,
                net_pnl=100 * sharpe,
                sharpe_ratio=sharpe,
                calmar_ratio=calmar,
                profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10},
            )

        searcher = _make_searcher(simulate_fn=unstable_simulate)
        param = _make_param(current=0.5)
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        # Robustness score too low for APPROVE
        assert report.routing != SearchRouting.APPROVE

    def test_baseline_at_current_value(self):
        """Baseline is computed at current_value, not proposed."""
        captured_params = []

        def capture_simulate(trades, missed, params, cost_multiplier=1.0):
            captured_params.append(dict(params))
            return SimulationMetrics(
                total_trades=50,
                win_count=25,
                loss_count=25,
                net_pnl=100,
                sharpe_ratio=1.0,
                calmar_ratio=1.5,
                profit_factor=1.5,
                pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=capture_simulate)
        param = _make_param(current=0.5)
        trades = _make_trades(50)
        searcher.search("s1", "bot1", param, 0.7, trades, [])
        # First simulate call should be baseline at current_value=0.5
        assert captured_params[0]["signal_strength_min"] == 0.5

    def test_exploration_summary_populated(self):
        """Report includes human-readable exploration summary."""
        def ok_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=50,
                win_count=25,
                loss_count=25,
                net_pnl=100,
                sharpe_ratio=1.0,
                calmar_ratio=1.5,
                profit_factor=1.5,
                pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=ok_simulate)
        param = _make_param()
        trades = _make_trades(50)
        report = searcher.search("s1", "bot1", param, 0.7, trades, [])
        assert report.exploration_summary
        assert "signal_strength_min" in report.exploration_summary
        assert "Routing:" in report.exploration_summary


# ─── Schema Tests ────────────────────────────────────────────────────

class TestSchemas:
    def test_parameter_search_report_defaults(self):
        report = ParameterSearchReport(
            suggestion_id="s1",
            bot_id="bot1",
            param_name="p1",
            original_proposed=0.7,
            current_value=0.5,
        )
        assert report.routing == SearchRouting.DISCARD
        assert report.searched_at is not None

    def test_candidate_result_defaults(self):
        result = CandidateResult(value=0.7)
        assert result.passes_safety is False
        assert result.composite_score == 0.0

    def test_search_routing_values(self):
        assert SearchRouting.APPROVE.value == "approve"
        assert SearchRouting.EXPERIMENT.value == "experiment"
        assert SearchRouting.DISCARD.value == "discard"


# ─── Composite Score Tests ───────────────────────────────────────────

class TestCompositeScore:
    def test_identical_metrics_score_near_expected(self):
        """Same metrics as baseline → composite ≈ 0.889 (all ratios = 1.0, no dd increase)."""
        metrics = SimulationMetrics(
            total_trades=50,
            win_count=25,
            loss_count=25,
            net_pnl=100.0,
            calmar_ratio=1.5,
            profit_factor=2.0,
            max_drawdown_pct=5.0,
            avg_win=20.0,
            avg_loss=10.0,
        )
        score = ParameterSearcher._composite_score(metrics, metrics)
        # 0.333*1 + 0.222*1 + 0.167*1 + 0.167*1 - 0.111*0 = 0.889
        assert abs(score - 0.889) < 0.01

    def test_better_calmar_increases_score(self):
        baseline = SimulationMetrics(
            total_trades=50, win_count=25, loss_count=25,
            net_pnl=100.0, calmar_ratio=1.5, profit_factor=2.0,
            max_drawdown_pct=5.0, avg_win=20.0, avg_loss=10.0,
        )
        better = SimulationMetrics(
            total_trades=50, win_count=25, loss_count=25,
            net_pnl=100.0, calmar_ratio=3.0, profit_factor=2.0,
            max_drawdown_pct=5.0, avg_win=20.0, avg_loss=10.0,
        )
        score = ParameterSearcher._composite_score(better, baseline)
        assert score > 0.889

    def test_negative_baseline_net_pnl_uses_neutral(self):
        """When baseline net_pnl <= 0, er_imp should be neutral (1.0)."""
        baseline = SimulationMetrics(
            total_trades=50, win_count=25, loss_count=25,
            net_pnl=-50.0, calmar_ratio=1.5, profit_factor=2.0,
            max_drawdown_pct=5.0, avg_win=20.0, avg_loss=10.0,
        )
        candidate = SimulationMetrics(
            total_trades=50, win_count=25, loss_count=25,
            net_pnl=100.0, calmar_ratio=1.5, profit_factor=2.0,
            max_drawdown_pct=5.0, avg_win=20.0, avg_loss=10.0,
        )
        score = ParameterSearcher._composite_score(candidate, baseline)
        # er_imp = 1.0 (neutral due to negative baseline), others = 1.0
        assert abs(score - 0.889) < 0.01


# ─── Regime-Conditional Search Tests ─────────────────────────────────

def _make_regime_trades(
    n_per_regime: int = 20,
    regimes: list[str] | None = None,
    param_name: str = "signal_strength_min",
    param_values: dict[str, float] | None = None,
    pnl_by_regime: dict[str, float] | None = None,
) -> list[TradeEvent]:
    """Create trades with strategy_params_at_entry for regime analysis.

    param_values maps regime → param value used in that regime's trades.
    pnl_by_regime maps regime → avg pnl for that regime's trades.
    """
    if regimes is None:
        regimes = ["trending", "ranging", "volatile"]
    if param_values is None:
        param_values = {r: 0.5 for r in regimes}
    if pnl_by_regime is None:
        pnl_by_regime = {r: 10.0 for r in regimes}

    trades: list[TradeEvent] = []
    for regime in regimes:
        pval = param_values.get(regime, 0.5)
        avg_pnl = pnl_by_regime.get(regime, 10.0)
        for i in range(n_per_regime):
            pnl = avg_pnl if i % 3 != 0 else -abs(avg_pnl) * 0.5
            trades.append(make_trade(
                trade_id=f"t_{regime}_{i}",
                pair="BTC/USD",
                entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                exit_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                entry_price=50000.0,
                position_size=0.1,
                pnl=pnl,
                pnl_pct=pnl / 50000.0,
                entry_signal_strength=0.8,
                market_regime=regime,
                strategy_params_at_entry={param_name: pval},
            ))
    return trades


class TestSearchWithRegimeAnalysis:
    """Tests for search_with_regime_analysis() method."""

    def test_regime_analysis_attached(self):
        """Report should have regime_analysis as a RegimeParameterAnalysis."""
        def ok_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=50, win_count=25, loss_count=25,
                net_pnl=100, sharpe_ratio=1.0, calmar_ratio=1.5,
                profit_factor=1.5, pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=ok_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        trades = _make_regime_trades(n_per_regime=20, param_name="signal_strength_min")
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert report.regime_analysis is not None
        assert isinstance(report.regime_analysis, RegimeParameterAnalysis)
        assert report.regime_analysis.param_name == "signal_strength_min"

    def test_low_sensitivity_no_enrichment(self):
        """When sensitivity <= threshold, summary is not enriched with regime info."""
        def ok_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=50, win_count=25, loss_count=25,
                net_pnl=100, sharpe_ratio=1.0, calmar_ratio=1.5,
                profit_factor=1.5, pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=ok_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # All regimes use same param value → sensitivity = 0
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.5, "ranging": 0.5, "volatile": 0.5},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert "Regime sensitivity:" not in report.exploration_summary

    def test_high_sensitivity_enriches_summary(self):
        """When sensitivity > threshold, regime info appended to summary."""
        def ok_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=50, win_count=25, loss_count=25,
                net_pnl=100, sharpe_ratio=1.0, calmar_ratio=1.5,
                profit_factor=1.5, pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=ok_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # Values must align with the regime analyzer's grid (11 points from 0.1 to 1.0,
        # step=0.09): [0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0]
        # stdev([0.1, 1.0, 0.55]) / 0.9 = 0.5 >> 0.3
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.1, "ranging": 1.0, "volatile": 0.55},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert report.regime_analysis is not None
        assert report.regime_analysis.regime_sensitivity > _REGIME_SENSITIVITY_THRESHOLD
        assert "Regime sensitivity:" in report.exploration_summary

    def test_discard_override_to_experiment(self):
        """Flat DISCARD + high sensitivity + positive regime → EXPERIMENT."""
        def bad_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=3, sharpe_ratio=-0.5,
                win_count=1, loss_count=2, net_pnl=-100,
            )

        searcher = _make_searcher(simulate_fn=bad_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # Values must align with the regime analyzer's grid (step=0.09):
        # [0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.0]
        # stdev([0.1, 1.0, 0.55]) / 0.9 = 0.5 >> 0.3
        # Trending: positive avg_pnl, 20 trades, optimal=0.1 != current 0.5
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.1, "ranging": 1.0, "volatile": 0.55},
            pnl_by_regime={"trending": 20.0, "ranging": -5.0, "volatile": 0.0},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        # Flat search returns DISCARD (all candidates fail safety with total_trades=3),
        # regime analysis overrides: trending has avg_pnl>0, count>=15, optimal!=current
        assert report.regime_analysis is not None
        assert report.regime_analysis.regime_sensitivity > _REGIME_SENSITIVITY_THRESHOLD
        assert report.routing == SearchRouting.EXPERIMENT
        assert "DISCARD→EXPERIMENT" in report.exploration_summary

    def test_discard_no_override_no_strong_regime(self):
        """DISCARD stays when no regime has positive avg_pnl."""
        def bad_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=3, sharpe_ratio=-0.5,
                win_count=1, loss_count=2, net_pnl=-100,
            )

        searcher = _make_searcher(simulate_fn=bad_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # Grid-aligned values; all regimes have negative avg_pnl → no rescue
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.1, "ranging": 1.0, "volatile": 0.55},
            pnl_by_regime={"trending": -10.0, "ranging": -5.0, "volatile": -8.0},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert report.routing == SearchRouting.DISCARD

    def test_approve_not_downgraded(self):
        """APPROVE routing is never downgraded by regime analysis."""
        def good_simulate(trades, missed, params, cost_multiplier=1.0):
            val = params.get("signal_strength_min", 0.5)
            sharpe = 2.0 if val >= 0.65 else 1.0
            calmar = 3.0 if val >= 0.65 else 1.5
            pf = 3.0 if val >= 0.65 else 1.5
            wr = 0.7 if val >= 0.65 else 0.5
            return SimulationMetrics(
                total_trades=50, win_count=int(50 * wr),
                loss_count=50 - int(50 * wr),
                net_pnl=sharpe * 100, sharpe_ratio=sharpe,
                calmar_ratio=calmar, profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=good_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # Grid-aligned values with high sensitivity
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.1, "ranging": 1.0, "volatile": 0.55},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert report.routing == SearchRouting.APPROVE

    def test_experiment_not_changed(self):
        """EXPERIMENT routing stays EXPERIMENT (not overridden)."""
        def marginal_simulate(trades, missed, params, cost_multiplier=1.0):
            val = params.get("signal_strength_min", 0.5)
            sharpe = 1.02 if val != 0.5 else 1.0
            calmar = 1.52 if val != 0.5 else 1.5
            pf = 1.52 if val != 0.5 else 1.5
            return SimulationMetrics(
                total_trades=50, win_count=25, loss_count=25,
                net_pnl=100, sharpe_ratio=sharpe,
                calmar_ratio=calmar, profit_factor=pf,
                pnl_by_regime={"trending": 80, "ranging": 20, "volatile": 10, "quiet": 5},
            )

        searcher = _make_searcher(simulate_fn=marginal_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        # Grid-aligned values with high sensitivity
        trades = _make_regime_trades(
            n_per_regime=20,
            param_values={"trending": 0.1, "ranging": 1.0, "volatile": 0.55},
        )
        report = searcher.search_with_regime_analysis(
            "s1", "bot1", param, 0.7, trades, [],
        )
        assert report.routing == SearchRouting.EXPERIMENT

    def test_empty_trades_no_crash(self):
        """Empty trade list → DISCARD with no regime analysis crash."""
        searcher = _make_searcher()
        param = _make_param()
        report = searcher.search_with_regime_analysis("s1", "bot1", param, 0.7, [], [])
        assert report.routing == SearchRouting.DISCARD
        assert report.regime_analysis is not None
        assert report.regime_analysis.regime_sensitivity == 0.0

    def test_regime_analyzer_crash_returns_flat_result(self):
        """If RegimeParameterAnalyzer.analyze() throws, flat result is still returned."""
        def ok_simulate(trades, missed, params, cost_multiplier=1.0):
            return SimulationMetrics(
                total_trades=50, win_count=25, loss_count=25,
                net_pnl=100, sharpe_ratio=1.0, calmar_ratio=1.5,
                profit_factor=1.5, pnl_by_regime={"trending": 50},
            )

        searcher = _make_searcher(simulate_fn=ok_simulate)
        param = _make_param(current=0.5, valid_range=(0.1, 1.0))
        trades = _make_regime_trades(n_per_regime=20)

        # Monkey-patch the analyzer to throw
        import trading_assistant.skills.parameter_searcher as ps_mod
        original_cls = ps_mod.RegimeParameterAnalyzer

        class CrashingAnalyzer:
            def analyze(self, *args, **kwargs):
                raise RuntimeError("boom")

        ps_mod.RegimeParameterAnalyzer = CrashingAnalyzer
        try:
            report = searcher.search_with_regime_analysis(
                "s1", "bot1", param, 0.7, trades, [],
            )
            # Flat search result preserved, regime_analysis absent
            assert report.regime_analysis is None
            assert report.routing in (
                SearchRouting.APPROVE, SearchRouting.EXPERIMENT, SearchRouting.DISCARD,
            )
            assert report.exploration_summary  # flat summary still present
        finally:
            ps_mod.RegimeParameterAnalyzer = original_cls

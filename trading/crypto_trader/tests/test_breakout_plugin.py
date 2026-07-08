"""Tests for BreakoutPlugin round 3 redesign — signal-first phase ordering."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.breakout_plugin import (
    HARD_REJECTS,
    PHASE_DIAGNOSTIC_MODULES,
    PHASE_GATE_CRITERIA,
    PHASE_NAMES,
    PHASE_SCORING_EMPHASIS,
    SCORING_CEILINGS,
    SCORING_WEIGHTS,
    BreakoutPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
)
from crypto_trader.optimize.types import (
    Experiment,
    GateCriterion,
    GreedyResult,
    GreedyRound,
    ScoredCandidate,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig


def _make_plugin() -> BreakoutPlugin:
    bc = BacktestConfig(
        start_date=date(2026, 2, 25),
        end_date=date(2026, 4, 18),
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10000.0,
    )
    return BreakoutPlugin(
        backtest_config=bc,
        base_config=BreakoutConfig(),
        data_dir=Path("data"),
    )


class TestPhaseNames:
    def test_signal_first_ordering(self):
        """Phase 1 is Signal & Direction (signal-first for round 3)."""
        assert PHASE_NAMES[1] == "Signal & Direction"
        assert PHASE_NAMES[2] == "Exit & Capture"
        assert PHASE_NAMES[3] == "Trail & Stop"
        assert PHASE_NAMES[4] == "Zone & Profile"
        assert PHASE_NAMES[5] == "Risk & Sizing"
        assert PHASE_NAMES[6] == "Finetune"


class TestScoringWeights:
    def test_returns_dominant_weights(self):
        """Round 2 uses returns-dominant weights for profitable baseline."""
        assert SCORING_WEIGHTS["returns"] == 0.30
        assert SCORING_WEIGHTS["coverage"] == 0.20
        assert SCORING_WEIGHTS["edge"] == 0.20
        assert SCORING_WEIGHTS["calmar"] == 0.15
        assert SCORING_WEIGHTS["capture"] == 0.15

    def test_weights_sum_to_1(self):
        """Scoring weights should sum to 1.0."""
        total = sum(SCORING_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_ceilings_for_profitable_baseline(self):
        """Scoring ceilings calibrated for profitable-baseline regime."""
        assert SCORING_CEILINGS["returns"] == 30.0
        assert SCORING_CEILINGS["edge"] == 10.0
        assert SCORING_CEILINGS["coverage"] == 15.0

    def test_hard_rejects_tightened(self):
        """Hard rejects tightened for profitable baseline."""
        assert HARD_REJECTS["total_trades"] == (">=", 5)
        assert HARD_REJECTS["profit_factor"] == (">=", 0.8)
        assert HARD_REJECTS["max_drawdown_pct"] == ("<=", 40.0)

    def test_phase_scoring_emphasis_all_phases(self):
        """All 6 phases have scoring emphasis that sums to 1.0."""
        for phase in range(1, 7):
            weights = PHASE_SCORING_EMPHASIS[phase]
            assert sum(weights.values()) == pytest.approx(1.0), \
                f"Phase {phase} weights don't sum to 1.0"


class TestGateCriteria:
    def test_phase1_signal_gates(self):
        """Phase 1 (Signal) gate: trades >= 5."""
        gates = PHASE_GATE_CRITERIA[1]
        assert len(gates) == 1
        assert gates[0].metric == "total_trades"
        assert gates[0].threshold == 5

    def test_phase3_trail_gates(self):
        """Phase 3 (Trail) gates require trades >= 5 and PF >= 0.8."""
        gates = PHASE_GATE_CRITERIA[3]
        metrics = {g.metric: g.threshold for g in gates}
        assert metrics["total_trades"] == 5
        assert metrics["profit_factor"] == 0.8

    def test_phase4_zone_gates(self):
        """Phase 4 (Zone) gates require trades >= 5 and PF >= 0.7."""
        gates = PHASE_GATE_CRITERIA[4]
        metrics = {g.metric: g.threshold for g in gates}
        assert metrics["total_trades"] == 5
        assert metrics["profit_factor"] == 0.7

    def test_phase5_dd_gate(self):
        """Phase 5 gates include drawdown <= 35% and PF >= 0.8."""
        gates = PHASE_GATE_CRITERIA[5]
        dd_gate = [g for g in gates if g.metric == "max_drawdown_pct"]
        assert len(dd_gate) == 1
        assert dd_gate[0].threshold == 35.0
        assert dd_gate[0].operator == "<="
        pf_gate = [g for g in gates if g.metric == "profit_factor"]
        assert len(pf_gate) == 1
        assert pf_gate[0].threshold == 0.8

    def test_phase6_gates(self):
        """Phase 6 (Finetune) gates: trades >= 5, PF >= 0.7."""
        gates = PHASE_GATE_CRITERIA[6]
        metrics = {g.metric: g.threshold for g in gates}
        assert metrics["total_trades"] == 5
        assert metrics["profit_factor"] == 0.7


class TestDiagnosticModules:
    def test_phase1_signal_modules(self):
        """Phase 1 (Signal & Direction) uses D4 (Signal), D5 (Env), D6 (Overview)."""
        modules = PHASE_DIAGNOSTIC_MODULES[1]
        assert "D4" in modules
        assert "D5" in modules
        assert "D6" in modules

    def test_phase2_exit_modules(self):
        """Phase 2 (Exit & Capture) uses D2 (Exit), D3 (Risk), D6 (Overview)."""
        modules = PHASE_DIAGNOSTIC_MODULES[2]
        assert "D2" in modules
        assert "D3" in modules
        assert "D6" in modules

    def test_phase3_trail_modules(self):
        """Phase 3 (Trail & Stop) uses D1 (Trail), D6 (Overview)."""
        modules = PHASE_DIAGNOSTIC_MODULES[3]
        assert "D1" in modules
        assert "D6" in modules

    def test_phase6_all_modules(self):
        """Phase 6 (Finetune) uses all diagnostic modules."""
        modules = PHASE_DIAGNOSTIC_MODULES[6]
        assert set(modules) == {"D1", "D2", "D3", "D4", "D5", "D6"}


class TestPhase1Candidates:
    def test_has_direction_filters(self):
        """Phase 1 has direction filter experiments (highest priority)."""
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert "eth_both" in names
        assert "eth_disabled" in names
        assert "sol_long_only" in names

    def test_has_confluence_quality(self):
        """Phase 1 has confluence quality experiments."""
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert any("conf" in n for n in names)

    def test_has_model2_reenable(self):
        """Phase 1 tests re-enabling Model 2 (disabled by default)."""
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert "model2_on" in names
        m2_on = [c for c in candidates if c.name == "model2_on"][0]
        assert m2_on.mutations["confirmation.enable_model2"] is True

    def test_has_model1_confirmation_gates(self):
        """Phase 1 has Model 1 confirmation quality experiments."""
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert "no_vol_gate" in names
        assert "vol_mult_13" in names
        assert "dir_close_on" in names

    def test_experiment_count(self):
        """Phase 1 has 23 experiments."""
        candidates = _phase1_candidates()
        assert len(candidates) == 23


class TestPhase2Candidates:
    def test_has_tp_experiments(self):
        """Phase 2 has TP1 and TP2 experiments."""
        candidates = _phase2_candidates()
        names = [c.name for c in candidates]
        assert any("tp1_r" in n for n in names)
        assert any("tp2_r" in n for n in names)

    def test_has_quick_exit_off(self):
        """Phase 2 has quick exit off toggle (now on by default)."""
        candidates = _phase2_candidates()
        names = [c.name for c in candidates]
        assert "quick_exit_off" in names

    def test_tp1_targets_around_new_default(self):
        """TP1 values centered around new 0.8R default (0.5-1.2)."""
        candidates = _phase2_candidates()
        tp1_exps = [c for c in candidates if c.name.startswith("tp1_r_")]
        tp1_values = [list(c.mutations.values())[0] for c in tp1_exps]
        assert 0.5 in tp1_values
        assert 1.2 in tp1_values

    def test_tp2_above_tp1(self):
        """TP2 values are above 1.0 (extension targets above TP1)."""
        candidates = _phase2_candidates()
        tp2_exps = [c for c in candidates if c.name.startswith("tp2_r_")]
        tp2_values = [list(c.mutations.values())[0] for c in tp2_exps]
        assert all(v >= 1.5 for v in tp2_values)

    def test_has_invalidation_min_bars(self):
        """Phase 2 has invalidation_min_bars experiments."""
        candidates = _phase2_candidates()
        names = [c.name for c in candidates]
        assert "invalidation_minbars_1" in names
        assert "invalidation_minbars_5" in names

    def test_experiment_count(self):
        """Phase 2 has 36 experiments."""
        candidates = _phase2_candidates()
        assert len(candidates) == 36


class TestPhase3Candidates:
    def test_has_trail_activation(self):
        """Phase 3 has trail activation R experiments."""
        candidates = _phase3_candidates()
        names = [c.name for c in candidates]
        assert any("trail_act_r" in n for n in names)

    def test_has_stop_params(self):
        """Phase 3 has stop ATR experiments."""
        candidates = _phase3_candidates()
        names = [c.name for c in candidates]
        assert any("stop_atr" in n for n in names)

    def test_experiment_count(self):
        """Phase 3 has 19 experiments."""
        candidates = _phase3_candidates()
        assert len(candidates) == 19


class TestPhase4Candidates:
    def test_has_zone_params(self):
        """Phase 4 has zone quality experiments."""
        candidates = _phase4_candidates()
        names = [c.name for c in candidates]
        assert any("min_bars_zone" in n for n in names)
        assert any("lookback" in n for n in names)

    def test_experiment_count(self):
        """Phase 4 has 17 experiments."""
        candidates = _phase4_candidates()
        assert len(candidates) == 17


class TestPhase5Candidates:
    def test_has_risk_and_limits(self):
        """Phase 5 has risk sizing and limit experiments."""
        candidates = _phase5_candidates()
        names = [c.name for c in candidates]
        assert any("risk_b" in n for n in names)
        assert any("consec_loss" in n for n in names)

    def test_experiment_count(self):
        """Phase 5 has 20 experiments."""
        candidates = _phase5_candidates()
        assert len(candidates) == 20


class TestDiagnosticGapFn:
    def test_no_tp_hits_detected(self):
        """Phase 2 (Exit) detects when no TP exits occur."""
        plugin = _make_plugin()

        mock_result = MagicMock()
        mock_result.trades = [MagicMock()]  # Non-empty trades list
        plugin._last_result = mock_result

        mock_insights = MagicMock()
        mock_insights.exit_attribution = {
            "protective_stop": {"n": 5, "avg_r": -0.5},
        }
        mock_insights.mfe_capture = {
            "avg_capture_pct": -0.40,
            "avg_giveback_pct": 1.40,
            "avg_mfe_r": 0.3,
        }

        with patch(
            "crypto_trader.backtest.diagnostics.extract_diagnostic_insights",
            return_value=mock_insights,
        ):
            gaps = plugin._diagnostic_gap_fn(2, {"total_trades": 5})

        # Should flag no TP hits AND negative capture
        assert any("No TP hits" in g for g in gaps)
        assert any("capture" in g.lower() for g in gaps)

    def test_high_giveback_detected_phase3(self):
        """Phase 3 (Trail) detects high giveback."""
        plugin = _make_plugin()

        mock_result = MagicMock()
        mock_result.trades = [MagicMock()] * 5
        plugin._last_result = mock_result

        mock_insights = MagicMock()
        mock_insights.exit_attribution = {
            "protective_stop": {"n": 5, "avg_r": -0.3},
        }
        mock_insights.mfe_capture = {
            "avg_giveback_pct": 0.70,
            "avg_capture_pct": 0.30,
            "avg_mfe_r": 0.8,
        }
        mock_insights.per_confirmation = {}

        with patch(
            "crypto_trader.backtest.diagnostics.extract_diagnostic_insights",
            return_value=mock_insights,
        ):
            gaps = plugin._diagnostic_gap_fn(3, {"total_trades": 5})

        # Should flag high giveback or high stop-out rate
        trail_gaps = [g for g in gaps if "D1" in g]
        assert len(trail_gaps) > 0

    def test_phase1_direction_gaps(self):
        """Phase 1 (Signal) detects value-destroying direction segments."""
        plugin = _make_plugin()

        mock_result = MagicMock()
        mock_result.trades = [MagicMock()] * 6
        plugin._last_result = mock_result

        mock_insights = MagicMock()
        mock_insights.direction = {
            "short": {"n": 6, "wr": 10, "avg_r": -0.35},
            "long": {"n": 3, "wr": 67, "avg_r": 0.5},
        }
        mock_insights.per_asset = {
            "ETH": {"n": 4, "wr": 0, "avg_r": -0.3},
        }

        with patch(
            "crypto_trader.backtest.diagnostics.extract_diagnostic_insights",
            return_value=mock_insights,
        ):
            gaps = plugin._diagnostic_gap_fn(1, {"total_trades": 9})

        assert any("short" in g.lower() or "direction" in g.lower() for g in gaps)


class TestSuggestExperimentsFn:
    def test_phase4_low_trades_suggestion(self):
        """Phase 4 suggests room relaxation when trade count is low."""
        plugin = _make_plugin()
        suggestions = plugin._suggest_experiments_fn(
            4, {"total_trades": 3}, [], None,
        )
        assert len(suggestions) > 0
        names = [s.name for s in suggestions]
        assert "aggressive_room" in names

    def test_phase2_no_tp_suggestion(self):
        """Phase 2 suggests extreme low TP when no TP hits."""
        plugin = _make_plugin()

        mock_result = MagicMock()
        mock_result.trades = [MagicMock()]
        plugin._last_result = mock_result

        mock_insights = MagicMock()
        mock_insights.exit_attribution = {
            "protective_stop": {"n": 5, "avg_r": -0.5},
        }
        mock_insights.mfe_capture = {"avg_capture_pct": 0.20}

        with patch(
            "crypto_trader.backtest.diagnostics.extract_diagnostic_insights",
            return_value=mock_insights,
        ):
            suggestions = plugin._suggest_experiments_fn(
                2, {"total_trades": 5}, [], None,
            )

        names = [s.name for s in suggestions]
        assert "extreme_low_tp" in names


class TestPhaseSpec:
    def test_phase_spec_structure(self):
        """get_phase_spec returns correctly structured PhaseSpec."""
        plugin = _make_plugin()
        spec = plugin.get_phase_spec(1, None)

        assert spec.phase_num == 1
        assert spec.name == "Signal & Direction"
        assert len(spec.candidates) == 23
        # Phase-specific scoring emphasis
        assert spec.scoring_weights == dict(PHASE_SCORING_EMPHASIS[1])
        assert spec.hard_rejects == dict(HARD_REJECTS)
        assert spec.min_delta == 0.005
        assert spec.gate_criteria is not None

    def test_phases_use_phase_specific_weights(self):
        """Each phase uses its PHASE_SCORING_EMPHASIS weights."""
        plugin = _make_plugin()
        for phase in range(1, 7):
            state = MagicMock()
            state.cumulative_mutations = {}
            spec = plugin.get_phase_spec(phase, state)
            expected = dict(PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS))
            assert spec.scoring_weights == expected, \
                f"Phase {phase} has wrong weights"


class TestTerminalMarkScoring:
    def test_plugin_no_longer_has_backtest_end_filter_attribute(self):
        plugin = _make_plugin()
        assert not hasattr(plugin, "_exclude_exit_reasons")

    def test_compute_final_metrics_keeps_terminal_mark_fields(self):
        plugin = _make_plugin()

        mock_result = MagicMock()
        mock_result.trades = []
        mock_result.terminal_marks = [MagicMock()]
        mock_result.metrics = MagicMock()

        metrics_dict = {
            "total_trades": 0.0,
            "terminal_mark_count": 1.0,
            "terminal_mark_pnl_net": 25.0,
            "realized_pnl_net": 0.0,
            "net_return_pct": 0.25,
            "max_drawdown_pct": 1.0,
            "sharpe_ratio": 3.0,
        }

        with patch.object(plugin, "_get_store", return_value=MagicMock()), \
             patch("crypto_trader.optimize.breakout_plugin.run", return_value=mock_result), \
             patch("crypto_trader.optimize.breakout_plugin.metrics_to_dict", return_value=metrics_dict):
            metrics = plugin.compute_final_metrics({})

        assert metrics["total_trades"] == 0.0
        assert metrics["terminal_mark_count"] == 1.0
        assert metrics["terminal_mark_pnl_net"] == 25.0
        assert metrics["sharpe_ratio"] == 3.0

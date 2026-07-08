"""Tests for adaptive optimization intelligence — Steps 1-5 of the plan.

Covers:
- extract_diagnostic_insights (structured data extraction)
- Policy callbacks (_diagnostic_gap_fn, _suggest_experiments_fn, _decide_action_fn, _redesign_scoring_weights_fn)
- Enhanced evaluation (build_evaluation_report with insights, format_dimension_text)
- Enhanced diagnostics (run_phase_diagnostics, run_enhanced_diagnostics with _last_result)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_trader.backtest.diagnostics import (
    DiagnosticInsights,
    extract_diagnostic_insights,
)
from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.optimize.evaluation import (
    build_evaluation_report,
    format_dimension_text,
)
from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.momentum_plugin import (
    MomentumPlugin,
    SCORING_WEIGHTS,
)
from crypto_trader.optimize.types import (
    Experiment,
    GateCriterion,
    GateResult,
    GreedyResult,
    GreedyRound,
    PhaseDecision,
    ScoredCandidate,
)
from crypto_trader.strategy.momentum.config import MomentumConfig


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_trade(
    symbol: str = "BTC",
    direction: Side = Side.LONG,
    pnl: float = 10.0,
    r_multiple: float = 0.5,
    bars_held: int = 3,
    setup_grade: SetupGrade = SetupGrade.B,
    exit_reason: str = "protective_stop",
    confirmation_type: str = "inside_bar_break",
    confluences: list[str] | None = None,
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    entry_time: datetime | None = None,
    exit_time: datetime | None = None,
    mae_r: float | None = -0.3,
    mfe_r: float | None = 0.8,
) -> Trade:
    if entry_time is None:
        entry_time = datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
    if exit_time is None:
        exit_time = entry_time + timedelta(hours=bars_held * 0.25)
    return Trade(
        trade_id=f"t_{symbol}_{pnl}",
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=0.01,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl=pnl,
        r_multiple=r_multiple,
        commission=0.5,
        bars_held=bars_held,
        setup_grade=setup_grade,
        exit_reason=exit_reason,
        confluences_used=confluences or [],
        confirmation_type=confirmation_type,
        entry_method="close",
        funding_paid=0.0,
        mae_r=mae_r,
        mfe_r=mfe_r,
    )


def _make_greedy_result(
    accepted_count: int = 0,
    final_score: float = 0.5,
    base_score: float = 0.45,
    accepted_names: list[str] | None = None,
) -> GreedyResult:
    accepted = []
    if accepted_names:
        for name in accepted_names:
            accepted.append(ScoredCandidate(
                experiment=Experiment(name, {f"param.{name}": 1.0}),
                score=final_score,
                metrics={},
            ))
    return GreedyResult(
        accepted_experiments=accepted,
        rejected_experiments=[],
        final_mutations={},
        final_score=final_score,
        base_score=base_score,
        accepted_count=accepted_count or len(accepted),
        rounds=[GreedyRound(1, 5, "test", final_score, 1.0, True)],
    )


def _make_gate_result(passed: bool = False) -> GateResult:
    return GateResult(
        passed=passed,
        criteria_results=[],
        failure_reasons=[] if passed else ["test failure"],
    )


def _make_phase_state(
    scoring_retries: dict[int, int] | None = None,
    diagnostic_retries: dict[int, int] | None = None,
    cumulative_mutations: dict | None = None,
    phase_results: dict | None = None,
):
    """Create a real PhaseState with specified retry counters."""
    from crypto_trader.optimize.phase_state import PhaseState
    state = PhaseState()
    if scoring_retries:
        state.scoring_retries = scoring_retries
    if diagnostic_retries:
        state.diagnostic_retries = diagnostic_retries
    if cumulative_mutations:
        state.cumulative_mutations = cumulative_mutations
    if phase_results:
        state.phase_results = phase_results
    return state


# ══════════════════════════════════════════════════════════════════════
# 6a. extract_diagnostic_insights tests
# ══════════════════════════════════════════════════════════════════════


class TestExtractDiagnosticInsights:
    """Tests for extract_diagnostic_insights."""

    def test_empty_trade_list(self):
        """Empty trade list returns zero/empty fields, no crash."""
        insights = extract_diagnostic_insights([])
        assert insights.n_trades == 0
        assert insights.win_rate == 0.0
        assert insights.mean_r == 0.0
        assert insights.profit_factor == 0.0
        assert insights.per_confirmation == {}
        assert insights.per_asset == {}
        assert insights.exit_attribution == {}
        assert insights.direction == {}
        assert insights.confluence == {}
        assert insights.grade == {}
        assert insights.worst_trades == []
        assert insights.best_trades == []
        assert insights.duration == {"avg_bars": 0.0, "avg_hours": 0.0}

    def test_single_winner(self):
        """Single winner with known values."""
        t = _make_trade(symbol="BTC", direction=Side.LONG, pnl=50.0,
                        r_multiple=1.5, setup_grade=SetupGrade.A,
                        confirmation_type="engulfing", confluences=["rsi_ob"])
        insights = extract_diagnostic_insights([t])

        assert insights.n_trades == 1
        assert insights.win_rate == 100.0
        assert insights.mean_r == 1.5
        assert insights.profit_factor == float("inf")

        # Per-confirmation
        assert "engulfing" in insights.per_confirmation
        assert insights.per_confirmation["engulfing"]["n"] == 1
        assert insights.per_confirmation["engulfing"]["wr"] == 100.0

        # Per-asset
        assert "BTC" in insights.per_asset
        assert insights.per_asset["BTC"]["n"] == 1
        assert insights.per_asset["BTC"]["long_wr"] == 100.0

        # Direction
        assert "long" in insights.direction
        assert insights.direction["long"]["n"] == 1

        # Grade
        assert "A" in insights.grade
        assert insights.grade["A"]["n"] == 1

        # Confluence
        assert 1 in insights.confluence  # 1 confluence used
        assert insights.confluence[1]["n"] == 1

    def test_mixed_winners_losers(self):
        """Mixed winners/losers compute correct aggregate stats."""
        trades = [
            _make_trade(pnl=50.0, r_multiple=1.5, mfe_r=2.0, mae_r=-0.2),
            _make_trade(pnl=30.0, r_multiple=0.8, mfe_r=1.2, mae_r=-0.4),
            _make_trade(pnl=-20.0, r_multiple=-0.5, mfe_r=0.3, mae_r=-0.8),
            _make_trade(pnl=-40.0, r_multiple=-1.0, mfe_r=0.1, mae_r=-1.0),
        ]
        insights = extract_diagnostic_insights(trades)

        assert insights.n_trades == 4
        assert insights.win_rate == 50.0
        assert insights.mean_r == pytest.approx(0.2, abs=0.01)

        # MFE capture should be computed
        assert insights.mfe_capture["avg_mfe_r"] > 0
        assert "avg_capture_pct" in insights.mfe_capture

        # R-stats
        assert insights.r_stats["mean"] == pytest.approx(0.2, abs=0.01)
        assert insights.r_stats["std"] > 0

    def test_multiple_confirmations(self):
        """Per-confirmation breakdown with correct avg_r per type."""
        trades = [
            _make_trade(pnl=50.0, r_multiple=1.5, confirmation_type="engulfing"),
            _make_trade(pnl=-20.0, r_multiple=-0.5, confirmation_type="engulfing"),
            _make_trade(pnl=30.0, r_multiple=0.8, confirmation_type="hammer"),
            _make_trade(pnl=-10.0, r_multiple=-0.3, confirmation_type="hammer"),
        ]
        insights = extract_diagnostic_insights(trades)

        assert "engulfing" in insights.per_confirmation
        assert "hammer" in insights.per_confirmation
        assert insights.per_confirmation["engulfing"]["n"] == 2
        assert insights.per_confirmation["hammer"]["n"] == 2
        # engulfing: avg_r = (1.5 + -0.5) / 2 = 0.5
        assert insights.per_confirmation["engulfing"]["avg_r"] == pytest.approx(0.5, abs=0.01)
        # hammer: avg_r = (0.8 + -0.3) / 2 = 0.25
        assert insights.per_confirmation["hammer"]["avg_r"] == pytest.approx(0.25, abs=0.01)

    def test_multiple_symbols(self):
        """Per-asset with long/short split."""
        trades = [
            _make_trade(symbol="BTC", direction=Side.LONG, pnl=50.0, r_multiple=1.5),
            _make_trade(symbol="BTC", direction=Side.SHORT, pnl=-20.0, r_multiple=-0.5),
            _make_trade(symbol="ETH", direction=Side.LONG, pnl=30.0, r_multiple=0.8),
        ]
        insights = extract_diagnostic_insights(trades)

        assert "BTC" in insights.per_asset
        assert "ETH" in insights.per_asset
        assert insights.per_asset["BTC"]["n"] == 2
        assert insights.per_asset["ETH"]["n"] == 1
        assert insights.per_asset["BTC"]["long_wr"] == 100.0
        assert insights.per_asset["BTC"]["short_wr"] == 0.0

    def test_confluence_grouping(self):
        """Confluence values correctly grouped."""
        trades = [
            _make_trade(pnl=50.0, r_multiple=1.5, confluences=["rsi", "vol"]),
            _make_trade(pnl=30.0, r_multiple=0.8, confluences=["rsi"]),
            _make_trade(pnl=-10.0, r_multiple=-0.3, confluences=[]),
        ]
        insights = extract_diagnostic_insights(trades)

        assert 0 in insights.confluence
        assert 1 in insights.confluence
        assert 2 in insights.confluence
        assert insights.confluence[0]["n"] == 1
        assert insights.confluence[1]["n"] == 1
        assert insights.confluence[2]["n"] == 1

    def test_worst_best_trade_ordering(self):
        """Worst/best trades ordered by R."""
        trades = [
            _make_trade(pnl=-50.0, r_multiple=-2.0),
            _make_trade(pnl=-10.0, r_multiple=-0.3),
            _make_trade(pnl=30.0, r_multiple=0.8),
            _make_trade(pnl=80.0, r_multiple=3.0),
        ]
        insights = extract_diagnostic_insights(trades)

        assert insights.worst_trades[0]["r_multiple"] == -2.0
        assert insights.worst_trades[1]["r_multiple"] == -0.3
        assert insights.best_trades[0]["r_multiple"] == 3.0
        assert insights.best_trades[1]["r_multiple"] == 0.8

    def test_concentration_with_winners(self):
        """Concentration computed correctly when there are profitable trades (net of commission)."""
        trades = [
            _make_trade(pnl=100.0, r_multiple=2.0),
            _make_trade(pnl=10.0, r_multiple=0.2),
            _make_trade(pnl=-5.0, r_multiple=-0.1),
        ]
        insights = extract_diagnostic_insights(trades)

        # _make_trade uses commission=0.5, so net_pnl = pnl - 0.5
        total_net_pnl = (100.0 - 0.5) + (10.0 - 0.5) + (-5.0 - 0.5)
        expected_top1 = (100.0 - 0.5) / total_net_pnl * 100.0
        assert insights.concentration["top1_pct"] == pytest.approx(expected_top1, abs=0.1)

    def test_exit_attribution(self):
        """Exit attribution with correct P&L share (net of commission)."""
        trades = [
            _make_trade(pnl=60.0, r_multiple=1.5, exit_reason="trailing_stop"),
            _make_trade(pnl=20.0, r_multiple=0.5, exit_reason="tp1"),
            _make_trade(pnl=-30.0, r_multiple=-0.8, exit_reason="protective_stop"),
        ]
        insights = extract_diagnostic_insights(trades)

        assert "trailing_stop" in insights.exit_attribution
        assert "tp1" in insights.exit_attribution
        assert "protective_stop" in insights.exit_attribution

        # _make_trade uses commission=0.5, so net_pnl = pnl - 0.5
        total_net = (60.0 - 0.5) + (20.0 - 0.5) + (-30.0 - 0.5)
        assert insights.exit_attribution["trailing_stop"]["pnl_share"] == pytest.approx(
            (60.0 - 0.5) / total_net, abs=0.01)

    def test_direction_both_sides(self):
        """Direction data includes both long and short."""
        trades = [
            _make_trade(direction=Side.LONG, pnl=50.0, r_multiple=1.5),
            _make_trade(direction=Side.SHORT, pnl=-20.0, r_multiple=-0.5),
        ]
        insights = extract_diagnostic_insights(trades)

        assert "long" in insights.direction
        assert "short" in insights.direction
        assert insights.direction["long"]["wr"] == 100.0
        assert insights.direction["short"]["wr"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# 6b. Policy callbacks tests
# ══════════════════════════════════════════════════════════════════════


class TestPolicyCallbacks:
    """Tests for MomentumPlugin policy callbacks."""

    def _make_plugin(self):
        """Create a minimal MomentumPlugin with mocked backtest dependencies."""
        from crypto_trader.optimize.momentum_plugin import MomentumPlugin

        plugin = MomentumPlugin.__new__(MomentumPlugin)
        plugin.backtest_config = BacktestConfig(symbols=["BTC"], initial_equity=10_000.0)
        plugin.base_config = MagicMock()
        plugin.data_dir = MagicMock()
        plugin.max_workers = None
        plugin._last_result = None
        plugin._cached_store = None
        return plugin

    def test_diagnostic_gap_fn_healthy_metrics(self):
        """Healthy metrics → empty gaps."""
        plugin = self._make_plugin()
        metrics = {
            "exit_efficiency": 0.50,
            "win_rate": 55,
            "total_trades": 20,
            "max_drawdown_pct": 20,
            "profit_factor": 1.5,
        }
        gaps = plugin._diagnostic_gap_fn(1, metrics)
        assert gaps == []

    def test_diagnostic_gap_fn_low_exit_efficiency(self):
        """Low exit_efficiency → includes MFE gap."""
        plugin = self._make_plugin()
        metrics = {"exit_efficiency": 0.20, "win_rate": 55, "total_trades": 20,
                    "max_drawdown_pct": 20, "profit_factor": 1.5}
        gaps = plugin._diagnostic_gap_fn(1, metrics)
        assert any("MFE" in g for g in gaps)

    def test_diagnostic_gap_fn_low_win_rate(self):
        """Low win_rate → includes confirmation gap."""
        plugin = self._make_plugin()
        metrics = {"exit_efficiency": 0.50, "win_rate": 35, "total_trades": 20,
                    "max_drawdown_pct": 20, "profit_factor": 1.5}
        gaps = plugin._diagnostic_gap_fn(1, metrics)
        assert any("confirmation" in g.lower() for g in gaps)

    def test_diagnostic_gap_fn_low_trades(self):
        """Low trade count → includes frequency gap."""
        plugin = self._make_plugin()
        metrics = {"exit_efficiency": 0.50, "win_rate": 55, "total_trades": 3,
                    "max_drawdown_pct": 20, "profit_factor": 1.5}
        gaps = plugin._diagnostic_gap_fn(1, metrics)
        assert any("frequency" in g.lower() or "fewer" in g.lower() for g in gaps)

    def test_suggest_experiments_fn_no_result(self):
        """No cached result → empty experiments."""
        plugin = self._make_plugin()
        plugin._last_result = None
        exps = plugin._suggest_experiments_fn(1, {}, [], MagicMock())
        assert exps == []

    def test_suggest_experiments_fn_poor_mfe_phase1(self):
        """Poor MFE capture in phase 1 → suggests trail experiments."""
        plugin = self._make_plugin()
        # Create mock result with trades that have poor MFE capture
        trades = [
            _make_trade(pnl=10.0, r_multiple=0.3, mfe_r=2.0, mae_r=-0.2),
            _make_trade(pnl=5.0, r_multiple=0.2, mfe_r=1.5, mae_r=-0.3),
        ]
        mock_result = MagicMock()
        mock_result.trades = trades
        plugin._last_result = mock_result

        exps = plugin._suggest_experiments_fn(1, {}, [], MagicMock())
        exp_names = [e.name for e in exps]
        assert any("TRAIL" in name for name in exp_names)

    def test_suggest_experiments_fn_bad_confirmation_phase3(self):
        """Underperforming confirmation in phase 3 → suggests disable."""
        plugin = self._make_plugin()
        trades = [
            _make_trade(pnl=-20.0, r_multiple=-0.5, confirmation_type="hammer"),
            _make_trade(pnl=-30.0, r_multiple=-0.8, confirmation_type="hammer"),
            _make_trade(pnl=50.0, r_multiple=1.5, confirmation_type="engulfing"),
        ]
        mock_result = MagicMock()
        mock_result.trades = trades
        plugin._last_result = mock_result

        exps = plugin._suggest_experiments_fn(3, {}, [], MagicMock())
        exp_names = [e.name for e in exps]
        assert any("HAMMER" in name.upper() for name in exp_names)
        # Verify it uses the CONFIRMATION_DISABLE_MAP
        disable_exps = [e for e in exps if "DISABLE" in e.name]
        if disable_exps:
            assert "confirmation.enable_hammer" in disable_exps[0].mutations

    def test_decide_action_fn_gate_passed(self):
        """Gate passed → advance."""
        plugin = self._make_plugin()
        state = _make_phase_state()
        decision = plugin._decide_action_fn(
            1, {}, state,
            _make_greedy_result(accepted_count=2),
            _make_gate_result(passed=True),
            {}, {}, 2, 1,
        )
        assert decision is not None
        assert decision.action == "advance"
        assert "passed" in decision.reason.lower()

    def test_decide_action_fn_nothing_accepted(self):
        """Nothing accepted → improve_diagnostics."""
        plugin = self._make_plugin()
        state = _make_phase_state()
        decision = plugin._decide_action_fn(
            1, {}, state,
            _make_greedy_result(accepted_count=0),
            _make_gate_result(passed=False),
            {}, {}, 2, 1,
        )
        assert decision is not None
        assert decision.action == "improve_diagnostics"

    def test_decide_action_fn_gate_failed_accepted(self):
        """Gate failed + experiments accepted → advance because scoring is immutable."""
        plugin = self._make_plugin()
        state = _make_phase_state()
        decision = plugin._decide_action_fn(
            1, {}, state,
            _make_greedy_result(accepted_count=3, accepted_names=["a", "b", "c"]),
            _make_gate_result(passed=False),
            {}, {}, 2, 1,
        )
        assert decision is not None
        assert decision.action == "advance"
        assert decision.scoring_weight_overrides is None
        assert "immutable" in decision.reason.lower()

    def test_decide_action_fn_budget_exhausted(self):
        """Accepted mutations still advance under immutable scoring."""
        plugin = self._make_plugin()
        state = _make_phase_state(scoring_retries={1: 2}, diagnostic_retries={1: 1})
        decision = plugin._decide_action_fn(
            1, {}, state,
            _make_greedy_result(accepted_count=2, accepted_names=["a", "b"]),
            _make_gate_result(passed=False),
            {}, {}, 2, 1,
        )
        assert decision is not None
        assert decision.action == "advance"
        assert "immutable" in decision.reason.lower()

    def test_redesign_scoring_weights_fn_with_weaknesses(self):
        """Round-3 replay does not redesign scoring weights."""
        plugin = self._make_plugin()

        result = plugin._redesign_scoring_weights_fn(
            phase=1,
            current_weights=dict(SCORING_WEIGHTS),
            metrics={},
            strengths=[],
            weaknesses=["total_trades", "exit_efficiency"],
        )
        assert result is None

    def test_redesign_scoring_weights_fn_no_weaknesses(self):
        """No weaknesses → still no redesign under immutable scoring."""
        plugin = self._make_plugin()
        result = plugin._redesign_scoring_weights_fn(
            phase=1,
            current_weights={},
            metrics={},
            strengths=["win_rate"],
            weaknesses=[],
        )
        assert result is None


# ══════════════════════════════════════════════════════════════════════
# 6c. Enhanced evaluation tests
# ══════════════════════════════════════════════════════════════════════


class TestEnhancedEvaluation:
    """Tests for build_evaluation_report with insights and format_dimension_text."""

    def _make_mock_metrics(self):
        """Create a mock PerformanceMetrics."""
        pm = MagicMock()
        pm.total_trades = 18
        pm.win_rate = 66.7
        pm.profit_factor = 2.5
        pm.a_setup_win_rate = 75.0
        pm.b_setup_win_rate = 50.0
        pm.avg_mae_r = -0.35
        pm.avg_bars_held = 5.2
        pm.max_drawdown_pct = 18.5
        pm.exit_efficiency = 0.45
        pm.avg_mfe_r = 1.2
        pm.sharpe_ratio = 1.5
        pm.calmar_ratio = 2.0
        return pm

    def test_build_evaluation_report_without_insights(self):
        """Without insights → backward-compatible (no assessment keys)."""
        pm = self._make_mock_metrics()
        report = build_evaluation_report(pm)

        assert "Signal Extraction" in report
        assert "Signal Discrimination" in report
        assert "Entry Mechanism" in report
        assert "Trade Management" in report
        assert "Exit Mechanism" in report

        # No assessment key without insights
        assert "assessment" not in report["Signal Extraction"]
        assert "assessment" not in report["Signal Discrimination"]

    def test_build_evaluation_report_with_insights(self):
        """With insights → enriched with assessments."""
        pm = self._make_mock_metrics()
        insights = DiagnosticInsights(
            n_trades=18, win_rate=66.7, mean_r=0.35, profit_factor=2.5,
            per_confirmation={"engulfing": {"n": 10, "wr": 70, "avg_r": 0.5, "total_r": 5.0, "pnl": 500}},
            per_asset={"BTC": {"n": 10, "wr": 70, "avg_r": 0.5, "long_wr": 80, "short_wr": 50, "long_avg_r": 0.6, "short_avg_r": 0.3}},
            exit_attribution={"trailing_stop": {"n": 10, "wr": 70, "avg_r": 0.5, "total_r": 5.0, "pnl_share": 0.6}},
            mfe_capture={"avg_mfe_r": 1.2, "avg_mae_r": -0.35, "avg_capture_pct": 0.45, "avg_giveback_pct": 0.3, "losers_with_mfe_pct": 30.0},
            direction={"long": {"n": 12, "wr": 75, "avg_r": 0.5}},
            confluence={1: {"n": 10, "wr": 70, "avg_r": 0.4}},
            grade={"A": {"n": 8, "wr": 75, "avg_r": 0.6}, "B": {"n": 10, "wr": 60, "avg_r": 0.2}},
            duration={"avg_bars": 5.2, "avg_hours": 1.3},
            concentration={"top1_pct": 25.0, "top20_pct": 60.0},
            r_stats={"mean": 0.35, "median": 0.3, "std": 0.8, "skew": 0.5},
            worst_trades=[], best_trades=[],
        )
        report = build_evaluation_report(pm, insights=insights)

        # Assessment keys present
        assert "assessment" in report["Signal Extraction"]
        assert "assessment" in report["Signal Discrimination"]
        assert "assessment" in report["Trade Management"]
        assert "assessment" in report["Exit Mechanism"]

        # Signal Extraction enriched
        assert "mean_r" in report["Signal Extraction"]
        assert "per_confirmation" in report["Signal Extraction"]

    def test_format_dimension_text_with_assessment(self):
        """Assessment key rendered as header line."""
        data = {
            "total_trades": 18,
            "assessment": "Alpha: capturing meaningful alpha",
            "win_rate": 66.7,
        }
        text = format_dimension_text("Signal Extraction", data)
        lines = text.split("\n")
        # Assessment should be first line
        assert lines[0].startswith(">> Alpha")

    def test_format_dimension_text_with_nested_dict(self):
        """Nested dict formatted correctly."""
        data = {
            "metric": 0.5,
            "breakdown": {"sub1": 0.3, "sub2": "text_value"},
        }
        text = format_dimension_text("Test", data)
        assert "breakdown:" in text
        assert "  sub1: 0.3000" in text
        assert "  sub2: text_value" in text

    def test_format_dimension_text_basic(self):
        """Basic formatting without assessment or nested dicts."""
        data = {"total_trades": 18, "win_rate": 66.7}
        text = format_dimension_text("Test", data)
        assert "total_trades: 18" in text
        assert "win_rate: 66.7000" in text

    def test_build_evaluation_report_with_bad_confirmations(self):
        """Insights with value-destroying confirmations → discrimination assessment."""
        pm = self._make_mock_metrics()
        insights = DiagnosticInsights(
            n_trades=10, win_rate=50, mean_r=0.1, profit_factor=1.1,
            per_confirmation={
                "engulfing": {"n": 5, "wr": 70, "avg_r": 0.5, "total_r": 2.5, "pnl": 250},
                "hammer": {"n": 5, "wr": 30, "avg_r": -0.3, "total_r": -1.5, "pnl": -150},
            },
            per_asset={}, exit_attribution={},
            mfe_capture={"avg_mfe_r": 0, "avg_mae_r": 0, "avg_capture_pct": 0.4, "avg_giveback_pct": 0, "losers_with_mfe_pct": 0},
            direction={}, confluence={},
            grade={"A": {"n": 5, "wr": 60, "avg_r": 0.3}, "B": {"n": 5, "wr": 40, "avg_r": -0.1}},
            duration={"avg_bars": 3, "avg_hours": 0.75},
            concentration={"top1_pct": 40, "top20_pct": 80},
            r_stats={"mean": 0.1, "median": 0.0, "std": 0.5, "skew": 0.2},
            worst_trades=[], best_trades=[],
        )
        report = build_evaluation_report(pm, insights=insights)

        # Discrimination should flag hammer
        disc = report["Signal Discrimination"]
        assert "value_destroying_confirmations" in disc
        assert "hammer" in disc["value_destroying_confirmations"]


# ══════════════════════════════════════════════════════════════════════
# 6d. Enhanced diagnostics tests
# ══════════════════════════════════════════════════════════════════════


class TestEnhancedDiagnostics:
    """Tests for run_phase_diagnostics and run_enhanced_diagnostics."""

    def _make_plugin(self):
        """Create a minimal MomentumPlugin."""
        from crypto_trader.optimize.momentum_plugin import MomentumPlugin

        plugin = MomentumPlugin.__new__(MomentumPlugin)
        plugin.backtest_config = BacktestConfig(symbols=["BTC"], initial_equity=10_000.0)
        plugin.base_config = MagicMock()
        plugin.data_dir = MagicMock()
        plugin.max_workers = None
        plugin._last_result = None
        plugin._cached_store = None
        return plugin

    def test_run_phase_diagnostics_with_cached_result(self):
        """Cached _last_result → includes per-confirmation/asset info."""
        plugin = self._make_plugin()
        trades = [
            _make_trade(pnl=50.0, r_multiple=1.5, confirmation_type="engulfing", symbol="BTC"),
            _make_trade(pnl=-20.0, r_multiple=-0.5, confirmation_type="hammer", symbol="ETH"),
        ]
        mock_result = MagicMock()
        mock_result.trades = trades
        plugin._last_result = mock_result

        text = plugin.run_phase_diagnostics(
            1, MagicMock(), {"total_trades": 2, "exit_efficiency": 0.5},
            _make_greedy_result(accepted_count=1, accepted_names=["test"]),
        )
        assert "engulfing" in text
        assert "hammer" in text
        assert "BTC" in text
        assert "ETH" in text
        assert "MFE capture" in text

    def test_run_phase_diagnostics_no_cached_result(self):
        """No cached result → falls back to key metrics."""
        plugin = self._make_plugin()
        plugin._last_result = None

        text = plugin.run_phase_diagnostics(
            1, MagicMock(), {"total_trades": 10, "win_rate": 55.0},
            _make_greedy_result(accepted_count=1, accepted_names=["test"]),
        )
        assert "total_trades" in text
        assert "Key metrics" in text

    def test_run_enhanced_diagnostics_with_cached_result(self):
        """Cached _last_result → includes full generate_diagnostics output."""
        plugin = self._make_plugin()
        trades = [
            _make_trade(pnl=50.0, r_multiple=1.5, symbol="BTC"),
            _make_trade(pnl=-20.0, r_multiple=-0.5, symbol="ETH"),
        ]
        mock_result = MagicMock()
        mock_result.trades = trades
        plugin._last_result = mock_result

        text = plugin.run_enhanced_diagnostics(
            1, MagicMock(), {},
            _make_greedy_result(accepted_count=1, accepted_names=["test"]),
        )
        # Should contain sections from generate_diagnostics
        assert "Overview" in text
        assert "Winner" in text or "Loser" in text

    def test_run_enhanced_diagnostics_no_cached_result(self):
        """No cached result → falls back to metrics."""
        plugin = self._make_plugin()
        plugin._last_result = None

        text = plugin.run_enhanced_diagnostics(
            1, MagicMock(), {"profit_factor": 1.5},
            _make_greedy_result(),
        )
        assert "profit_factor" in text
        assert "All metrics" in text

    def test_compute_final_metrics_caches_result(self):
        """compute_final_metrics sets _last_result."""
        plugin = self._make_plugin()
        mock_result = MagicMock()
        mock_result.metrics = MagicMock()
        mock_metrics_dict = {"total_trades": 10, "win_rate": 55.0}

        with patch("crypto_trader.optimize.momentum_plugin.run", return_value=mock_result), \
             patch("crypto_trader.optimize.momentum_plugin.apply_mutations", return_value=MagicMock()), \
             patch("crypto_trader.optimize.momentum_plugin.metrics_to_dict", return_value=mock_metrics_dict):
            result = plugin.compute_final_metrics({})

        assert plugin._last_result is mock_result
        assert result == mock_metrics_dict

    def test_get_phase_spec_has_callbacks(self):
        """get_phase_spec wires all 4 callbacks."""
        plugin = self._make_plugin()
        spec = plugin.get_phase_spec(1, MagicMock())

        assert spec.analysis_policy.diagnostic_gap_fn is not None
        assert spec.analysis_policy.suggest_experiments_fn is not None
        assert spec.analysis_policy.decide_action_fn is not None
        assert spec.analysis_policy.redesign_scoring_weights_fn is not None


# ══════════════════════════════════════════════════════════════════════
# Deep diagnostic_gap_fn tests (Step 3)
# ══════════════════════════════════════════════════════════════════════


class TestDeepDiagnosticGapFn:
    """Tests for insights-driven _diagnostic_gap_fn."""

    def _make_plugin(self):
        return MomentumPlugin(BacktestConfig(), MomentumConfig())

    def test_insights_high_giveback_phase1(self):
        """High alpha giveback flagged for trail phase."""
        plugin = self._make_plugin()
        # Create trades with high giveback (mfe_r=2.0, r_multiple=0.3 → low capture)
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=10, r_multiple=0.3, mfe_r=2.0),
            _make_trade(pnl=5, r_multiple=0.2, mfe_r=1.5),
        ]
        gaps = plugin._diagnostic_gap_fn(1, {})
        assert any("D1" in g and "giveback" in g.lower() for g in gaps)

    def test_insights_bad_confirmations_phase3(self):
        """Negative-R confirmations flagged for signal phase."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=-10, r_multiple=-0.8, confirmation_type="micro_shift"),
            _make_trade(pnl=-5, r_multiple=-0.5, confirmation_type="micro_shift"),
            _make_trade(pnl=20, r_multiple=1.0, confirmation_type="engulfing"),
        ]
        gaps = plugin._diagnostic_gap_fn(3, {})
        assert any("D4" in g and "micro_shift" in g for g in gaps)

    def test_insights_concentration_phase5(self):
        """Concentration risk flagged for risk phase."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        # One big winner and one small trade → top1_pct > 0.5
        plugin._last_result.trades = [
            _make_trade(pnl=100, r_multiple=3.0),
            _make_trade(pnl=1, r_multiple=0.05),
        ]
        gaps = plugin._diagnostic_gap_fn(5, {})
        assert any("D3" in g and "concentration" in g.lower() for g in gaps)

    def test_phase_specific_filtering(self):
        """Trail gaps only flagged for phases 1-2, not 3-4."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=10, r_multiple=0.3, mfe_r=2.0),
            _make_trade(pnl=5, r_multiple=0.2, mfe_r=1.5),
        ]
        # Phase 3 should NOT flag trail/giveback issues
        gaps_p3 = plugin._diagnostic_gap_fn(3, {})
        assert not any("giveback" in g.lower() for g in gaps_p3)

    def test_healthy_metrics_no_gaps(self):
        """Healthy trades produce no gaps."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        # Good trades: moderate capture, no bad confirmations
        plugin._last_result.trades = [
            _make_trade(pnl=10, r_multiple=0.8, mfe_r=1.0,
                        confirmation_type="engulfing"),
            _make_trade(pnl=8, r_multiple=0.6, mfe_r=0.9,
                        confirmation_type="hammer"),
            _make_trade(pnl=-3, r_multiple=-0.3, mfe_r=0.2,
                        confirmation_type="inside_bar_break"),
        ]
        # Use a phase that checks everything
        gaps = plugin._diagnostic_gap_fn(1, {"max_drawdown_pct": 10})
        # With good data and low DD, should have few or no gaps
        assert not any("concentration" in g.lower() for g in gaps)

    def test_fallback_no_last_result(self):
        """Falls back to metric thresholds when no _last_result."""
        plugin = self._make_plugin()
        plugin._last_result = None
        gaps = plugin._diagnostic_gap_fn(1, {
            "exit_efficiency": 0.2,
            "win_rate": 30,
            "total_trades": 3,
        })
        assert len(gaps) >= 2
        assert any("exit efficiency" in g.lower() for g in gaps)
        assert any("win rate" in g.lower() or "confirmation" in g.lower() for g in gaps)


# ══════════════════════════════════════════════════════════════════════
# Full suggest_experiments_fn coverage tests (Step 4)
# ══════════════════════════════════════════════════════════════════════


class TestSuggestExperimentsFnExtended:
    """Tests for extended _suggest_experiments_fn (phases 5-6)."""

    def _make_plugin(self):
        return MomentumPlugin(BacktestConfig(), MomentumConfig())

    def test_phase5_high_dd_suggests_risk_reduction(self):
        """Phase 5 with high DD suggests risk_pct_b reduction."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=-20, r_multiple=-1.0),
            _make_trade(pnl=-15, r_multiple=-0.8),
            _make_trade(pnl=5, r_multiple=0.3),
        ]
        exps = plugin._suggest_experiments_fn(
            5, {"max_drawdown_pct": 35}, [], MagicMock())
        names = [e.name for e in exps]
        assert any("risk" in n.lower() or "RISK" in n for n in names)

    def test_phase6_direction_imbalance(self):
        """Phase 6 with direction imbalance suggests symbol filter."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(direction=Side.LONG, pnl=20, r_multiple=1.0),
            _make_trade(direction=Side.LONG, pnl=15, r_multiple=0.8),
            _make_trade(direction=Side.SHORT, pnl=-10, r_multiple=-0.8),
            _make_trade(direction=Side.SHORT, pnl=-12, r_multiple=-0.9),
            _make_trade(direction=Side.SHORT, pnl=-8, r_multiple=-0.6),
        ]
        exps = plugin._suggest_experiments_fn(
            6, {}, [], MagicMock())
        names = [e.name for e in exps]
        # Should suggest per-asset direction filter (e.g. SUGG_BTC_LONG_ONLY)
        assert any("LONG_ONLY" in n or "SHORT_ONLY" in n for n in names)

    def test_phase1_short_duration_wider_activation(self):
        """Phase 1 with short avg duration suggests wider trail activation."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(bars_held=2, exit_reason="protective_stop"),
            _make_trade(bars_held=3, exit_reason="protective_stop"),
            _make_trade(bars_held=2, exit_reason="protective_stop"),
        ]
        exps = plugin._suggest_experiments_fn(
            1, {}, [], MagicMock())
        names = [e.name for e in exps]
        assert any("activation" in n.lower() or "ACTIVATION" in n or
                    "stop" in n.lower() or "STOP" in n for n in names)

    def test_no_experiments_when_healthy(self):
        """No diagnostic-driven experiments when all metrics healthy."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=10, r_multiple=0.8, mfe_r=1.0, bars_held=8,
                        confirmation_type="engulfing", exit_reason="tp1"),
            _make_trade(pnl=8, r_multiple=0.6, mfe_r=0.8, bars_held=10,
                        confirmation_type="hammer", exit_reason="trailing_stop"),
        ]
        exps = plugin._suggest_experiments_fn(
            5, {"max_drawdown_pct": 10}, [], MagicMock())
        # With healthy trades, phase 5 shouldn't suggest much
        # (no high DD, no concentration)
        assert len(exps) <= 2

    def test_phase2_high_stopout_share(self):
        """Phase 2 with high stop-out share suggests stop buffer increase."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=-10, r_multiple=-1.0, exit_reason="protective_stop"),
            _make_trade(pnl=-8, r_multiple=-0.9, exit_reason="protective_stop"),
            _make_trade(pnl=-7, r_multiple=-0.8, exit_reason="protective_stop"),
            _make_trade(pnl=5, r_multiple=0.5, exit_reason="tp1"),
        ]
        exps = plugin._suggest_experiments_fn(
            2, {}, [], MagicMock())
        names = [e.name for e in exps]
        assert any("stop" in n.lower() or "STOP" in n or
                    "atr" in n.lower() or "ATR" in n for n in names)

    def test_all_phases_covered(self):
        """Every phase (1-6) can generate experiments without error."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=-10, r_multiple=-1.0),
            _make_trade(pnl=20, r_multiple=1.5),
        ]
        for phase in range(1, 7):
            exps = plugin._suggest_experiments_fn(
                phase, {"max_drawdown_pct": 40}, [], MagicMock())
            assert isinstance(exps, list)


# ══════════════════════════════════════════════════════════════════════
# Extra analysis callbacks tests (Step 7)
# ══════════════════════════════════════════════════════════════════════


class TestExtraAnalysisCallbacks:
    """Tests for _build_extra_analysis_fn and _format_extra_analysis_fn."""

    def _make_plugin(self):
        return MomentumPlugin(BacktestConfig(), MomentumConfig())

    def test_build_extra_analysis_returns_dict(self):
        """_build_extra_analysis_fn returns a dict for each phase group."""
        plugin = self._make_plugin()
        plugin._last_result = MagicMock()
        plugin._last_result.trades = [
            _make_trade(pnl=10, r_multiple=0.5),
            _make_trade(pnl=-5, r_multiple=-0.3),
        ]
        for phase in (1, 3, 5):
            result = plugin._build_extra_analysis_fn(
                phase, {"total_trades": 2}, MagicMock(), MagicMock())
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_format_extra_analysis_returns_text(self):
        """_format_extra_analysis_fn renders dict into text."""
        plugin = self._make_plugin()
        data = {"trail_activations": 5, "avg_buffer": 0.3}
        text = plugin._format_extra_analysis_fn(data)
        assert isinstance(text, str)
        assert "trail_activations" in text

    def test_get_phase_spec_includes_extra_callbacks(self):
        """get_phase_spec wires both extra analysis callbacks."""
        plugin = self._make_plugin()
        spec = plugin.get_phase_spec(1, MagicMock())
        assert spec.analysis_policy.build_extra_analysis_fn is not None
        assert spec.analysis_policy.format_extra_analysis_fn is not None

    def test_no_last_result_returns_empty_dict(self):
        """Returns empty dict when _last_result is None."""
        plugin = self._make_plugin()
        plugin._last_result = None
        result = plugin._build_extra_analysis_fn(
            1, {}, MagicMock(), MagicMock())
        assert isinstance(result, dict)
        assert len(result) == 0

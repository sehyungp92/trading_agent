"""Tests for SignalHealthAnalyzer — signal health schemas and analyzer logic."""
from __future__ import annotations

import pytest

from trading_assistant.schemas.signal_health import ComponentHealth, SignalHealthReport
from trading_assistant.skills.signal_health_analyzer import SignalHealthAnalyzer
from tests.factories import make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    pnl: float = 100.0,
    signal_evolution: list[dict] | None = None,
):
    return make_trade(
        trade_id=f"t-{pnl}", bot_id="test_bot", pair="NQ", pnl=pnl,
        signal_evolution=signal_evolution,
    )


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestSignalHealthSchemas:
    def test_component_health_defaults(self):
        ch = ComponentHealth(component_name="momentum")
        assert ch.trade_count == 0
        assert ch.stability == 0.0
        assert ch.win_correlation == 0.0

    def test_component_health_round_trip(self):
        ch = ComponentHealth(
            component_name="momentum",
            trade_count=10,
            avg_entry_value=0.7,
            avg_exit_value=0.5,
            avg_range=0.3,
            stability=0.85,
            win_correlation=0.45,
            trend_during_trade=-0.2,
        )
        data = ch.model_dump(mode="json")
        restored = ComponentHealth(**data)
        assert restored.component_name == "momentum"
        assert restored.stability == 0.85

    def test_report_round_trip(self):
        report = SignalHealthReport(
            bot_id="bot_a",
            date="2026-03-01",
            components=[
                ComponentHealth(component_name="momentum", trade_count=5),
            ],
            total_trades_with_data=5,
            coverage_pct=100.0,
        )
        data = report.model_dump(mode="json")
        restored = SignalHealthReport(**data)
        assert len(restored.components) == 1
        assert restored.coverage_pct == 100.0


# ---------------------------------------------------------------------------
# Analyzer — empty / no-data cases
# ---------------------------------------------------------------------------

class TestSignalHealthAnalyzerEmpty:
    def test_empty_trade_list(self):
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute([])
        assert report.total_trades_with_data == 0
        assert report.coverage_pct == 0.0
        assert report.components == []

    def test_trades_without_signal_evolution(self):
        trades = [_make_trade(pnl=50.0), _make_trade(pnl=-20.0)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        assert report.total_trades_with_data == 0
        assert report.components == []

    def test_trades_with_too_few_bars(self):
        """Signal evolution with < 2 bars should be skipped."""
        trades = [_make_trade(signal_evolution=[{"bar": 1, "momentum": 0.5}])]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        assert report.total_trades_with_data == 0


# ---------------------------------------------------------------------------
# Analyzer — basic component extraction
# ---------------------------------------------------------------------------

class TestSignalHealthAnalyzerBasic:
    def test_single_component_single_trade(self):
        evo = [
            {"bar": 1, "momentum": 0.8},
            {"bar": 2, "momentum": 0.7},
            {"bar": 3, "momentum": 0.6},
        ]
        trades = [_make_trade(pnl=100.0, signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.total_trades_with_data == 1
        assert report.coverage_pct == 100.0
        assert len(report.components) == 1

        comp = report.components[0]
        assert comp.component_name == "momentum"
        assert comp.trade_count == 1
        assert comp.avg_entry_value == 0.8
        assert comp.avg_exit_value == 0.6
        assert comp.avg_range == pytest.approx(0.2, abs=0.001)
        # Trend: exit - entry = -0.2
        assert comp.trend_during_trade == pytest.approx(-0.2, abs=0.001)

    def test_multiple_components(self):
        evo = [
            {"bar": 1, "momentum": 0.8, "volume": 1000},
            {"bar": 2, "momentum": 0.7, "volume": 1200},
        ]
        trades = [_make_trade(pnl=50.0, signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert len(report.components) == 2
        comp_names = {c.component_name for c in report.components}
        assert comp_names == {"momentum", "volume"}

    def test_coverage_pct_partial(self):
        evo = [
            {"bar": 1, "momentum": 0.8},
            {"bar": 2, "momentum": 0.7},
        ]
        trades = [
            _make_trade(pnl=100.0, signal_evolution=evo),
            _make_trade(pnl=-50.0),  # no signal evolution
        ]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        assert report.coverage_pct == 50.0
        assert report.total_trades_with_data == 1


# ---------------------------------------------------------------------------
# Analyzer — stability metric
# ---------------------------------------------------------------------------

class TestSignalHealthAnalyzerStability:
    def test_perfectly_stable_component(self):
        """Same value across all bars → stability = 1.0."""
        evo = [
            {"bar": 1, "momentum": 0.5},
            {"bar": 2, "momentum": 0.5},
            {"bar": 3, "momentum": 0.5},
        ]
        trades = [_make_trade(signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        # Range = 0 → stability defaults to 1.0
        assert comp.stability == 1.0

    def test_volatile_component(self):
        """Wide value swings → lower stability."""
        evo = [
            {"bar": 1, "momentum": 0.0},
            {"bar": 2, "momentum": 1.0},
            {"bar": 3, "momentum": 0.0},
        ]
        trades = [_make_trade(signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        # High stdev relative to range → lower stability
        assert comp.stability < 0.7


# ---------------------------------------------------------------------------
# Analyzer — win correlation (Pearson)
# ---------------------------------------------------------------------------

class TestSignalHealthAnalyzerWinCorrelation:
    def test_positive_correlation(self):
        """Higher entry values → higher PnL should give positive correlation."""
        trades = [
            _make_trade(pnl=100.0, signal_evolution=[
                {"bar": 1, "momentum": 0.9}, {"bar": 2, "momentum": 0.8},
            ]),
            _make_trade(pnl=50.0, signal_evolution=[
                {"bar": 1, "momentum": 0.7}, {"bar": 2, "momentum": 0.6},
            ]),
            _make_trade(pnl=-20.0, signal_evolution=[
                {"bar": 1, "momentum": 0.3}, {"bar": 2, "momentum": 0.2},
            ]),
        ]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        assert comp.win_correlation > 0.5

    def test_no_correlation_with_two_trades(self):
        """Pearson requires >=3 trades, so 2 trades should give 0.0."""
        trades = [
            _make_trade(pnl=100.0, signal_evolution=[
                {"bar": 1, "momentum": 0.9}, {"bar": 2, "momentum": 0.8},
            ]),
            _make_trade(pnl=-50.0, signal_evolution=[
                {"bar": 1, "momentum": 0.3}, {"bar": 2, "momentum": 0.2},
            ]),
        ]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        assert comp.win_correlation == 0.0

    def test_constant_entry_values_zero_correlation(self):
        """All same entry value → zero variance → correlation = 0."""
        trades = [
            _make_trade(pnl=100.0, signal_evolution=[
                {"bar": 1, "momentum": 0.5}, {"bar": 2, "momentum": 0.4},
            ]),
            _make_trade(pnl=-50.0, signal_evolution=[
                {"bar": 1, "momentum": 0.5}, {"bar": 2, "momentum": 0.3},
            ]),
            _make_trade(pnl=25.0, signal_evolution=[
                {"bar": 1, "momentum": 0.5}, {"bar": 2, "momentum": 0.6},
            ]),
        ]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        assert comp.win_correlation == 0.0


# ---------------------------------------------------------------------------
# Analyzer — trend during trade
# ---------------------------------------------------------------------------

class TestSignalHealthAnalyzerTrend:
    def test_positive_trend(self):
        """Signal increasing during trade → positive trend."""
        evo = [
            {"bar": 1, "momentum": 0.3},
            {"bar": 2, "momentum": 0.5},
            {"bar": 3, "momentum": 0.7},
        ]
        trades = [_make_trade(signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        assert comp.trend_during_trade > 0

    def test_negative_trend(self):
        """Signal decreasing during trade → negative trend."""
        evo = [
            {"bar": 1, "momentum": 0.9},
            {"bar": 2, "momentum": 0.7},
            {"bar": 3, "momentum": 0.4},
        ]
        trades = [_make_trade(signal_evolution=evo)]
        analyzer = SignalHealthAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        comp = report.components[0]
        assert comp.trend_during_trade < 0


# ---------------------------------------------------------------------------
# Analyzer — Pearson helper edge cases
# ---------------------------------------------------------------------------

class TestPearsonHelper:
    def test_fewer_than_3_returns_zero(self):
        assert SignalHealthAnalyzer._pearson([1.0, 2.0], [3.0, 4.0]) == 0.0

    def test_perfect_positive_correlation(self):
        r = SignalHealthAnalyzer._pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative_correlation(self):
        r = SignalHealthAnalyzer._pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r == pytest.approx(-1.0, abs=0.001)

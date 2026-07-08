"""Tests for the comprehensive diagnostics module."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from crypto_trader.backtest.metrics import PerformanceMetrics
from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.backtest.diagnostics import (
    generate_diagnostics,
    generate_phase_diagnostics,
    DIAGNOSTIC_MODULES,
    _SECTION_FUNCTIONS,
    _group_stats,
    _s01_overview,
    _s02_winner_loser_profiles,
    _s03_mfe_capture,
    _s04_stop_calibration,
    _s05_exit_attribution,
    _s06_streaks,
    _s07_drawdown,
    _s08_rolling_expectancy,
    _s09_per_asset,
    _s11_confirmation,
    _s12_confluence,
    _s13_timing,
    _s14_duration,
    _s15_concentration,
    _s16_worst_trades,
    _s17_interactions,
    _s18_friction,
    _s19_weekly_pnl,
    _s20_best_trades,
    _s21_risk_sizing,
    _s22_entry_method,
    _verdict,
    _r_distribution,
    _skew,
)


# ── Fixtures ─────────────────────────────────────────────────────────

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


@pytest.fixture
def mixed_trades() -> list[Trade]:
    """A realistic mix of winners and losers across symbols."""
    base = datetime(2026, 4, 14, 8, 0, tzinfo=timezone.utc)
    return [
        # BTC winners
        _make_trade("BTC", Side.LONG, 50.0, 0.8, 2, SetupGrade.A, "protective_stop",
                    "inside_bar_break", ["rsi_ob", "volume_spike"], entry_time=base),
        _make_trade("BTC", Side.LONG, 20.0, 0.3, 4, SetupGrade.B, "protective_stop",
                    "inside_bar_break", ["rsi_ob"], entry_time=base + timedelta(hours=4)),
        # BTC loser
        _make_trade("BTC", Side.LONG, -30.0, -0.6, 1, SetupGrade.B, "structure_break",
                    "micro_structure_shift", [], mae_r=-0.7, mfe_r=0.1,
                    entry_time=base + timedelta(hours=8)),
        # ETH winners
        _make_trade("ETH", Side.SHORT, 40.0, 0.6, 3, SetupGrade.A, "reversal_candle",
                    "hammer", ["volume_spike"], entry_time=base + timedelta(hours=12)),
        _make_trade("ETH", Side.LONG, 15.0, 0.2, 5, SetupGrade.B, "protective_stop",
                    "inside_bar_break", [], entry_time=base + timedelta(hours=16)),
        # ETH losers
        _make_trade("ETH", Side.LONG, -25.0, -0.5, 2, SetupGrade.B, "protective_stop",
                    "micro_structure_shift", [], mae_r=-0.6, mfe_r=0.15,
                    entry_time=base + timedelta(hours=20)),
        _make_trade("ETH", Side.LONG, -10.0, -0.3, 1, SetupGrade.B, "protective_stop",
                    "micro_structure_shift", [], mae_r=-0.4, mfe_r=0.05,
                    entry_time=base + timedelta(hours=24)),
        # SOL big winner
        _make_trade("SOL", Side.LONG, 100.0, 2.5, 6, SetupGrade.A, "reversal_candle",
                    "inside_bar_break", ["rsi_ob", "volume_spike"], mae_r=-0.2, mfe_r=3.0,
                    entry_time=base + timedelta(hours=28)),
    ]


# ── Tests ────────────────────────────────────────────────────────────


class TestGroupStats:
    def test_empty(self):
        assert "n=0" in _group_stats([])

    def test_basic(self, mixed_trades):
        result = _group_stats(mixed_trades)
        assert "n=8" in result
        assert "WR=" in result
        assert "Mean R=" in result
        assert "PF=" in result


class TestOverview:
    def test_overview_flags_low_sample(self, mixed_trades):
        result = _s01_overview(mixed_trades, 10000)
        assert "LOW SAMPLE" in result

    def test_overview_shows_pnl(self, mixed_trades):
        result = _s01_overview(mixed_trades, 10000)
        assert "Realized P&L" in result
        assert "Net Liq P&L" in result
        assert "Win Rate" in result

    def test_overview_shows_risk_adjusted_metrics_when_provided(self, mixed_trades):
        pm = PerformanceMetrics(
            max_drawdown_pct=4.117762015658249,
            sharpe_ratio=2.9466486019333233,
            calmar_ratio=4.661703820252063,
        )
        result = _s01_overview(mixed_trades, 10000, performance_metrics=pm)
        assert "Max DD:         4.12%" in result
        assert "Sharpe Ratio:   2.95" in result
        assert "Calmar Ratio:   4.66" in result

    def test_overview_empty(self):
        result = _s01_overview([], 10000)
        assert "No realized trades or terminal marks" in result


class TestWinnerLoserProfiles:
    def test_side_by_side(self, mixed_trades):
        result = _s02_winner_loser_profiles(mixed_trades)
        assert "Winners" in result
        assert "Losers" in result
        assert "Delta" in result
        assert "Avg R" in result
        assert "Avg MFE R" in result

    def test_grade_distribution(self, mixed_trades):
        result = _s02_winner_loser_profiles(mixed_trades)
        assert "A-grade" in result
        assert "B-grade" in result

    def test_only_winners(self):
        trades = [_make_trade(pnl=10, r_multiple=0.5)]
        result = _s02_winner_loser_profiles(trades)
        assert "Need both" in result


class TestMFECapture:
    def test_capture_by_exit(self, mixed_trades):
        result = _s03_mfe_capture(mixed_trades)
        assert "Capture" in result
        assert "Giveback" in result
        assert "protective_stop" in result

    def test_capture_headline_is_explicitly_split(self):
        trades = [
            _make_trade(pnl=20, r_multiple=0.5, mfe_r=1.0),
            _make_trade(pnl=-10, r_multiple=-0.2, mfe_r=0.8),
        ]
        result = _s03_mfe_capture(trades)
        assert "Winner capture efficiency" in result
        assert "All-trades capture" in result

    def test_winner_giveback(self, mixed_trades):
        result = _s03_mfe_capture(mixed_trades)
        assert "Winner Giveback" in result

    def test_losers_with_positive_mfe(self, mixed_trades):
        result = _s03_mfe_capture(mixed_trades)
        assert "Losers with positive MFE" in result


class TestStopCalibration:
    def test_stop_frequency(self, mixed_trades):
        result = _s04_stop_calibration(mixed_trades)
        assert "Stop exits" in result
        assert "Non-stop exits" in result

    def test_mae_distribution(self, mixed_trades):
        result = _s04_stop_calibration(mixed_trades)
        assert "MAE Distribution" in result


class TestExitAttribution:
    def test_all_reasons_present(self, mixed_trades):
        result = _s05_exit_attribution(mixed_trades)
        assert "protective_stop" in result
        assert "reversal_candle" in result
        assert "structure_break" in result
        assert "Share" in result


class TestStreaks:
    def test_streaks_computed(self, mixed_trades):
        result = _s06_streaks(mixed_trades)
        assert "Max win streak" in result
        assert "Max loss streak" in result
        assert "Sequence" in result

    def test_trend_detection(self, mixed_trades):
        result = _s06_streaks(mixed_trades)
        assert any(t in result for t in ["IMPROVING", "DEGRADING", "STABLE"])


class TestDrawdown:
    def test_r_drawdown(self, mixed_trades):
        result = _s07_drawdown(mixed_trades)
        assert "Max R-drawdown" in result
        assert "Final cum R" in result

    def test_episodes(self):
        # Create a sequence with a clear drawdown
        trades = [
            _make_trade(pnl=50, r_multiple=1.0),
            _make_trade(pnl=-30, r_multiple=-0.6),
            _make_trade(pnl=-20, r_multiple=-0.4),
            _make_trade(pnl=60, r_multiple=1.2),
        ]
        result = _s07_drawdown(trades)
        assert "Drawdown Episodes" in result or "Max R-drawdown" in result


class TestRollingExpectancy:
    def test_rolling(self, mixed_trades):
        result = _s08_rolling_expectancy(mixed_trades)
        assert "Current" in result
        assert "Best" in result
        assert "Worst" in result

    def test_too_few_trades(self):
        trades = [_make_trade()]
        result = _s08_rolling_expectancy(trades)
        assert "need >= 5" in result


class TestPerAsset:
    def test_all_symbols(self, mixed_trades):
        result = _s09_per_asset(mixed_trades)
        assert "BTC" in result
        assert "ETH" in result
        assert "SOL" in result

    def test_direction_split(self, mixed_trades):
        result = _s09_per_asset(mixed_trades)
        assert "Long" in result
        assert "Short" in result


class TestConfirmation:
    def test_all_types(self, mixed_trades):
        result = _s11_confirmation(mixed_trades)
        assert "inside_bar_break" in result
        assert "micro_structure_shift" in result
        assert "hammer" in result


class TestConfluence:
    def test_monotonicity_check(self, mixed_trades):
        result = _s12_confluence(mixed_trades)
        assert "Confluences" in result
        # Should check for monotonic assessment
        assert "onotonic" in result  # Monotonic or Non-monotonic


class TestTiming:
    def test_hour_and_dow(self, mixed_trades):
        result = _s13_timing(mixed_trades)
        assert "Entry Hour" in result
        assert "Day of Week" in result
        assert "Session Performance" in result


class TestDuration:
    def test_buckets(self, mixed_trades):
        result = _s14_duration(mixed_trades)
        assert "bars" in result.lower()
        assert "Duration Buckets" in result


class TestConcentration:
    def test_risk_assessment(self, mixed_trades):
        result = _s15_concentration(mixed_trades)
        assert "Largest single winner" in result
        assert "Concentration Risk" in result
        assert "SOL" in result  # SOL has biggest winner

    def test_no_profit(self):
        trades = [_make_trade(pnl=-10, r_multiple=-0.3)]
        result = _s15_concentration(trades)
        assert "no profitable" in result


class TestWorstTrades:
    def test_autopsy(self, mixed_trades):
        result = _s16_worst_trades(mixed_trades)
        assert "#1 worst" in result
        assert "Exit reason" in result
        assert "Common Patterns" in result


class TestInteractions:
    def test_grade_direction(self, mixed_trades):
        result = _s17_interactions(mixed_trades)
        assert "Grade" in result
        assert "Direction" in result

    def test_confirmation_symbol(self, mixed_trades):
        result = _s17_interactions(mixed_trades)
        assert "Confirmation" in result


class TestFriction:
    def test_totals(self, mixed_trades):
        result = _s18_friction(mixed_trades)
        assert "Total commissions" in result
        assert "Total funding" in result
        assert "Total friction" in result

    def test_per_symbol(self, mixed_trades):
        result = _s18_friction(mixed_trades)
        assert "Per-Symbol Friction" in result
        assert "BTC" in result
        assert "ETH" in result

    def test_funding_direction(self):
        t = Trade(
            trade_id="t_fund", symbol="BTC", direction=Side.LONG,
            entry_price=100.0, exit_price=110.0, qty=1.0,
            entry_time=datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
            pnl=10.0, r_multiple=0.5, commission=2.0, bars_held=16,
            setup_grade=SetupGrade.B, exit_reason="protective_stop",
            confluences_used=[], confirmation_type="inside_bar_break",
            entry_method="close", funding_paid=1.5, mae_r=-0.2, mfe_r=0.8,
        )
        result = _s18_friction([t])
        assert "Funding direction" in result
        assert "adverse" in result

    def test_single_trade(self):
        trade = _make_trade(pnl=10.0)
        result = _s18_friction([trade])
        assert "Total commissions" in result

    def test_empty(self):
        result = _s18_friction([])
        assert "No trades" in result

    def test_with_funding(self):
        t = Trade(
            trade_id="t1", symbol="BTC", direction=Side.LONG,
            entry_price=100.0, exit_price=110.0, qty=1.0,
            entry_time=datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
            pnl=10.0, r_multiple=0.5, commission=2.0, bars_held=16,
            setup_grade=SetupGrade.B, exit_reason="protective_stop",
            confluences_used=[], confirmation_type="inside_bar_break",
            entry_method="close", funding_paid=3.5, mae_r=-0.2, mfe_r=0.8,
        )
        result = _s18_friction([t])
        assert "adverse" in result or "favorable" in result


class TestWeeklyPnl:
    def test_calendar(self, mixed_trades):
        result = _s19_weekly_pnl(mixed_trades)
        assert "Weekly P&L" in result
        assert "Cum P&L" in result

    def test_positive_negative_weeks(self, mixed_trades):
        result = _s19_weekly_pnl(mixed_trades)
        assert "Positive weeks" in result
        assert "Best week" in result

    def test_empty(self):
        result = _s19_weekly_pnl([])
        assert "No trades" in result

    def test_multi_week_spread(self):
        base = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)
        trades = [
            _make_trade(pnl=20.0, r_multiple=0.5, entry_time=base,
                        exit_time=base + timedelta(hours=1)),
            _make_trade(pnl=-10.0, r_multiple=-0.3,
                        entry_time=base + timedelta(days=7),
                        exit_time=base + timedelta(days=7, hours=2)),
            _make_trade(pnl=30.0, r_multiple=1.0,
                        entry_time=base + timedelta(days=14),
                        exit_time=base + timedelta(days=14, hours=3)),
        ]
        result = _s19_weekly_pnl(trades)
        assert result.count("W") >= 3  # Multiple weeks in output


class TestBestTrades:
    def test_autopsy(self, mixed_trades):
        result = _s20_best_trades(mixed_trades)
        assert "#1 best" in result
        assert "SOL" in result  # SOL has the biggest winner
        assert "Exit reason" in result
        assert "captured" in result  # capture efficiency shown

    def test_common_patterns(self, mixed_trades):
        result = _s20_best_trades(mixed_trades)
        assert "Common Patterns in Best" in result

    def test_no_winners(self):
        trades = [_make_trade(pnl=-10, r_multiple=-0.3)]
        result = _s20_best_trades(trades)
        assert "No winning" in result

    def test_single_winner(self):
        trade = _make_trade(pnl=50, r_multiple=1.5, mfe_r=2.0)
        result = _s20_best_trades([trade])
        assert "#1 best" in result
        assert "captured" in result


class TestRiskSizing:
    def test_position_values(self, mixed_trades):
        result = _s21_risk_sizing(mixed_trades)
        assert "Position value" in result
        assert "Avg" in result
        assert "Range" in result

    def test_r_vs_dollar_alignment(self, mixed_trades):
        result = _s21_risk_sizing(mixed_trades)
        assert "R-Multiple vs Dollar" in result

    def test_rank_correlation(self, mixed_trades):
        result = _s21_risk_sizing(mixed_trades)
        assert "rank correlation" in result

    def test_worst_overlap(self, mixed_trades):
        result = _s21_risk_sizing(mixed_trades)
        assert "Worst-3 overlap" in result

    def test_empty(self):
        result = _s21_risk_sizing([])
        assert "No trades" in result

    def test_divergence_detection(self):
        # Create a trade where R is positive but pnl is negative (e.g. due to high commissions)
        t = Trade(
            trade_id="t_div", symbol="BTC", direction=Side.LONG,
            entry_price=100.0, exit_price=101.0, qty=1.0,
            entry_time=datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
            pnl=-5.0, r_multiple=0.2, commission=6.0, bars_held=4,
            setup_grade=SetupGrade.B, exit_reason="protective_stop",
            confluences_used=[], confirmation_type="inside_bar_break",
            entry_method="close", funding_paid=0.0, mae_r=-0.1, mfe_r=0.3,
        )
        result = _s21_risk_sizing([t])
        assert "disagree" in result


class TestEntryMethod:
    def test_method_breakdown(self, mixed_trades):
        result = _s22_entry_method(mixed_trades)
        assert "close" in result  # All test trades use "close" method

    def test_no_method_data(self):
        t = _make_trade()
        # Override entry_method to None
        t2 = Trade(
            trade_id=t.trade_id, symbol=t.symbol, direction=t.direction,
            entry_price=t.entry_price, exit_price=t.exit_price, qty=t.qty,
            entry_time=t.entry_time, exit_time=t.exit_time,
            pnl=t.pnl, r_multiple=t.r_multiple, commission=t.commission,
            bars_held=t.bars_held, setup_grade=t.setup_grade,
            exit_reason=t.exit_reason, confluences_used=t.confluences_used,
            confirmation_type=t.confirmation_type, entry_method=None,
            funding_paid=t.funding_paid, mae_r=t.mae_r, mfe_r=t.mfe_r,
        )
        result = _s22_entry_method([t2])
        assert "No entry method" in result or "unknown" in result

    def test_multiple_methods(self):
        t1 = _make_trade(pnl=20.0, r_multiple=0.5)
        t2 = Trade(
            trade_id="t_break", symbol="ETH", direction=Side.LONG,
            entry_price=100.0, exit_price=105.0, qty=0.5,
            entry_time=datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 14, 14, 0, tzinfo=timezone.utc),
            pnl=15.0, r_multiple=0.3, commission=0.5, bars_held=4,
            setup_grade=SetupGrade.B, exit_reason="protective_stop",
            confluences_used=[], confirmation_type="inside_bar_break",
            entry_method="break", funding_paid=0.0, mae_r=-0.1, mfe_r=0.6,
        )
        result = _s22_entry_method([t1, t2])
        assert "close" in result
        assert "break" in result


class TestRDistribution:
    def test_histogram(self, mixed_trades):
        result = _r_distribution(mixed_trades)
        assert "█" in result
        assert "Mean" in result
        assert "Median" in result
        assert "Skew" in result


class TestSkew:
    def test_symmetric(self):
        assert abs(_skew([1.0, 2.0, 3.0, 4.0, 5.0])) < 0.5

    def test_right_skew(self):
        assert _skew([1.0, 1.0, 1.0, 1.0, 10.0]) > 0

    def test_too_few(self):
        assert _skew([1.0]) == 0.0


class TestVerdict:
    def test_strengths_and_weaknesses(self, mixed_trades):
        result = _verdict(mixed_trades, 10000)
        assert "STRENGTHS" in result
        assert "WEAKNESSES" in result
        assert "RECOMMENDATIONS" in result

    def test_flags_low_sample(self, mixed_trades):
        result = _verdict(mixed_trades, 10000)
        assert "sample size" in result.lower() or "LOW SAMPLE" in result or "sample" in result.lower()

    def test_flags_underperforming_confirmation(self, mixed_trades):
        result = _verdict(mixed_trades, 10000)
        assert "micro_structure_shift" in result


class TestGenerateDiagnostics:
    def test_full_report(self, mixed_trades):
        report = generate_diagnostics(mixed_trades, initial_equity=10000)
        # Check all sections present
        assert "1. Overview" in report
        assert "2. Winner vs Loser" in report
        assert "3. MFE/MAE" in report
        assert "4. Stop Calibration" in report
        assert "5. Exit Reason" in report
        assert "6. Streak" in report
        assert "7. R-Curve" in report
        assert "8. Rolling" in report
        assert "9. Per-Asset" in report
        assert "10. Direction" in report
        assert "11. Confirmation" in report
        assert "12. Confluence" in report
        assert "13. Session" in report
        assert "14. Trade Duration" in report
        assert "15. Concentration" in report
        assert "16. Worst Trades" in report
        assert "17. Interaction" in report
        assert "18. Friction" in report
        assert "19. Weekly P&L" in report
        assert "20. Best Trades" in report
        assert "21. Risk & Sizing" in report
        assert "22. Entry Method" in report
        assert "VERDICT" in report

    def test_empty_trades(self):
        assert "No trades" in generate_diagnostics([])

    def test_single_trade(self):
        trade = _make_trade()
        report = generate_diagnostics([trade])
        assert "1. Overview" in report


# ══════════════════════════════════════════════════════════════════════
# Phase-targeted diagnostics (generate_phase_diagnostics)
# ══════════════════════════════════════════════════════════════════════


class TestDiagnosticModules:
    def test_all_section_functions_covered(self):
        """Every section function in _SECTION_FUNCTIONS appears in at least one module."""
        all_fns_in_modules: set[str] = set()
        for fns in DIAGNOSTIC_MODULES.values():
            all_fns_in_modules.update(fns)
        for fn_name in _SECTION_FUNCTIONS:
            assert fn_name in all_fns_in_modules, f"{fn_name} not in any diagnostic module"

    def test_module_keys_valid(self):
        """All module IDs follow D1-D6 naming."""
        for key in DIAGNOSTIC_MODULES:
            assert key.startswith("D") and key[1:].isdigit()

    def test_all_module_functions_exist(self):
        """Every function listed in DIAGNOSTIC_MODULES exists in _SECTION_FUNCTIONS."""
        for mod_id, fns in DIAGNOSTIC_MODULES.items():
            for fn_name in fns:
                assert fn_name in _SECTION_FUNCTIONS, (
                    f"{fn_name} in {mod_id} not found in _SECTION_FUNCTIONS"
                )


class TestGeneratePhaseDiagnostics:
    def test_single_module(self):
        """Single module produces relevant sections."""
        trades = [_make_trade(), _make_trade(pnl=-5, r_multiple=-0.5)]
        report = generate_phase_diagnostics(trades, ["D1"])
        # D1 has MFE capture and stop calibration
        assert "MFE" in report or "3." in report
        # D6 always included
        assert "1. Overview" in report

    def test_multiple_modules(self):
        """Multiple modules include sections from all."""
        trades = [_make_trade(), _make_trade(pnl=-5, r_multiple=-0.5)]
        report = generate_phase_diagnostics(trades, ["D1", "D4"])
        # D1 sections
        assert "3." in report or "MFE" in report
        # D4 sections
        assert "11." in report or "Confirmation" in report

    def test_d6_always_included(self):
        """D6 overview sections appear even when not explicitly requested."""
        trades = [_make_trade()]
        report = generate_phase_diagnostics(trades, ["D1"])
        assert "1. Overview" in report
        assert "VERDICT" in report

    def test_overview_receives_initial_equity(self):
        """_s01_overview and _verdict receive initial_equity parameter."""
        trades = [_make_trade()]
        # Should not raise — equity sections get the parameter
        report = generate_phase_diagnostics(trades, ["D6"], initial_equity=50_000.0)
        assert "Overview" in report

    def test_deduplication_across_modules(self):
        """Shared functions between modules only appear once."""
        trades = [_make_trade()]
        # D1 and D2 don't share functions, but D6 is always added
        report = generate_phase_diagnostics(trades, ["D1", "D2", "D6"])
        # Count overview occurrences — should be exactly 1
        assert report.count("1. Overview") == 1

    def test_empty_trades(self):
        """Empty trade list returns 'No trades' message."""
        report = generate_phase_diagnostics([], ["D1"])
        assert "No trades" in report

    def test_custom_title(self):
        """Custom title appears in output."""
        trades = [_make_trade()]
        report = generate_phase_diagnostics(trades, ["D6"], title="Phase 1 Test")
        assert "PHASE 1 TEST" in report

    def test_overview_first_verdict_last(self):
        """Overview appears before other sections; verdict is last section."""
        trades = [_make_trade(), _make_trade(pnl=-5, r_multiple=-0.5)]
        report = generate_phase_diagnostics(trades, ["D1"])
        overview_pos = report.find("1. Overview")
        verdict_pos = report.find("VERDICT")
        # D1 has MFE capture (section 3)
        mfe_pos = report.find("3.")
        assert overview_pos < mfe_pos, "Overview should appear before D1 sections"
        assert verdict_pos > mfe_pos, "Verdict should appear after D1 sections"

    def test_module_ids_in_header(self):
        """Module IDs appear in the header."""
        trades = [_make_trade()]
        report = generate_phase_diagnostics(trades, ["D1", "D4"])
        assert "D1" in report
        assert "D4" in report
        assert "D6" in report  # always included

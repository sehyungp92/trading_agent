from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backtests.swing.analysis.metrics import PerformanceMetrics
from backtests.swing.auto.greedy_optimize import (
    _drawdown_quality,
    _portfolio_synergy_score,
    _static_initial_strategy_net_profit,
)
from backtests.swing.auto.portfolio_synergy.run_latest_two_rounds import (
    _profit_factor,
    _static_initial_risk_return,
)
from backtests.swing.auto.scoring import composite_score
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.data.preprocessing import NumpyBars
from backtests.swing.engine.unified_portfolio_engine import (
    UnifiedPortfolioData,
    _combined_child_open_mtm_pnl,
    _normalised_trade_copy,
    _normalised_trade_pnl,
    _normalised_trade_r,
    _scaled_tpc_trade,
    _tpc_open_mtm_pnl,
)


def test_static_initial_risk_return_uses_strategy_slot_units() -> None:
    config = UnifiedBacktestConfig(initial_equity=25_000.0)
    config = replace(
        config,
        atrss=replace(config.atrss, unit_risk_pct=0.016),
        helix=replace(config.helix, unit_risk_pct=0.009),
        tpc=replace(config.tpc, unit_risk_pct=0.005),
    )
    strategy_results = {
        "ATRSS": SimpleNamespace(total_r=10.0),
        "AKC_HELIX": SimpleNamespace(total_r=20.0),
        "TPC": SimpleNamespace(total_r=-4.0),
    }

    pnl, by_strategy = _static_initial_risk_return(
        config,
        strategy_results,
        initial_equity=25_000.0,
    )

    assert pnl == pytest.approx(8_000.0)
    assert by_strategy["ATRSS"]["initial_unit_risk_dollars"] == pytest.approx(400.0)
    assert by_strategy["AKC_HELIX"]["initial_unit_risk_dollars"] == pytest.approx(225.0)
    assert by_strategy["TPC"]["initial_unit_risk_dollars"] == pytest.approx(125.0)
    assert by_strategy["ATRSS"]["static_risk_pnl"] == pytest.approx(4_000.0)
    assert by_strategy["AKC_HELIX"]["static_risk_pnl"] == pytest.approx(4_500.0)
    assert by_strategy["TPC"]["static_risk_pnl"] == pytest.approx(-500.0)


def test_greedy_static_net_profit_matches_portfolio_synergy_basis() -> None:
    config = UnifiedBacktestConfig(initial_equity=25_000.0)
    config = replace(
        config,
        atrss=replace(config.atrss, unit_risk_pct=0.016),
        helix=replace(config.helix, unit_risk_pct=0.009),
        tpc=replace(config.tpc, unit_risk_pct=0.005),
    )
    result = SimpleNamespace(
        strategy_results={
            "ATRSS": SimpleNamespace(total_r=10.0),
            "AKC_HELIX": SimpleNamespace(total_r=20.0),
            "TPC": SimpleNamespace(total_r=-4.0),
        }
    )

    assert _static_initial_strategy_net_profit(config, result, 25_000.0) == pytest.approx(8_000.0)


def test_composite_score_net_profit_override_takes_precedence() -> None:
    metrics = PerformanceMetrics(
        total_trades=100,
        profit_factor=2.0,
        max_drawdown_pct=0.10,
        calmar=2.0,
        net_profit=10.0,
    )

    compounded = composite_score(
        metrics,
        initial_equity=100.0,
        equity_curve=np.array([100.0, 400.0]),
    )
    static = composite_score(
        metrics,
        initial_equity=100.0,
        equity_curve=np.array([100.0, 400.0]),
        net_profit_override=0.0,
    )

    assert compounded.net_profit_component == pytest.approx(1.0)
    assert static.net_profit_component == pytest.approx(0.0)
    assert static.total < compounded.total


def test_composite_score_can_apply_stricter_drawdown_profile() -> None:
    base_kwargs = dict(
        total_trades=100,
        profit_factor=2.0,
        calmar=2.0,
        net_profit=400.0,
    )
    low_dd = composite_score(
        PerformanceMetrics(max_drawdown_pct=0.10, **base_kwargs),
        initial_equity=100.0,
        max_drawdown_hard_pct=0.18,
        drawdown_score_scale_pct=0.15,
        drawdown_penalty_start_pct=0.12,
        drawdown_penalty_full_pct=0.15,
        drawdown_penalty_weight=0.25,
    )
    high_dd = composite_score(
        PerformanceMetrics(max_drawdown_pct=0.16, **base_kwargs),
        initial_equity=100.0,
        max_drawdown_hard_pct=0.18,
        drawdown_score_scale_pct=0.15,
        drawdown_penalty_start_pct=0.12,
        drawdown_penalty_full_pct=0.15,
        drawdown_penalty_weight=0.25,
    )
    rejected = composite_score(
        PerformanceMetrics(max_drawdown_pct=0.181, **base_kwargs),
        initial_equity=100.0,
        max_drawdown_hard_pct=0.18,
        drawdown_score_scale_pct=0.15,
    )

    assert not low_dd.rejected
    assert not high_dd.rejected
    assert high_dd.total < low_dd.total
    assert rejected.rejected


def test_portfolio_synergy_score_rejects_high_drawdown_and_rewards_balance() -> None:
    config = UnifiedBacktestConfig(initial_equity=25_000.0)
    config = replace(
        config,
        atrss=replace(config.atrss, unit_risk_pct=0.016),
        helix=replace(config.helix, unit_risk_pct=0.009),
        tpc=replace(config.tpc, unit_risk_pct=0.005),
    )
    balanced_result = SimpleNamespace(
        combined_equity=np.array([25_000.0, 30_000.0, 45_000.0]),
        combined_timestamps=np.array(["2021-01-01", "2023-01-01", "2026-01-01"], dtype="datetime64[ns]"),
        strategy_results={
            "ATRSS": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=300, entries_accepted_by_portfolio=120),
            "AKC_HELIX": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=300, entries_accepted_by_portfolio=120),
            "TPC": SimpleNamespace(total_trades=80, total_r=80.0, entry_signals_fired=180, entries_accepted_by_portfolio=80),
        },
    )
    concentrated_result = SimpleNamespace(
        combined_equity=balanced_result.combined_equity,
        combined_timestamps=balanced_result.combined_timestamps,
        strategy_results={
            "ATRSS": SimpleNamespace(total_trades=1, total_r=1.0, entry_signals_fired=300, entries_accepted_by_portfolio=1),
            "AKC_HELIX": SimpleNamespace(total_trades=319, total_r=319.0, entry_signals_fired=400, entries_accepted_by_portfolio=319),
            "TPC": SimpleNamespace(total_trades=0, total_r=0.0, entry_signals_fired=180, entries_accepted_by_portfolio=0),
        },
    )
    metrics = PerformanceMetrics(
        total_trades=320,
        profit_factor=2.5,
        max_drawdown_pct=0.10,
        sharpe=1.6,
        net_profit=50_000.0,
    )
    kwargs = {"max_drawdown_hard_pct": 0.15, "min_profit_factor": 1.75, "min_trades": 120}

    balanced = _portfolio_synergy_score(
        metrics,
        config,
        balanced_result,
        25_000.0,
        net_profit_override=50_000.0,
        scoring_kwargs=kwargs,
    )
    concentrated = _portfolio_synergy_score(
        metrics,
        config,
        concentrated_result,
        25_000.0,
        net_profit_override=50_000.0,
        scoring_kwargs=kwargs,
    )
    rejected = _portfolio_synergy_score(
        PerformanceMetrics(total_trades=320, profit_factor=2.5, max_drawdown_pct=0.151),
        config,
        balanced_result,
        25_000.0,
        net_profit_override=50_000.0,
        scoring_kwargs=kwargs,
    )

    assert not balanced.rejected
    assert balanced.total > concentrated.total
    assert rejected.rejected


def test_portfolio_synergy_drawdown_comfort_stops_rewarding_unneeded_conservatism() -> None:
    assert _drawdown_quality(0.09, hard_dd=0.15, comfort_pct=0.12) == pytest.approx(1.0)
    assert _drawdown_quality(0.119, hard_dd=0.15, comfort_pct=0.12) == pytest.approx(1.0)
    assert _drawdown_quality(0.135, hard_dd=0.15, comfort_pct=0.12) == pytest.approx(0.5)


def test_portfolio_synergy_alpha_target_keeps_high_returns_discriminating() -> None:
    config = UnifiedBacktestConfig(initial_equity=25_000.0)
    result = SimpleNamespace(
        combined_equity=np.array([25_000.0, 45_000.0]),
        combined_timestamps=np.array(["2021-01-01", "2026-01-01"], dtype="datetime64[ns]"),
        strategy_results={
            "ATRSS": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=300, entries_accepted_by_portfolio=120),
            "AKC_HELIX": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=300, entries_accepted_by_portfolio=120),
            "TPC": SimpleNamespace(total_trades=80, total_r=80.0, entry_signals_fired=180, entries_accepted_by_portfolio=80),
        },
    )
    metrics = PerformanceMetrics(
        total_trades=320,
        profit_factor=2.5,
        max_drawdown_pct=0.10,
        sharpe=1.6,
        net_profit=0.0,
    )
    kwargs = {
        "max_drawdown_hard_pct": 0.15,
        "drawdown_comfort_pct": 0.12,
        "alpha_return_target_pct": 350.0,
        "min_profit_factor": 1.75,
        "min_trades": 120,
    }

    lower_return = _portfolio_synergy_score(
        metrics,
        config,
        result,
        25_000.0,
        net_profit_override=60_000.0,
        scoring_kwargs=kwargs,
    )
    higher_return = _portfolio_synergy_score(
        metrics,
        config,
        result,
        25_000.0,
        net_profit_override=70_000.0,
        scoring_kwargs=kwargs,
    )

    assert higher_return.total > lower_return.total


def test_portfolio_mtm_curve_uses_only_child_open_delta() -> None:
    engine_groups = [
        (
            "atrss",
            {
                "QQQ": SimpleNamespace(equity=100_000.0, equity_curve=[130_000.0]),
                "GLD": SimpleNamespace(equity=100_000.0, equity_curve=[85_000.0]),
            },
        )
    ]

    assert _combined_child_open_mtm_pnl(engine_groups) == pytest.approx(15_000.0)


def test_portfolio_trade_copy_uses_static_strategy_risk_basis() -> None:
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    config = replace(config, atrss=replace(config.atrss, unit_risk_pct=0.016))
    source_trade = SimpleNamespace(qty=10, pnl_dollars=9_999.0, r_multiple=2.0, portfolio_size_mult=0.5)

    normalised = _normalised_trade_copy("ATRSS", source_trade, config, 50_000.0)

    assert normalised.source_pnl_dollars == pytest.approx(9_999.0)
    assert normalised.source_r_multiple == pytest.approx(2.0)
    assert normalised.r_multiple == pytest.approx(1.0)
    assert normalised.pnl_dollars == pytest.approx(800.0)
    assert normalised.portfolio_normalised is True


def test_scaled_tpc_trade_preserves_source_r_and_reports_portfolio_risk_basis() -> None:
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    config = replace(config, tpc=replace(config.tpc, unit_risk_pct=0.004))
    source_trade = SimpleNamespace(
        symbol="QQQ",
        qty=100,
        entry_price=100.0,
        initial_stop=99.0,
        pnl_dollars=1_000.0,
        commission=20.0,
        r_multiple=2.0,
    )

    adjusted, ratio, new_qty = _scaled_tpc_trade(source_trade, 0.55)
    normalised = _normalised_trade_copy("TPC", adjusted, config, 50_000.0)

    assert ratio == pytest.approx(0.55)
    assert new_qty == 55
    assert adjusted.qty == 55
    assert adjusted.pnl_dollars == pytest.approx(550.0)
    assert adjusted.commission == pytest.approx(11.0)
    assert adjusted.r_multiple == pytest.approx(2.0)
    assert adjusted.portfolio_size_mult == pytest.approx(0.55)
    assert _normalised_trade_r(adjusted) == pytest.approx(1.1)
    assert _normalised_trade_pnl("TPC", adjusted, config, 50_000.0) == pytest.approx(220.0)
    assert normalised.source_pnl_dollars == pytest.approx(550.0)
    assert normalised.source_r_multiple == pytest.approx(2.0)
    assert normalised.pnl_dollars == pytest.approx(220.0)


def test_portfolio_synergy_profit_factor_is_fee_net() -> None:
    trades = [
        SimpleNamespace(pnl_dollars=100.0, commission=10.0),
        SimpleNamespace(pnl_dollars=-50.0, commission=5.0),
    ]

    assert _profit_factor(trades) == pytest.approx(90.0 / 55.0)


def test_tpc_open_mtm_uses_accepted_trade_size_once() -> None:
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    config = replace(config, tpc=replace(config.tpc, unit_risk_pct=0.004))
    source_trade = SimpleNamespace(
        symbol="QQQ",
        qty=100,
        direction=1,
        entry_price=100.0,
        initial_stop=99.0,
        pnl_dollars=0.0,
        commission=0.0,
        r_multiple=0.0,
    )
    accepted_trade, ratio, _ = _scaled_tpc_trade(source_trade, 0.5)
    data = UnifiedPortfolioData(
        etf_15m={
            "QQQ": NumpyBars(
                opens=np.array([100.0]),
                highs=np.array([102.0]),
                lows=np.array([99.5]),
                closes=np.array([102.0]),
                volumes=np.array([1_000.0]),
                times=np.array(["2024-01-02T15:00:00"], dtype="datetime64[ns]"),
            )
        }
    )
    ts_key = int(data.etf_15m["QQQ"].times.astype("datetime64[ns]").astype(np.int64)[0])

    mtm = _tpc_open_mtm_pnl(
        data,
        {1: accepted_trade},
        {1: 100.0 * ratio},
        config,
        50_000.0,
        ts_key,
    )

    assert mtm == pytest.approx(200.0)


def test_portfolio_synergy_capture_uses_entry_requests_when_present() -> None:
    config = UnifiedBacktestConfig(initial_equity=25_000.0)
    metrics = PerformanceMetrics(
        total_trades=320,
        profit_factor=2.5,
        max_drawdown_pct=0.10,
        sharpe=1.6,
        net_profit=0.0,
    )
    kwargs = {
        "max_drawdown_hard_pct": 0.15,
        "min_profit_factor": 1.75,
        "min_trades": 120,
        "accept_rate_target": 0.50,
    }
    fallback_result = SimpleNamespace(
        combined_timestamps=np.array(["2021-01-01", "2026-01-01"], dtype="datetime64[ns]"),
        strategy_results={
            "ATRSS": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=120, entries_accepted_by_portfolio=120),
            "AKC_HELIX": SimpleNamespace(total_trades=120, total_r=120.0, entry_signals_fired=120, entries_accepted_by_portfolio=120),
            "TPC": SimpleNamespace(total_trades=80, total_r=80.0, entry_signals_fired=80, entries_accepted_by_portfolio=80),
        },
    )
    request_result = SimpleNamespace(
        combined_timestamps=fallback_result.combined_timestamps,
        strategy_results={
            sid: SimpleNamespace(
                total_trades=sr.total_trades,
                total_r=sr.total_r,
                entry_signals_fired=sr.entry_signals_fired,
                entry_requests=sr.entry_signals_fired * 4,
                entries_accepted_by_portfolio=sr.entries_accepted_by_portfolio,
            )
            for sid, sr in fallback_result.strategy_results.items()
        },
    )

    fallback = _portfolio_synergy_score(
        metrics,
        config,
        fallback_result,
        25_000.0,
        net_profit_override=60_000.0,
        scoring_kwargs=kwargs,
    )
    request_adjusted = _portfolio_synergy_score(
        metrics,
        config,
        request_result,
        25_000.0,
        net_profit_override=60_000.0,
        scoring_kwargs=kwargs,
    )

    assert request_adjusted.total < fallback.total

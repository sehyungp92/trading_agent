from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from backtests.momentum.analysis.family_portfolio_diagnostics import _portfolio_mtm_metrics
from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
    SCORE_WEIGHTS,
    apply_portfolio_mutations,
    score_metrics,
)
from backtests.momentum.engine.family_portfolio_engine import (
    FamilyDynamicRiskConfig,
    FamilyPortfolioBacktester,
    FamilyPortfolioResult,
    FamilyPortfolioTrade,
    FamilySignalFilterCondition,
    FamilySignalFilterRule,
    make_controlled_aggressive_family_config,
)
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig


@pytest.mark.asyncio
async def test_portfolio_rule_checker_uses_replay_clock_for_nqdtc_direction_filter() -> None:
    now = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)  # 10:00 ET
    config = PortfolioRulesConfig(
        nqdtc_direction_filter_enabled=True,
        nqdtc_oppose_size_mult=0.0,
        directional_cap_R=0,
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value={
            "last_direction": "SHORT",
            "signal_date": now.date(),
        }),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 50_000.0,
        now_provider=lambda: now,
    )

    result = await checker.check_entry("VdubusNQ_v4", "LONG", 1.0)

    assert not result.approved
    assert "nqdtc_direction_filter" in result.denial_reason


def test_family_portfolio_replay_uses_all_four_and_shared_rule_blocks() -> None:
    cfg = make_controlled_aggressive_family_config(50_000.0)
    cfg = replace(
        cfg,
        max_total_positions=1,
        strategy_allocations=tuple(
            replace(allocation, base_risk_pct=0.001)
            for allocation in cfg.strategy_allocations
        ),
        rules=replace(cfg.rules, directional_cap_R=0, max_family_contracts_mnq_eq=0),
    )
    trades = {
        "NQ_REGIME": [_trade("BUY", 1, 10.0)],
        "VdubusNQ_v4": [_trade("BUY", 1, 11.0)],
        "NQDTC_v2.1": [_trade("BUY", 1, 12.0)],
        "DownturnDominator_v1": [_trade("SELL", -1, 13.0)],
    }

    result = FamilyPortfolioBacktester(cfg).run(trades)

    assert result.metrics["active_strategies"] >= 1
    assert result.rule_blocks["max_total_positions"] == 3
    assert result.metrics["blocked_trades"] == 3
    assert result.replay_architecture == "canonical_replay_bundle_live_rule_adapter"
    assert result.action_count == 8
    assert result.trade_outcome_count == 4
    assert result.replay_source_fingerprint


def test_family_score_has_no_more_than_seven_components() -> None:
    assert len(SCORE_WEIGHTS) <= 7
    scored = score_metrics({
        "net_profit": 60_000.0,
        "trades_per_month": 24.0,
        "max_drawdown_pct": 0.10,
        "profit_factor": 2.4,
        "calmar": 4.0,
        "active_strategies": 4.0,
        "min_strategy_trades": 25.0,
        "block_rate": 0.20,
        "max_concurrent": 4.0,
    })

    assert len(scored["components"]) == len(SCORE_WEIGHTS)
    assert not scored["rejected"]


def test_family_score_treats_frequency_shortfall_as_soft_warning() -> None:
    scored = score_metrics({
        "net_profit": 60_000.0,
        "trades_per_month": 15.0,
        "max_drawdown_pct": 0.10,
        "profit_factor": 2.4,
        "calmar": 4.0,
        "active_strategies": 4.0,
        "min_strategy_trades": 25.0,
        "block_rate": 0.20,
        "max_concurrent": 4.0,
    })

    assert not scored["rejected"]
    assert "frequency_below_18_trades_per_month" in scored["soft_warnings"]
    assert scored["score"] > 0.0


def test_family_mutations_can_update_live_rules_and_allocations() -> None:
    cfg = make_controlled_aggressive_family_config(50_000.0)
    updated = apply_portfolio_mutations(
        cfg,
        {
            "allocation.NQ_REGIME.base_risk_pct": 0.007,
            "config.heat_cap_R": 5.25,
            "rules.directional_cap_R": 3.75,
        },
    )

    assert updated.allocation_for("NQ_REGIME").base_risk_pct == 0.007
    assert updated.heat_cap_R == 5.25
    assert updated.rules.directional_cap_R == 3.75

    dotted = apply_portfolio_mutations(
        cfg,
        {
            "allocation.NQDTC_v2.1.max_concurrent": 2,
        },
    )

    assert dotted.allocation_for("NQDTC_v2.1").max_concurrent == 2


def test_dynamic_risk_can_fit_trade_to_remaining_heat() -> None:
    cfg = make_controlled_aggressive_family_config(50_000.0)
    cfg = replace(
        cfg,
        heat_cap_R=1.0,
        dynamic_risk=FamilyDynamicRiskConfig(
            enabled=True,
            fit_to_remaining_heat=True,
            max_trade_risk_R=1.0,
        ),
        strategy_allocations=tuple(
            replace(allocation, base_risk_pct=0.006, max_concurrent=3)
            for allocation in cfg.strategy_allocations
        ),
        rules=replace(cfg.rules, directional_cap_R=0, max_family_contracts_mnq_eq=0),
    )
    trade = _trade("BUY", 1, 10.0)
    trade.initial_stop = 0.0
    trades = {"NQ_REGIME": [trade]}

    result = FamilyPortfolioBacktester(cfg).run(trades)

    assert result.metrics["total_trades"] == 1
    assert result.trades[0].normalized_risk_R <= 1.0
    assert result.trades[0].metadata["portfolio_entry_context"]["initial_base_qty"] > result.trades[0].portfolio_qty


def test_signal_filter_rules_block_matching_entry_metadata() -> None:
    cfg = make_controlled_aggressive_family_config(50_000.0)
    cfg = replace(
        cfg,
        signal_filter_rules=(
            FamilySignalFilterRule(
                name="block_low_score_nqdtc",
                strategy_id="NQDTC_v2.1",
                conditions=(
                    FamilySignalFilterCondition("metadata.score_at_entry", "lte", 2.5),
                ),
            ),
        ),
        rules=replace(cfg.rules, directional_cap_R=0, max_family_contracts_mnq_eq=0),
    )
    trade = _trade("BUY", 1, 10.0)
    trade.score_at_entry = 2.5
    trades = {"NQDTC_v2.1": [trade]}

    result = FamilyPortfolioBacktester(cfg).run(trades)

    assert result.metrics["total_trades"] == 0
    assert result.rule_blocks["signal_filter:block_low_score_nqdtc"] == 1


def test_family_replay_bundle_matches_legacy_trade_input_path() -> None:
    cfg = make_controlled_aggressive_family_config(50_000.0)
    cfg = replace(cfg, rules=replace(cfg.rules, directional_cap_R=0, max_family_contracts_mnq_eq=0))
    trades = {
        "NQ_REGIME": [_trade("BUY", 1, 10.0)],
        "VdubusNQ_v4": [_trade("SELL", -1, 11.0)],
    }
    backtester = FamilyPortfolioBacktester(cfg)

    legacy_result = backtester.run(trades)
    bundle = backtester.build_replay_bundle(trades)
    bundle_result = backtester.run_bundle(bundle)
    replay_contract = bundle.metadata["replay_contract"]

    assert bundle.source_fingerprint == legacy_result.replay_source_fingerprint
    assert bundle_result.replay_source_fingerprint == bundle.source_fingerprint
    assert bundle_result.metrics == legacy_result.metrics
    assert bundle_result.strategy_trade_counts == legacy_result.strategy_trade_counts
    assert replay_contract["uses_live_portfolio_rules"] is True
    assert replay_contract["source_strategy_execution_simulation"] is False
    assert replay_contract["decision_stream_status"] == "not_provided_completed_trade_replay"
    assert bundle_result.replay_bundle_metadata["replay_contract"] == replay_contract
    assert [trade.adjusted_pnl for trade in bundle_result.trades] == [
        trade.adjusted_pnl for trade in legacy_result.trades
    ]


def test_portfolio_diagnostics_max_dd_uses_bar_close_mark_to_market() -> None:
    cfg = make_controlled_aggressive_family_config(1_000.0)
    entry = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    peak = datetime(2025, 1, 2, 15, 5, tzinfo=timezone.utc)
    trough = datetime(2025, 1, 2, 15, 10, tzinfo=timezone.utc)
    exit_time = datetime(2025, 1, 2, 15, 15, tzinfo=timezone.utc)
    trade = FamilyPortfolioTrade(
        strategy_id="NQ_REGIME",
        direction=1,
        entry_time=entry,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=110.0,
        initial_stop=95.0,
        raw_pnl_dollars=20.0,
        raw_qty=1,
        r_multiple=1.0,
        portfolio_qty=1,
        adjusted_pnl=20.0,
        risk_dollars=10.0,
    )
    result = FamilyPortfolioResult(
        trades=[trade],
        initial_equity=1_000.0,
        metrics={
            "net_profit": 20.0,
            "net_return_pct": 0.02,
            "max_drawdown_pct": 0.0,
            "cagr": 0.0,
            "calmar": 0.0,
        },
    )
    price_bars = {
        "source": "synthetic",
        "timeframe": "5m",
        "bars": pd.DataFrame(
            {"close": [100.0, 150.0, 100.0, 110.0]},
            index=pd.DatetimeIndex([entry, peak, trough, exit_time]),
        ),
    }

    metrics = _portfolio_mtm_metrics(cfg, result, price_bars)

    assert metrics["risk_basis"] == "bar_close_mark_to_market"
    assert metrics["max_drawdown_pct"] == pytest.approx(100.0 / 1100.0)
    assert metrics["realized_daily_max_drawdown_pct"] == 0.0


def test_portfolio_diagnostics_reconciles_same_timestamp_trade_pnl() -> None:
    cfg = make_controlled_aggressive_family_config(1_000.0)
    ts = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    trade = FamilyPortfolioTrade(
        strategy_id="NQ_REGIME",
        direction=1,
        entry_time=ts,
        exit_time=ts,
        entry_price=100.0,
        exit_price=95.0,
        initial_stop=95.0,
        raw_pnl_dollars=-10.0,
        raw_qty=1,
        r_multiple=-1.0,
        portfolio_qty=1,
        adjusted_pnl=-10.0,
        risk_dollars=10.0,
    )
    result = FamilyPortfolioResult(
        trades=[trade],
        initial_equity=1_000.0,
        equity_curve=pd.Series([1_000.0]).to_numpy(),
        metrics={"net_return_pct": 0.0, "max_drawdown_pct": 0.0, "cagr": 0.0, "calmar": 0.0},
    )
    price_bars = {
        "source": "synthetic",
        "timeframe": "5m",
        "bars": pd.DataFrame({"close": [95.0]}, index=pd.DatetimeIndex([ts])),
    }

    metrics = _portfolio_mtm_metrics(cfg, result, price_bars)

    assert metrics["final_equity"] == 990.0
    assert metrics["net_return_pct"] == pytest.approx(-0.01)
    assert metrics["same_timestamp_trade_count"] == 1
    assert metrics["same_timestamp_adjusted_pnl"] == -10.0


def _trade(side: str, direction: int, minute: float):
    entry = datetime(2025, 1, 2, 15, int(minute), tzinfo=timezone.utc)
    exit_time = datetime(2025, 1, 2, 16, int(minute), tzinfo=timezone.utc)
    return SimpleNamespace(
        side=side,
        direction=direction,
        entry_time=entry,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=102.0 if direction > 0 else 98.0,
        initial_stop=98.0 if direction > 0 else 102.0,
        pnl_dollars=20.0,
        pnl=20.0,
        qty=5,
        entry_contracts=5,
        avg_entry=100.0,
        r_multiple=1.0,
        commission=6.2,
    )

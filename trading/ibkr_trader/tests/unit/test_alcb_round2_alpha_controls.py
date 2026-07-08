from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backtests.stock.auto.alcb.time_utils import hydrate_time_mutations
from backtests.stock.auto.alcb.phase_candidates import (
    PHASE_FOCUS,
    get_phase_candidates,
    is_small_sample_overfit_candidate,
    sanitize_round2_seed,
)
from backtests.stock.auto.config_mutator import mutate_alcb_config
from backtests.stock.auto.alcb.phase_scoring import (
    compute_alcb_phase_metrics,
    enrich_alcb_phase_metrics,
    score_alcb_phase,
)
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.auto.alcb.worker import phase_reject_reason
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine, _PendingEntry
from strategies.stock.alcb.config import StrategySettings
from strategies.stock.alcb.models import Direction, MomentumSetup
from strategies.stock.alcb.risk import (
    conditional_entry_blocked,
    conditional_entry_size_mult,
)


def _trade(
    r_multiple: float,
    *,
    hold_bars: int,
    exit_reason: str = "EOD_FLATTEN",
    momentum_score: int = 5,
    rvol: float = 2.5,
    risk_dollars: float = 100.0,
):
    entry = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        r_multiple=r_multiple,
        hold_bars=hold_bars,
        exit_reason=exit_reason,
        entry_type="OR_BREAKOUT",
        entry_time=entry,
        fill_time=entry,
        exit_time=entry + timedelta(minutes=hold_bars * 5),
        entry_price=10.0,
        risk_per_share=1.0,
        quantity=risk_dollars,
        metadata={
            "momentum_score": momentum_score,
            "rvol_at_entry": rvol,
            "avwap_at_entry": 9.98,
            "or_high": 10.0,
            "or_low": 9.9,
            "mfe_r": max(r_multiple, 0.0) + 0.2,
        },
    )


class _Pos(SimpleNamespace):
    def unrealized_r(self, price: float) -> float:
        return (price - self.entry_price) / self.risk_per_share


def _bar(close: float, *, high: float | None = None, low: float | None = None, volume: float = 100.0):
    high_value = high if high is not None else close
    low_value = low if low is not None else close
    return SimpleNamespace(
        open=close,
        high=high_value,
        low=low_value,
        close=close,
        volume=volume,
    )


def test_round2_seed_preserves_previous_optimized_config_exactly():
    seed = {
        "param_overrides.sector_mult_consumer_disc": 1.2,
        "param_overrides.sector_mult_financials": 0.5,
        "param_overrides.sector_mult_industrials": 0.5,
        "param_overrides.momentum_size_mult_score_4": 1.15,
        "param_overrides.momentum_size_mult_score_7_plus": 1.25,
        "param_overrides.breakout_distance_cap_r": 1.1,
    }

    sanitized = sanitize_round2_seed(seed)

    assert sanitized == seed
    assert sanitized is not seed


def test_round2_candidates_exclude_small_sample_sector_and_weekday_fits():
    sector_weekday_candidate = {
        "param_overrides.tuesday_sizing_mult": 0.75,
        "param_overrides.sector_entry_size_mults": {
            "Financials:OR_BREAKOUT": 0.25,
        },
    }

    assert is_small_sample_overfit_candidate("r2_tuesday075_fin025", sector_weekday_candidate)

    all_candidates = [
        (name, mutations)
        for phase in PHASE_FOCUS
        for name, mutations in get_phase_candidates(phase)
    ]

    assert all_candidates
    assert not any(
        is_small_sample_overfit_candidate(name, mutations)
        for name, mutations in all_candidates
    )
    assert "r2_consumer_disc_120_fin025" not in {name for name, _ in all_candidates}
    assert "r2_tuesday075_fin025" not in {name for name, _ in all_candidates}


def test_round2_optimized_param_overrides_match_live_defaults():
    optimized_path = Path("backtests/output/stock/alcb/round_2/optimized_config.json")
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    backtest_config = mutate_alcb_config(ALCBBacktestConfig(), hydrate_time_mutations(optimized))
    backtest_settings = StrategySettings(**backtest_config.param_overrides)
    live_settings = StrategySettings()

    mismatches = {}
    for key in optimized:
        if not key.startswith("param_overrides."):
            continue
        field = key.split(".", 1)[1]
        backtest_value = getattr(backtest_settings, field)
        live_value = getattr(live_settings, field)
        if backtest_value != live_value:
            mismatches[field] = (backtest_value, live_value)

    assert mismatches == {}


def test_conditional_entry_blocklist_uses_completed_bar_cohorts():
    settings = StrategySettings(
        block_entry_bars=(9,),
        entry_score_blocklist=("COMBINED_BREAKOUT:5", "*:3"),
        sector_entry_blocklist=("Financials:OR_BREAKOUT",),
    )

    assert conditional_entry_blocked("Consumer Discretionary", "OR_BREAKOUT", 9, 6, settings)
    assert conditional_entry_blocked("Industrials", "COMBINED_BREAKOUT", 12, 5, settings)
    assert conditional_entry_blocked("Healthcare", "PDH_BREAKOUT", 12, 3, settings)
    assert conditional_entry_blocked("Financials", "OR_BREAKOUT", 14, 6, settings)
    assert not conditional_entry_blocked("Financials", "PDH_BREAKOUT", 14, 6, settings)


def test_conditional_entry_size_mult_combines_bar_score_and_sector_overlays():
    settings = StrategySettings(
        entry_bar_size_mults={"11": 0.50},
        entry_score_size_mults={"OR_BREAKOUT:5": 0.80},
        sector_entry_size_mults={"Financials:OR_BREAKOUT": 0.25},
    )

    mult = conditional_entry_size_mult("Financials", "OR_BREAKOUT", 11, 5, settings)

    # 0.50 (bar=11) * 0.80 (OR_BREAKOUT:5) * 0.55 (default detail !bar_vol_surge) * 0.25 (sector)
    assert mult == pytest.approx(0.055)


def test_conditional_entry_detail_controls_use_completed_score_components():
    settings = StrategySettings(
        entry_detail_blocklist=("OR_BREAKOUT:5:!bar_vol_surge",),
        entry_detail_size_mults={
            "*:5:!adx_trending": 0.70,
            "OR_BREAKOUT:strong_cpr": 1.10,
        },
    )

    weak_detail = {"strong_cpr": 1}
    strong_detail = {"bar_vol_surge": 1, "adx_trending": 1, "strong_cpr": 1}

    assert conditional_entry_blocked("Technology", "OR_BREAKOUT", 12, 5, settings, weak_detail)
    assert not conditional_entry_blocked("Technology", "OR_BREAKOUT", 12, 5, settings, strong_detail)
    # 0.75 (default OR_BREAKOUT:5 score mult) * 0.70 (!adx_trending) * 1.10 (strong_cpr) = 0.5775
    assert round(conditional_entry_size_mult("Technology", "OR_BREAKOUT", 12, 5, settings, weak_detail), 2) == 0.58
    # 0.75 (default OR_BREAKOUT:5) * 1.10 (strong_cpr) = 0.825
    assert conditional_entry_size_mult("Technology", "OR_BREAKOUT", 12, 5, settings, strong_detail) == pytest.approx(0.825)


def test_entry_confirmation_uses_completed_confirmation_bar_not_fill_bar():
    engine = object.__new__(ALCBIntradayEngine)
    setup = MomentumSetup(
        symbol="TST",
        or_high=10.0,
        or_low=9.5,
        or_volume=1000.0,
        prior_day_high=10.2,
        prior_day_low=9.0,
        prior_day_close=9.8,
        breakout_level=10.0,
        entry_type="OR_BREAKOUT",
        rvol_at_entry=2.5,
        momentum_score=5,
        score_detail={},
        avwap_at_entry=9.95,
    )
    pending = _PendingEntry(
        symbol="TST",
        item=SimpleNamespace(sector="Technology"),
        entry_type="OR_BREAKOUT",
        signal_time=datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc),
        signal_bar_index=0,
        signal_price=10.0,
        signal_low=9.8,
        signal_risk_per_share=1.0,
        or_low=9.5,
        daily_atr=1.0,
        avwap_at_signal=9.95,
        expected_volume_5m=100.0,
        signal_rvol=2.5,
        momentum_score=5,
        score_detail={},
        orb_quality_score=70.0,
        gap_size_mult=1.0,
        time_size_mult=1.0,
        setup=setup,
        regime_tier="A",
        opened_date=datetime(2025, 1, 2, tzinfo=timezone.utc).date(),
        reentry_sequence=0,
    )
    settings = StrategySettings(
        entry_confirmation_bars=1,
        entry_confirmation_min_current_r=0.05,
        entry_confirmation_min_mfe_r=0.10,
        entry_confirmation_max_mae_r=0.50,
        entry_confirmation_min_rvol_ratio=0.30,
        entry_confirmation_require_above_breakout=True,
        entry_confirmation_require_above_avwap=True,
    )
    session_bars = {
        "TST": [
            _bar(10.0, high=10.1, low=9.9, volume=250.0),
            _bar(10.08, high=10.14, low=9.94, volume=120.0),
            _bar(9.70, high=9.75, low=9.60, volume=10.0),
        ]
    }

    assert engine._entry_confirmation_passed(pending, session_bars, settings)


def test_maturation_stop_requires_configured_failed_check_count():
    engine = object.__new__(ALCBIntradayEngine)
    setup = SimpleNamespace(breakout_level=10.0)
    pos = _Pos(
        symbol="TST",
        direction=Direction.LONG,
        entry_price=10.0,
        risk_per_share=1.0,
        max_favorable=10.08,
        max_adverse=9.82,
        momentum_setup=setup,
        entry_expected_volume_5m=100.0,
        entry_signal_rvol=2.5,
    )
    bar = _bar(9.98, high=10.02, low=9.92, volume=70.0)
    settings = StrategySettings(
        maturation_stop_bars=3,
        maturation_stop_min_failed_checks=2,
        maturation_stop_min_current_r=0.00,
        maturation_stop_min_mfe_r=0.05,
        maturation_stop_min_rvol_ratio=0.40,
        maturation_stop_require_above_breakout=True,
    )

    assert engine._maturation_failed(pos, bar, {"TST": [bar]}, settings)

    settings_one_fail_short = StrategySettings(
        maturation_stop_bars=3,
        maturation_stop_min_failed_checks=2,
        maturation_stop_min_current_r=0.00,
    )

    assert not engine._maturation_failed(pos, bar, {"TST": [bar]}, settings_one_fail_short)


def test_profit_protection_tracks_round1_short_hold_and_bad_exit_cohorts():
    trades = [
        _trade(-1.0, hold_bars=20, exit_reason="FLOW_REVERSAL"),
        _trade(-0.8, hold_bars=16, exit_reason="MFE_CONVICTION"),
        _trade(1.2, hold_bars=30),
    ]

    metrics = enrich_alcb_phase_metrics({
        **compute_alcb_phase_metrics(trades),
        "total_trades": len(trades),
        "expectancy": sum(t.r_multiple for t in trades) / len(trades),
        "max_drawdown_pct": 0.02,
    })

    assert metrics["short_hold_24_total_r"] == -1.8
    assert metrics["flow_reversal_total_r"] == -1.0
    assert metrics["mfe_conviction_total_r"] == -0.8
    assert metrics["profit_protection"] < 1.0
    assert metrics["flow_mfe_exit_inverse"] < 1.0


def test_sizing_alignment_penalizes_losers_sized_larger_than_winners():
    trades = [
        _trade(1.0, hold_bars=30, risk_dollars=80.0),
        _trade(-0.5, hold_bars=12, risk_dollars=120.0),
    ]

    metrics = compute_alcb_phase_metrics(trades)

    assert metrics["winner_avg_risk_dollars"] == 80.0
    assert metrics["loser_avg_risk_dollars"] == 120.0
    assert metrics["sizing_alignment"] < 1.0


def test_phase4_rejects_candidates_that_do_not_repair_short_hold_drag():
    metrics = enrich_alcb_phase_metrics({
        "total_trades": 100,
        "expectancy": 1.1,
        "net_profit": 9000.0,
        "expectancy_dollar": 15.5,
        "trades_per_month": 22.0,
        "profit_factor": 1.8,
        "max_drawdown_pct": 0.03,
        "short_hold_24_total_r": -190.0,
        "flow_reversal_total_r": -10.0,
        "mfe_conviction_total_r": -10.0,
    })

    reason = phase_reject_reason(
        metrics,
        {"min_short_hold_24_drag_inverse": 0.30},
        phase=4,
    )

    assert "short_hold_24_drag_inverse" in reason


def test_phase4_score_rewards_real_frequency_when_return_quality_is_preserved():
    base_metrics = enrich_alcb_phase_metrics({
        "total_trades": 560,
        "expectancy": 0.195,
        "net_profit": 9000.0,
        "expectancy_dollar": 16.0,
        "trades_per_month": 21.5,
        "profit_factor": 1.8,
        "max_drawdown_pct": 0.03,
        "late_entry_quality": 0.82,
        "timing_quality": 0.60,
        "signal_quality": 0.60,
    })
    higher_frequency = dict(base_metrics)
    higher_frequency.update({
        "total_trades": 610,
        "trades_per_month": 23.4,
        "net_profit": 9800.0,
    })
    higher_frequency = enrich_alcb_phase_metrics(higher_frequency)

    assert score_alcb_phase(4, higher_frequency) > score_alcb_phase(4, base_metrics)

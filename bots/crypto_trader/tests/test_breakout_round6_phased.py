"""Tests for the breakout round-6 phased optimization module."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.breakout_round6_phased import (
    ROUND6_HARD_REJECTS,
    ROUND6_IMMUTABLE_SCORING_WEIGHTS,
    ROUND6_PHASE_GATE_CRITERIA,
    BreakoutRound6PhasedPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    load_breakout_strategy,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig


def _make_plugin() -> BreakoutRound6PhasedPlugin:
    bc = BacktestConfig(
        start_date=date(2026, 1, 4),
        end_date=date(2026, 4, 20),
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10000.0,
    )
    return BreakoutRound6PhasedPlugin(
        backtest_config=bc,
        base_config=BreakoutConfig(),
        data_dir=Path("data"),
        max_workers=2,
    )


def test_weights_sum_to_one():
    assert sum(ROUND6_IMMUTABLE_SCORING_WEIGHTS.values()) == pytest.approx(1.0)


def test_hard_rejects_protect_sample_and_quality():
    assert ROUND6_HARD_REJECTS["total_trades"] == (">=", 14.0)
    assert ROUND6_HARD_REJECTS["profit_factor"] == (">=", 1.15)
    assert ROUND6_HARD_REJECTS["max_drawdown_pct"] == ("<=", 12.0)


def test_phase1_candidates_prioritize_capture_harvesting():
    names = [c.name for c in _phase1_candidates()]
    assert "tp1_frac_025" in names
    assert "trail_tight_009" in names
    assert "tp1_025_trail_009" in names


def test_phase2_candidates_include_selective_relaxed_body_branch():
    candidates = {c.name: c for c in _phase2_candidates()}
    assert "relaxed_strong_pockets" in candidates
    assert "relaxed_btc_sol_short" in candidates
    relaxed = candidates["relaxed_strong_pockets"]
    assert relaxed.mutations["setup.relaxed_body_enabled"] is True
    assert relaxed.mutations["symbol_filter.eth_relaxed_body_direction"] == "long_only"
    assert relaxed.mutations["symbol_filter.sol_relaxed_body_direction"] == "short_only"


def test_phase3_candidates_focus_on_zone_age_expansion():
    names = [c.name for c in _phase3_candidates()]
    assert names == ["zone_age_28", "zone_age_30", "zone_age_32", "zone_age_36"]


def test_phase4_candidates_pressure_test_entry_controls():
    names = [c.name for c in _phase4_candidates()]
    assert "retest_rejection_on" in names
    assert "strict_retest_stack" in names
    assert "model2_break_entry" in names
    assert "model2_off" in names


def test_phase5_candidates_skip_risk_finetuning():
    candidates = _phase5_candidates(
        {
            "exits.tp1_frac": 0.25,
            "balance.max_zone_age_bars": 30,
            "risk.risk_pct_b": 0.02,
            "setup.relaxed_body_risk_scale": 0.5,
        }
    )
    names = [c.name for c in candidates]
    assert any(name.startswith("finetune_tp1_frac_") for name in names)
    assert any(name.startswith("finetune_max_zone_age_bars_") for name in names)
    assert not any("risk_pct_b" in name for name in names)
    assert not any("risk_scale" in name for name in names)


def test_phase_gate_criteria_tighten_into_later_phases():
    assert any(g.metric == "net_return_pct" for g in ROUND6_PHASE_GATE_CRITERIA[1])
    assert any(g.metric == "total_trades" and g.threshold == 16.0 for g in ROUND6_PHASE_GATE_CRITERIA[3])
    assert any(g.metric == "profit_factor" and g.threshold == 1.75 for g in ROUND6_PHASE_GATE_CRITERIA[5])


def test_plugin_uses_five_phases_and_immutable_weights():
    plugin = _make_plugin()
    spec = plugin.get_phase_spec(1, state=None)
    assert plugin.num_phases == 5
    assert spec.name == "Capture & Monetization"
    assert spec.scoring_weights == ROUND6_IMMUTABLE_SCORING_WEIGHTS
    assert spec.max_rounds == 4


def test_load_breakout_strategy_round_trip(tmp_path: Path):
    path = tmp_path / "optimized_config.json"
    payload = {
        "strategy": BreakoutConfig().to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    cfg = load_breakout_strategy(path)
    assert isinstance(cfg, BreakoutConfig)
    assert cfg.symbols == ["BTC", "ETH", "SOL"]

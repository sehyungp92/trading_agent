"""Tests for the breakout round-5 phased optimization module."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.breakout_round5_phased import (
    IMMUTABLE_HARD_REJECTS,
    IMMUTABLE_SCORING_WEIGHTS,
    PHASE_GATE_CRITERIA,
    BreakoutRound5PhasedPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    load_breakout_strategy,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig


def _make_plugin() -> BreakoutRound5PhasedPlugin:
    bc = BacktestConfig(
        start_date=date(2026, 1, 4),
        end_date=date(2026, 4, 20),
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10000.0,
    )
    return BreakoutRound5PhasedPlugin(
        backtest_config=bc,
        base_config=BreakoutConfig(),
        data_dir=Path("data"),
        max_workers=2,
    )


def test_weights_sum_to_one():
    assert sum(IMMUTABLE_SCORING_WEIGHTS.values()) == pytest.approx(1.0)


def test_hard_rejects_prioritize_quality_and_sample():
    assert IMMUTABLE_HARD_REJECTS["total_trades"] == (">=", 14.0)
    assert IMMUTABLE_HARD_REJECTS["profit_factor"] == (">=", 1.10)
    assert IMMUTABLE_HARD_REJECTS["max_drawdown_pct"] == ("<=", 12.0)


def test_phase1_candidates_include_symbol_and_quality_filters():
    names = [c.name for c in _phase1_candidates()]
    assert "eth_long_only" in names
    assert "sol_short_only" in names
    assert "require_vol_surge" in names
    assert "room_r_b_16" in names


def test_phase2_candidates_include_strict_retest_stacks():
    candidates = {c.name: c for c in _phase2_candidates()}
    assert "strict_retest_stack" in candidates
    assert candidates["strict_retest_stack"].mutations["confirmation.retest_require_rejection"] is True
    assert candidates["strict_retest_stack"].mutations["confirmation.retest_require_volume_decline"] is True
    assert "model2_break_entry_strict" in candidates


def test_phase3_candidates_include_capture_repairs():
    names = [c.name for c in _phase3_candidates()]
    assert "quick_exit_on" in names
    assert "structure_trail_off" in names
    assert "tp1_frac_025" in names
    assert "stop_use_farther_off" in names


def test_phase4_candidates_include_structural_frequency_expansion():
    names = [c.name for c in _phase4_candidates()]
    assert "min_bars_zone_4" in names
    assert "lookback_48" in names
    assert "reentry_max_2" in names
    assert "max_trades_per_day_5" in names


def test_phase5_candidates_are_risk_last():
    names = [c.name for c in _phase5_candidates()]
    assert "risk_b_023" in names
    assert "risk_a_0125" in names
    assert "risk_a_plus_015" in names


def test_phase_gate_criteria_tighten_into_later_phases():
    assert any(g.metric == "avg_mae_r" for g in PHASE_GATE_CRITERIA[2])
    assert any(g.metric == "exit_efficiency" for g in PHASE_GATE_CRITERIA[3])
    assert any(g.metric == "sharpe_ratio" for g in PHASE_GATE_CRITERIA[5])


def test_plugin_uses_immutable_weights_and_named_phases():
    plugin = _make_plugin()
    spec = plugin.get_phase_spec(1, state=None)
    assert spec.name == "Signal Discrimination"
    assert spec.scoring_weights == IMMUTABLE_SCORING_WEIGHTS
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

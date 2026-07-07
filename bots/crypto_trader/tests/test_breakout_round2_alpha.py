"""Tests for the alpha-focused breakout round-2 optimizer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.breakout_round2_alpha import (
    OPTIMIZATION_END_DATE,
    ROUND2_ALPHA_HARD_REJECTS,
    ROUND2_ALPHA_PHASE_GATE_CRITERIA,
    ROUND2_ALPHA_SCORING_CEILINGS,
    ROUND2_ALPHA_SCORING_WEIGHTS,
    BreakoutRound2AlphaPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    _phase6_candidates,
    load_breakout_strategy,
)
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.strategy.breakout.config import BreakoutConfig
from scripts.run_breakout_round2_alpha import _build_promoted_state, _select_promoted_phase


def _make_plugin() -> BreakoutRound2AlphaPlugin:
    bc = BacktestConfig(
        start_date=date(2025, 12, 1),
        end_date=OPTIMIZATION_END_DATE,
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10000.0,
    )
    return BreakoutRound2AlphaPlugin(
        backtest_config=bc,
        base_config=BreakoutConfig(),
        data_dir=Path("data"),
        max_workers=2,
    )


def test_score_is_immutable_and_limited_to_seven_components():
    assert len(ROUND2_ALPHA_SCORING_WEIGHTS) == 7
    assert sum(ROUND2_ALPHA_SCORING_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(ROUND2_ALPHA_SCORING_WEIGHTS) == set(ROUND2_ALPHA_SCORING_CEILINGS)


def test_holdout_cutoff_excludes_post_april_twentieth_period():
    assert OPTIMIZATION_END_DATE == date(2026, 4, 20)


def test_hard_rejects_prioritize_real_edge_and_capture():
    assert ROUND2_ALPHA_HARD_REJECTS["total_trades"] == (">=", 12.0)
    assert ROUND2_ALPHA_HARD_REJECTS["profit_factor"] == (">=", 2.0)
    assert ROUND2_ALPHA_HARD_REJECTS["expectancy_r"] == (">=", 0.25)
    assert ROUND2_ALPHA_HARD_REJECTS["exit_efficiency"] == (">=", 0.50)
    assert ROUND2_ALPHA_HARD_REJECTS["max_drawdown_pct"] == ("<=", 8.0)


def test_signal_candidates_include_discrimination_not_just_frequency_expansion():
    names = [c.name for c in _phase1_candidates()]
    assert "quality_stack_strict" in names
    assert "no_countertrend" in names
    assert "model1_vol_135" in names
    assert "room_r_b_16" in names


def test_variant_candidates_audit_relaxed_and_symbol_direction_pockets():
    candidates = {c.name: c for c in _phase2_candidates()}
    assert candidates["relaxed_body_off"].mutations["setup.relaxed_body_enabled"] is False
    assert candidates["relaxed_selective_pockets"].mutations["symbol_filter.eth_relaxed_body_direction"] == "long_only"
    assert candidates["relaxed_selective_pockets"].mutations["symbol_filter.sol_relaxed_body_direction"] == "short_only"
    assert "sol_disabled" in candidates


def test_entry_candidates_include_retest_and_break_entry_variants():
    candidates = {c.name: c for c in _phase3_candidates()}
    assert candidates["strict_retest_stack"].mutations["confirmation.retest_require_rejection"] is True
    assert candidates["strict_retest_stack"].mutations["confirmation.retest_require_volume_decline"] is True
    assert candidates["strict_break_entry"].mutations["entry.model2_entry_on_break"] is True
    assert candidates["strict_break_entry"].mutations["entry.max_bars_after_signal"] == 2


def test_capture_and_frequency_candidates_target_known_diagnostic_gaps():
    phase4_names = [c.name for c in _phase4_candidates()]
    phase5_names = [c.name for c in _phase5_candidates()]
    assert "failure_handling_stack" in phase4_names
    assert "early_lock_055_010" in phase4_names
    assert "guarded_reentry_2" in phase5_names
    assert "max_trades_per_day_5" in phase5_names


def test_finetune_skips_risk_and_risk_scale_parameters():
    candidates = _phase6_candidates(
        {
            "exits.tp1_frac": 0.3,
            "balance.max_zone_age_bars": 24,
            "risk.risk_pct_b": 0.015,
            "setup.relaxed_body_risk_scale": 0.5,
        }
    )
    names = [c.name for c in candidates]
    assert any(name.startswith("finetune_tp1_frac_") for name in names)
    assert any(name.startswith("finetune_max_zone_age_bars_") for name in names)
    assert not any("risk_pct_b" in name for name in names)
    assert not any("risk_scale" in name for name in names)


def test_phase_gates_reflect_aggressive_but_capped_risk_stance():
    assert any(
        g.metric == "max_drawdown_pct" and g.operator == "<=" and g.threshold == 8.0
        for gates in ROUND2_ALPHA_PHASE_GATE_CRITERIA.values()
        for g in gates
    )
    assert any(g.metric == "exit_efficiency" and g.threshold >= 0.58 for g in ROUND2_ALPHA_PHASE_GATE_CRITERIA[4])
    assert any(g.metric == "total_trades" and g.threshold == 15.0 for g in ROUND2_ALPHA_PHASE_GATE_CRITERIA[6])


def test_plugin_uses_six_phases_and_immutable_alpha_score():
    plugin = _make_plugin()
    spec = plugin.get_phase_spec(1, state=None)
    assert plugin.num_phases == 6
    assert spec.name == "Broad Signal Discrimination"
    assert spec.scoring_weights == ROUND2_ALPHA_SCORING_WEIGHTS
    assert spec.max_rounds == 4
    assert spec.prune_threshold == 0.0


def test_round_promotion_prefers_latest_gate_passing_phase():
    state = PhaseState(
        completed_phases=[1, 2, 3],
        cumulative_mutations={"setup.body_ratio": 0.8},
        phase_results={
            1: {"final_mutations": {"setup.body_ratio": 0.7}},
            2: {"final_mutations": {"setup.body_ratio": 0.75}},
            3: {"final_mutations": {"setup.body_ratio": 0.8}},
        },
        phase_metrics={
            1: {"total_trades": 16.0},
            2: {"total_trades": 15.0},
            3: {"total_trades": 13.0},
        },
        phase_gate_results={
            1: {"passed": True},
            2: {"passed": True},
            3: {"passed": False, "failure_reasons": ["total_trades"]},
        },
    )

    promoted_phase = _select_promoted_phase(state)
    promoted_state = _build_promoted_state(state, promoted_phase)

    assert promoted_phase == 2
    assert promoted_state.completed_phases == [1, 2]
    assert promoted_state.cumulative_mutations == {"setup.body_ratio": 0.75}
    assert 3 not in promoted_state.phase_metrics


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

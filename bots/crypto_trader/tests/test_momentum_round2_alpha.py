"""Tests for the alpha-focused momentum round-2 optimizer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.optimize.momentum_round2_alpha import (
    OPTIMIZATION_END_DATE,
    ROUND2_ALPHA_HARD_REJECTS,
    ROUND2_ALPHA_PHASE_GATE_CRITERIA,
    ROUND2_ALPHA_SCORING_CEILINGS,
    ROUND2_ALPHA_SCORING_WEIGHTS,
    MomentumRound2AlphaPlugin,
    _phase1_candidates,
    _phase2_candidates,
    _phase3_candidates,
    _phase4_candidates,
    _phase5_candidates,
    _phase6_candidates,
    load_momentum_strategy,
)
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.strategy.momentum.config import MomentumConfig
from scripts.run_momentum_round2_alpha import _build_promoted_state, _select_promoted_phase


def _make_plugin() -> MomentumRound2AlphaPlugin:
    bc = BacktestConfig(
        start_date=date(2025, 12, 1),
        end_date=OPTIMIZATION_END_DATE,
        symbols=["BTC", "ETH", "SOL"],
        initial_equity=10_000.0,
    )
    return MomentumRound2AlphaPlugin(
        backtest_config=bc,
        base_config=MomentumConfig(),
        data_dir=Path("data"),
        max_workers=2,
    )


def test_score_is_immutable_and_limited_to_seven_components():
    assert len(ROUND2_ALPHA_SCORING_WEIGHTS) == 7
    assert sum(ROUND2_ALPHA_SCORING_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(ROUND2_ALPHA_SCORING_WEIGHTS) == set(ROUND2_ALPHA_SCORING_CEILINGS)


def test_holdout_cutoff_excludes_post_april_twentieth_period():
    assert OPTIMIZATION_END_DATE == date(2026, 4, 20)


def test_hard_rejects_prioritize_real_alpha_capture_and_entry_quality():
    assert ROUND2_ALPHA_HARD_REJECTS["total_trades"] == (">=", 18.0)
    assert ROUND2_ALPHA_HARD_REJECTS["profit_factor"] == (">=", 1.50)
    assert ROUND2_ALPHA_HARD_REJECTS["expectancy_r"] == (">=", 0.20)
    assert ROUND2_ALPHA_HARD_REJECTS["exit_efficiency"] == (">=", 0.42)
    assert ROUND2_ALPHA_HARD_REJECTS["avg_mae_r"] == (">=", -0.45)
    assert ROUND2_ALPHA_HARD_REJECTS["max_drawdown_pct"] == ("<=", 8.0)


def test_signal_candidates_are_broad_structural_filters_not_time_fits():
    candidates = {candidate.name: candidate for candidate in _phase1_candidates()}
    assert "inside_as_weak_gate2" in candidates
    assert "weak_volume_110" in candidates
    assert "zone_prox_035" in candidates
    assert "quality_stack_light" in candidates
    assert not any("hour" in name or "weekday" in name for name in candidates)


def test_entry_candidates_include_trigger_aware_modes():
    candidates = {candidate.name: candidate for candidate in _phase2_candidates()}
    assert candidates["entry_confirm_specific_ttl2"].mutations["entry.mode"] == "confirmation_specific"
    assert candidates["entry_break_ttl1"].mutations["entry.mode"] == "break"
    assert candidates["fib_050_confirm_specific_ttl2"].mutations["setup.fib_high"] == 0.50


def test_failure_and_capture_candidates_target_diagnostic_gaps():
    phase3 = {candidate.name: candidate for candidate in _phase3_candidates()}
    phase4 = {candidate.name: candidate for candidate in _phase4_candidates()}
    assert "failure_control_stack" in phase3
    assert "proof_lock_040_flat" in phase3
    assert "mfe_retrace_150_gb100_min075_b6" in phase4
    assert "runner_trigger_125" in phase4


def test_frequency_candidates_expand_structurally_not_by_symbol_disabling():
    candidates = {candidate.name: candidate for candidate in _phase5_candidates()}
    assert "min_b_conf_0" in candidates
    assert "guarded_reentry_2" in candidates
    assert "daily_max_6" in candidates
    assert not any("disabled" in name or "short_only" in name or "long_only" in name for name in candidates)


def test_phase6_allows_modest_risk_scaling_but_finetune_skips_risk():
    candidates = _phase6_candidates(
        {
            "exits.tp1_frac": 0.16,
            "trail.runner_trigger_r": 1.5,
            "risk.risk_pct_b": 0.011,
        }
    )
    names = [candidate.name for candidate in candidates]
    assert "risk_b_0110" in names
    assert "gross_risk_045" in names
    assert any(name.startswith("finetune_tp1_frac_") for name in names)
    assert any(name.startswith("finetune_runner_trigger_r_") for name in names)
    assert not any(name.startswith("finetune_risk_pct_b_") for name in names)


def test_phase_gates_reflect_aggressive_but_capped_risk_stance():
    assert any(
        criterion.metric == "max_drawdown_pct"
        and criterion.operator == "<="
        and criterion.threshold == 8.0
        for gates in ROUND2_ALPHA_PHASE_GATE_CRITERIA.values()
        for criterion in gates
    )
    assert any(
        criterion.metric == "total_trades" and criterion.threshold == 21.0
        for criterion in ROUND2_ALPHA_PHASE_GATE_CRITERIA[5]
    )
    assert any(
        criterion.metric == "exit_efficiency" and criterion.threshold >= 0.50
        for criterion in ROUND2_ALPHA_PHASE_GATE_CRITERIA[4]
    )


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
        cumulative_mutations={"entry.mode": "break"},
        phase_results={
            1: {"final_mutations": {"entry.mode": "close"}},
            2: {"final_mutations": {"entry.mode": "confirmation_specific"}},
            3: {"final_mutations": {"entry.mode": "break"}},
        },
        phase_metrics={
            1: {"total_trades": 22.0},
            2: {"total_trades": 21.0},
            3: {"total_trades": 17.0},
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
    assert promoted_state.cumulative_mutations == {"entry.mode": "confirmation_specific"}
    assert 3 not in promoted_state.phase_metrics


def test_load_momentum_strategy_round_trip(tmp_path: Path):
    path = tmp_path / "optimized_config.json"
    payload = {"strategy": MomentumConfig().to_dict()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    cfg = load_momentum_strategy(path)
    assert isinstance(cfg, MomentumConfig)
    assert cfg.symbols == ["BTC", "ETH", "SOL"]

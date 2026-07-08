"""Momentum canonical round 2 phased optimization from the round-1 baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crypto_trader.optimize.momentum_plugin import MomentumPlugin
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.momentum.config import MomentumConfig

PHASE_NAMES: dict[int, str] = {
    1: "Signal Discrimination",
    2: "Entry Quality",
    3: "Capture & Management",
    4: "Frequency Expansion",
    5: "Risk Validation & Scaling",
    6: "Finetune",
}

IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "coverage": 0.14,
    "returns": 0.18,
    "edge": 0.22,
    "capture": 0.26,
    "entry_quality": 0.12,
    "risk": 0.08,
}

IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 26.0,
    "returns": 12.0,
    "edge": 3.0,
    "risk": 8.0,
}

IMMUTABLE_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 18.0),
    "profit_factor": (">=", 1.20),
    "max_drawdown_pct": ("<=", 10.0),
    "avg_mae_r": (">=", -0.40),
    "exit_efficiency": (">=", 0.43),
}

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.20),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.45),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.20),
        GateCriterion(metric="avg_mae_r", operator=">=", threshold=-0.40),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.46),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    3: [
        GateCriterion(metric="net_return_pct", operator=">=", threshold=7.0),
        GateCriterion(metric="total_trades", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.25),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.48),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    4: [
        GateCriterion(metric="net_return_pct", operator=">=", threshold=7.0),
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.20),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.48),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    5: [
        GateCriterion(metric="net_return_pct", operator=">=", threshold=7.5),
        GateCriterion(metric="total_trades", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.25),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.2),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.48),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    6: [
        GateCriterion(metric="net_return_pct", operator=">=", threshold=8.0),
        GateCriterion(metric="total_trades", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.25),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.2),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.50),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
}


def load_momentum_strategy(path: Path) -> MomentumConfig:
    """Load a momentum config from an optimized config JSON artifact."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return MomentumConfig.from_dict(payload["strategy"])


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp("hammer_off", {"confirmation.enable_hammer": False}),
        _exp(
            "weak_volume_105",
            {
                "confirmation.enforce_volume_on_weak_confirmations": True,
                "confirmation.volume_threshold_mult": 1.05,
            },
        ),
        _exp(
            "weak_volume_110",
            {
                "confirmation.enforce_volume_on_weak_confirmations": True,
                "confirmation.volume_threshold_mult": 1.10,
            },
        ),
        _exp("weak_gate_3", {"confirmation.min_confluences_for_weak": 3}),
        _exp("weak_gate_4", {"confirmation.min_confluences_for_weak": 4}),
        _exp("micro_shift_4", {"confirmation.micro_shift_min_bars": 4}),
        _exp("micro_shift_5", {"confirmation.micro_shift_min_bars": 5}),
        _exp("min_b_conf_1", {"setup.min_confluences_b": 1}),
        _exp("min_b_conf_2", {"setup.min_confluences_b": 2}),
        _exp("room_b_14", {"setup.min_room_b": 1.4}),
        _exp(
            "zone_prox_050",
            {
                "confirmation.require_zone_proximity": True,
                "confirmation.zone_proximity_atr": 0.5,
            },
        ),
        _exp(
            "zone_prox_025",
            {
                "confirmation.require_zone_proximity": True,
                "confirmation.zone_proximity_atr": 0.25,
            },
        ),
        _exp("reject_extended_reaction_on", {"setup.reject_extended_reaction": True}),
        _exp("outside_session_require_a", {"session.reduced_window_require_a": True}),
        _exp("rsi_pullback_35", {"setup.rsi_pullback_threshold": 35.0}),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp(
            "entry_break_ttl1",
            {"entry.mode": "break", "entry.max_bars_after_confirmation": 1},
        ),
        _exp(
            "entry_break_ttl2",
            {"entry.mode": "break", "entry.max_bars_after_confirmation": 2},
        ),
        _exp(
            "entry_confirm_specific_ttl1",
            {"entry.mode": "confirmation_specific", "entry.max_bars_after_confirmation": 1},
        ),
        _exp(
            "entry_confirm_specific_ttl2",
            {"entry.mode": "confirmation_specific", "entry.max_bars_after_confirmation": 2},
        ),
        _exp("fib_high_050", {"setup.fib_high": 0.5}),
        _exp(
            "fib_050_confirm_specific_ttl2",
            {
                "setup.fib_high": 0.5,
                "entry.mode": "confirmation_specific",
                "entry.max_bars_after_confirmation": 2,
            },
        ),
        _exp(
            "fib_050_break_ttl2",
            {
                "setup.fib_high": 0.5,
                "entry.mode": "break",
                "entry.max_bars_after_confirmation": 2,
            },
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp(
            "trail_activate_5_r03",
            {
                "trail.trail_activation_bars": 5,
                "trail.trail_activation_r": 0.3,
            },
        ),
        _exp(
            "trail_activate_4_r03",
            {
                "trail.trail_activation_bars": 4,
                "trail.trail_activation_r": 0.3,
            },
        ),
        _exp(
            "trail_activate_4_r04",
            {
                "trail.trail_activation_bars": 4,
                "trail.trail_activation_r": 0.4,
            },
        ),
        _exp("trail_generous", {"trail.trail_use_tightest": False}),
        _exp("trail_structure_only", {"trail.trail_behind_structure": True, "trail.trail_behind_ema": False}),
        _exp("trail_ema_only", {"trail.trail_behind_structure": False, "trail.trail_behind_ema": True}),
        _exp("trail_lock_025", {"trail.trail_buffer_tight": 0.25}),
        _exp("trail_r_ceiling_15", {"trail.trail_r_ceiling": 1.5}),
        _exp(
            "mfe_floor_100",
            {
                "trail.trail_mfe_floor_enabled": True,
                "trail.trail_mfe_floor_threshold": 1.0,
                "trail.trail_mfe_floor_buffer": 0.5,
            },
        ),
        _exp(
            "quick_exit_5",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 5,
                "exits.quick_exit_max_mfe_r": 0.20,
                "exits.quick_exit_max_r": -0.15,
            },
        ),
        _exp(
            "quick_exit_4",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.25,
                "exits.quick_exit_max_r": -0.10,
            },
        ),
        _exp(
            "tp_balance_100_20_200_35",
            {
                "exits.tp1_r": 1.0,
                "exits.tp1_frac": 0.20,
                "exits.tp2_r": 2.0,
                "exits.tp2_frac": 0.35,
            },
        ),
        _exp(
            "tp_balance_080_20_200_35",
            {
                "exits.tp1_r": 0.8,
                "exits.tp1_frac": 0.20,
                "exits.tp2_r": 2.0,
                "exits.tp2_frac": 0.35,
            },
        ),
        _exp(
            "be_accept_1_buffer_010",
            {"exits.be_acceptance_bars": 1, "exits.be_buffer_r": 0.10},
        ),
        _exp(
            "be_accept_1_buffer_000",
            {"exits.be_acceptance_bars": 1, "exits.be_buffer_r": 0.0},
        ),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp("bias_h1_1", {"bias.min_1h_conditions": 1}),
        _exp("bias_h4_1", {"bias.min_4h_conditions": 1}),
        _exp("h1_adx_12", {"bias.h1_adx_threshold": 12.0}),
        _exp("adx_chop_8", {"filters.adx_chop_threshold": 8.0}),
        _exp("adx_chop_7", {"filters.adx_chop_threshold": 7.0}),
        _exp("reentry_cooldown_2", {"reentry.cooldown_bars": 2}),
        _exp("reentry_max_3", {"reentry.max_reentries": 3}),
        _exp("daily_max_5", {"daily_limits.max_trades_per_day": 5}),
        _exp("max_positions_4", {"risk.max_concurrent_positions": 4}),
        _exp("room_b_125", {"setup.min_room_b": 1.25}),
        _exp("rsi_pullback_45", {"setup.rsi_pullback_threshold": 45.0}),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp("sol_long_only", {"symbol_filter.sol_direction": "long_only"}),
        _exp("risk_a_0225", {"risk.risk_pct_a": 0.0225}),
        _exp("risk_a_025", {"risk.risk_pct_a": 0.025}),
        _exp("risk_b_014", {"risk.risk_pct_b": 0.014}),
        _exp("risk_b_015", {"risk.risk_pct_b": 0.015}),
        _exp("corr_risk_018", {"risk.max_correlated_risk": 0.018}),
        _exp("corr_risk_022", {"risk.max_correlated_risk": 0.022}),
        _exp("gross_risk_045", {"risk.max_gross_risk": 0.045}),
        _exp("gross_risk_050", {"risk.max_gross_risk": 0.05}),
        _exp("daily_loss_020", {"daily_limits.max_daily_loss_pct": 0.020}),
        _exp("max_consecutive_losses_3", {"daily_limits.max_consecutive_losses": 3}),
    ]


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for key, val in cumulative.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue

        is_int = isinstance(val, int) and not isinstance(val, bool)
        for mult in (0.90, 0.95, 1.05, 1.10):
            raw = val * mult
            if is_int:
                new_val = int(round(raw))
                if new_val <= 0:
                    continue
            else:
                new_val = round(raw, 6)

            if new_val == val:
                continue

            suffix = str(new_val).replace(".", "_").replace("-", "neg_")
            experiments.append(_exp(f"finetune_{key.split('.')[-1]}_{suffix}", {key: new_val}))
    return experiments


_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
}


class MomentumRound5PhasedPlugin(MomentumPlugin):
    """Canonical round-2 momentum optimization seeded from the round-1 winner."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 12.0,
            "total_trades": 28.0,
            "profit_factor": 2.6,
            "exit_efficiency": 0.55,
            "avg_mae_r": -0.28,
            "max_drawdown_pct": 8.0,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 6:
            cumulative = state.cumulative_mutations if state else {}
            candidates = _phase6_candidates(cumulative)
        else:
            gen = _PHASE_CANDIDATES.get(phase)
            candidates = gen() if gen else []

        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            focus_metrics=["total_trades", "profit_factor", "exit_efficiency"],
            diagnostic_gap_fn=lambda p, m: self._diagnostic_gap_fn(p, m),
            suggest_experiments_fn=lambda p, m, w, s: self._suggest_experiments_fn(p, m, w, s),
            decide_action_fn=lambda *args: self._decide_action_fn(*args),
            redesign_scoring_weights_fn=lambda *args: None,
            build_extra_analysis_fn=lambda p, m, s, g: self._build_extra_analysis_fn(p, m, s, g),
            format_extra_analysis_fn=lambda d: self._format_extra_analysis_fn(d),
        )

        return PhaseSpec(
            phase_num=phase,
            name=PHASE_NAMES.get(phase, f"Phase {phase}"),
            candidates=candidates,
            scoring_weights=dict(IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(IMMUTABLE_HARD_REJECTS),
            min_delta=0.003,
            max_rounds=4 if phase < 6 else 3,
            prune_threshold=0.0,
            gate_criteria=list(PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda _m, _p=phase: list(PHASE_GATE_CRITERIA[_p]),
            analysis_policy=policy,
            focus=PHASE_NAMES.get(phase, ""),
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = IMMUTABLE_SCORING_CEILINGS

        def evaluate_fn(
            candidates: list[Experiment],
            current_mutations: dict[str, Any],
        ) -> list[ScoredCandidate]:
            return evaluate_parallel(
                candidates=candidates,
                current_mutations=current_mutations,
                cumulative_mutations=cumulative_mutations,
                base_config=self.base_config,
                backtest_config=self.backtest_config,
                data_dir=self.data_dir,
                scoring_weights=scoring_weights,
                hard_rejects=hard_rejects,
                phase=phase,
                max_workers=self.max_workers,
                ceilings=ceilings,
            )

        return evaluate_fn

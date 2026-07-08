"""Momentum round 2 alpha-focused phased optimization.

This round is seeded from the current parity-aligned momentum round_1 optimized
config, capped before the post-2026-04-20 holdout, and uses only normal
``MomentumConfig`` mutations consumed by the shared strategy implementation.
The search deliberately favors broad signal/entry/management structures over
small-sample time, symbol, or trade-id fitting.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
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

OPTIMIZATION_END_DATE = date(2026, 4, 20)
DEFAULT_OPTIMIZATION_START_DATE = date(2025, 12, 1)

PHASE_NAMES: dict[int, str] = {
    1: "Broad Signal Discrimination",
    2: "Entry Architecture",
    3: "Proof & Failure Control",
    4: "Runner Capture",
    5: "Structural Frequency",
    6: "Risk Scaling & Finetune",
}

# Seven components max by design. The score leaves headroom above the current
# baseline so it can distinguish real improvements in return, frequency, and
# capture instead of saturating on an already-strong PF/Calmar profile.
ROUND2_ALPHA_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.24,
    "coverage": 0.18,
    "expectancy": 0.16,
    "edge": 0.13,
    "capture": 0.12,
    "entry_quality": 0.09,
    "risk": 0.08,
}

ROUND2_ALPHA_SCORING_CEILINGS: dict[str, float] = {
    "returns": 20.0,
    "coverage": 42.0,
    "expectancy": 0.65,
    "edge": 3.5,
    "capture": 0.62,
    "entry_quality": 0.75,
    "risk": 8.0,
}

# Alias used by the optimizer contract collector.
IMMUTABLE_SCORING_CEILINGS = ROUND2_ALPHA_SCORING_CEILINGS

ROUND2_ALPHA_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 18.0),
    "profit_factor": (">=", 1.50),
    "expectancy_r": (">=", 0.20),
    "exit_efficiency": (">=", 0.42),
    "avg_mae_r": (">=", -0.45),
    "max_drawdown_pct": ("<=", 8.0),
}

ROUND2_ALPHA_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=19.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.0),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.30),
        GateCriterion(metric="avg_mae_r", operator=">=", threshold=-0.42),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=19.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.0),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.30),
        GateCriterion(metric="avg_mae_r", operator=">=", threshold=-0.40),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=19.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.1),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.32),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.46),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=19.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.1),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.34),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.50),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=21.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.8),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.28),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.46),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    6: [
        GateCriterion(metric="total_trades", operator=">=", threshold=21.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.8),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.28),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.46),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
}


def load_momentum_strategy(path: Path) -> MomentumConfig:
    """Load a momentum config from an optimized config JSON artifact."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return MomentumConfig.from_dict(payload["strategy"])


def _baseline_start_date(baseline_path: Path | None) -> date:
    if baseline_path is None or not baseline_path.exists():
        return DEFAULT_OPTIMIZATION_START_DATE
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_OPTIMIZATION_START_DATE

    start = (
        payload.get("metadata", {})
        .get("contract", {})
        .get("backtest_config", {})
        .get("start_date")
    )
    if not isinstance(start, str):
        start = payload.get("metadata", {}).get("data_window", {}).get("start_date")
    if isinstance(start, str):
        try:
            return date.fromisoformat(start)
        except ValueError:
            return DEFAULT_OPTIMIZATION_START_DATE
    return DEFAULT_OPTIMIZATION_START_DATE


def build_backtest_config(
    data_dir: Path,
    baseline_path: Path | None = None,
) -> tuple[BacktestConfig, dict[str, str]]:
    """Build a live-parity config capped before the post-2026-04-20 holdout."""
    start_date = _baseline_start_date(baseline_path)
    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=["BTC", "ETH", "SOL"],
        start_date=start_date,
        end_date=OPTIMIZATION_END_DATE,
    )
    metadata = {
        "start_date": bt_cfg.start_date.isoformat(),
        "end_date": bt_cfg.end_date.isoformat(),
        "holdout_excluded_after": OPTIMIZATION_END_DATE.isoformat(),
        "baseline_start_source": str(baseline_path) if baseline_path else "default",
    }
    return bt_cfg, metadata


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    inside_weak = ["micro_structure_shift", "shooting_star", "inside_bar_break"]
    inside_base_weak = [
        "micro_structure_shift",
        "shooting_star",
        "inside_bar_break",
        "base_break",
    ]
    return [
        _exp("base_break_off", {"confirmation.enable_base_break": False}),
        _exp("inside_as_weak_gate2", {
            "confirmation.weak_confirmations": inside_weak,
            "confirmation.min_confluences_for_weak": 2,
        }),
        _exp("inside_as_weak_gate3", {
            "confirmation.weak_confirmations": inside_weak,
            "confirmation.min_confluences_for_weak": 3,
        }),
        _exp("inside_base_as_weak_gate2", {
            "confirmation.weak_confirmations": inside_base_weak,
            "confirmation.min_confluences_for_weak": 2,
        }),
        _exp("weak_volume_105", {
            "confirmation.enforce_volume_on_weak_confirmations": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
        _exp("weak_volume_110", {
            "confirmation.enforce_volume_on_weak_confirmations": True,
            "confirmation.volume_threshold_mult": 1.10,
        }),
        _exp("trigger_volume_105", {
            "confirmation.enforce_volume_on_trigger": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
        _exp("zone_prox_050", {
            "confirmation.require_zone_proximity": True,
            "confirmation.zone_proximity_atr": 0.50,
        }),
        _exp("zone_prox_035", {
            "confirmation.require_zone_proximity": True,
            "confirmation.zone_proximity_atr": 0.35,
        }),
        _exp("micro_shift_4", {"confirmation.micro_shift_min_bars": 4}),
        _exp("micro_shift_5", {"confirmation.micro_shift_min_bars": 5}),
        _exp("hammer_wick_25", {"confirmation.hammer_wick_ratio": 2.5}),
        _exp("rsi_pullback_35", {"setup.rsi_pullback_threshold": 35.0}),
        _exp("room_b_140", {"setup.min_room_b": 1.40}),
        _exp("quality_stack_light", {
            "confirmation.weak_confirmations": inside_weak,
            "confirmation.min_confluences_for_weak": 2,
            "confirmation.enforce_volume_on_weak_confirmations": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
        _exp("quality_stack_location", {
            "confirmation.require_zone_proximity": True,
            "confirmation.zone_proximity_atr": 0.50,
            "setup.min_room_b": 1.35,
        }),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp("entry_confirm_specific_ttl1", {
            "entry.mode": "confirmation_specific",
            "entry.max_bars_after_confirmation": 1,
        }),
        _exp("entry_confirm_specific_ttl2", {
            "entry.mode": "confirmation_specific",
            "entry.max_bars_after_confirmation": 2,
        }),
        _exp("entry_confirm_specific_ttl3", {
            "entry.mode": "confirmation_specific",
            "entry.max_bars_after_confirmation": 3,
        }),
        _exp("entry_break_ttl1", {
            "entry.mode": "break",
            "entry.max_bars_after_confirmation": 1,
        }),
        _exp("entry_break_ttl2", {
            "entry.mode": "break",
            "entry.max_bars_after_confirmation": 2,
        }),
        _exp("entry_close_ttl1", {
            "entry.mode": "close",
            "entry.max_bars_after_confirmation": 1,
        }),
        _exp("fib_high_050", {"setup.fib_high": 0.50}),
        _exp("fib_high_0786", {"setup.fib_high": 0.786}),
        _exp("fib_050_confirm_specific_ttl2", {
            "setup.fib_high": 0.50,
            "entry.mode": "confirmation_specific",
            "entry.max_bars_after_confirmation": 2,
        }),
        _exp("fib_0786_confirm_specific_ttl2", {
            "setup.fib_high": 0.786,
            "entry.mode": "confirmation_specific",
            "entry.max_bars_after_confirmation": 2,
        }),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp("proof_lock_035_neg010", {
            "exits.proof_lock_enabled": True,
            "exits.proof_lock_trigger_r": 0.35,
            "exits.proof_lock_stop_r": -0.10,
            "exits.proof_lock_min_bars": 2,
        }),
        _exp("proof_lock_040_flat", {
            "exits.proof_lock_enabled": True,
            "exits.proof_lock_trigger_r": 0.40,
            "exits.proof_lock_stop_r": 0.00,
            "exits.proof_lock_min_bars": 2,
        }),
        _exp("proof_lock_050_pos010", {
            "exits.proof_lock_enabled": True,
            "exits.proof_lock_trigger_r": 0.50,
            "exits.proof_lock_stop_r": 0.10,
            "exits.proof_lock_min_bars": 2,
        }),
        _exp("quick_exit_3_020_neg010", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 3,
            "exits.quick_exit_max_mfe_r": 0.20,
            "exits.quick_exit_max_r": -0.10,
        }),
        _exp("quick_exit_4_030_000", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 4,
            "exits.quick_exit_max_mfe_r": 0.30,
            "exits.quick_exit_max_r": 0.00,
        }),
        _exp("followthrough_off", {"exits.followthrough_exit_enabled": False}),
        _exp("followthrough_all_flat", {
            "exits.followthrough_exit_enabled": True,
            "exits.followthrough_peak_r": 0.35,
            "exits.followthrough_bars": 4,
            "exits.followthrough_floor_r": 0.00,
            "exits.followthrough_scope": "all",
        }),
        _exp("followthrough_continuation_flat", {
            "exits.followthrough_exit_enabled": True,
            "exits.followthrough_peak_r": 0.35,
            "exits.followthrough_bars": 4,
            "exits.followthrough_floor_r": 0.00,
            "exits.followthrough_scope": "continuation",
        }),
        _exp("failure_control_stack", {
            "exits.proof_lock_enabled": True,
            "exits.proof_lock_trigger_r": 0.40,
            "exits.proof_lock_stop_r": 0.00,
            "exits.proof_lock_min_bars": 2,
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 4,
            "exits.quick_exit_max_mfe_r": 0.30,
            "exits.quick_exit_max_r": 0.00,
        }),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp("mfe_retrace_125_gb075_min050_b5", {
            "exits.mfe_retrace_exit_enabled": True,
            "exits.mfe_retrace_trigger_r": 1.25,
            "exits.mfe_retrace_giveback_r": 0.75,
            "exits.mfe_retrace_min_r": 0.50,
            "exits.mfe_retrace_min_bars": 5,
        }),
        _exp("mfe_retrace_150_gb100_min075_b6", {
            "exits.mfe_retrace_exit_enabled": True,
            "exits.mfe_retrace_trigger_r": 1.50,
            "exits.mfe_retrace_giveback_r": 1.00,
            "exits.mfe_retrace_min_r": 0.75,
            "exits.mfe_retrace_min_bars": 6,
        }),
        _exp("mfe_retrace_200_gb125_min100_b8", {
            "exits.mfe_retrace_exit_enabled": True,
            "exits.mfe_retrace_trigger_r": 2.00,
            "exits.mfe_retrace_giveback_r": 1.25,
            "exits.mfe_retrace_min_r": 1.00,
            "exits.mfe_retrace_min_bars": 8,
        }),
        _exp("mfe_retrace_150_inflection", {
            "exits.mfe_retrace_exit_enabled": True,
            "exits.mfe_retrace_trigger_r": 1.50,
            "exits.mfe_retrace_giveback_r": 1.00,
            "exits.mfe_retrace_min_r": 0.75,
            "exits.mfe_retrace_min_bars": 6,
            "exits.mfe_retrace_scope": "inflection",
        }),
        _exp("runner_trigger_125", {"trail.runner_trigger_r": 1.25}),
        _exp("runner_trigger_175", {"trail.runner_trigger_r": 1.75}),
        _exp("runner_tight_020", {"trail.runner_trail_buffer_tight": 0.20}),
        _exp("runner_tight_030", {"trail.runner_trail_buffer_tight": 0.30}),
        _exp("runner_ceiling_100", {"trail.runner_trail_r_ceiling": 1.00}),
        _exp("runner_ceiling_150", {"trail.runner_trail_r_ceiling": 1.50}),
        _exp("trail_mfe_basis", {"trail.trail_r_basis": "mfe"}),
        _exp("trail_activation_4_r025", {
            "trail.trail_activation_bars": 4,
            "trail.trail_activation_r": 0.25,
        }),
        _exp("trail_activation_6_r035", {
            "trail.trail_activation_bars": 6,
            "trail.trail_activation_r": 0.35,
        }),
        _exp("tp_balance_100_20_200_30", {
            "exits.tp1_r": 1.00,
            "exits.tp1_frac": 0.20,
            "exits.tp2_r": 2.00,
            "exits.tp2_frac": 0.30,
        }),
        _exp("tp_balance_080_16_200_25", {
            "exits.tp1_r": 0.80,
            "exits.tp1_frac": 0.16,
            "exits.tp2_r": 2.00,
            "exits.tp2_frac": 0.25,
        }),
        _exp("reversal_080_130", {
            "exits.reversal_body_atr_mult": 0.80,
            "exits.reversal_volume_mult": 1.30,
        }),
        _exp("structure_125", {"exits.structure_break_body_atr_mult": 1.25}),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp("min_b_conf_0", {"setup.min_confluences_b": 0}),
        _exp("bias_h1_1", {"bias.min_1h_conditions": 1}),
        _exp("bias_h4_1", {"bias.min_4h_conditions": 1}),
        _exp("h1_adx_8", {"bias.h1_adx_threshold": 8.0}),
        _exp("adx_chop_8", {"filters.adx_chop_threshold": 8.0}),
        _exp("rsi_pullback_45", {"setup.rsi_pullback_threshold": 45.0}),
        _exp("fib_high_0786", {"setup.fib_high": 0.786}),
        _exp("room_b_110", {"setup.min_room_b": 1.10}),
        _exp("reentry_cooldown_2", {"reentry.cooldown_bars": 2}),
        _exp("reentry_max_3", {"reentry.max_reentries": 3}),
        _exp("guarded_reentry_2", {
            "reentry.cooldown_bars": 2,
            "reentry.min_confluences_override": 1,
        }),
        _exp("daily_max_6", {"daily_limits.max_trades_per_day": 6}),
        _exp("max_positions_4", {"risk.max_concurrent_positions": 4}),
        _exp("frequency_stack_quality", {
            "setup.min_confluences_b": 0,
            "setup.min_room_b": 1.15,
            "reentry.cooldown_bars": 2,
            "reentry.min_confluences_override": 1,
        }),
    ]


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    experiments = [
        _exp("risk_b_0095", {"risk.risk_pct_b": 0.0095}),
        _exp("risk_b_0110", {"risk.risk_pct_b": 0.0110}),
        _exp("risk_b_0125", {"risk.risk_pct_b": 0.0125}),
        _exp("risk_a_0200", {"risk.risk_pct_a": 0.0200}),
        _exp("risk_a_0215", {"risk.risk_pct_a": 0.0215}),
        _exp("gross_risk_045", {"risk.max_gross_risk": 0.045}),
        _exp("gross_risk_050", {"risk.max_gross_risk": 0.050}),
        _exp("corr_risk_022", {"risk.max_correlated_risk": 0.022}),
        _exp("corr_risk_025", {"risk.max_correlated_risk": 0.025}),
    ]

    for key, val in cumulative.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if key.startswith("risk."):
            continue

        is_int = isinstance(val, int) and not isinstance(val, bool)
        for mult in (0.95, 1.05):
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


class MomentumRound2AlphaPlugin(MomentumPlugin):
    """Round-2 optimizer focused on alpha extraction under parity constraints."""

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 16.0,
            "total_trades": 28.0,
            "expectancy_r": 0.45,
            "profit_factor": 2.5,
            "exit_efficiency": 0.55,
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
            focus_metrics=[
                "net_return_pct",
                "total_trades",
                "expectancy_r",
                "exit_efficiency",
                "avg_mae_r",
            ],
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
            scoring_weights=dict(ROUND2_ALPHA_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND2_ALPHA_HARD_REJECTS),
            min_delta=0.003,
            max_rounds=4 if phase < 6 else 3,
            prune_threshold=0.0,
            gate_criteria=list(ROUND2_ALPHA_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda _m, _p=phase: list(ROUND2_ALPHA_PHASE_GATE_CRITERIA[_p]),
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
        ceilings = ROUND2_ALPHA_SCORING_CEILINGS

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
                strategy_type="momentum",
                ceilings=ceilings,
            )

        return evaluate_fn


__all__ = [
    "OPTIMIZATION_END_DATE",
    "PHASE_NAMES",
    "ROUND2_ALPHA_HARD_REJECTS",
    "ROUND2_ALPHA_PHASE_GATE_CRITERIA",
    "ROUND2_ALPHA_SCORING_CEILINGS",
    "ROUND2_ALPHA_SCORING_WEIGHTS",
    "MomentumRound2AlphaPlugin",
    "build_backtest_config",
    "load_momentum_strategy",
]

"""Momentum round 3 phased optimization focused on profit-lock and exit architecture."""

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
    1: "Proof Lock & Failure Control",
    2: "Trail Regime",
    3: "Scale-Out Architecture",
    4: "Exit Trigger Calibration",
    5: "Finetune",
}

IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "capture": 0.32,
    "edge": 0.20,
    "returns": 0.18,
    "risk": 0.12,
    "coverage": 0.10,
    "entry_quality": 0.08,
}

IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 28.0,
    "returns": 16.0,
    "edge": 4.5,
    "risk": 8.0,
}

IMMUTABLE_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 20.0),
    "profit_factor": (">=", 1.30),
    "exit_efficiency": (">=", 0.45),
    "avg_mae_r": (">=", -0.35),
    "max_drawdown_pct": ("<=", 8.0),
}

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.50),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.49),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=11.5),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.60),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.51),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=11.8),
        GateCriterion(metric="avg_bars_held", operator=">=", threshold=9.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.60),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.52),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=12.0),
        GateCriterion(metric="avg_bars_held", operator=">=", threshold=9.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.70),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.53),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=12.3),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.70),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.54),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=12.5),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=3.5),
    ],
}


def load_momentum_strategy(path: Path) -> MomentumConfig:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return MomentumConfig.from_dict(payload["strategy"])


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp(
            "proof_lock_soft",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.50,
                "exits.proof_lock_stop_r": -0.10,
                "exits.proof_lock_min_bars": 2,
            },
        ),
        _exp(
            "proof_lock_flat",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.50,
                "exits.proof_lock_stop_r": 0.00,
                "exits.proof_lock_min_bars": 2,
            },
        ),
        _exp(
            "proof_lock_positive",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.60,
                "exits.proof_lock_stop_r": 0.10,
                "exits.proof_lock_min_bars": 2,
            },
        ),
        _exp(
            "followthrough_soft",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": -0.10,
            },
        ),
        _exp(
            "followthrough_flat",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
            },
        ),
        _exp(
            "followthrough_flat_inflection_only",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
                "exits.followthrough_scope": "inflection",
            },
        ),
        _exp(
            "followthrough_flat_continuation_only",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
                "exits.followthrough_scope": "continuation",
            },
        ),
        _exp(
            "mfe_retrace_150_gb125_min075_b6",
            {
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 1.50,
                "exits.mfe_retrace_giveback_r": 1.25,
                "exits.mfe_retrace_min_r": 0.75,
                "exits.mfe_retrace_min_bars": 6,
            },
        ),
        _exp(
            "mfe_retrace_200_gb150_min100_b8",
            {
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 2.00,
                "exits.mfe_retrace_giveback_r": 1.50,
                "exits.mfe_retrace_min_r": 1.00,
                "exits.mfe_retrace_min_bars": 8,
            },
        ),
        _exp(
            "mfe_retrace_250_gb175_min125_b10",
            {
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 2.50,
                "exits.mfe_retrace_giveback_r": 1.75,
                "exits.mfe_retrace_min_r": 1.25,
                "exits.mfe_retrace_min_bars": 10,
            },
        ),
        _exp(
            "mfe_retrace_200_gb150_min100_b8_inflection_only",
            {
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 2.00,
                "exits.mfe_retrace_giveback_r": 1.50,
                "exits.mfe_retrace_min_r": 1.00,
                "exits.mfe_retrace_min_bars": 8,
                "exits.mfe_retrace_scope": "inflection",
            },
        ),
        _exp(
            "mfe_retrace_200_gb150_min100_b8_continuation_only",
            {
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 2.00,
                "exits.mfe_retrace_giveback_r": 1.50,
                "exits.mfe_retrace_min_r": 1.00,
                "exits.mfe_retrace_min_bars": 8,
                "exits.mfe_retrace_scope": "continuation",
            },
        ),
        _exp(
            "followthrough_flat__mfe_retrace_150_gb125",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 1.50,
                "exits.mfe_retrace_giveback_r": 1.25,
                "exits.mfe_retrace_min_r": 0.75,
                "exits.mfe_retrace_min_bars": 6,
            },
        ),
        _exp(
            "followthrough_flat__mfe_retrace_200_gb150",
            {
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
                "exits.mfe_retrace_exit_enabled": True,
                "exits.mfe_retrace_trigger_r": 2.00,
                "exits.mfe_retrace_giveback_r": 1.50,
                "exits.mfe_retrace_min_r": 1.00,
                "exits.mfe_retrace_min_bars": 8,
            },
        ),
        _exp(
            "proof_lock_flat_followthrough_soft",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.50,
                "exits.proof_lock_stop_r": 0.00,
                "exits.proof_lock_min_bars": 2,
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": -0.10,
            },
        ),
        _exp(
            "proof_lock_positive_followthrough_flat",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.60,
                "exits.proof_lock_stop_r": 0.10,
                "exits.proof_lock_min_bars": 2,
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
            },
        ),
        _exp(
            "proof_lock_flat_followthrough_flat",
            {
                "exits.proof_lock_enabled": True,
                "exits.proof_lock_trigger_r": 0.50,
                "exits.proof_lock_stop_r": 0.00,
                "exits.proof_lock_min_bars": 2,
                "exits.followthrough_exit_enabled": True,
                "exits.followthrough_peak_r": 0.35,
                "exits.followthrough_bars": 4,
                "exits.followthrough_floor_r": 0.00,
            },
        ),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp(
            "ema20_tight025",
            {
                "trail.trail_mode": "ema",
                "trail.trail_ema_period": 20,
                "trail.trail_buffer_tight": 0.25,
            },
        ),
        _exp(
            "ema25_tight025",
            {
                "trail.trail_mode": "ema",
                "trail.trail_ema_period": 25,
                "trail.trail_buffer_tight": 0.25,
            },
        ),
        _exp(
            "ema30_wide175",
            {
                "trail.trail_mode": "ema",
                "trail.trail_ema_period": 30,
                "trail.trail_buffer_wide": 1.75,
            },
        ),
        _exp(
            "activate4_r025",
            {
                "trail.trail_activation_bars": 4,
                "trail.trail_activation_r": 0.25,
            },
        ),
        _exp(
            "activate6_r035",
            {
                "trail.trail_activation_bars": 6,
                "trail.trail_activation_r": 0.35,
            },
        ),
        _exp(
            "chandelier20_2_5",
            {
                "trail.trail_mode": "chandelier",
                "trail.trail_chandelier_lookback": 20,
                "trail.trail_chandelier_atr_mult": 2.5,
            },
        ),
        _exp(
            "chandelier24_2_75",
            {
                "trail.trail_mode": "chandelier",
                "trail.trail_chandelier_lookback": 24,
                "trail.trail_chandelier_atr_mult": 2.75,
            },
        ),
        _exp(
            "hybrid_ema25_ch20_2_75",
            {
                "trail.trail_mode": "hybrid",
                "trail.trail_ema_period": 25,
                "trail.trail_chandelier_lookback": 20,
                "trail.trail_chandelier_atr_mult": 2.75,
            },
        ),
        _exp(
            "hybrid_ema20_ch18_2_5",
            {
                "trail.trail_mode": "hybrid",
                "trail.trail_ema_period": 20,
                "trail.trail_chandelier_lookback": 18,
                "trail.trail_chandelier_atr_mult": 2.5,
            },
        ),
        _exp(
            "ema25_floor_100_060",
            {
                "trail.trail_mode": "ema",
                "trail.trail_mfe_floor_enabled": True,
                "trail.trail_mfe_floor_threshold": 1.0,
                "trail.trail_mfe_floor_buffer": 0.6,
            },
        ),
        _exp(
            "ema25_mfe_basis",
            {
                "trail.trail_mode": "ema",
                "trail.trail_r_basis": "mfe",
            },
        ),
        _exp(
            "ema25_mfe_basis_ceiling125",
            {
                "trail.trail_mode": "ema",
                "trail.trail_r_basis": "mfe",
                "trail.trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "ema25_mfe_basis_activate4_r025",
            {
                "trail.trail_mode": "ema",
                "trail.trail_r_basis": "mfe",
                "trail.trail_activation_bars": 4,
                "trail.trail_activation_r": 0.25,
            },
        ),
        _exp(
            "ema25_mfe_basis_buffer_125_020",
            {
                "trail.trail_mode": "ema",
                "trail.trail_r_basis": "mfe",
                "trail.trail_buffer_wide": 1.25,
                "trail.trail_buffer_tight": 0.20,
                "trail.trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "ema25_mfe_basis_buffer_110_015_activate4",
            {
                "trail.trail_mode": "ema",
                "trail.trail_r_basis": "mfe",
                "trail.trail_buffer_wide": 1.10,
                "trail.trail_buffer_tight": 0.15,
                "trail.trail_r_ceiling": 1.00,
                "trail.trail_activation_bars": 4,
                "trail.trail_activation_r": 0.20,
            },
        ),
        _exp(
            "runner_trail_all_150_mild",
            {
                "trail.runner_trail_enabled": True,
                "trail.runner_trail_scope": "all",
                "trail.runner_trigger_r": 1.5,
                "trail.runner_trail_r_basis": "mfe",
                "trail.runner_trail_buffer_wide": 1.25,
                "trail.runner_trail_buffer_tight": 0.25,
                "trail.runner_trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "runner_trail_all_200_mild",
            {
                "trail.runner_trail_enabled": True,
                "trail.runner_trail_scope": "all",
                "trail.runner_trigger_r": 2.0,
                "trail.runner_trail_r_basis": "mfe",
                "trail.runner_trail_buffer_wide": 1.25,
                "trail.runner_trail_buffer_tight": 0.25,
                "trail.runner_trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "runner_trail_inflection_150_mild",
            {
                "trail.runner_trail_enabled": True,
                "trail.runner_trail_scope": "inflection",
                "trail.runner_trigger_r": 1.5,
                "trail.runner_trail_r_basis": "mfe",
                "trail.runner_trail_buffer_wide": 1.25,
                "trail.runner_trail_buffer_tight": 0.25,
                "trail.runner_trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "runner_trail_continuation_150_mild",
            {
                "trail.runner_trail_enabled": True,
                "trail.runner_trail_scope": "continuation",
                "trail.runner_trigger_r": 1.5,
                "trail.runner_trail_r_basis": "mfe",
                "trail.runner_trail_buffer_wide": 1.25,
                "trail.runner_trail_buffer_tight": 0.25,
                "trail.runner_trail_r_ceiling": 1.25,
            },
        ),
        _exp(
            "runner_trail_all_150_tight",
            {
                "trail.runner_trail_enabled": True,
                "trail.runner_trail_scope": "all",
                "trail.runner_trigger_r": 1.5,
                "trail.runner_trail_r_basis": "mfe",
                "trail.runner_trail_buffer_wide": 1.10,
                "trail.runner_trail_buffer_tight": 0.20,
                "trail.runner_trail_r_ceiling": 1.00,
            },
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp(
            "tp_080_10__180_15__be1_000",
            {
                "exits.tp1_r": 0.8,
                "exits.tp1_frac": 0.10,
                "exits.tp2_r": 1.8,
                "exits.tp2_frac": 0.15,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.0,
            },
        ),
        _exp(
            "tp_080_15__200_20__be1_000",
            {
                "exits.tp1_r": 0.8,
                "exits.tp1_frac": 0.15,
                "exits.tp2_r": 2.0,
                "exits.tp2_frac": 0.20,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.0,
            },
        ),
        _exp(
            "tp_100_10__200_15__be1_010",
            {
                "exits.tp1_r": 1.0,
                "exits.tp1_frac": 0.10,
                "exits.tp2_r": 2.0,
                "exits.tp2_frac": 0.15,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.10,
            },
        ),
        _exp(
            "tp_100_15__220_20__be1_010",
            {
                "exits.tp1_r": 1.0,
                "exits.tp1_frac": 0.15,
                "exits.tp2_r": 2.2,
                "exits.tp2_frac": 0.20,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.10,
            },
        ),
        _exp(
            "tp1_only_090_15__be1_000",
            {
                "exits.tp1_r": 0.9,
                "exits.tp1_frac": 0.15,
                "exits.tp2_frac": 0.0,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.0,
            },
        ),
        _exp(
            "tp1_only_100_15__be1_010",
            {
                "exits.tp1_r": 1.0,
                "exits.tp1_frac": 0.15,
                "exits.tp2_frac": 0.0,
                "exits.be_acceptance_bars": 1,
                "exits.be_buffer_r": 0.10,
            },
        ),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp(
            "reversal_0_8atr_1_3vol",
            {
                "exits.reversal_body_atr_mult": 0.8,
                "exits.reversal_volume_mult": 1.3,
            },
        ),
        _exp(
            "reversal_1_0atr_1_3vol",
            {
                "exits.reversal_body_atr_mult": 1.0,
                "exits.reversal_volume_mult": 1.3,
            },
        ),
        _exp(
            "structure_1_25atr",
            {"exits.structure_break_body_atr_mult": 1.25},
        ),
        _exp(
            "structure_1_75atr",
            {"exits.structure_break_body_atr_mult": 1.75},
        ),
        _exp(
            "reversal_0_8atr_1_3vol__structure_1_25atr",
            {
                "exits.reversal_body_atr_mult": 0.8,
                "exits.reversal_volume_mult": 1.3,
                "exits.structure_break_body_atr_mult": 1.25,
            },
        ),
        _exp(
            "reversal_1_0atr_1_2vol__structure_1_5atr",
            {
                "exits.reversal_body_atr_mult": 1.0,
                "exits.reversal_volume_mult": 1.2,
                "exits.structure_break_body_atr_mult": 1.5,
            },
        ),
        _exp("reversal_off", {"exits.enable_reversal_candle_exit": False}),
        _exp("structure_off", {"exits.enable_structure_break_exit": False}),
    ]


def _phase5_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for key, val in cumulative.items():
        if not (key.startswith("exits.") or key.startswith("trail.")):
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue

        is_int = isinstance(val, int) and not isinstance(val, bool)
        for mult in (0.95, 1.05, 0.90, 1.10):
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
}


class MomentumRound3ExitPhasedPlugin(MomentumPlugin):
    """Round-3 optimizer seeded from round 2 with an exit-only mandate."""

    @property
    def num_phases(self) -> int:
        return 5

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 13.5,
            "total_trades": 24.0,
            "profit_factor": 3.0,
            "exit_efficiency": 0.55,
            "avg_mae_r": -0.28,
            "max_drawdown_pct": 8.0,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 5:
            cumulative = state.cumulative_mutations if state else {}
            candidates = _phase5_candidates(cumulative)
        else:
            gen = _PHASE_CANDIDATES.get(phase)
            candidates = gen() if gen else []

        policy = PhaseAnalysisPolicy(
            max_scoring_retries=0,
            max_diagnostic_retries=0,
            focus_metrics=["exit_efficiency", "profit_factor", "net_return_pct"],
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
            min_delta=0.002,
            max_rounds=4 if phase < 5 else 3,
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

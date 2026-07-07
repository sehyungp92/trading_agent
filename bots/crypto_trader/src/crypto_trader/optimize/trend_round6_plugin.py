"""Round 6 phased-auto plugin for the trend strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.trend_plugin import TrendPlugin
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)

ROUND6_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "coverage": 0.12,
    "edge": 0.18,
    "capture": 0.16,
    "calmar": 0.12,
    "sharpe": 0.08,
    "risk": 0.07,
    "entry_quality": 0.05,
}

ROUND6_PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    1: {
        "returns": 0.18,
        "coverage": 0.14,
        "edge": 0.22,
        "capture": 0.12,
        "calmar": 0.10,
        "sharpe": 0.08,
        "risk": 0.06,
        "entry_quality": 0.10,
    },
    2: {
        "returns": 0.18,
        "coverage": 0.12,
        "edge": 0.18,
        "capture": 0.14,
        "calmar": 0.08,
        "sharpe": 0.08,
        "risk": 0.06,
        "entry_quality": 0.16,
    },
    3: {
        "returns": 0.18,
        "coverage": 0.10,
        "edge": 0.14,
        "capture": 0.22,
        "calmar": 0.10,
        "sharpe": 0.06,
        "risk": 0.08,
        "entry_quality": 0.12,
    },
    4: {
        "returns": 0.24,
        "coverage": 0.10,
        "edge": 0.16,
        "capture": 0.22,
        "calmar": 0.10,
        "sharpe": 0.06,
        "risk": 0.07,
        "entry_quality": 0.05,
    },
    5: {
        "returns": 0.22,
        "coverage": 0.12,
        "edge": 0.18,
        "capture": 0.14,
        "calmar": 0.12,
        "sharpe": 0.08,
        "risk": 0.09,
        "entry_quality": 0.05,
    },
    6: dict(ROUND6_SCORING_WEIGHTS),
}

ROUND6_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 30.0,
    "coverage": 120.0,
    "edge": 1.0,
    "calmar": 2.5,
    "sharpe": 3.5,
    "risk": 18.0,
}

ROUND6_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 18.0),
    "total_trades": (">=", 45.0),
    "profit_factor": (">=", 1.10),
}

ROUND6_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion("total_trades", ">=", 70.0),
        GateCriterion("net_return_pct", ">=", 18.0),
        GateCriterion("profit_factor", ">=", 1.25),
        GateCriterion("max_drawdown_pct", "<=", 15.0),
    ],
    2: [
        GateCriterion("total_trades", ">=", 70.0),
        GateCriterion("net_return_pct", ">=", 18.0),
        GateCriterion("profit_factor", ">=", 1.25),
        GateCriterion("max_drawdown_pct", "<=", 15.0),
        GateCriterion("avg_mae_r", "<=", 0.45),
    ],
    3: [
        GateCriterion("total_trades", ">=", 70.0),
        GateCriterion("profit_factor", ">=", 1.25),
        GateCriterion("max_drawdown_pct", "<=", 15.0),
        GateCriterion("exit_efficiency", ">=", 0.45),
    ],
    4: [
        GateCriterion("total_trades", ">=", 70.0),
        GateCriterion("profit_factor", ">=", 1.30),
        GateCriterion("max_drawdown_pct", "<=", 14.0),
        GateCriterion("exit_efficiency", ">=", 0.48),
    ],
    5: [
        GateCriterion("total_trades", ">=", 70.0),
        GateCriterion("profit_factor", ">=", 1.30),
        GateCriterion("max_drawdown_pct", "<=", 14.0),
        GateCriterion("sharpe_ratio", ">=", 2.70),
        GateCriterion("calmar_ratio", ">=", 1.80),
    ],
    6: [
        GateCriterion("total_trades", ">=", 75.0),
        GateCriterion("net_return_pct", ">=", 18.0),
        GateCriterion("profit_factor", ">=", 1.35),
        GateCriterion("max_drawdown_pct", "<=", 13.5),
        GateCriterion("exit_efficiency", ">=", 0.48),
        GateCriterion("sharpe_ratio", ">=", 2.70),
        GateCriterion("calmar_ratio", ">=", 1.80),
    ],
}

ROUND6_PHASE_NAMES: dict[int, str] = {
    1: "Signal Discrimination",
    2: "Entry Architecture",
    3: "Early Management",
    4: "Exit Harvesting",
    5: "Regime & Allocation",
    6: "Risk & Polish",
}


def _phase1_candidates() -> list[Experiment]:
    return [
        Experiment("disable_ema_reclaim", {"confirmation.enable_ema_reclaim": False}),
        Experiment("disable_hammer", {"confirmation.enable_hammer": False}),
        Experiment("min_confluences_1", {"setup.min_confluences": 1}),
        Experiment("min_confluences_2", {"setup.min_confluences": 2}),
        Experiment("pullback_retrace_618", {"setup.pullback_max_retrace": 0.618}),
        Experiment("impulse_atr_1_0", {"setup.impulse_min_atr_move": 1.0}),
        Experiment("strict_orderly", {"setup.strict_orderly_pullback": True}),
        Experiment("weighted_score_basic", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.5,
            "setup.min_setup_score_a": 2.5,
        }),
        Experiment("weighted_score_tight", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.8,
            "setup.min_setup_score_a": 2.8,
        }),
        Experiment("weekly_room_1_0", {
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.0,
        }),
        Experiment("weekly_room_1_5", {
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.5,
        }),
        Experiment("strict_quality_bundle", {
            "setup.strict_orderly_pullback": True,
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.6,
            "setup.min_setup_score_a": 2.6,
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.0,
            "confirmation.require_confirmation_for_b": True,
        }),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        Experiment("entry_break", {"entry.mode": "break"}),
        Experiment("entry_hybrid_grade", {"entry.mode": "hybrid_grade"}),
        Experiment("confirm_all", {"confirmation.require_confirmation": True}),
        Experiment("confirm_b_only", {"confirmation.require_confirmation_for_b": True}),
        Experiment("volume_trigger_gate", {"confirmation.enforce_volume_on_trigger": True}),
        Experiment("volume_confirm_bundle", {
            "confirmation.require_volume_confirm": True,
            "confirmation.enforce_volume_on_trigger": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
        Experiment("break_ttl_1", {
            "entry.mode": "break",
            "entry.max_bars_after_confirmation": 1,
        }),
        Experiment("hybrid_confirm_bundle", {
            "entry.mode": "hybrid_grade",
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enforce_volume_on_trigger": True,
        }),
        Experiment("structure_only_confirmations", {
            "entry.mode": "hybrid_grade",
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_ema_reclaim": False,
            "confirmation.enable_hammer": False,
        }),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        Experiment("scratch_025_flat_4", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.25,
            "exits.scratch_floor_r": 0.0,
            "exits.scratch_min_bars": 4,
        }),
        Experiment("scratch_025_neg_3", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.25,
            "exits.scratch_floor_r": -0.05,
            "exits.scratch_min_bars": 3,
        }),
        Experiment("scratch_035_lock_4", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.35,
            "exits.scratch_floor_r": 0.10,
            "exits.scratch_min_bars": 4,
        }),
        Experiment("quick_exit_off", {"exits.quick_exit_enabled": False}),
        Experiment("quick_exit_lighter", {
            "exits.quick_exit_bars": 10,
            "exits.quick_exit_max_mfe_r": 0.25,
            "exits.quick_exit_max_r": -0.10,
        }),
        Experiment("time_stop_reduce_12", {
            "exits.time_stop_bars": 12,
            "exits.time_stop_action": "reduce",
            "exits.time_stop_min_progress_r": 0.2,
        }),
        Experiment("scratch_plus_reduce", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.25,
            "exits.scratch_floor_r": 0.0,
            "exits.scratch_min_bars": 4,
            "exits.time_stop_bars": 12,
            "exits.time_stop_action": "reduce",
            "exits.time_stop_min_progress_r": 0.2,
        }),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        Experiment("trail_tight_0_2", {"trail.trail_buffer_tight": 0.2}),
        Experiment("trail_wide_1_0", {"trail.trail_buffer_wide": 1.0}),
        Experiment("trail_ceiling_1_2", {"trail.trail_r_ceiling": 1.2}),
        Experiment("trail_structure_on", {"trail.structure_trail_enabled": True}),
        Experiment("tp2_1_8", {"exits.tp2_r": 1.8}),
        Experiment("tp2_frac_0_4", {"exits.tp2_frac": 0.4}),
        Experiment("runner_0_35", {"exits.runner_frac": 0.35}),
        Experiment("ema_failsafe_15", {"exits.ema_failsafe_period": 15}),
        Experiment("capture_bundle", {
            "trail.trail_buffer_tight": 0.2,
            "trail.trail_buffer_wide": 1.0,
            "trail.trail_r_ceiling": 1.2,
            "trail.structure_trail_enabled": True,
            "exits.tp2_r": 1.8,
            "exits.tp2_frac": 0.4,
        }),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        Experiment("require_structure", {"regime.require_structure": True}),
        Experiment("b_adx_10", {"regime.b_min_adx": 10.0}),
        Experiment("h1_adx_24", {"regime.h1_min_adx": 24.0}),
        Experiment("relative_strength_0", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.0,
        }),
        Experiment("relative_strength_1pct", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.01,
        }),
        Experiment("funding_on_5bps", {
            "filters.funding_filter_enabled": True,
            "filters.funding_extreme_threshold": 0.0005,
        }),
        Experiment("reentry_off", {"reentry.enabled": False}),
        Experiment("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        Experiment("eth_short_only", {"symbol_filter.eth_direction": "short_only"}),
        Experiment("btc_long_only", {"symbol_filter.btc_direction": "long_only"}),
        Experiment("quality_regime_bundle", {
            "regime.require_structure": True,
            "regime.b_min_adx": 10.0,
            "regime.h1_min_adx": 24.0,
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.0,
        }),
    ]


def _phase6_candidates() -> list[Experiment]:
    return [
        Experiment("risk_a_0_012", {"risk.risk_pct_a": 0.012}),
        Experiment("risk_b_0_015", {"risk.risk_pct_b": 0.015}),
        Experiment("risk_b_0_020", {"risk.risk_pct_b": 0.020}),
        Experiment("max_trades_8", {"limits.max_trades_per_day": 8}),
        Experiment("max_trades_12", {"limits.max_trades_per_day": 12}),
        Experiment("max_positions_4", {"limits.max_concurrent_positions": 4}),
        Experiment("balanced_scale_bundle", {
            "risk.risk_pct_a": 0.012,
            "risk.risk_pct_b": 0.015,
            "limits.max_trades_per_day": 12,
        }),
    ]


ROUND6_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
    6: _phase6_candidates,
}


class Round6TrendPlugin(TrendPlugin):
    """Trend plugin variant for the round 6 phased-auto run."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 22.0,
            "total_trades": 90.0,
            "profit_factor": 1.55,
            "max_drawdown_pct": 12.0,
            "sharpe_ratio": 3.0,
            "calmar_ratio": 2.2,
            "exit_efficiency": 0.60,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        generator = ROUND6_PHASE_CANDIDATES.get(phase)
        if generator is None:
            raise ValueError(f"Unknown phase: {phase}")

        return PhaseSpec(
            phase_num=phase,
            name=ROUND6_PHASE_NAMES[phase],
            candidates=generator(),
            scoring_weights=dict(ROUND6_PHASE_SCORING_EMPHASIS.get(phase, ROUND6_SCORING_WEIGHTS)),
            hard_rejects=dict(ROUND6_HARD_REJECTS),
            min_delta=0.005,
            max_rounds=4,
            prune_threshold=0.0,
            gate_criteria=list(ROUND6_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda metrics, _phase=phase: self._gate_criteria_fn(metrics, _phase),
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=1,
                max_diagnostic_retries=1,
                focus_metrics=[
                    "net_return_pct",
                    "total_trades",
                    "profit_factor",
                    "exit_efficiency",
                    "max_drawdown_pct",
                ],
                diagnostic_gap_fn=lambda phase_num, metrics: self._diagnostic_gap_fn(phase_num, metrics),
                suggest_experiments_fn=lambda phase_num, metrics, weaknesses, replay_state:
                    self._suggest_experiments_fn(phase_num, metrics, weaknesses, replay_state),
                decide_action_fn=lambda *args: self._decide_action_fn(*args),
                redesign_scoring_weights_fn=lambda *args: self._redesign_scoring_weights_fn(*args),
                build_extra_analysis_fn=lambda phase_num, metrics, replay_state, greedy_result:
                    self._build_extra_analysis_fn(phase_num, metrics, replay_state, greedy_result),
                format_extra_analysis_fn=lambda extra: self._format_extra_analysis_fn(extra),
            ),
            focus=ROUND6_PHASE_NAMES[phase],
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = dict(ROUND6_IMMUTABLE_SCORING_CEILINGS)

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
                strategy_type="trend",
                ceilings=ceilings,
            )

        return evaluate_fn


__all__ = [
    "ROUND6_HARD_REJECTS",
    "ROUND6_IMMUTABLE_SCORING_CEILINGS",
    "ROUND6_PHASE_CANDIDATES",
    "ROUND6_PHASE_GATE_CRITERIA",
    "ROUND6_PHASE_NAMES",
    "ROUND6_PHASE_SCORING_EMPHASIS",
    "ROUND6_SCORING_WEIGHTS",
    "Round6TrendPlugin",
]

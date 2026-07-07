"""Round 2 phased-auto plugin for the trend strategy."""

from __future__ import annotations

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

ROUND2_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "coverage": 0.22,
    "edge": 0.20,
    "capture": 0.14,
    "risk": 0.08,
    "entry_quality": 0.06,
    "calmar": 0.05,
    "sharpe": 0.03,
}

ROUND2_PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    1: {
        "returns": 0.18,
        "coverage": 0.16,
        "edge": 0.26,
        "capture": 0.10,
        "risk": 0.08,
        "entry_quality": 0.12,
        "calmar": 0.06,
        "sharpe": 0.04,
    },
    2: {
        "returns": 0.18,
        "coverage": 0.18,
        "edge": 0.20,
        "capture": 0.10,
        "risk": 0.08,
        "entry_quality": 0.18,
        "calmar": 0.05,
        "sharpe": 0.03,
    },
    3: {
        "returns": 0.18,
        "coverage": 0.14,
        "edge": 0.18,
        "capture": 0.22,
        "risk": 0.08,
        "entry_quality": 0.10,
        "calmar": 0.06,
        "sharpe": 0.04,
    },
    4: {
        "returns": 0.20,
        "coverage": 0.12,
        "edge": 0.18,
        "capture": 0.24,
        "risk": 0.08,
        "entry_quality": 0.06,
        "calmar": 0.07,
        "sharpe": 0.05,
    },
    5: {
        "returns": 0.22,
        "coverage": 0.18,
        "edge": 0.20,
        "capture": 0.12,
        "risk": 0.10,
        "entry_quality": 0.06,
        "calmar": 0.07,
        "sharpe": 0.05,
    },
    6: dict(ROUND2_SCORING_WEIGHTS),
}

ROUND2_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 30.0,
    "coverage": 40.0,
    "edge": 2.0,
    "calmar": 4.0,
    "sharpe": 4.0,
    "risk": 12.0,
}

ROUND2_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 12.0),
    "total_trades": (">=", 18.0),
    "profit_factor": (">=", 1.40),
    "expectancy_r": (">=", 0.10),
}

ROUND2_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion("total_trades", ">=", 18.0),
        GateCriterion("expectancy_r", ">=", 0.20),
        GateCriterion("profit_factor", ">=", 1.60),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
    ],
    2: [
        GateCriterion("total_trades", ">=", 18.0),
        GateCriterion("expectancy_r", ">=", 0.20),
        GateCriterion("profit_factor", ">=", 1.60),
        GateCriterion("avg_mae_r", "<=", 0.35),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
    ],
    3: [
        GateCriterion("total_trades", ">=", 18.0),
        GateCriterion("profit_factor", ">=", 1.60),
        GateCriterion("exit_efficiency", ">=", 0.48),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
    ],
    4: [
        GateCriterion("total_trades", ">=", 18.0),
        GateCriterion("profit_factor", ">=", 1.60),
        GateCriterion("exit_efficiency", ">=", 0.50),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
    ],
    5: [
        GateCriterion("total_trades", ">=", 20.0),
        GateCriterion("net_return_pct", ">=", 12.0),
        GateCriterion("profit_factor", ">=", 1.60),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
        GateCriterion("sharpe_ratio", ">=", 3.0),
    ],
    6: [
        GateCriterion("total_trades", ">=", 24.0),
        GateCriterion("net_return_pct", ">=", 14.0),
        GateCriterion("profit_factor", ">=", 1.65),
        GateCriterion("max_drawdown_pct", "<=", 12.0),
        GateCriterion("exit_efficiency", ">=", 0.50),
        GateCriterion("sharpe_ratio", ">=", 3.0),
        GateCriterion("calmar_ratio", ">=", 2.4),
    ],
}

ROUND2_PHASE_NAMES: dict[int, str] = {
    1: "Signal Quality",
    2: "Entry Architecture",
    3: "Failed Follow-Through",
    4: "Winner Capture",
    5: "Regime & Allocation",
    6: "Risk & Reentry",
}


def _phase1_candidates() -> list[Experiment]:
    return [
        Experiment("disable_hammer", {"confirmation.enable_hammer": False}),
        Experiment("min_confluences_2", {"setup.min_confluences": 2}),
        Experiment("weighted_score_loose", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.35,
            "setup.min_setup_score_a": 2.05,
        }),
        Experiment("weighted_score_balanced", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.50,
            "setup.min_setup_score_a": 2.20,
        }),
        Experiment("weekly_room_1_0", {
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.0,
        }),
        Experiment("weekly_room_1_25", {
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.25,
        }),
        Experiment("quality_floor_bundle", {
            "confirmation.enable_hammer": False,
            "setup.min_confluences": 2,
        }),
        Experiment("weighted_room_bundle", {
            "confirmation.enable_hammer": False,
            "setup.min_confluences": 2,
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.35,
            "setup.min_setup_score_a": 2.05,
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.0,
        }),
        Experiment("quality_tight_bundle", {
            "confirmation.enable_hammer": False,
            "setup.min_confluences": 2,
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.50,
            "setup.min_setup_score_a": 2.20,
            "setup.weekly_room_filter_enabled": True,
            "setup.min_weekly_room_r": 1.25,
        }),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        Experiment("entry_break", {"entry.mode": "break"}),
        Experiment("entry_hybrid_grade", {"entry.mode": "hybrid_grade"}),
        Experiment("confirm_b_only", {"confirmation.require_confirmation_for_b": True}),
        Experiment("confirm_all", {"confirmation.require_confirmation": True}),
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
        }),
        Experiment("structure_only_confirmations", {
            "entry.mode": "hybrid_grade",
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_hammer": False,
            "confirmation.enable_ema_reclaim": False,
        }),
        Experiment("conservative_quality_bundle", {
            "entry.mode": "break",
            "confirmation.require_confirmation": True,
            "confirmation.require_volume_confirm": True,
            "confirmation.enforce_volume_on_trigger": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        Experiment("scratch_030_lock_2", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.30,
            "exits.scratch_floor_r": 0.05,
            "exits.scratch_min_bars": 2,
        }),
        Experiment("scratch_040_lock_3", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.40,
            "exits.scratch_floor_r": 0.10,
            "exits.scratch_min_bars": 3,
        }),
        Experiment("quick_exit_lighter", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 8,
            "exits.quick_exit_max_mfe_r": 0.25,
            "exits.quick_exit_max_r": -0.05,
        }),
        Experiment("quick_exit_balanced", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 10,
            "exits.quick_exit_max_mfe_r": 0.35,
            "exits.quick_exit_max_r": -0.10,
        }),
        Experiment("time_stop_exit_8", {
            "exits.time_stop_bars": 8,
            "exits.time_stop_min_progress_r": 0.15,
            "exits.time_stop_action": "exit",
        }),
        Experiment("time_stop_exit_10", {
            "exits.time_stop_bars": 10,
            "exits.time_stop_min_progress_r": 0.20,
            "exits.time_stop_action": "exit",
        }),
        Experiment("salvage_bundle_soft", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.30,
            "exits.scratch_floor_r": 0.05,
            "exits.scratch_min_bars": 2,
            "exits.time_stop_bars": 8,
            "exits.time_stop_min_progress_r": 0.15,
            "exits.time_stop_action": "exit",
        }),
        Experiment("salvage_bundle_balanced", {
            "exits.scratch_exit_enabled": True,
            "exits.scratch_peak_r": 0.40,
            "exits.scratch_floor_r": 0.10,
            "exits.scratch_min_bars": 3,
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 10,
            "exits.quick_exit_max_mfe_r": 0.35,
            "exits.quick_exit_max_r": -0.10,
        }),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        Experiment("structure_trail_on", {"trail.structure_trail_enabled": True}),
        Experiment("trail_activation_0_5", {
            "trail.trail_activation_r": 0.5,
            "trail.trail_activation_bars": 4,
        }),
        Experiment("trail_activation_0_6_bars6", {
            "trail.trail_activation_r": 0.6,
            "trail.trail_activation_bars": 6,
        }),
        Experiment("trail_buffers_1_0_0_2", {
            "trail.trail_buffer_wide": 1.0,
            "trail.trail_buffer_tight": 0.2,
        }),
        Experiment("trail_ceiling_1_2", {"trail.trail_r_ceiling": 1.2}),
        Experiment("tp2_1_8", {"exits.tp2_r": 1.8}),
        Experiment("tp2_1_75_frac_0_4", {
            "exits.tp2_r": 1.75,
            "exits.tp2_frac": 0.4,
        }),
        Experiment("be_min_bars_2", {"exits.be_min_bars_above": 2}),
        Experiment("capture_bundle", {
            "trail.structure_trail_enabled": True,
            "trail.trail_activation_r": 0.5,
            "trail.trail_activation_bars": 4,
            "trail.trail_buffer_wide": 1.0,
            "trail.trail_buffer_tight": 0.2,
        }),
        Experiment("harvest_bundle", {
            "trail.structure_trail_enabled": True,
            "trail.trail_activation_r": 0.5,
            "trail.trail_activation_bars": 4,
            "exits.tp2_r": 1.75,
            "exits.tp2_frac": 0.4,
            "exits.be_min_bars_above": 2,
        }),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        Experiment("require_structure", {"regime.require_structure": True}),
        Experiment("b_adx_rising", {"regime.b_adx_rising_required": True}),
        Experiment("h1_regime_off", {"regime.h1_regime_enabled": False}),
        Experiment("relative_strength_24_0", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.0,
        }),
        Experiment("relative_strength_48_0", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 48,
            "filters.relative_strength_min_delta": 0.0,
        }),
        Experiment("relative_strength_24_0_005", {
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.005,
        }),
        Experiment("funding_on_10bps", {
            "filters.funding_filter_enabled": True,
            "filters.funding_extreme_threshold": 0.001,
        }),
        Experiment("reentry_off", {"reentry.enabled": False}),
        Experiment("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        Experiment("quality_regime_bundle", {
            "regime.require_structure": True,
            "regime.b_adx_rising_required": True,
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.0,
        }),
        Experiment("regime_filter_bundle", {
            "regime.require_structure": True,
            "regime.b_adx_rising_required": True,
            "filters.relative_strength_filter_enabled": True,
            "filters.relative_strength_lookback": 24,
            "filters.relative_strength_min_delta": 0.005,
            "filters.funding_filter_enabled": True,
            "filters.funding_extreme_threshold": 0.001,
        }),
    ]


def _phase6_candidates() -> list[Experiment]:
    return [
        Experiment("risk_reset_12_10", {
            "risk.risk_pct_a": 0.012,
            "risk.risk_pct_b": 0.010,
        }),
        Experiment("risk_reset_14_11", {
            "risk.risk_pct_a": 0.014,
            "risk.risk_pct_b": 0.011,
        }),
        Experiment("risk_reset_15_12", {
            "risk.risk_pct_a": 0.015,
            "risk.risk_pct_b": 0.012,
        }),
        Experiment("max_trades_12", {"limits.max_trades_per_day": 12}),
        Experiment("max_positions_4", {"limits.max_concurrent_positions": 4}),
        Experiment("reentry_scratch_half", {
            "reentry.enabled": True,
            "reentry.cooldown_bars": 1,
            "reentry.max_loss_r": 0.5,
            "reentry.max_reentries": 1,
            "reentry.min_confluences_override": 2,
            "reentry.max_wait_bars": 4,
            "reentry.require_same_direction": True,
            "reentry.only_after_scratch_exit": True,
            "reentry.risk_scale": 0.5,
        }),
        Experiment("reentry_scratch_075", {
            "reentry.enabled": True,
            "reentry.cooldown_bars": 1,
            "reentry.max_loss_r": 0.5,
            "reentry.max_reentries": 1,
            "reentry.min_confluences_override": 2,
            "reentry.max_wait_bars": 6,
            "reentry.require_same_direction": True,
            "reentry.only_after_scratch_exit": True,
            "reentry.risk_scale": 0.75,
        }),
        Experiment("balanced_scale_bundle", {
            "risk.risk_pct_a": 0.014,
            "risk.risk_pct_b": 0.011,
            "limits.max_trades_per_day": 12,
        }),
    ]


ROUND2_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
    6: _phase6_candidates,
}


class Round2TrendPlugin(TrendPlugin):
    """Trend plugin variant for the round 2 phased-auto run."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 18.0,
            "total_trades": 36.0,
            "profit_factor": 1.8,
            "max_drawdown_pct": 10.0,
            "sharpe_ratio": 3.5,
            "calmar_ratio": 2.8,
            "exit_efficiency": 0.55,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        generator = ROUND2_PHASE_CANDIDATES.get(phase)
        if generator is None:
            raise ValueError(f"Unknown phase: {phase}")

        return PhaseSpec(
            phase_num=phase,
            name=ROUND2_PHASE_NAMES[phase],
            candidates=generator(),
            scoring_weights=dict(ROUND2_PHASE_SCORING_EMPHASIS.get(phase, ROUND2_SCORING_WEIGHTS)),
            hard_rejects=dict(ROUND2_HARD_REJECTS),
            gate_criteria=list(ROUND2_PHASE_GATE_CRITERIA[phase]),
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
            min_delta=0.004,
            focus=ROUND2_PHASE_NAMES[phase],
            max_rounds=4,
            prune_threshold=0.0,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = dict(ROUND2_IMMUTABLE_SCORING_CEILINGS)

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
    "ROUND2_HARD_REJECTS",
    "ROUND2_IMMUTABLE_SCORING_CEILINGS",
    "ROUND2_PHASE_CANDIDATES",
    "ROUND2_PHASE_GATE_CRITERIA",
    "ROUND2_PHASE_NAMES",
    "ROUND2_PHASE_SCORING_EMPHASIS",
    "ROUND2_SCORING_WEIGHTS",
    "Round2TrendPlugin",
]

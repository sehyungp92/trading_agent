"""Round 3 phased-auto plugin for the trend strategy."""

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

ROUND3_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.21,
    "coverage": 0.20,
    "edge": 0.18,
    "capture": 0.16,
    "risk": 0.08,
    "entry_quality": 0.05,
    "calmar": 0.07,
    "sharpe": 0.05,
}

ROUND3_PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    1: {
        "returns": 0.18,
        "coverage": 0.14,
        "edge": 0.26,
        "capture": 0.10,
        "risk": 0.08,
        "entry_quality": 0.10,
        "calmar": 0.08,
        "sharpe": 0.06,
    },
    2: {
        "returns": 0.18,
        "coverage": 0.18,
        "edge": 0.22,
        "capture": 0.12,
        "risk": 0.08,
        "entry_quality": 0.08,
        "calmar": 0.08,
        "sharpe": 0.06,
    },
    3: {
        "returns": 0.18,
        "coverage": 0.16,
        "edge": 0.18,
        "capture": 0.12,
        "risk": 0.08,
        "entry_quality": 0.16,
        "calmar": 0.07,
        "sharpe": 0.05,
    },
    4: {
        "returns": 0.22,
        "coverage": 0.14,
        "edge": 0.16,
        "capture": 0.22,
        "risk": 0.08,
        "entry_quality": 0.04,
        "calmar": 0.08,
        "sharpe": 0.06,
    },
    5: {
        "returns": 0.20,
        "coverage": 0.24,
        "edge": 0.16,
        "capture": 0.12,
        "risk": 0.08,
        "entry_quality": 0.06,
        "calmar": 0.08,
        "sharpe": 0.06,
    },
    6: dict(ROUND3_SCORING_WEIGHTS),
}

ROUND3_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 40.0,
    "coverage": 60.0,
    "edge": 2.8,
    "calmar": 6.0,
    "sharpe": 6.0,
    "risk": 10.0,
}

ROUND3_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 10.0),
    "total_trades": (">=", 30.0),
    "net_return_pct": (">=", 18.0),
    "profit_factor": (">=", 1.75),
    "expectancy_r": (">=", 0.20),
}

ROUND3_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion("total_trades", ">=", 30.0),
        GateCriterion("net_return_pct", ">=", 24.0),
        GateCriterion("profit_factor", ">=", 2.10),
        GateCriterion("max_drawdown_pct", "<=", 9.5),
    ],
    2: [
        GateCriterion("total_trades", ">=", 34.0),
        GateCriterion("net_return_pct", ">=", 26.0),
        GateCriterion("profit_factor", ">=", 2.00),
        GateCriterion("max_drawdown_pct", "<=", 9.5),
        GateCriterion("exit_efficiency", ">=", 0.54),
    ],
    3: [
        GateCriterion("total_trades", ">=", 34.0),
        GateCriterion("net_return_pct", ">=", 26.0),
        GateCriterion("profit_factor", ">=", 2.00),
        GateCriterion("max_drawdown_pct", "<=", 9.5),
        GateCriterion("avg_mae_r", "<=", 0.35),
    ],
    4: [
        GateCriterion("total_trades", ">=", 36.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 1.95),
        GateCriterion("max_drawdown_pct", "<=", 9.5),
        GateCriterion("exit_efficiency", ">=", 0.56),
    ],
    5: [
        GateCriterion("total_trades", ">=", 40.0),
        GateCriterion("net_return_pct", ">=", 30.0),
        GateCriterion("profit_factor", ">=", 1.90),
        GateCriterion("max_drawdown_pct", "<=", 10.0),
        GateCriterion("sharpe_ratio", ">=", 4.50),
    ],
    6: [
        GateCriterion("total_trades", ">=", 43.0),
        GateCriterion("net_return_pct", ">=", 32.0),
        GateCriterion("profit_factor", ">=", 1.85),
        GateCriterion("max_drawdown_pct", "<=", 10.0),
        GateCriterion("exit_efficiency", ">=", 0.56),
        GateCriterion("calmar_ratio", ">=", 4.50),
    ],
}

ROUND3_PHASE_NAMES: dict[int, str] = {
    1: "Signal Discrimination",
    2: "Reentry Calibration",
    3: "Entry Architecture",
    4: "Exit Harvest",
    5: "Coverage Expansion",
    6: "Allocation & Polish",
}


def _reentry_bundle(
    *,
    cooldown_bars: int,
    max_loss_r: float,
    max_reentries: int,
    min_confluences_override: int,
    max_wait_bars: int,
    risk_scale: float,
) -> dict[str, Any]:
    return {
        "reentry.enabled": True,
        "reentry.cooldown_bars": cooldown_bars,
        "reentry.max_loss_r": max_loss_r,
        "reentry.max_reentries": max_reentries,
        "reentry.min_confluences_override": min_confluences_override,
        "reentry.max_wait_bars": max_wait_bars,
        "reentry.require_same_direction": True,
        "reentry.only_after_scratch_exit": True,
        "reentry.risk_scale": risk_scale,
    }


def _phase1_candidates() -> list[Experiment]:
    return [
        Experiment("disable_hammer", {"confirmation.enable_hammer": False}),
        Experiment("btc_long_only", {"symbol_filter.btc_direction": "long_only"}),
        Experiment("sol_short_only", {"symbol_filter.sol_direction": "short_only"}),
        Experiment("weighted_score_b_1_50_a_2_25", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.50,
            "setup.min_setup_score_a": 2.25,
        }),
        Experiment("weighted_score_b_1_60_a_2_35", {
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.60,
            "setup.min_setup_score_a": 2.35,
        }),
        Experiment("confirm_b_structure_only", {
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_hammer": False,
            "confirmation.enable_engulfing": False,
            "confirmation.enable_ema_reclaim": False,
            "confirmation.enable_structure_break": True,
        }),
        Experiment("confirm_b_ema_or_structure", {
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_hammer": False,
            "confirmation.enable_engulfing": False,
        }),
        Experiment("directional_filter_bundle", {
            "confirmation.enable_hammer": False,
            "symbol_filter.btc_direction": "long_only",
            "symbol_filter.sol_direction": "short_only",
        }),
        Experiment("quality_directional_bundle", {
            "confirmation.enable_hammer": False,
            "symbol_filter.btc_direction": "long_only",
            "symbol_filter.sol_direction": "short_only",
            "setup.use_weighted_confluence": True,
            "setup.min_setup_score_b": 1.50,
            "setup.min_setup_score_a": 2.25,
        }),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        Experiment("reentry_wait4_conf1_risk075", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.35,
            max_reentries=1,
            min_confluences_override=1,
            max_wait_bars=4,
            risk_scale=0.75,
        )),
        Experiment("reentry_wait6_conf1_risk050", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.35,
            max_reentries=1,
            min_confluences_override=1,
            max_wait_bars=6,
            risk_scale=0.50,
        )),
        Experiment("reentry_wait8_conf1_risk075", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.50,
            max_reentries=1,
            min_confluences_override=1,
            max_wait_bars=8,
            risk_scale=0.75,
        )),
        Experiment("reentry_wait6_conf0_risk075", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.35,
            max_reentries=1,
            min_confluences_override=0,
            max_wait_bars=6,
            risk_scale=0.75,
        )),
        Experiment("reentry_wait8_conf0_risk050", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.35,
            max_reentries=1,
            min_confluences_override=0,
            max_wait_bars=8,
            risk_scale=0.50,
        )),
        Experiment("reentry_wait4_conf2_risk075", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.35,
            max_reentries=1,
            min_confluences_override=2,
            max_wait_bars=4,
            risk_scale=0.75,
        )),
        Experiment("reentry_wait6_double_risk050", _reentry_bundle(
            cooldown_bars=1,
            max_loss_r=0.50,
            max_reentries=2,
            min_confluences_override=1,
            max_wait_bars=6,
            risk_scale=0.50,
        )),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        Experiment("confirm_preferred", {"entry.mode": "confirm_preferred"}),
        Experiment("reentry_break", {"entry.mode": "reentry_break"}),
        Experiment("reentry_confirm_preferred", {"entry.mode": "reentry_confirm_preferred"}),
        Experiment("hybrid_grade", {"entry.mode": "hybrid_grade"}),
        Experiment("confirm_preferred_ttl1", {
            "entry.mode": "confirm_preferred",
            "entry.max_bars_after_confirmation": 1,
        }),
        Experiment("confirm_preferred_b_confirm", {
            "entry.mode": "confirm_preferred",
            "confirmation.require_confirmation_for_b": True,
        }),
        Experiment("confirm_preferred_volume", {
            "entry.mode": "confirm_preferred",
            "confirmation.require_volume_confirm": True,
            "confirmation.enforce_volume_on_trigger": True,
            "confirmation.volume_threshold_mult": 1.05,
        }),
        Experiment("confirm_preferred_structure_only", {
            "entry.mode": "confirm_preferred",
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_hammer": False,
            "confirmation.enable_engulfing": False,
            "confirmation.enable_ema_reclaim": False,
            "confirmation.enable_structure_break": True,
        }),
        Experiment("reentry_break_structure_bundle", {
            "entry.mode": "reentry_break",
            "confirmation.require_confirmation_for_b": True,
            "confirmation.enable_hammer": False,
            "confirmation.enable_engulfing": False,
        }),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        Experiment("trail_activation_0_5_bars4", {
            "trail.trail_activation_r": 0.5,
            "trail.trail_activation_bars": 4,
        }),
        Experiment("trail_buffers_1_0_0_15", {
            "trail.trail_buffer_wide": 1.0,
            "trail.trail_buffer_tight": 0.15,
        }),
        Experiment("structure_trail_on", {"trail.structure_trail_enabled": True}),
        Experiment("be_min_bars_2", {"exits.be_min_bars_above": 2}),
        Experiment("tp2_1_65", {"exits.tp2_r": 1.65}),
        Experiment("tp2_1_70_frac_0_45", {
            "exits.tp2_r": 1.70,
            "exits.tp2_frac": 0.45,
        }),
        Experiment("scratch_025_flat_2", {
            "exits.scratch_peak_r": 0.25,
            "exits.scratch_floor_r": 0.0,
            "exits.scratch_min_bars": 2,
        }),
        Experiment("quick_exit_10_025_010", {
            "exits.quick_exit_bars": 10,
            "exits.quick_exit_max_mfe_r": 0.25,
            "exits.quick_exit_max_r": -0.10,
        }),
        Experiment("harvest_bundle", {
            "trail.structure_trail_enabled": True,
            "trail.trail_activation_r": 0.5,
            "trail.trail_activation_bars": 4,
            "trail.trail_buffer_wide": 1.0,
            "trail.trail_buffer_tight": 0.15,
            "exits.tp2_r": 1.70,
            "exits.tp2_frac": 0.45,
            "exits.be_min_bars_above": 2,
        }),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        Experiment("orderly_volume_1_00", {"setup.orderly_max_countertrend_volume_ratio": 1.00}),
        Experiment("orderly_body_0_90", {"setup.orderly_max_body_frac": 0.90}),
        Experiment("pullback_bars_18", {"setup.pullback_max_bars": 18}),
        Experiment("pullback_retrace_0_80", {"setup.pullback_max_retrace": 0.80}),
        Experiment("impulse_atr_0_75", {"setup.impulse_min_atr_move": 0.75}),
        Experiment("regime_b_adx_7", {"regime.b_min_adx": 7.0}),
        Experiment("h1_adx_20", {"regime.h1_min_adx": 20.0}),
        Experiment("max_trades_12", {"limits.max_trades_per_day": 12}),
        Experiment("coverage_bundle_soft", {
            "setup.orderly_max_countertrend_volume_ratio": 1.00,
            "setup.orderly_max_body_frac": 0.90,
            "setup.pullback_max_bars": 18,
        }),
        Experiment("coverage_bundle_regime", {
            "regime.b_min_adx": 7.0,
            "regime.h1_min_adx": 20.0,
            "limits.max_trades_per_day": 12,
        }),
    ]


def _phase6_candidates() -> list[Experiment]:
    return [
        Experiment("risk_b_0_019", {"risk.risk_pct_b": 0.019}),
        Experiment("risk_b_0_020", {"risk.risk_pct_b": 0.020}),
        Experiment("max_trades_12", {"limits.max_trades_per_day": 12}),
        Experiment("max_positions_4", {"limits.max_concurrent_positions": 4}),
        Experiment("risk_b_0_019_maxtrades12", {
            "risk.risk_pct_b": 0.019,
            "limits.max_trades_per_day": 12,
        }),
        Experiment("risk_b_0_020_maxtrades12", {
            "risk.risk_pct_b": 0.020,
            "limits.max_trades_per_day": 12,
        }),
    ]


ROUND3_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
    6: _phase6_candidates,
}


class Round3TrendPlugin(TrendPlugin):
    """Trend plugin variant for the round-3 phased run."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 36.0,
            "total_trades": 50.0,
            "profit_factor": 2.20,
            "max_drawdown_pct": 9.0,
            "sharpe_ratio": 4.80,
            "calmar_ratio": 4.80,
            "exit_efficiency": 0.58,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        generator = ROUND3_PHASE_CANDIDATES.get(phase)
        if generator is None:
            raise ValueError(f"Unknown phase: {phase}")

        return PhaseSpec(
            phase_num=phase,
            name=ROUND3_PHASE_NAMES[phase],
            candidates=generator(),
            scoring_weights=dict(ROUND3_PHASE_SCORING_EMPHASIS.get(phase, ROUND3_SCORING_WEIGHTS)),
            hard_rejects=dict(ROUND3_HARD_REJECTS),
            gate_criteria=list(ROUND3_PHASE_GATE_CRITERIA[phase]),
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=1,
                max_diagnostic_retries=1,
                focus_metrics=[
                    "total_trades",
                    "net_return_pct",
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
            max_rounds=4,
            prune_threshold=0.0,
            focus=ROUND3_PHASE_NAMES[phase],
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = dict(ROUND3_IMMUTABLE_SCORING_CEILINGS)

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
    "ROUND3_HARD_REJECTS",
    "ROUND3_IMMUTABLE_SCORING_CEILINGS",
    "ROUND3_PHASE_CANDIDATES",
    "ROUND3_PHASE_GATE_CRITERIA",
    "ROUND3_PHASE_NAMES",
    "ROUND3_PHASE_SCORING_EMPHASIS",
    "ROUND3_SCORING_WEIGHTS",
    "Round3TrendPlugin",
]

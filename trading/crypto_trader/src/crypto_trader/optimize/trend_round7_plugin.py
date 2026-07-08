"""Round 7 phased-auto plugin for the trend strategy."""

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

ROUND7_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.18,
    "coverage": 0.28,
    "edge": 0.18,
    "capture": 0.12,
    "calmar": 0.08,
    "sharpe": 0.06,
    "risk": 0.05,
    "entry_quality": 0.05,
}

ROUND7_PHASE_SCORING_EMPHASIS: dict[int, dict[str, float]] = {
    1: {
        "returns": 0.16,
        "coverage": 0.32,
        "edge": 0.18,
        "capture": 0.10,
        "calmar": 0.06,
        "sharpe": 0.05,
        "risk": 0.05,
        "entry_quality": 0.08,
    },
    2: {
        "returns": 0.16,
        "coverage": 0.30,
        "edge": 0.18,
        "capture": 0.10,
        "calmar": 0.06,
        "sharpe": 0.05,
        "risk": 0.05,
        "entry_quality": 0.10,
    },
    3: {
        "returns": 0.17,
        "coverage": 0.30,
        "edge": 0.18,
        "capture": 0.10,
        "calmar": 0.07,
        "sharpe": 0.05,
        "risk": 0.05,
        "entry_quality": 0.08,
    },
    4: {
        "returns": 0.18,
        "coverage": 0.28,
        "edge": 0.18,
        "capture": 0.11,
        "calmar": 0.07,
        "sharpe": 0.05,
        "risk": 0.05,
        "entry_quality": 0.08,
    },
    5: {
        "returns": 0.18,
        "coverage": 0.26,
        "edge": 0.20,
        "capture": 0.10,
        "calmar": 0.08,
        "sharpe": 0.06,
        "risk": 0.06,
        "entry_quality": 0.06,
    },
    6: dict(ROUND7_SCORING_WEIGHTS),
}

ROUND7_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "returns": 35.0,
    "coverage": 75.0,
    "edge": 2.0,
    "calmar": 4.0,
    "sharpe": 5.0,
    "risk": 12.0,
}

ROUND7_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "max_drawdown_pct": ("<=", 9.0),
    "total_trades": (">=", 50.0),
    "profit_factor": (">=", 2.05),
    "net_return_pct": (">=", 28.0),
    "exit_efficiency": (">=", 0.55),
    "expectancy_r": (">=", 0.24),
}

ROUND7_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
    ],
    2: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
        GateCriterion("avg_mae_r", "<=", 0.40),
    ],
    3: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
        GateCriterion("exit_efficiency", ">=", 0.55),
    ],
    4: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
        GateCriterion("exit_efficiency", ">=", 0.56),
    ],
    5: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
        GateCriterion("sharpe_ratio", ">=", 4.0),
    ],
    6: [
        GateCriterion("total_trades", ">=", 50.0),
        GateCriterion("net_return_pct", ">=", 28.0),
        GateCriterion("profit_factor", ">=", 2.05),
        GateCriterion("max_drawdown_pct", "<=", 9.0),
        GateCriterion("exit_efficiency", ">=", 0.56),
        GateCriterion("sharpe_ratio", ">=", 4.0),
        GateCriterion("calmar_ratio", ">=", 3.5),
    ],
}

ROUND7_PHASE_NAMES: dict[int, str] = {
    1: "Scratch Reentry Recovery",
    2: "Reentry Calibration",
    3: "Orderly Micro Relaxation",
    4: "Orderly Recovery Bundles",
    5: "Regime Frequency",
    6: "Risk & Final Polish",
}


def _scratch_reentry_bundle(
    *,
    cooldown_bars: int,
    max_loss_r: float,
    max_reentries: int,
    min_confluences_override: int,
    max_wait_bars: int,
    risk_scale: float,
) -> dict[str, Any]:
    return {
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
        Experiment(
            "scratch_reentry_wait4_conf1_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.25,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=4,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait4_conf1_fullrisk",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.25,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=4,
                risk_scale=1.0,
            ),
        ),
        Experiment(
            "scratch_reentry_wait6_conf1_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=6,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait6_conf0_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=0,
                max_wait_bars=6,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait6_double_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=2,
                min_confluences_override=1,
                max_wait_bars=6,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait8_conf1_risk050",
            _scratch_reentry_bundle(
                cooldown_bars=2,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=8,
                risk_scale=0.50,
            ),
        ),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        Experiment(
            "scratch_reentry_wait8_conf1_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=8,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait12_conf1_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=12,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait8_conf0_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=0,
                max_wait_bars=8,
                risk_scale=0.75,
            ),
        ),
        Experiment(
            "scratch_reentry_wait6_conf1_risk050",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=6,
                risk_scale=0.50,
            ),
        ),
        Experiment(
            "scratch_reentry_wait6_conf1_fullrisk",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.35,
                max_reentries=1,
                min_confluences_override=1,
                max_wait_bars=6,
                risk_scale=1.0,
            ),
        ),
        Experiment(
            "scratch_reentry_wait8_double_risk075",
            _scratch_reentry_bundle(
                cooldown_bars=1,
                max_loss_r=0.50,
                max_reentries=2,
                min_confluences_override=1,
                max_wait_bars=8,
                risk_scale=0.75,
            ),
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        Experiment("orderly_volume_1_00", {"setup.orderly_max_countertrend_volume_ratio": 1.00}),
        Experiment("orderly_volume_1_05", {"setup.orderly_max_countertrend_volume_ratio": 1.05}),
        Experiment("orderly_body_0_90", {"setup.orderly_max_body_frac": 0.90}),
        Experiment("orderly_body_0_95", {"setup.orderly_max_body_frac": 0.95}),
        Experiment("orderly_countertrend_1", {"setup.orderly_min_countertrend_bars": 1}),
        Experiment("pullback_bars_18", {"setup.pullback_max_bars": 18}),
        Experiment("pullback_retrace_0_80", {"setup.pullback_max_retrace": 0.80}),
        Experiment("impulse_atr_0_75", {"setup.impulse_min_atr_move": 0.75}),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        Experiment("orderly_vol1_body090", {
            "setup.orderly_max_countertrend_volume_ratio": 1.00,
            "setup.orderly_max_body_frac": 0.90,
        }),
        Experiment("orderly_vol1_pullback18", {
            "setup.orderly_max_countertrend_volume_ratio": 1.00,
            "setup.pullback_max_bars": 18,
        }),
        Experiment("orderly_body095_retrace080", {
            "setup.orderly_max_body_frac": 0.95,
            "setup.pullback_max_retrace": 0.80,
        }),
        Experiment("orderly_vol105_body090_pullback18", {
            "setup.orderly_max_countertrend_volume_ratio": 1.05,
            "setup.orderly_max_body_frac": 0.90,
            "setup.pullback_max_bars": 18,
        }),
        Experiment("orderly_counter1_vol1", {
            "setup.orderly_min_countertrend_bars": 1,
            "setup.orderly_max_countertrend_volume_ratio": 1.00,
        }),
        Experiment("orderly_soft_bundle", {
            "setup.orderly_max_countertrend_volume_ratio": 1.00,
            "setup.orderly_max_body_frac": 0.90,
            "setup.pullback_max_retrace": 0.80,
        }),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        Experiment("b_adx_7", {"regime.b_min_adx": 7.0}),
        Experiment("h1_adx_20", {"regime.h1_min_adx": 20.0}),
        Experiment("adx_looser_bundle", {
            "regime.b_min_adx": 7.0,
            "regime.h1_min_adx": 20.0,
        }),
        Experiment("max_trades_12", {"limits.max_trades_per_day": 12}),
        Experiment("b_adx_7_disable_hammer", {
            "regime.b_min_adx": 7.0,
            "confirmation.enable_hammer": False,
        }),
        Experiment("h1_adx_20_disable_hammer", {
            "regime.h1_min_adx": 20.0,
            "confirmation.enable_hammer": False,
        }),
    ]


def _phase6_candidates() -> list[Experiment]:
    return [
        Experiment("risk_b_0_019", {"risk.risk_pct_b": 0.019}),
        Experiment("risk_b_0_020", {"risk.risk_pct_b": 0.020}),
        Experiment("risk_b_0_019_maxtrades12", {
            "risk.risk_pct_b": 0.019,
            "limits.max_trades_per_day": 12,
        }),
        Experiment("risk_b_0_019_disable_hammer", {
            "risk.risk_pct_b": 0.019,
            "confirmation.enable_hammer": False,
        }),
        Experiment("risk_b_0_020_disable_hammer", {
            "risk.risk_pct_b": 0.020,
            "confirmation.enable_hammer": False,
        }),
        Experiment("max_trades_12_disable_hammer", {
            "limits.max_trades_per_day": 12,
            "confirmation.enable_hammer": False,
        }),
    ]


ROUND7_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
    6: _phase6_candidates,
}


class Round7TrendPlugin(TrendPlugin):
    """Trend plugin variant for the round 7 phased-auto run."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 30.0,
            "total_trades": 72.0,
            "profit_factor": 2.30,
            "max_drawdown_pct": 8.0,
            "sharpe_ratio": 4.25,
            "calmar_ratio": 3.75,
            "exit_efficiency": 0.60,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        generator = ROUND7_PHASE_CANDIDATES.get(phase)
        if generator is None:
            raise ValueError(f"Unknown phase: {phase}")

        return PhaseSpec(
            phase_num=phase,
            name=ROUND7_PHASE_NAMES[phase],
            candidates=generator(),
            scoring_weights=dict(ROUND7_PHASE_SCORING_EMPHASIS.get(phase, ROUND7_SCORING_WEIGHTS)),
            hard_rejects=dict(ROUND7_HARD_REJECTS),
            min_delta=0.004,
            max_rounds=4,
            prune_threshold=0.0,
            gate_criteria=list(ROUND7_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda metrics, _phase=phase: self._gate_criteria_fn(metrics, _phase),
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
            focus=ROUND7_PHASE_NAMES[phase],
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float],
        hard_rejects: dict[str, tuple[str, float]],
    ) -> EvaluateFn:
        ceilings = dict(ROUND7_IMMUTABLE_SCORING_CEILINGS)

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
    "ROUND7_HARD_REJECTS",
    "ROUND7_IMMUTABLE_SCORING_CEILINGS",
    "ROUND7_PHASE_CANDIDATES",
    "ROUND7_PHASE_GATE_CRITERIA",
    "ROUND7_PHASE_NAMES",
    "ROUND7_PHASE_SCORING_EMPHASIS",
    "ROUND7_SCORING_WEIGHTS",
    "Round7TrendPlugin",
]

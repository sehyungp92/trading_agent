"""Breakout canonical round 2 phased optimization from the round-1 baseline.

This canonical round 2 is designed to improve the quality of captured alpha
rather than simply scaling risk. It starts from the canonical round-1
optimized config and keeps a single immutable score across all phases so each
phase is judged against the same definition of quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crypto_trader.optimize.breakout_plugin import BreakoutPlugin
from crypto_trader.optimize.breakout_round3_pre_round1 import build_backtest_config
from crypto_trader.optimize.parallel import evaluate_parallel
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GateCriterion,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)
from crypto_trader.strategy.breakout.config import BreakoutConfig

PHASE_NAMES: dict[int, str] = {
    1: "Signal Discrimination",
    2: "Entry Quality",
    3: "Capture & Management",
    4: "Frequency Expansion",
    5: "Risk & Sizing",
    6: "Finetune",
}

IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "coverage": 0.25,
    "returns": 0.20,
    "edge": 0.18,
    "capture": 0.15,
    "sharpe": 0.12,
    "risk": 0.10,
}

IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 24.0,
    "returns": 12.0,
    "edge": 2.5,
    "sharpe": 2.0,
    "risk": 12.0,
}

IMMUTABLE_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 14.0),
    "profit_factor": (">=", 1.10),
    "max_drawdown_pct": ("<=", 12.0),
}

PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.15),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.15),
        GateCriterion(metric="avg_mae_r", operator=">=", threshold=-0.40),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.20),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.20),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.35),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=1.40),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    6: [
        GateCriterion(metric="net_return_pct", operator=">=", threshold=9.0),
        GateCriterion(metric="total_trades", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.35),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=1.40),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
}


def load_breakout_strategy(path: Path) -> BreakoutConfig:
    """Load a breakout config from an optimized config JSON artifact."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return BreakoutConfig.from_dict(payload["strategy"])


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp("eth_long_only", {"symbol_filter.eth_direction": "long_only"}),
        _exp("eth_disabled", {"symbol_filter.eth_direction": "disabled"}),
        _exp("sol_short_only", {"symbol_filter.sol_direction": "short_only"}),
        _exp("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        _exp("min_b_conf_1", {"setup.min_confluences_b": 1}),
        _exp("min_a_conf_3", {"setup.min_confluences_a": 3}),
        _exp("min_a_plus_conf_5", {"setup.min_confluences_a_plus": 5}),
        _exp("body_ratio_045", {"setup.body_ratio_min": 0.45}),
        _exp("body_ratio_050", {"setup.body_ratio_min": 0.50}),
        _exp("body_ratio_055", {"setup.body_ratio_min": 0.55}),
        _exp("require_vol_surge", {"setup.require_volume_surge": True}),
        _exp("model1_vol_115", {"confirmation.model1_min_volume_mult": 1.15}),
        _exp("model1_vol_130", {"confirmation.model1_min_volume_mult": 1.30}),
        _exp("room_r_b_14", {"setup.min_room_r_b": 1.4}),
        _exp("room_r_b_16", {"setup.min_room_r_b": 1.6}),
        _exp("room_r_a_20", {"setup.min_room_r_a": 2.0}),
        _exp("room_r_a_22", {"setup.min_room_r_a": 2.2}),
        _exp("h4_adx_15", {"context.h4_adx_threshold": 15.0}),
        _exp("no_countertrend", {"context.allow_countertrend": False}),
        _exp("min_breakout_atr_03", {"setup.min_breakout_atr": 0.3}),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp("model2_off", {"confirmation.enable_model2": False}),
        _exp("retest_bars_3", {"confirmation.retest_max_bars": 3}),
        _exp("retest_bars_4", {"confirmation.retest_max_bars": 4}),
        _exp("retest_zone_025", {"confirmation.retest_zone_atr": 0.25}),
        _exp("retest_zone_035", {"confirmation.retest_zone_atr": 0.35}),
        _exp("retest_rejection_on", {"confirmation.retest_require_rejection": True}),
        _exp(
            "retest_volume_decline_on",
            {"confirmation.retest_require_volume_decline": True},
        ),
        _exp(
            "strict_retest_stack",
            {
                "confirmation.retest_require_rejection": True,
                "confirmation.retest_require_volume_decline": True,
                "confirmation.retest_max_bars": 4,
                "confirmation.retest_zone_atr": 0.35,
            },
        ),
        _exp(
            "model2_break_entry",
            {
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
            },
        ),
        _exp(
            "model2_break_entry_strict",
            {
                "confirmation.retest_require_rejection": True,
                "confirmation.retest_require_volume_decline": True,
                "confirmation.retest_max_bars": 4,
                "confirmation.retest_zone_atr": 0.35,
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
            },
        ),
        _exp("entry_ttl_1", {"entry.max_bars_after_signal": 1}),
        _exp("entry_ttl_2", {"entry.max_bars_after_signal": 2}),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp("quick_exit_on", {"exits.quick_exit_enabled": True}),
        _exp(
            "quick_exit_stack",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.3,
                "exits.quick_exit_max_r": -0.1,
            },
        ),
        _exp(
            "quick_exit_bars_6",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 6,
            },
        ),
        _exp("trail_act_r_025", {"trail.trail_activation_r": 0.25}),
        _exp("trail_act_r_035", {"trail.trail_activation_r": 0.35}),
        _exp("trail_act_bars_3", {"trail.trail_activation_bars": 3}),
        _exp("trail_act_bars_4", {"trail.trail_activation_bars": 4}),
        _exp("trail_tight_010", {"trail.trail_buffer_tight": 0.10}),
        _exp("trail_tight_015", {"trail.trail_buffer_tight": 0.15}),
        _exp("trail_tight_020", {"trail.trail_buffer_tight": 0.20}),
        _exp("structure_trail_off", {"trail.structure_trail_enabled": False}),
        _exp("tp1_frac_025", {"exits.tp1_frac": 0.25}),
        _exp("tp1_frac_040", {"exits.tp1_frac": 0.40}),
        _exp("be_buffer_010", {"exits.be_buffer_r": 0.10}),
        _exp("be_buffer_020", {"exits.be_buffer_r": 0.20}),
        _exp("be_buffer_035", {"exits.be_buffer_r": 0.35}),
        _exp("time_stop_reduce", {"exits.time_stop_action": "reduce"}),
        _exp("time_stop_12", {"exits.time_stop_bars": 12}),
        _exp("stop_use_farther_off", {"stops.use_farther": False}),
        _exp("stop_atr_08", {"stops.atr_mult": 0.8}),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp("min_bars_zone_4", {"balance.min_bars_in_zone": 4}),
        _exp("min_bars_zone_5", {"balance.min_bars_in_zone": 5}),
        _exp("zone_width_10", {"balance.zone_width_atr": 1.0}),
        _exp("lookback_24", {"profile.lookback_bars": 24}),
        _exp("lookback_48", {"profile.lookback_bars": 48}),
        _exp("zone_age_36", {"balance.max_zone_age_bars": 36}),
        _exp("dedup_02", {"balance.dedup_atr_frac": 0.2}),
        _exp("lvn_runway_020", {"setup.min_lvn_runway_atr": 0.20}),
        _exp("lvn_runway_025", {"setup.min_lvn_runway_atr": 0.25}),
        _exp("reentry_cooldown_2", {"reentry.cooldown_bars": 2}),
        _exp("reentry_cooldown_4", {"reentry.cooldown_bars": 4}),
        _exp("reentry_max_2", {"reentry.max_reentries": 2}),
        _exp("max_trades_per_day_5", {"limits.max_trades_per_day": 5}),
        _exp("max_concurrent_4", {"limits.max_concurrent_positions": 4}),
        _exp("hvn_thresh_12", {"profile.hvn_threshold_pct": 1.2}),
        _exp("min_breakout_atr_015", {"setup.min_breakout_atr": 0.15}),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp("risk_b_020", {"risk.risk_pct_b": 0.020}),
        _exp("risk_b_023", {"risk.risk_pct_b": 0.023}),
        _exp("risk_b_024", {"risk.risk_pct_b": 0.024}),
        _exp("risk_b_026", {"risk.risk_pct_b": 0.026}),
        _exp("risk_a_010", {"risk.risk_pct_a": 0.010}),
        _exp("risk_a_0125", {"risk.risk_pct_a": 0.0125}),
        _exp("risk_a_015", {"risk.risk_pct_a": 0.015}),
        _exp("risk_a_plus_010", {"risk.risk_pct_a_plus": 0.010}),
        _exp("risk_a_plus_0125", {"risk.risk_pct_a_plus": 0.0125}),
        _exp("risk_a_plus_015", {"risk.risk_pct_a_plus": 0.015}),
        _exp("max_consecutive_losses_3", {"limits.max_consecutive_losses": 3}),
        _exp("max_daily_loss_020", {"limits.max_daily_loss_pct": 0.020}),
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


class BreakoutRound5PhasedPlugin(BreakoutPlugin):
    """Round-5 breakout optimization seeded from the round-4 winner."""

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 10.0,
            "total_trades": 22.0,
            "profit_factor": 1.6,
            "exit_efficiency": 0.60,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 10.0,
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
                strategy_type="breakout",
                ceilings=ceilings,
            )

        return evaluate_fn

"""Broad breakout round seeded from the live round_2 baseline.

This round keeps the full BTC/ETH/SOL universe and both directions. The search
space is restricted to broad structural and incremental changes that can
improve signal quality, entry quality, failure handling, and trade frequency
without leaning on asset-specific pruning.
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
    1: "Global Branch Audit",
    2: "Signal Discrimination",
    3: "Context Discipline",
    4: "Entry Architecture",
    5: "Failure Handling",
    6: "Structural Frequency",
    7: "Finetune",
}

ROUND8_IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "coverage": 0.20,
    "returns": 0.18,
    "capture": 0.16,
    "entry_quality": 0.14,
    "edge": 0.12,
    "sharpe": 0.08,
    "calmar": 0.06,
    "risk": 0.06,
}

ROUND8_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 28.0,
    "returns": 25.0,
    "edge": 4.5,
    "sharpe": 3.5,
    "calmar": 5.5,
    "risk": 10.0,
}

ROUND8_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 14.0),
    "profit_factor": (">=", 1.45),
    "expectancy_r": (">=", 0.25),
    "exit_efficiency": (">=", 0.50),
    "max_drawdown_pct": ("<=", 9.0),
}

ROUND8_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=15.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.8),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=15.5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.9),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.31),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.0),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=15.5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.9),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.31),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.0),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=15.8),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.9),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.32),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.55),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=16.3),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.0),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.33),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.56),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    6: [
        GateCriterion(metric="total_trades", operator=">=", threshold=18.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=16.8),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.95),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.32),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.55),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
    ],
    7: [
        GateCriterion(metric="total_trades", operator=">=", threshold=18.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=17.5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.0),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.33),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.2),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.56),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.5),
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
        _exp("relaxed_body_off", {"setup.relaxed_body_enabled": False}),
        _exp("relaxed_body_min_040", {"setup.relaxed_body_min": 0.40}),
        _exp(
            "relaxed_body_conf_6",
            {"setup.relaxed_body_min_confluences": 6},
        ),
        _exp("relaxed_body_room_16", {"setup.relaxed_body_min_room_r": 1.6}),
        _exp(
            "relaxed_body_strict_core",
            {
                "setup.relaxed_body_min": 0.40,
                "setup.relaxed_body_min_confluences": 6,
            },
        ),
        _exp(
            "relaxed_body_full_strict",
            {
                "setup.relaxed_body_min": 0.40,
                "setup.relaxed_body_min_confluences": 6,
                "setup.relaxed_body_min_room_r": 1.6,
            },
        ),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp("body_ratio_070", {"setup.body_ratio_min": 0.70}),
        _exp("model1_vol_110", {"confirmation.model1_min_volume_mult": 1.10}),
        _exp("model1_vol_115", {"confirmation.model1_min_volume_mult": 1.15}),
        _exp("model1_vol_120", {"confirmation.model1_min_volume_mult": 1.20}),
        _exp("min_conf_b_1", {"setup.min_confluences_b": 1}),
        _exp("require_vol_surge", {"setup.require_volume_surge": True}),
        _exp("min_bo_atr_025", {"setup.min_breakout_atr": 0.25}),
        _exp("min_bo_atr_030", {"setup.min_breakout_atr": 0.30}),
        _exp(
            "quality_bundle_light",
            {
                "confirmation.model1_min_volume_mult": 1.10,
                "setup.min_confluences_b": 1,
            },
        ),
        _exp(
            "quality_bundle_strict",
            {
                "confirmation.model1_min_volume_mult": 1.10,
                "setup.min_confluences_b": 1,
                "setup.min_breakout_atr": 0.25,
            },
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp("no_countertrend", {"context.allow_countertrend": False}),
        _exp("h4_adx_15", {"context.h4_adx_threshold": 15.0}),
        _exp("h4_adx_18", {"context.h4_adx_threshold": 18.0}),
        _exp(
            "no_countertrend_h4_adx_15",
            {
                "context.allow_countertrend": False,
                "context.h4_adx_threshold": 15.0,
            },
        ),
        _exp(
            "no_countertrend_h4_adx_18",
            {
                "context.allow_countertrend": False,
                "context.h4_adx_threshold": 18.0,
            },
        ),
    ]


def _phase4_candidates() -> list[Experiment]:
    strict_retest = {
        "confirmation.retest_require_rejection": True,
        "confirmation.retest_require_volume_decline": True,
        "confirmation.retest_zone_atr": 0.35,
        "confirmation.retest_max_bars": 4,
    }
    return [
        _exp(
            "retest_rejection_on",
            {"confirmation.retest_require_rejection": True},
        ),
        _exp(
            "retest_volume_decline_on",
            {"confirmation.retest_require_volume_decline": True},
        ),
        _exp("retest_zone_035", {"confirmation.retest_zone_atr": 0.35}),
        _exp("retest_bars_4", {"confirmation.retest_max_bars": 4}),
        _exp("entry_ttl_2", {"entry.max_bars_after_signal": 2}),
        _exp("strict_retest_stack", strict_retest),
        _exp(
            "model2_break_entry",
            {
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
            },
        ),
        _exp(
            "strict_break_entry",
            {
                **strict_retest,
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
                "entry.max_bars_after_signal": 2,
            },
        ),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp(
            "quick_exit_4_040_000",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.40,
                "exits.quick_exit_max_r": 0.0,
            },
        ),
        _exp(
            "quick_exit_4_050_neg005",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.50,
                "exits.quick_exit_max_r": -0.05,
            },
        ),
        _exp(
            "quick_exit_5_040_000",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 5,
                "exits.quick_exit_max_mfe_r": 0.40,
                "exits.quick_exit_max_r": 0.0,
            },
        ),
        _exp(
            "quick_exit_5_050_neg005",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 5,
                "exits.quick_exit_max_mfe_r": 0.50,
                "exits.quick_exit_max_r": -0.05,
            },
        ),
        _exp(
            "early_lock_055_000",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.55,
                "exits.early_lock_stop_r": 0.0,
            },
        ),
        _exp(
            "early_lock_060_005",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.60,
                "exits.early_lock_stop_r": 0.05,
            },
        ),
        _exp(
            "time_stop_reduce_12",
            {
                "exits.time_stop_action": "reduce",
                "exits.time_stop_bars": 12,
                "exits.time_stop_min_progress_r": 0.25,
            },
        ),
        _exp(
            "quick_exit_4_040_000_early_lock_055_000",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.40,
                "exits.quick_exit_max_r": 0.0,
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.55,
                "exits.early_lock_stop_r": 0.0,
            },
        ),
    ]


def _phase6_candidates() -> list[Experiment]:
    return [
        _exp("lookback_40", {"profile.lookback_bars": 40}),
        _exp("lookback_42", {"profile.lookback_bars": 42}),
        _exp("zone_age_40", {"balance.max_zone_age_bars": 40}),
        _exp("zone_age_42", {"balance.max_zone_age_bars": 42}),
        _exp(
            "lookback_40_zone_40",
            {
                "profile.lookback_bars": 40,
                "balance.max_zone_age_bars": 40,
            },
        ),
        _exp("min_bars_zone_5", {"balance.min_bars_in_zone": 5}),
        _exp(
            "guarded_reentry_2",
            {
                "reentry.cooldown_bars": 2,
                "reentry.min_confluences_override": 1,
            },
        ),
        _exp(
            "guarded_reentry_2_max2",
            {
                "reentry.cooldown_bars": 2,
                "reentry.max_reentries": 2,
                "reentry.min_confluences_override": 1,
            },
        ),
    ]


def _phase7_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for key, val in cumulative.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if key.startswith("risk.") or key.startswith("limits."):
            continue
        if "risk_scale" in key:
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
            experiments.append(
                _exp(f"finetune_{key.split('.')[-1]}_{suffix}", {key: new_val})
            )
    return experiments


_PHASE_CANDIDATES = {
    1: _phase1_candidates,
    2: _phase2_candidates,
    3: _phase3_candidates,
    4: _phase4_candidates,
    5: _phase5_candidates,
    6: _phase6_candidates,
}


class BreakoutRound8FromRound2Plugin(BreakoutPlugin):
    """Broad breakout phased optimizer seeded from the live round_2 baseline."""

    @property
    def num_phases(self) -> int:
        return 7

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 20.0,
            "total_trades": 24.0,
            "profit_factor": 2.4,
            "expectancy_r": 0.40,
            "exit_efficiency": 0.60,
            "max_drawdown_pct": 8.5,
        }

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        if phase == 7:
            cumulative = state.cumulative_mutations if state else {}
            candidates = _phase7_candidates(cumulative)
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
            scoring_weights=dict(ROUND8_IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND8_HARD_REJECTS),
            min_delta=0.006,
            max_rounds=4 if phase < 7 else 3,
            prune_threshold=0.0,
            gate_criteria=list(ROUND8_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda _m, _p=phase: list(ROUND8_PHASE_GATE_CRITERIA[_p]),
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
        ceilings = ROUND8_IMMUTABLE_SCORING_CEILINGS

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


__all__ = [
    "PHASE_NAMES",
    "ROUND8_HARD_REJECTS",
    "ROUND8_IMMUTABLE_SCORING_CEILINGS",
    "ROUND8_IMMUTABLE_SCORING_WEIGHTS",
    "ROUND8_PHASE_GATE_CRITERIA",
    "BreakoutRound8FromRound2Plugin",
    "build_backtest_config",
    "load_breakout_strategy",
]

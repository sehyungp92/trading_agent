"""Breakout post-diagnostics round seeded from the promoted round_1 baseline.

This round starts from the current live breakout round_1 optimized config,
keeps the maximum common BTC/ETH/SOL window, and focuses on:

1. Re-earning the higher-alpha revalidated branch that was stripped from the
   cleaned seed.
2. Improving signal discrimination, especially body quality and weak
   symbol-direction pockets.
3. Tightening retest entry quality so added trades reflect better alpha rather
   than broader noise.
4. Capturing more of the move once in the trade.
5. Expanding trade count only through structural changes that survived
   revalidation well enough to merit re-test.

Risk-tier changes are intentionally excluded from this round so that score gains
come from real alpha extraction rather than leverage.
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
    1: "Re-Earn Revalidated Alpha",
    2: "Signal Discrimination",
    3: "Entry Architecture",
    4: "Capture & Management",
    5: "Structural Frequency",
    6: "Finetune",
}

ROUND7_IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "coverage": 0.23,
    "returns": 0.22,
    "edge": 0.15,
    "capture": 0.13,
    "entry_quality": 0.09,
    "sharpe": 0.08,
    "calmar": 0.05,
    "risk": 0.05,
}

ROUND7_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 18.0,
    "returns": 18.0,
    "edge": 3.5,
    "sharpe": 3.0,
    "calmar": 4.0,
    "risk": 10.0,
}

ROUND7_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 11.0),
    "profit_factor": (">=", 1.20),
    "max_drawdown_pct": ("<=", 10.0),
}

ROUND7_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=11.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=10.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.0),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=11.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=11.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.2),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.0),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=12.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.2),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.1),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=13.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.3),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.52),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=13.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=14.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.4),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.3),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    6: [
        GateCriterion(metric="total_trades", operator=">=", threshold=13.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=14.5),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.4),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.3),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.53),
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


def _relaxed_selective_branch() -> dict[str, Any]:
    return {
        "setup.relaxed_body_enabled": True,
        "setup.relaxed_body_min": 0.35,
        "setup.relaxed_body_min_confluences": 5,
        "setup.relaxed_body_min_room_r": 1.4,
        "setup.relaxed_body_require_volume_surge": True,
        "setup.relaxed_body_risk_scale": 0.5,
        "symbol_filter.btc_relaxed_body_direction": "both",
        "symbol_filter.eth_relaxed_body_direction": "long_only",
        "symbol_filter.sol_relaxed_body_direction": "short_only",
    }


def _phase1_candidates() -> list[Experiment]:
    full_branch = {
        "trail.trail_buffer_tight": 0.09,
        "exits.tp1_frac": 0.25,
        "balance.max_zone_age_bars": 30,
        **_relaxed_selective_branch(),
    }
    return [
        _exp("trail_tight_009", {"trail.trail_buffer_tight": 0.09}),
        _exp("tp1_frac_025", {"exits.tp1_frac": 0.25}),
        _exp("tp1_frac_020", {"exits.tp1_frac": 0.20}),
        _exp("zone_age_30", {"balance.max_zone_age_bars": 30}),
        _exp("relaxed_selective", _relaxed_selective_branch()),
        _exp(
            "branch_core_bundle",
            {
                "trail.trail_buffer_tight": 0.09,
                "exits.tp1_frac": 0.25,
                "balance.max_zone_age_bars": 30,
            },
        ),
        _exp("branch_full_bundle", full_branch),
        _exp(
            "branch_full_bundle_tp1_020",
            {
                **full_branch,
                "exits.tp1_frac": 0.20,
            },
        ),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp("body_ratio_060", {"setup.body_ratio_min": 0.60}),
        _exp("body_ratio_065", {"setup.body_ratio_min": 0.65}),
        _exp("eth_long_only", {"symbol_filter.eth_direction": "long_only"}),
        _exp("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        _exp("sol_short_only", {"symbol_filter.sol_direction": "short_only"}),
        _exp("no_countertrend", {"context.allow_countertrend": False}),
        _exp("model1_vol_110", {"confirmation.model1_min_volume_mult": 1.10}),
        _exp("model1_vol_115", {"confirmation.model1_min_volume_mult": 1.15}),
        _exp(
            "quality_stack_060_sol_disabled",
            {
                "setup.body_ratio_min": 0.60,
                "symbol_filter.eth_direction": "long_only",
                "symbol_filter.sol_direction": "disabled",
            },
        ),
        _exp(
            "quality_stack_065_sol_short",
            {
                "setup.body_ratio_min": 0.65,
                "symbol_filter.eth_direction": "long_only",
                "symbol_filter.sol_direction": "short_only",
            },
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    strict_stack = {
        "confirmation.retest_require_rejection": True,
        "confirmation.retest_require_volume_decline": True,
        "confirmation.retest_zone_atr": 0.35,
        "confirmation.retest_max_bars": 4,
    }
    return [
        _exp("retest_rejection_on", {"confirmation.retest_require_rejection": True}),
        _exp(
            "retest_volume_decline_on",
            {"confirmation.retest_require_volume_decline": True},
        ),
        _exp("retest_zone_035", {"confirmation.retest_zone_atr": 0.35}),
        _exp("retest_bars_4", {"confirmation.retest_max_bars": 4}),
        _exp("strict_retest_stack", strict_stack),
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
                **strict_stack,
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
                "entry.max_bars_after_signal": 2,
            },
        ),
        _exp("entry_ttl_2", {"entry.max_bars_after_signal": 2}),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp(
            "trail_activation_035_bars4",
            {
                "trail.trail_activation_r": 0.35,
                "trail.trail_activation_bars": 4,
            },
        ),
        _exp(
            "trail_activation_030_bars4",
            {
                "trail.trail_activation_r": 0.30,
                "trail.trail_activation_bars": 4,
            },
        ),
        _exp("trail_tight_007", {"trail.trail_buffer_tight": 0.07}),
        _exp("trail_tight_004", {"trail.trail_buffer_tight": 0.04}),
        _exp(
            "early_lock_035_0",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.35,
                "exits.early_lock_stop_r": 0.0,
            },
        ),
        _exp(
            "early_lock_045_01",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.45,
                "exits.early_lock_stop_r": 0.1,
            },
        ),
        _exp(
            "trail_activation_030_tight_007",
            {
                "trail.trail_activation_r": 0.30,
                "trail.trail_activation_bars": 4,
                "trail.trail_buffer_tight": 0.07,
            },
        ),
        _exp(
            "early_lock_035_tight_007",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.35,
                "exits.early_lock_stop_r": 0.0,
                "trail.trail_buffer_tight": 0.07,
            },
        ),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp("lookback_38", {"profile.lookback_bars": 38}),
        _exp("zone_age_32", {"balance.max_zone_age_bars": 32}),
        _exp("zone_age_36", {"balance.max_zone_age_bars": 36}),
        _exp(
            "lookback_38_zone_32",
            {
                "profile.lookback_bars": 38,
                "balance.max_zone_age_bars": 32,
            },
        ),
        _exp("reentry_cooldown_2", {"reentry.cooldown_bars": 2}),
        _exp("reentry_max_2", {"reentry.max_reentries": 2}),
    ]


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
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
}


class BreakoutRound7FromRound1Plugin(BreakoutPlugin):
    """Focused breakout phased optimizer seeded from the live round_1 baseline."""

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 14.5,
            "total_trades": 13.0,
            "profit_factor": 2.4,
            "sharpe_ratio": 2.3,
            "exit_efficiency": 0.53,
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
            focus_metrics=[
                "net_return_pct",
                "total_trades",
                "profit_factor",
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
            scoring_weights=dict(ROUND7_IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND7_HARD_REJECTS),
            min_delta=0.0025,
            max_rounds=4 if phase < 6 else 3,
            prune_threshold=0.0,
            gate_criteria=list(ROUND7_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda _m, _p=phase: list(ROUND7_PHASE_GATE_CRITERIA[_p]),
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
        ceilings = ROUND7_IMMUTABLE_SCORING_CEILINGS

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
    "ROUND7_HARD_REJECTS",
    "ROUND7_IMMUTABLE_SCORING_CEILINGS",
    "ROUND7_IMMUTABLE_SCORING_WEIGHTS",
    "ROUND7_PHASE_GATE_CRITERIA",
    "BreakoutRound7FromRound1Plugin",
    "build_backtest_config",
    "load_breakout_strategy",
]

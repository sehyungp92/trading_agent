"""Breakout canonical round 3 phased optimization from the round-2 baseline.

This canonical round 3 is intentionally narrower than canonical round 2. It
targets the three highest-signal opportunities observed in the post-round-2
validation work:

1. Better monetization of the existing winners via a smaller TP1 fraction and a
   slightly tighter adaptive trail.
2. A selective relaxed-body supplemental branch in the strongest
   symbol-direction pockets rather than broad global loosening.
3. A structural extension of balance-zone age that improved both expected
   returns and trade count when combined with the stronger entry/exit regime.

The immutable score stays focused on the user's joint objective: maximize
returns and trading frequency without letting low-sample profit-factor spikes
or pure leverage changes dominate the round.
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
    1: "Capture & Monetization",
    2: "Selective Supplemental Entries",
    3: "Structural Frequency Expansion",
    4: "Entry Architecture Controls",
    5: "Finetune",
}

ROUND6_IMMUTABLE_SCORING_WEIGHTS: dict[str, float] = {
    "coverage": 0.26,
    "returns": 0.24,
    "edge": 0.18,
    "sharpe": 0.14,
    "risk": 0.10,
    "capture": 0.08,
}

ROUND6_IMMUTABLE_SCORING_CEILINGS: dict[str, float] = {
    "coverage": 20.0,
    "returns": 20.0,
    "edge": 4.0,
    "sharpe": 3.5,
    "risk": 12.0,
}

ROUND6_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 14.0),
    "profit_factor": (">=", 1.15),
    "max_drawdown_pct": ("<=", 12.0),
}

ROUND6_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=14.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=15.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.30),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=15.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=16.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.35),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.60),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=12.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.50),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.70),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.50),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.70),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=10.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=16.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=18.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=1.75),
        GateCriterion(metric="sharpe_ratio", operator=">=", threshold=2.70),
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


def _relaxed_body_branch(
    name: str,
    *,
    btc_direction: str = "disabled",
    eth_direction: str = "disabled",
    sol_direction: str = "disabled",
    relaxed_body_min: float = 0.35,
) -> Experiment:
    return _exp(
        name,
        {
            "setup.relaxed_body_enabled": True,
            "setup.relaxed_body_min": relaxed_body_min,
            "setup.relaxed_body_min_confluences": 5,
            "setup.relaxed_body_min_room_r": 1.4,
            "setup.relaxed_body_require_volume_surge": True,
            "setup.relaxed_body_risk_scale": 0.5,
            "symbol_filter.btc_relaxed_body_direction": btc_direction,
            "symbol_filter.eth_relaxed_body_direction": eth_direction,
            "symbol_filter.sol_relaxed_body_direction": sol_direction,
        },
    )


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp("tp1_frac_025", {"exits.tp1_frac": 0.25}),
        _exp("tp1_frac_035", {"exits.tp1_frac": 0.35}),
        _exp("tp1_frac_040", {"exits.tp1_frac": 0.40}),
        _exp("trail_tight_009", {"trail.trail_buffer_tight": 0.09}),
        _exp("trail_tight_095", {"trail.trail_buffer_tight": 0.095}),
        _exp(
            "tp1_025_trail_009",
            {
                "exits.tp1_frac": 0.25,
                "trail.trail_buffer_tight": 0.09,
            },
        ),
        _exp(
            "tp1_040_trail_009",
            {
                "exits.tp1_frac": 0.40,
                "trail.trail_buffer_tight": 0.09,
            },
        ),
        _exp("quick_exit_on", {"exits.quick_exit_enabled": True}),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _relaxed_body_branch(
            "relaxed_strong_pockets",
            btc_direction="both",
            eth_direction="long_only",
            sol_direction="short_only",
        ),
        _relaxed_body_branch(
            "relaxed_btc_sol_short",
            btc_direction="both",
            sol_direction="short_only",
        ),
        _relaxed_body_branch("relaxed_btc", btc_direction="both"),
        _relaxed_body_branch("relaxed_sol_short", sol_direction="short_only"),
        _relaxed_body_branch(
            "relaxed_strong_pockets_body_040",
            btc_direction="both",
            eth_direction="long_only",
            sol_direction="short_only",
            relaxed_body_min=0.40,
        ),
    ]


def _phase3_candidates() -> list[Experiment]:
    return [
        _exp("zone_age_28", {"balance.max_zone_age_bars": 28}),
        _exp("zone_age_30", {"balance.max_zone_age_bars": 30}),
        _exp("zone_age_32", {"balance.max_zone_age_bars": 32}),
        _exp("zone_age_36", {"balance.max_zone_age_bars": 36}),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
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
            },
        ),
        _exp(
            "model2_break_entry",
            {
                "entry.model2_entry_on_close": False,
                "entry.model2_entry_on_break": True,
            },
        ),
        _exp("model2_off", {"confirmation.enable_model2": False}),
        _exp("retest_bars_4", {"confirmation.retest_max_bars": 4}),
    ]


def _phase5_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for key, val in cumulative.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if key.startswith("risk.") or "risk_scale" in key:
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
}


class BreakoutRound6PhasedPlugin(BreakoutPlugin):
    """Round-6 breakout optimizer seeded from the round-5 winner."""

    @property
    def num_phases(self) -> int:
        return 5

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 18.0,
            "total_trades": 18.0,
            "profit_factor": 2.5,
            "sharpe_ratio": 2.5,
            "exit_efficiency": 0.55,
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
            focus_metrics=["net_return_pct", "total_trades", "profit_factor", "sharpe_ratio"],
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
            scoring_weights=dict(ROUND6_IMMUTABLE_SCORING_WEIGHTS),
            hard_rejects=dict(ROUND6_HARD_REJECTS),
            min_delta=0.003,
            max_rounds=4 if phase < 5 else 3,
            prune_threshold=0.0,
            gate_criteria=list(ROUND6_PHASE_GATE_CRITERIA[phase]),
            gate_criteria_fn=lambda _m, _p=phase: list(ROUND6_PHASE_GATE_CRITERIA[_p]),
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
        ceilings = ROUND6_IMMUTABLE_SCORING_CEILINGS

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

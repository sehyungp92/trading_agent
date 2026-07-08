"""Breakout round 2 alpha-focused phased optimization.

This round is seeded from the latest parity-aligned breakout round_1 optimized
config and deliberately avoids backtest-only strategy mechanics. All
experiments are expressed as normal ``BreakoutConfig`` mutations consumed by
the shared strategy implementation, preserving live/backtest parity while
testing signal quality, entry controls, capture, and structural frequency.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.optimize.breakout_plugin import BreakoutPlugin
from crypto_trader.optimize.breakout_round3_pre_round1 import (
    SYMBOLS,
    detect_common_window,
)
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

OPTIMIZATION_END_DATE = date(2026, 4, 20)

PHASE_NAMES: dict[int, str] = {
    1: "Broad Signal Discrimination",
    2: "Variant & Direction Audit",
    3: "Entry Architecture",
    4: "Capture & Failure Handling",
    5: "Structural Frequency",
    6: "Finetune",
}

# Seven components max by design. Ceilings leave headroom above the current
# strong baseline so PF/Sharpe/return spikes do not saturate the score.
ROUND2_ALPHA_SCORING_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "coverage": 0.20,
    "expectancy": 0.16,
    "edge": 0.13,
    "capture": 0.12,
    "sharpe": 0.10,
    "risk": 0.07,
}

ROUND2_ALPHA_SCORING_CEILINGS: dict[str, float] = {
    "returns": 45.0,
    "coverage": 26.0,
    "expectancy": 1.0,
    "edge": 8.0,
    "capture": 0.85,
    "sharpe": 5.0,
    "risk": 8.0,
}

ROUND2_ALPHA_HARD_REJECTS: dict[str, tuple[str, float]] = {
    "total_trades": (">=", 12.0),
    "profit_factor": (">=", 2.0),
    "expectancy_r": (">=", 0.25),
    "exit_efficiency": (">=", 0.50),
    "max_drawdown_pct": ("<=", 8.0),
}

ROUND2_ALPHA_PHASE_GATE_CRITERIA: dict[int, list[GateCriterion]] = {
    1: [
        GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.5),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.40),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    2: [
        GateCriterion(metric="total_trades", operator=">=", threshold=12.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.5),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.42),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    3: [
        GateCriterion(metric="total_trades", operator=">=", threshold=13.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=20.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.5),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.55),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    4: [
        GateCriterion(metric="total_trades", operator=">=", threshold=13.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=21.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.6),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    5: [
        GateCriterion(metric="total_trades", operator=">=", threshold=15.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=22.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.5),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.40),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
    6: [
        GateCriterion(metric="total_trades", operator=">=", threshold=15.0),
        GateCriterion(metric="net_return_pct", operator=">=", threshold=23.0),
        GateCriterion(metric="profit_factor", operator=">=", threshold=2.6),
        GateCriterion(metric="expectancy_r", operator=">=", threshold=0.45),
        GateCriterion(metric="exit_efficiency", operator=">=", threshold=0.58),
        GateCriterion(metric="max_drawdown_pct", operator="<=", threshold=8.0),
    ],
}


def load_breakout_strategy(path: Path) -> BreakoutConfig:
    """Load a breakout config from an optimized config JSON artifact."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return BreakoutConfig.from_dict(payload["strategy"])


def build_backtest_config(data_dir: Path) -> tuple[BacktestConfig, dict[str, str]]:
    """Build a live-parity config capped before the post-2026-04-20 holdout."""
    start_dt, end_dt = detect_common_window(data_dir)
    capped_end = min(end_dt.date(), OPTIMIZATION_END_DATE)
    bt_cfg = build_backtest_config_from_profile(
        profile=LIVE_PARITY_PROFILE,
        symbols=list(SYMBOLS),
        start_date=start_dt.date(),
        end_date=capped_end,
    )
    metadata = {
        "common_start_utc": start_dt.isoformat(),
        "common_end_utc": end_dt.isoformat(),
        "start_date": bt_cfg.start_date.isoformat(),
        "end_date": bt_cfg.end_date.isoformat(),
        "holdout_excluded_after": OPTIMIZATION_END_DATE.isoformat(),
    }
    return bt_cfg, metadata


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name=name, mutations=mutations)


def _phase1_candidates() -> list[Experiment]:
    return [
        _exp("body_ratio_070", {"setup.body_ratio_min": 0.70}),
        _exp("body_ratio_075", {"setup.body_ratio_min": 0.75}),
        _exp("min_conf_b_1", {"setup.min_confluences_b": 1}),
        _exp("min_conf_b_2", {"setup.min_confluences_b": 2}),
        _exp("model1_vol_125", {"confirmation.model1_min_volume_mult": 1.25}),
        _exp("model1_vol_135", {"confirmation.model1_min_volume_mult": 1.35}),
        _exp("min_breakout_atr_025", {"setup.min_breakout_atr": 0.25}),
        _exp("min_breakout_atr_030", {"setup.min_breakout_atr": 0.30}),
        _exp("room_r_b_14", {"setup.min_room_r_b": 1.4}),
        _exp("room_r_b_16", {"setup.min_room_r_b": 1.6}),
        _exp("no_countertrend", {"context.allow_countertrend": False}),
        _exp("h4_adx_15", {"context.h4_adx_threshold": 15.0}),
        _exp(
            "quality_stack_light",
            {
                "setup.body_ratio_min": 0.70,
                "confirmation.model1_min_volume_mult": 1.25,
                "setup.min_breakout_atr": 0.25,
            },
        ),
        _exp(
            "quality_stack_strict",
            {
                "setup.body_ratio_min": 0.75,
                "setup.min_confluences_b": 1,
                "confirmation.model1_min_volume_mult": 1.25,
                "setup.min_breakout_atr": 0.25,
            },
        ),
    ]


def _phase2_candidates() -> list[Experiment]:
    return [
        _exp("relaxed_body_off", {"setup.relaxed_body_enabled": False}),
        _exp("relaxed_body_min_045", {"setup.relaxed_body_min": 0.45}),
        _exp("relaxed_body_min_050", {"setup.relaxed_body_min": 0.50}),
        _exp("relaxed_body_conf_6", {"setup.relaxed_body_min_confluences": 6}),
        _exp("relaxed_body_room_16", {"setup.relaxed_body_min_room_r": 1.6}),
        _exp(
            "relaxed_body_strict",
            {
                "setup.relaxed_body_min": 0.45,
                "setup.relaxed_body_min_confluences": 6,
                "setup.relaxed_body_min_room_r": 1.6,
            },
        ),
        _exp("eth_relaxed_long_only", {"symbol_filter.eth_relaxed_body_direction": "long_only"}),
        _exp("sol_relaxed_short_only", {"symbol_filter.sol_relaxed_body_direction": "short_only"}),
        _exp(
            "relaxed_selective_pockets",
            {
                "symbol_filter.btc_relaxed_body_direction": "both",
                "symbol_filter.eth_relaxed_body_direction": "long_only",
                "symbol_filter.sol_relaxed_body_direction": "short_only",
            },
        ),
        _exp("eth_long_only", {"symbol_filter.eth_direction": "long_only"}),
        _exp("sol_short_only", {"symbol_filter.sol_direction": "short_only"}),
        _exp("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
    ]


def _phase3_candidates() -> list[Experiment]:
    strict_retest = {
        "confirmation.retest_require_rejection": True,
        "confirmation.retest_require_volume_decline": True,
        "confirmation.retest_zone_atr": 0.35,
        "confirmation.retest_max_bars": 4,
    }
    return [
        _exp("model2_off", {"confirmation.enable_model2": False}),
        _exp("retest_rejection_on", {"confirmation.retest_require_rejection": True}),
        _exp("retest_volume_decline_on", {"confirmation.retest_require_volume_decline": True}),
        _exp("retest_zone_030", {"confirmation.retest_zone_atr": 0.30}),
        _exp("retest_zone_035", {"confirmation.retest_zone_atr": 0.35}),
        _exp("retest_bars_2", {"confirmation.retest_max_bars": 2}),
        _exp("retest_bars_4", {"confirmation.retest_max_bars": 4}),
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
        _exp("entry_ttl_2", {"entry.max_bars_after_signal": 2}),
    ]


def _phase4_candidates() -> list[Experiment]:
    return [
        _exp("quick_exit_on", {"exits.quick_exit_enabled": True}),
        _exp(
            "quick_exit_4_045_neg005",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.45,
                "exits.quick_exit_max_r": -0.05,
            },
        ),
        _exp(
            "quick_exit_5_055_000",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 5,
                "exits.quick_exit_max_mfe_r": 0.55,
                "exits.quick_exit_max_r": 0.0,
            },
        ),
        _exp(
            "early_lock_045_000",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.45,
                "exits.early_lock_stop_r": 0.0,
            },
        ),
        _exp(
            "early_lock_055_010",
            {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.55,
                "exits.early_lock_stop_r": 0.1,
            },
        ),
        _exp("trail_activation_035", {"trail.trail_activation_r": 0.35}),
        _exp("trail_activation_bars_4", {"trail.trail_activation_bars": 4}),
        _exp("trail_tight_006", {"trail.trail_buffer_tight": 0.06}),
        _exp("trail_tight_008", {"trail.trail_buffer_tight": 0.08}),
        _exp("trail_tight_010", {"trail.trail_buffer_tight": 0.10}),
        _exp("tp1_frac_010", {"exits.tp1_frac": 0.10}),
        _exp("tp1_frac_030", {"exits.tp1_frac": 0.30}),
        _exp("be_buffer_015", {"exits.be_buffer_r": 0.15}),
        _exp("be_buffer_045", {"exits.be_buffer_r": 0.45}),
        _exp(
            "failure_handling_stack",
            {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.45,
                "exits.quick_exit_max_r": -0.05,
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.55,
                "exits.early_lock_stop_r": 0.0,
            },
        ),
    ]


def _phase5_candidates() -> list[Experiment]:
    return [
        _exp("lookback_32", {"profile.lookback_bars": 32}),
        _exp("lookback_40", {"profile.lookback_bars": 40}),
        _exp("lookback_46", {"profile.lookback_bars": 46}),
        _exp("hvn_threshold_12", {"profile.hvn_threshold_pct": 1.2}),
        _exp("hvn_threshold_14", {"profile.hvn_threshold_pct": 1.4}),
        _exp("lvn_threshold_04", {"profile.lvn_threshold_pct": 0.4}),
        _exp("lvn_threshold_06", {"profile.lvn_threshold_pct": 0.6}),
        _exp("min_bars_zone_5", {"balance.min_bars_in_zone": 5}),
        _exp("min_bars_zone_8", {"balance.min_bars_in_zone": 8}),
        _exp("zone_age_30", {"balance.max_zone_age_bars": 30}),
        _exp("zone_age_42", {"balance.max_zone_age_bars": 42}),
        _exp("zone_age_48", {"balance.max_zone_age_bars": 48}),
        _exp("reentry_cooldown_2", {"reentry.cooldown_bars": 2}),
        _exp("guarded_reentry_2", {"reentry.cooldown_bars": 2, "reentry.min_confluences_override": 1}),
        _exp("max_trades_per_day_5", {"limits.max_trades_per_day": 5}),
    ]


def _phase6_candidates(cumulative: dict[str, Any]) -> list[Experiment]:
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
    5: _phase5_candidates,
}


class BreakoutRound2AlphaPlugin(BreakoutPlugin):
    """Round-2 optimizer focused on alpha extraction under parity constraints."""

    @property
    def num_phases(self) -> int:
        return 6

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {
            "net_return_pct": 30.0,
            "total_trades": 18.0,
            "expectancy_r": 0.55,
            "profit_factor": 3.0,
            "exit_efficiency": 0.60,
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
                strategy_type="breakout",
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
    "BreakoutRound2AlphaPlugin",
    "build_backtest_config",
    "load_breakout_strategy",
]

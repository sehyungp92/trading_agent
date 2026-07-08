from __future__ import annotations

from copy import deepcopy
from typing import Any

INITIAL_EQUITY = 25_000.0
ROUND_NAME = "round_1_dynamic_stock_synergy"
RISK_STANCE = "aggressive_controlled"
DEFAULT_PROFILE = "default"
BLOCKED_ALPHA_ROUND3_PROFILE = "blocked_alpha_round3"

STRATEGY_ORDER = (
    "IARIC_V5R1",
    "ALCB_R3",
)

SCORE_WEIGHTS: dict[str, float] = {
    "alpha_return": 0.27,
    "trade_frequency": 0.23,
    "drawdown_control": 0.19,
    "profit_factor_quality": 0.12,
    "synergy_capture": 0.08,
    "allocation_balance": 0.06,
    "robustness": 0.05,
}

BLOCKED_ALPHA_SCORE_WEIGHTS: dict[str, float] = {
    "alpha_return": 0.23,
    "trade_frequency": 0.20,
    "drawdown_control": 0.20,
    "profit_factor_quality": 0.12,
    "synergy_capture": 0.15,
    "allocation_balance": 0.05,
    "robustness": 0.05,
}

ROUND_TARGETS: dict[str, float] = {
    "initial_equity": INITIAL_EQUITY,
    "min_active_trades_per_month": 52.0,
    "min_total_r_per_month": 13.0,
    "min_profit_factor": 2.60,
    "target_max_drawdown_pct": 0.08,
    "hard_max_drawdown_pct": 0.10,
    "min_active_strategies": 2.0,
    "max_single_strategy_trade_share": 0.72,
    "max_single_strategy_risk_share": 0.72,
    "min_trade_capture_ratio": 0.84,
    "max_positive_alpha_block_rate": 0.18,
}

BLOCKED_ALPHA_ROUND_TARGETS: dict[str, float] = {
    **ROUND_TARGETS,
    "min_active_trades_per_month": 48.0,
    "min_total_r_per_month": 9.0,
    "min_profit_factor": 1.95,
    "target_max_drawdown_pct": 0.055,
    "hard_max_drawdown_pct": 0.060,
    "min_trade_capture_ratio": 0.83,
    "max_positive_alpha_block_rate": 0.17,
}

SEED_PORTFOLIO_CONFIG: dict[str, Any] = {
    "initial_equity": INITIAL_EQUITY,
    "risk_stance": RISK_STANCE,
    "portfolio_rules": {
        "reference_risk_pct": 0.0060,
        "heat_cap_R": 6.00,
        "max_total_active_positions": 12,
        "max_symbol_heat_R": 2.20,
        "max_long_heat_R": 6.25,
        "portfolio_daily_stop_R": 3.50,
        "portfolio_weekly_stop_R": 8.00,
        "max_single_strategy_trade_share": 0.72,
        "max_single_strategy_risk_share": 0.72,
        "drawdown_tiers": (
            (0.04, 1.00),
            (0.07, 0.75),
            (0.10, 0.40),
            (0.13, 0.00),
        ),
    },
    "strategy_allocations": {
        "IARIC_V5R1": {
            "unit_risk_pct": 0.0080,
            "max_heat_R": 4.60,
            "max_concurrent": 9,
            "daily_stop_R": 2.75,
            "priority": 0,
            "role": "primary pullback alpha and frequency engine",
        },
        "ALCB_R3": {
            "unit_risk_pct": 0.0065,
            "max_heat_R": 3.25,
            "max_concurrent": 6,
            "daily_stop_R": 2.35,
            "priority": 1,
            "role": "intraday momentum complement and early-session alpha sleeve",
        },
    },
    "dynamic_allocation": {
        "enabled": True,
        "lookback_trades": 60,
        "min_mult": 0.65,
        "max_mult": 1.22,
        "positive_expectancy_boost": 0.10,
        "negative_expectancy_cut": 0.18,
        "drawdown_risk_floor": 0.40,
    },
    "cross_strategy_rules": {
        "priority_order": list(STRATEGY_ORDER),
        "candidate_rank_mode": "diagnostic_alpha_score",
        "same_symbol_policy": "half_size",
        "same_symbol_size_mult": 0.50,
        "same_sector_heat_cap_R": 3.80,
        "iaric_priority_headroom_R": 1.15,
    },
    "strategy_filters": {
        "ALCB_R3": {
            "financials_size_mult": 0.65,
            "pdh_size_mult": 1.10,
            "score5_no_surge_mult": 0.70,
        },
        "IARIC_V5R1": {
            "delayed_confirm_min_quality": 0.0,
            "gap_up_size_mult": 0.85,
            "carry_route_size_mult": 0.75,
        },
    },
    "validation": {
        "walk_forward_required": True,
        "min_positive_slices": 4,
        "completed_bar_and_fee_parity_required": True,
    },
}

PHASE_FOCUS: dict[int, str] = {
    1: "Latest diagnostics ingestion and two-strategy baseline integrity",
    2: "Alpha/frequency capacity expansion under heat caps",
    3: "Dynamic synergistic unit-risk allocation",
    4: "Cross-strategy routing, symbol collision, and sector heat",
    5: "Blocked-candidate discrimination using live-known quality proxies",
    6: "Aggressive drawdown governors and cascade containment",
    7: "Final blend, robustness gates, and no-overfit acceptance",
}

PHASE_GATES: dict[int, dict[str, float]] = {
    1: {
        "min_active_strategies": 2.0,
        "min_active_trades_per_month": 40.0,
        "hard_max_drawdown_pct": 0.10,
    },
    2: {
        "min_active_trades_per_month": 44.0,
        "min_total_r_per_month": 12.5,
        "hard_max_drawdown_pct": 0.10,
    },
    3: {
        "min_total_r_per_month": 13.0,
        "max_single_strategy_risk_share": 0.72,
        "target_max_drawdown_pct": 0.08,
    },
    4: {
        "min_trade_capture_ratio": 0.78,
        "max_positive_alpha_block_rate": 0.24,
        "min_profit_factor": 2.55,
    },
    5: {
        "max_positive_alpha_block_rate": 0.22,
        "min_candidate_discrimination": 0.58,
        "min_trade_capture_ratio": 0.78,
    },
    6: {
        "target_max_drawdown_pct": 0.08,
        "max_daily_loss_R": 3.50,
        "max_weekly_loss_R": 8.00,
    },
    7: {
        "min_active_strategies": 2.0,
        "min_active_trades_per_month": 52.0,
        "min_total_r_per_month": 13.0,
        "min_profit_factor": 2.60,
        "hard_max_drawdown_pct": 0.10,
        "max_single_strategy_trade_share": 0.72,
        "max_single_strategy_risk_share": 0.72,
        "min_trade_capture_ratio": 0.82,
        "max_positive_alpha_block_rate": 0.18,
        "min_candidate_discrimination": 0.58,
        "min_positive_slices": 4.0,
    },
}


def get_phase_candidates(phase: int, profile: str = DEFAULT_PROFILE) -> list[dict[str, Any]]:
    candidates = get_profile_phase_candidates(phase, profile=profile)
    return deepcopy(candidates)


def get_profile_phase_candidates(phase: int, profile: str = DEFAULT_PROFILE) -> list[dict[str, Any]]:
    candidate_map = _profile_value(profile, _PHASE_CANDIDATES, _BLOCKED_ALPHA_PHASE_CANDIDATES)
    candidates = candidate_map.get(phase)
    if candidates is None:
        raise ValueError(f"Unknown stock portfolio synergy phase {phase} for profile {profile!r}")
    return deepcopy(candidates)


def get_phase_focus(phase: int, profile: str = DEFAULT_PROFILE) -> str:
    focus_map = _profile_value(profile, PHASE_FOCUS, _BLOCKED_ALPHA_PHASE_FOCUS)
    return focus_map[phase]


def get_phase_gates(phase: int, profile: str = DEFAULT_PROFILE) -> dict[str, float]:
    gate_map = _profile_value(profile, PHASE_GATES, _BLOCKED_ALPHA_PHASE_GATES)
    return dict(gate_map[phase])


def get_score_weights(profile: str = DEFAULT_PROFILE) -> dict[str, float]:
    if profile == DEFAULT_PROFILE:
        return dict(SCORE_WEIGHTS)
    if profile == BLOCKED_ALPHA_ROUND3_PROFILE:
        return dict(BLOCKED_ALPHA_SCORE_WEIGHTS)
    raise ValueError(f"Unknown stock portfolio synergy profile: {profile!r}")


def get_round_targets(profile: str = DEFAULT_PROFILE) -> dict[str, float]:
    if profile == DEFAULT_PROFILE:
        return dict(ROUND_TARGETS)
    if profile == BLOCKED_ALPHA_ROUND3_PROFILE:
        return dict(BLOCKED_ALPHA_ROUND_TARGETS)
    raise ValueError(f"Unknown stock portfolio synergy profile: {profile!r}")


def phase_summary(profile: str = DEFAULT_PROFILE) -> list[dict[str, Any]]:
    focus_map = _profile_value(profile, PHASE_FOCUS, _BLOCKED_ALPHA_PHASE_FOCUS)
    gate_map = _profile_value(profile, PHASE_GATES, _BLOCKED_ALPHA_PHASE_GATES)
    candidate_map = _profile_value(profile, _PHASE_CANDIDATES, _BLOCKED_ALPHA_PHASE_CANDIDATES)
    return [
        {
            "phase": phase,
            "focus": focus_map[phase],
            "gate": gate_map[phase],
            "candidate_count": len(candidate_map[phase]),
            "candidate_names": [candidate["name"] for candidate in candidate_map[phase]],
        }
        for phase in sorted(focus_map)
    ]


def _profile_value(profile: str, default_value, blocked_alpha_value):
    if profile == DEFAULT_PROFILE:
        return default_value
    if profile == BLOCKED_ALPHA_ROUND3_PROFILE:
        return blocked_alpha_value
    raise ValueError(f"Unknown stock portfolio synergy profile: {profile!r}")


_PHASE_CANDIDATES: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "name": "seed_controlled_aggressive_25k",
            "mutations": SEED_PORTFOLIO_CONFIG,
            "rationale": "Use the latest ALCB R3 and IARIC V5R1 outputs together with $25k starting equity.",
        },
        {
            "name": "seed_iaric_primary_headroom",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.priority": 0,
                "strategy_allocations.ALCB_R3.priority": 1,
                "cross_strategy_rules.iaric_priority_headroom_R": 1.35,
            },
            "rationale": "IARIC has the cleanest latest PF/DD/frequency profile and should reserve first capacity.",
        },
        {
            "name": "seed_balanced_no_monopoly",
            "mutations": {
                "portfolio_rules.max_single_strategy_trade_share": 0.68,
                "portfolio_rules.max_single_strategy_risk_share": 0.68,
            },
            "rationale": "Control probe to keep the two-strategy blend from collapsing into IARIC-only throughput.",
        },
    ],
    2: [
        {
            "name": "positions_13_capacity",
            "mutations": {"portfolio_rules.max_total_active_positions": 13},
            "rationale": "Permit overlap between IARIC pullbacks and ALCB early breakouts without removing all scarcity.",
        },
        {
            "name": "positions_14_probe",
            "mutations": {"portfolio_rules.max_total_active_positions": 14},
            "rationale": "Upper frequency probe; must earn its keep through capture rate and DD gates.",
        },
        {
            "name": "heat_cap_6_5",
            "mutations": {"portfolio_rules.heat_cap_R": 6.50},
            "rationale": "Controlled-aggressive heat expansion above seed while still below unconstrained demand.",
        },
        {
            "name": "heat_cap_7_0_probe",
            "mutations": {"portfolio_rules.heat_cap_R": 7.00},
            "rationale": "Probe maximum alpha extraction only if blocked positive-alpha and drawdown remain acceptable.",
        },
        {
            "name": "iaric_max_concurrent_10",
            "mutations": {"strategy_allocations.IARIC_V5R1.max_concurrent": 10},
            "rationale": "IARIC V5R1 proved it can carry 1015 trades at 2.2% DD; test restoring full slot count.",
        },
        {
            "name": "strategy_heat_5_4_4_0",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.max_heat_R": 5.40,
                "strategy_allocations.ALCB_R3.max_heat_R": 4.00,
            },
            "rationale": "Directly target the dominant strategy-heat blocker without lifting portfolio DD limits.",
        },
        {
            "name": "share_cap_85_actual_balance_guard",
            "mutations": {
                "portfolio_rules.max_single_strategy_trade_share": 0.85,
                "portfolio_rules.max_single_strategy_risk_share": 0.82,
            },
            "rationale": "Relax early share throttling while final gates still require the realized blend to stay balanced.",
        },
    ],
    3: [
        {
            "name": "risk_iaric_090",
            "mutations": {"strategy_allocations.IARIC_V5R1.unit_risk_pct": 0.0090},
            "rationale": "Lean into IARIC's high PF and clean walk-forward with a modest risk overweight.",
        },
        {
            "name": "risk_alcb_0725",
            "mutations": {"strategy_allocations.ALCB_R3.unit_risk_pct": 0.00725},
            "rationale": "ALCB has higher latest avg R than IARIC, so test a restrained size lift.",
        },
        {
            "name": "risk_balanced_080_070",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.unit_risk_pct": 0.0080,
                "strategy_allocations.ALCB_R3.unit_risk_pct": 0.0070,
            },
            "rationale": "Balanced controlled-aggressive risk if ALCB complements IARIC without DD expansion.",
        },
        {
            "name": "dynamic_allocation_faster",
            "mutations": {
                "dynamic_allocation.lookback_trades": 40,
                "dynamic_allocation.positive_expectancy_boost": 0.14,
                "dynamic_allocation.negative_expectancy_cut": 0.22,
            },
            "rationale": "Let recent realized edge shift risk faster while keeping min/max multipliers bounded.",
        },
        {
            "name": "dynamic_allocation_tighter_bounds",
            "mutations": {
                "dynamic_allocation.min_mult": 0.70,
                "dynamic_allocation.max_mult": 1.15,
            },
            "rationale": "Control overreaction risk if faster dynamic allocation increases churn or DD.",
        },
    ],
    4: [
        {
            "name": "same_symbol_half_size_keep_frequency",
            "mutations": {
                "cross_strategy_rules.same_symbol_policy": "half_size",
                "cross_strategy_rules.same_symbol_size_mult": 0.50,
            },
            "rationale": "Preserve both signals when they agree on a ticker, but avoid double full-size heat.",
        },
        {
            "name": "same_symbol_best_rank_only",
            "mutations": {"cross_strategy_rules.same_symbol_policy": "best_rank_only"},
            "rationale": "Test whether conflict pruning beats half-size overlap when the same name appears in both sleeves.",
        },
        {
            "name": "sector_heat_3_5",
            "mutations": {"cross_strategy_rules.same_sector_heat_cap_R": 3.50},
            "rationale": "Technology dominates both strategies; cap sector crowding before it becomes hidden beta.",
        },
        {
            "name": "sector_heat_5_0_capacity",
            "mutations": {"cross_strategy_rules.same_sector_heat_cap_R": 5.00},
            "rationale": "Probe whether the sector cap is blocking high-quality overlap more than it is reducing beta.",
        },
        {
            "name": "rank_expected_alpha_per_heat",
            "mutations": {"cross_strategy_rules.candidate_rank_mode": "expected_alpha_per_heat"},
            "rationale": "When capacity is scarce, prefer live-known quality per unit heat rather than raw priority.",
        },
        {
            "name": "rank_frequency_first_control",
            "mutations": {"cross_strategy_rules.candidate_rank_mode": "frequency_first"},
            "rationale": "Control probe to ensure the alpha-ranking layer is not merely overfitting sparse wins.",
        },
    ],
    5: [
        {
            "name": "alcb_pdh_priority",
            "mutations": {"strategy_filters.ALCB_R3.pdh_size_mult": 1.20},
            "rationale": "ALCB PDH breakout has the strongest latest entry-type avg R but low frequency.",
        },
        {
            "name": "alcb_score5_no_surge_dampen",
            "mutations": {"strategy_filters.ALCB_R3.score5_no_surge_mult": 0.55},
            "rationale": "Target ALCB's weak score monotonicity without blocking whole entry families.",
        },
        {
            "name": "alcb_score45_no_surge_dampen",
            "mutations": {"strategy_filters.ALCB_R3.score5_no_surge_mult": 0.45},
            "rationale": "More selective ALCB score/no-surge sizing probe that improved PF and blocked-alpha capture in diagnostics.",
        },
        {
            "name": "iaric_gap_up_dampen",
            "mutations": {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.70},
            "rationale": "IARIC gap-down pullbacks dominate gap-up expectancy; damp gap-up entries instead of hard blocking.",
        },
        {
            "name": "iaric_gap_up_60_dampen",
            "mutations": {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.60},
            "rationale": "Stronger IARIC gap-up dampening probe for reducing positive-alpha blocks without deleting the route.",
        },
        {
            "name": "blocked_alpha_ranker",
            "mutations": {"cross_strategy_rules.candidate_rank_mode": "diagnostic_alpha_score"},
            "rationale": "Use diagnostic live-known features to reduce positive-alpha blocks and avoid hindsight ranking.",
        },
    ],
    6: [
        {
            "name": "daily_stop_3_25",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 3.25},
            "rationale": "Slightly tighter daily cascade control for an otherwise aggressive risk stance.",
        },
        {
            "name": "daily_stop_3_75_probe",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 3.75},
            "rationale": "Probe looser daily loss tolerance only if frequency and return improve materially.",
        },
        {
            "name": "drawdown_tiers_aggressive_controlled",
            "mutations": {
                "portfolio_rules.drawdown_tiers": (
                    (0.04, 1.00),
                    (0.07, 0.75),
                    (0.10, 0.40),
                    (0.13, 0.00),
                )
            },
            "rationale": "Press near equity highs, then force de-risking before max DD can snowball.",
        },
        {
            "name": "weekly_stop_7R",
            "mutations": {"portfolio_rules.portfolio_weekly_stop_R": 7.00},
            "rationale": "Reduce clustered weekly damage while leaving ordinary high-frequency days alone.",
        },
    ],
    7: [
        {
            "name": "final_share_cap_70",
            "mutations": {
                "portfolio_rules.max_single_strategy_trade_share": 0.70,
                "portfolio_rules.max_single_strategy_risk_share": 0.70,
            },
            "rationale": "Final book remains synergistic only if neither sleeve monopolizes trades or risk.",
        },
        {
            "name": "final_walk_forward_required",
            "mutations": {
                "validation.walk_forward_required": True,
                "validation.min_positive_slices": 4,
            },
            "rationale": "Reject blends whose return comes from one narrow time slice.",
        },
        {
            "name": "final_live_parity_required",
            "mutations": {"validation.completed_bar_and_fee_parity_required": True},
            "rationale": "No adoption if the portfolio alpha depends on non-causal bars or optimistic costs.",
        },
    ],
}

_BLOCKED_ALPHA_PHASE_FOCUS: dict[int, str] = {
    1: "Phase 2 + Phase 5 blend integrity under a 6% MTM drawdown cap",
    2: "Capacity relief for profitable blocked candidates",
    3: "Risk-scaled capture without drawdown expansion",
    4: "Scarce-capacity ranking, symbol, and sector routing",
    5: "Quality dampeners that preserve capture while rejecting weak cohorts",
    6: "Drawdown containment for the higher-capture blend",
    7: "Final blocked-alpha guardrail validation",
}

_BLOCKED_ALPHA_PHASE_GATES: dict[int, dict[str, float]] = {
    1: {
        "min_active_strategies": 2.0,
        "min_trade_capture_ratio": 0.82,
        "max_positive_alpha_block_rate": 0.18,
        "hard_max_drawdown_pct": 0.060,
    },
    2: {
        "min_trade_capture_ratio": 0.84,
        "max_positive_alpha_block_rate": 0.17,
        "hard_max_drawdown_pct": 0.060,
    },
    3: {
        "min_total_r_per_month": 9.0,
        "max_single_strategy_risk_share": 0.72,
        "hard_max_drawdown_pct": 0.060,
    },
    4: {
        "min_trade_capture_ratio": 0.84,
        "max_positive_alpha_block_rate": 0.17,
        "min_candidate_discrimination": 0.57,
        "hard_max_drawdown_pct": 0.060,
    },
    5: {
        "max_positive_alpha_block_rate": 0.17,
        "min_candidate_discrimination": 0.57,
        "min_profit_factor": 1.95,
        "hard_max_drawdown_pct": 0.060,
    },
    6: {
        "target_max_drawdown_pct": 0.055,
        "max_daily_loss_R": 3.50,
        "max_weekly_loss_R": 6.00,
        "hard_max_drawdown_pct": 0.060,
    },
    7: {
        "min_active_strategies": 2.0,
        "min_active_trades_per_month": 48.0,
        "min_total_r_per_month": 9.0,
        "min_profit_factor": 1.95,
        "hard_max_drawdown_pct": 0.060,
        "max_single_strategy_trade_share": 0.72,
        "max_single_strategy_risk_share": 0.72,
        "min_trade_capture_ratio": 0.83,
        "max_positive_alpha_block_rate": 0.17,
        "min_candidate_discrimination": 0.57,
        "min_positive_slices": 4.0,
    },
}

_BLOCKED_ALPHA_PHASE_CANDIDATES: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "name": "round3_seed_blend_noop",
            "mutations": {},
            "rationale": "Validate the Phase 2 capacity plus Phase 5 quality blend before adding new risk.",
        },
    ],
    2: [
        {
            "name": "strategy_heat_5_7_4_2",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.max_heat_R": 5.70,
                "strategy_allocations.ALCB_R3.max_heat_R": 4.20,
            },
            "rationale": "Target the largest remaining profitable blocker while staying below the 6% MTM DD cap.",
        },
        {
            "name": "strategy_heat_6_0_4_4",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.max_heat_R": 6.00,
                "strategy_allocations.ALCB_R3.max_heat_R": 4.40,
            },
            "rationale": "Upper strategy-heat probe; hard rejected if drawdown expands past the round cap.",
        },
        {
            "name": "portfolio_heat_6_8",
            "mutations": {"portfolio_rules.heat_cap_R": 6.80},
            "rationale": "Relieve portfolio heat only modestly because the seed already captures most entries.",
        },
        {
            "name": "trade_share_88_guarded",
            "mutations": {
                "portfolio_rules.max_single_strategy_trade_share": 0.88,
                "portfolio_rules.max_single_strategy_risk_share": 0.82,
            },
            "rationale": "Reduce early trade-share blocks while preserving the final realized balance guard.",
        },
        {
            "name": "sector_heat_4_4_capacity",
            "mutations": {"cross_strategy_rules.same_sector_heat_cap_R": 4.40},
            "rationale": "Test whether remaining sector scarcity is blocking positive R more than reducing beta.",
        },
        {
            "name": "symbol_heat_2_6_capacity",
            "mutations": {"portfolio_rules.max_symbol_heat_R": 2.60},
            "rationale": "Allow limited same-name conviction when both sleeves agree without lifting full portfolio heat.",
        },
    ],
    3: [
        {
            "name": "risk_slightly_lower_more_slots",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.unit_risk_pct": 0.0075,
                "strategy_allocations.ALCB_R3.unit_risk_pct": 0.0061,
            },
            "rationale": "Lower per-trade heat to admit more candidates while testing whether breadth offsets smaller size.",
        },
        {
            "name": "risk_alcb_0068_iaric_0078",
            "mutations": {
                "strategy_allocations.IARIC_V5R1.unit_risk_pct": 0.0078,
                "strategy_allocations.ALCB_R3.unit_risk_pct": 0.0068,
            },
            "rationale": "Slightly rebalance risk toward the complementary ALCB sleeve without exceeding heat caps.",
        },
        {
            "name": "dynamic_faster_defensive",
            "mutations": {
                "dynamic_allocation.lookback_trades": 40,
                "dynamic_allocation.positive_expectancy_boost": 0.12,
                "dynamic_allocation.negative_expectancy_cut": 0.24,
                "dynamic_allocation.max_mult": 1.18,
            },
            "rationale": "Let recent edge respond faster while making the downside cut larger than the upside boost.",
        },
        {
            "name": "dynamic_tighter_bounds",
            "mutations": {
                "dynamic_allocation.min_mult": 0.70,
                "dynamic_allocation.max_mult": 1.14,
            },
            "rationale": "Control overreaction risk if dynamic sizing is degrading drawdown or blocked-candidate quality.",
        },
    ],
    4: [
        {
            "name": "rank_expected_alpha_per_heat",
            "mutations": {"cross_strategy_rules.candidate_rank_mode": "expected_alpha_per_heat"},
            "rationale": "Rank scarce slots by live-known quality per heat unit rather than raw priority.",
        },
        {
            "name": "rank_frequency_first_control",
            "mutations": {"cross_strategy_rules.candidate_rank_mode": "frequency_first"},
            "rationale": "Control probe to ensure the alpha-ranking layer is not merely overfitting quality labels.",
        },
        {
            "name": "same_symbol_best_rank_only",
            "mutations": {"cross_strategy_rules.same_symbol_policy": "best_rank_only"},
            "rationale": "Block lower-ranked same-symbol overlap if it removes more losers than winners.",
        },
        {
            "name": "same_symbol_35_size",
            "mutations": {
                "cross_strategy_rules.same_symbol_policy": "half_size",
                "cross_strategy_rules.same_symbol_size_mult": 0.35,
            },
            "rationale": "Keep overlap but reduce same-name heat when profitable blocks are mostly capacity-driven.",
        },
        {
            "name": "iaric_priority_headroom_1_35",
            "mutations": {"cross_strategy_rules.iaric_priority_headroom_R": 1.35},
            "rationale": "Give IARIC modest first-right capacity when it competes with ALCB in crowded sessions.",
        },
    ],
    5: [
        {
            "name": "alcb_score5_no_surge_45",
            "mutations": {"strategy_filters.ALCB_R3.score5_no_surge_mult": 0.45},
            "rationale": "Stronger ALCB weak-score dampener; must improve blocked-alpha quality without killing capture.",
        },
        {
            "name": "iaric_gap_up_50_dampen",
            "mutations": {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.50},
            "rationale": "Further damp IARIC gap-up entries if they consume heat that blocks higher-quality trades.",
        },
        {
            "name": "iaric_gap_up_70_less_dampen",
            "mutations": {"strategy_filters.IARIC_V5R1.gap_up_size_mult": 0.70},
            "rationale": "Undo some gap-up dampening if the stronger seed is over-pruning profitable candidates.",
        },
        {
            "name": "alcb_financials_55",
            "mutations": {"strategy_filters.ALCB_R3.financials_size_mult": 0.55},
            "rationale": "Test whether a known weaker ALCB sector is still consuming scarce portfolio heat.",
        },
        {
            "name": "pdh_125_score5_45",
            "mutations": {
                "strategy_filters.ALCB_R3.pdh_size_mult": 1.25,
                "strategy_filters.ALCB_R3.score5_no_surge_mult": 0.45,
            },
            "rationale": "Shift ALCB risk toward stronger PDH breakouts while dampening weaker score-5/no-surge setups.",
        },
    ],
    6: [
        {
            "name": "drawdown_tiers_5_6_guard",
            "mutations": {
                "portfolio_rules.drawdown_tiers": (
                    (0.035, 1.00),
                    (0.050, 0.70),
                    (0.060, 0.45),
                    (0.075, 0.00),
                )
            },
            "rationale": "Start de-risking before the round's 6% MTM drawdown ceiling is reached.",
        },
        {
            "name": "daily_stop_3_0",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 3.00},
            "rationale": "Tighter daily loss guard for the higher-capture blend.",
        },
        {
            "name": "weekly_stop_6R",
            "mutations": {"portfolio_rules.portfolio_weekly_stop_R": 6.00},
            "rationale": "Reduce clustered weekly damage while keeping the portfolio aggressive near highs.",
        },
        {
            "name": "portfolio_heat_6_4_guard",
            "mutations": {"portfolio_rules.heat_cap_R": 6.40},
            "rationale": "Back off heat slightly if earlier capacity lifts increase drawdown too much.",
        },
    ],
    7: [
        {
            "name": "final_noop_guard",
            "mutations": {},
            "rationale": "Validate the final blend against blocked-alpha and 6% MTM drawdown gates.",
        },
        {
            "name": "final_trade_share_86",
            "mutations": {
                "portfolio_rules.max_single_strategy_trade_share": 0.86,
                "portfolio_rules.max_single_strategy_risk_share": 0.80,
            },
            "rationale": "Slightly tighten final share caps if the blend captured alpha by over-concentrating one sleeve.",
        },
        {
            "name": "final_dd_tiers_5_6_guard",
            "mutations": {
                "portfolio_rules.drawdown_tiers": (
                    (0.035, 1.00),
                    (0.050, 0.70),
                    (0.060, 0.45),
                    (0.075, 0.00),
                )
            },
            "rationale": "Final protection against accepting a high-capture blend with unstable drawdown behavior.",
        },
    ],
}

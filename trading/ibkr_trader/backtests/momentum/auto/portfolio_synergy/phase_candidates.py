from __future__ import annotations

from copy import deepcopy
from typing import Any

INITIAL_EQUITY = 50_000.0
ROUND_NAME = "round_1_alpha_frequency_synergy"
RISK_STANCE = "aggressive_controlled"

STRATEGY_ORDER = (
    "NQ_REGIME",
    "VdubusNQ_v4",
    "NQDTC_v2.1",
    "DownturnDominator_v1",
)

SCORE_WEIGHTS: dict[str, float] = {
    "alpha_return": 0.28,
    "trade_frequency": 0.22,
    "drawdown_control": 0.18,
    "profit_factor_quality": 0.12,
    "strategy_coverage": 0.10,
    "capital_efficiency": 0.06,
    "robustness": 0.04,
}

ROUND_TARGETS: dict[str, float] = {
    "initial_equity": INITIAL_EQUITY,
    "min_trades_per_month": 22.0,
    "min_total_r_per_month": 14.0,
    "min_profit_factor": 2.4,
    "max_drawdown_pct": 0.18,
    "min_active_strategies": 4.0,
    "max_single_strategy_risk_share": 0.40,
}

SEED_PORTFOLIO_CONFIG: dict[str, Any] = {
    "initial_equity": INITIAL_EQUITY,
    "risk_stance": RISK_STANCE,
    "portfolio_rules": {
        "heat_cap_R": 4.0,
        "max_total_positions": 4,
        "directional_cap_R": 3.5,
        "directional_cap_long_R": 3.25,
        "directional_cap_short_R": 4.0,
        "portfolio_daily_stop_R": 2.25,
        "portfolio_weekly_stop_R": 6.0,
        "priority_headroom_R": 1.0,
        "priority_reserve_threshold": 1,
        "reference_unit_risk_dollars": 250.0,
        "target_leverage": 12.0,
        "drawdown_tiers": (
            (0.08, 1.00),
            (0.12, 0.65),
            (0.16, 0.35),
            (0.20, 0.00),
        ),
    },
    "strategy_allocations": {
        "NQ_REGIME": {
            "base_risk_pct": 0.0065,
            "daily_stop_R": 2.25,
            "max_concurrent": 2,
            "priority": 0,
            "role": "primary alpha and frequency engine",
        },
        "VdubusNQ_v4": {
            "base_risk_pct": 0.0045,
            "daily_stop_R": 2.25,
            "max_concurrent": 1,
            "priority": 1,
            "role": "high-frequency VWAP/failure complement",
        },
        "NQDTC_v2.1": {
            "base_risk_pct": 0.0035,
            "daily_stop_R": 2.0,
            "max_concurrent": 1,
            "priority": 2,
            "continuation_size_mult": 0.70,
            "role": "range and directional confirmation engine",
        },
        "DownturnDominator_v1": {
            "base_risk_pct": 0.0040,
            "daily_stop_R": 2.0,
            "max_concurrent": 1,
            "priority": 3,
            "role": "correction and range ballast",
        },
    },
    "cross_strategy_rules": {
        "priority_order": list(STRATEGY_ORDER),
        "nq_regime_candidate_led": True,
        "vdubus_nqdtc_direction_filter": {
            "enabled": True,
            "agree_size_mult": 1.25,
            "oppose_size_mult": 0.50,
        },
        "downturn_regime_reserve": {
            "enabled": True,
            "correction_priority": 1,
            "neutral_range_priority": 3,
        },
    },
}

PHASE_FOCUS: dict[int, str] = {
    1: "Baseline four-strategy inclusion and diagnostic integrity",
    2: "Alpha/frequency capacity expansion",
    3: "Conviction-weighted risk allocation",
    4: "Cross-strategy routing and conflict rules",
    5: "Regime/session specialization",
    6: "Aggressive risk governors and drawdown containment",
    7: "Final blend, contribution balance, and robustness gates",
}

PHASE_GATES: dict[int, dict[str, float]] = {
    1: {
        "min_active_strategies": 4.0,
        "min_trades_per_month": 18.0,
        "max_drawdown_pct": 0.20,
    },
    2: {
        "min_trades_per_month": 22.0,
        "min_total_r_per_month": 13.0,
        "max_drawdown_pct": 0.20,
    },
    3: {
        "min_total_r_per_month": 14.0,
        "max_single_strategy_risk_share": 0.40,
        "max_drawdown_pct": 0.19,
    },
    4: {
        "min_trade_capture_ratio": 0.82,
        "max_positive_alpha_block_rate": 0.20,
        "min_profit_factor": 2.35,
    },
    5: {
        "min_active_strategies": 4.0,
        "min_downturn_correction_trade_share": 0.08,
    },
    6: {
        "max_drawdown_pct": 0.18,
        "max_daily_loss_R": 2.25,
        "max_weekly_loss_R": 6.0,
    },
    7: {
        "min_trades_per_month": 22.0,
        "min_total_r_per_month": 14.0,
        "min_profit_factor": 2.4,
        "max_drawdown_pct": 0.18,
        "min_positive_years": 3.0,
    },
}


def get_phase_candidates(phase: int) -> list[dict[str, Any]]:
    candidates = _PHASE_CANDIDATES.get(phase)
    if candidates is None:
        raise ValueError(f"Unknown portfolio synergy phase: {phase}")
    return deepcopy(candidates)


def phase_summary() -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "focus": PHASE_FOCUS[phase],
            "gate": PHASE_GATES[phase],
            "candidate_count": len(_PHASE_CANDIDATES[phase]),
            "candidate_names": [candidate["name"] for candidate in _PHASE_CANDIDATES[phase]],
        }
        for phase in sorted(PHASE_FOCUS)
    ]


_PHASE_CANDIDATES: dict[int, list[dict[str, Any]]] = {
    1: [
        {
            "name": "seed_aggressive_controlled_50k",
            "mutations": SEED_PORTFOLIO_CONFIG,
            "rationale": "Use all active strategies with NQ_REGIME/Vdubus as the frequency core.",
        },
        {
            "name": "seed_nq_regime_heavy",
            "mutations": {
                "strategy_allocations.NQ_REGIME.base_risk_pct": 0.0075,
                "strategy_allocations.NQ_REGIME.max_concurrent": 2,
            },
            "rationale": "Lean harder into the cleanest latest diagnostic: 497 trades, 10.47 trades/month, 3.2% DD.",
        },
        {
            "name": "seed_frequency_balanced",
            "mutations": {
                "strategy_allocations.VdubusNQ_v4.base_risk_pct": 0.0050,
                "strategy_allocations.NQDTC_v2.1.base_risk_pct": 0.0040,
                "portfolio_rules.max_total_positions": 5,
            },
            "rationale": "Lift Vdubus/NQDTC throughput while preserving active strategy participation.",
        },
    ],
    2: [
        {
            "name": "capacity_positions_5",
            "mutations": {"portfolio_rules.max_total_positions": 5},
            "rationale": "Allow one slot per active strategy before conflict rules start rationing alpha.",
        },
        {
            "name": "capacity_positions_6_probe",
            "mutations": {"portfolio_rules.max_total_positions": 6},
            "rationale": "Probe whether NQ_REGIME can run two independent modules without crowding the rest.",
        },
        {
            "name": "heat_cap_4_0",
            "mutations": {"portfolio_rules.heat_cap_R": 4.0},
            "rationale": "Aggressive enough to increase frequency, still capped below the sum of per-strategy desire.",
        },
        {
            "name": "heat_cap_4_5_probe",
            "mutations": {"portfolio_rules.heat_cap_R": 4.5},
            "rationale": "Upper-bound test for alpha extraction, accepted only if DD and positive-alpha blocking stay controlled.",
        },
        {
            "name": "directional_asym_short_room",
            "mutations": {
                "portfolio_rules.directional_cap_long_R": 3.25,
                "portfolio_rules.directional_cap_short_R": 4.25,
            },
            "rationale": "Give Downturn/NQDTC shorts more room because they diversify long Nasdaq momentum risk.",
        },
    ],
    3: [
        {
            "name": "risk_nq_regime_075",
            "mutations": {"strategy_allocations.NQ_REGIME.base_risk_pct": 0.0075},
            "rationale": "NQ_REGIME has the strongest PF/DD/frequency profile and should be the first overweight probe.",
        },
        {
            "name": "risk_vdubus_055",
            "mutations": {"strategy_allocations.VdubusNQ_v4.base_risk_pct": 0.0055},
            "rationale": "Vdubus carries the best legacy return engine but needs fast-death gates to earn this size.",
        },
        {
            "name": "risk_downturn_045",
            "mutations": {"strategy_allocations.DownturnDominator_v1.base_risk_pct": 0.0045},
            "rationale": "Downturn has low DD and correction PnL, so test modestly higher ballast size.",
        },
        {
            "name": "risk_no_single_strategy_dominance",
            "mutations": {"portfolio_rules.max_single_strategy_risk_share": 0.40},
            "rationale": "Force the optimizer to keep the round synergistic instead of collapsing into one engine.",
        },
    ],
    4: [
        {
            "name": "nq_regime_candidate_led_priority",
            "mutations": {
                "cross_strategy_rules.nq_regime_candidate_led": True,
                "portfolio_rules.priority_headroom_R": 1.25,
            },
            "rationale": "Let NQ_REGIME reserve headroom because it has the cleanest frequency and DD profile.",
        },
        {
            "name": "vdubus_nqdtc_agree_boost",
            "mutations": {
                "cross_strategy_rules.vdubus_nqdtc_direction_filter.enabled": True,
                "cross_strategy_rules.vdubus_nqdtc_direction_filter.agree_size_mult": 1.25,
                "cross_strategy_rules.vdubus_nqdtc_direction_filter.oppose_size_mult": 0.50,
            },
            "rationale": "Use NQDTC as confirmation without fully blocking Vdubus frequency on opposed reads.",
        },
        {
            "name": "downturn_correction_priority",
            "mutations": {
                "cross_strategy_rules.downturn_regime_reserve.enabled": True,
                "cross_strategy_rules.downturn_regime_reserve.correction_priority": 1,
            },
            "rationale": "Promote Downturn only when its correction/range edge is most likely to offset long-momentum risk.",
        },
    ],
    5: [
        {
            "name": "nq_regime_transition_throttle",
            "mutations": {"regime_rules.TRANSITION.max_new_positions": 2},
            "rationale": "NQ_REGIME selected only 5.0% of decisions; transition regimes need capacity, but not unlimited capacity.",
        },
        {
            "name": "vdubus_fast_death_suppressor",
            "mutations": {"strategy_filters.VdubusNQ_v4.fast_death_guard": True},
            "rationale": "Vdubus 1-4 bar trades had 2% WR and negative PnL; suppress them before granting more size.",
        },
        {
            "name": "downturn_neutral_range_only",
            "mutations": {"strategy_filters.DownturnDominator_v1.allowed_regimes": ("neutral", "range", "correction")},
            "rationale": "Latest Downturn alpha came from correction/range, while aligned/emerging bear buckets lost money.",
        },
    ],
    6: [
        {
            "name": "portfolio_daily_stop_225",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 2.25},
            "rationale": "Aggressive but not reckless; gives NQ_REGIME/Vdubus room while clipping cascade days.",
        },
        {
            "name": "portfolio_daily_stop_250_probe",
            "mutations": {"portfolio_rules.portfolio_daily_stop_R": 2.50},
            "rationale": "Probe looser daily loss tolerance only if alpha/frequency gains are material.",
        },
        {
            "name": "drawdown_tiers_controlled_aggressive",
            "mutations": {
                "portfolio_rules.drawdown_tiers": (
                    (0.08, 1.00),
                    (0.12, 0.65),
                    (0.16, 0.35),
                    (0.20, 0.00),
                )
            },
            "rationale": "Let the book press at highs but force de-risking before drawdown accelerates.",
        },
        {
            "name": "weekly_stop_6R",
            "mutations": {"portfolio_rules.portfolio_weekly_stop_R": 6.0},
            "rationale": "Control clustered losing weeks without choking ordinary two-loss days.",
        },
    ],
    7: [
        {
            "name": "final_contribution_cap_40pct",
            "mutations": {"portfolio_rules.max_single_strategy_risk_share": 0.40},
            "rationale": "A synergy round fails if one strategy contributes nearly all selected risk.",
        },
        {
            "name": "final_min_all_strategies_active",
            "mutations": {"portfolio_rules.min_active_strategies": 4},
            "rationale": "Require all four momentum engines to survive the blend unless diagnostics prove one is harmful.",
        },
        {
            "name": "final_walk_forward_robustness_gate",
            "mutations": {"validation.walk_forward_required": True, "validation.min_positive_years": 3},
            "rationale": "Frequency and return are not enough; the final blend must avoid one-period overfit.",
        },
        {
            "name": "final_live_parity_gate",
            "mutations": {"validation.completed_bar_and_fee_parity_required": True},
            "rationale": "Reject candidates whose alpha depends on unavailable context, optimistic fills, or friction drift.",
        },
    ],
}
